"""Render sequential frozen-policy runs as one synchronized three-row MP4."""

import argparse
import copy
from functools import lru_cache
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from gym.utils.helpers import get_load_path, set_seed
from gym.utils.task_registry import task_registry
from scripts.evaluate_go2_composition import blend_alpha, configure


RUNS = (
    ("a_only", "A only - Walking"),
    ("immediate", "A -> B - Immediate Switching"),
    ("blend_025", "A -> B - 0.25 s Blending"),
)


@lru_cache
def font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label_frame(frame, label, time_s, status):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0, 165))
    draw.text((12, 6), label, font=font(20), fill="white")
    text = f"t = {time_s:4.1f} s   {status}"
    width = draw.textbbox((0, 0), text, font=font(16))[2]
    draw.text((image.width - width - 12, 8), text, font=font(16), fill="white")
    return np.asarray(image)


def run_status(method, time_s, crouch_time, stand_time):
    if method == "a_only":
        return "WALK (A)"
    if time_s < crouch_time:
        return "WALK (A)"
    if method == "blend_025" and time_s < crouch_time + 0.25:
        alpha = blend_alpha(method, 0.25, time_s - crouch_time)
        return f"BLEND alpha={alpha:.2f}"
    if time_s < stand_time:
        return "CROUCH (B)"
    return "STAND (B)"


def render_frame(renderer, backend, camera, label, time_s, status):
    data = backend._datas[0]
    camera.lookat[:] = (data.qpos[0], data.qpos[1], data.qpos[2] - 0.08)
    renderer.update_scene(data, camera=camera)
    return label_frame(renderer.render(), label, time_s, status)


@torch.no_grad()
def simulate_run(method, label, args):
    set_seed(args.seed)
    trot_env_cfg, trot_train_cfg = copy.deepcopy(
        task_registry.get_cfgs("go2trot", original_cfg=True, load_run=args.trot_run)
    )
    height_env_cfg, height_train_cfg = copy.deepcopy(
        task_registry.get_cfgs("go2height", original_cfg=True, load_run=args.height_run)
    )
    configure(
        trot_env_cfg, trot_train_cfg, num_envs=1, seconds=args.duration, seed=args.seed
    )
    configure(
        height_env_cfg,
        height_train_cfg,
        num_envs=1,
        seconds=args.duration,
        seed=args.seed,
    )

    trot_path = get_load_path("go2trot", args.trot_run, args.checkpoint)
    height_path = get_load_path("go2height", args.height_run, args.checkpoint)
    env = task_registry.make_env("go2height", height_env_cfg, "cpu", True, "mujoco")
    renderer = None
    try:
        trot_runner = task_registry.make_alg_runner(env, trot_train_cfg)
        height_runner = task_registry.make_alg_runner(env, height_train_cfg)
        trot_runner.load(trot_path, load_optimizer=False)
        height_runner.load(height_path, load_optimizer=False)
        trot_runner.switch_to_eval()
        height_runner.switch_to_eval()

        env.phase[:] = 0.6
        env.phase_frequency[:] = 2.0
        env._update_phase_observation()
        env._update_gait_reference()
        initial_state = torch.cat(
            (env.root_states.flatten(), env.dof_pos.flatten(), env.phase.flatten())
        ).cpu()

        renderer = mujoco.Renderer(env._backend._mjm, height=216, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = 145
        camera.elevation = -18
        camera.distance = 1.35

        frames = []
        action_checks = []
        next_frame = 0
        frame_count = round(args.duration * args.fps)
        total_steps = round(args.duration / env.dt)
        for step in range(total_steps):
            time_s = step * env.dt
            env.commands.zero_()
            if method == "a_only" or time_s < args.crouch_time:
                env.commands[:, 0] = args.speed
            env.base_height_command[:] = (
                0.20 if args.crouch_time <= time_s < args.stand_time else 0.35
            )

            trot_actions = trot_runner.get_inference_actions()
            height_actions = height_runner.get_inference_actions()
            if method == "a_only":
                alpha = 0.0
            elif method == "immediate":
                alpha = blend_alpha("direct", 0.0, time_s - args.crouch_time)
            else:
                alpha = blend_alpha("blend_025", 0.25, time_s - args.crouch_time)
            actions = trot_actions + alpha * (height_actions - trot_actions)
            if args.crouch_time <= time_s < args.crouch_time + 0.25:
                action_checks.append(
                    {
                        "time_s": round(time_s, 3),
                        "alpha": alpha,
                        "mean_abs_applied_minus_a": float(
                            (actions - trot_actions).abs().mean()
                        ),
                        "mean_abs_applied_minus_b": float(
                            (actions - height_actions).abs().mean()
                        ),
                    }
                )
            height_runner.set_actions(height_runner.actor_cfg["actions"], actions)
            env.step()

            while (
                next_frame < frame_count
                and (next_frame + 1) / args.fps <= (step + 1) * env.dt + 1e-9
            ):
                frame_time = (next_frame + 1) / args.fps
                frames.append(
                    render_frame(
                        renderer,
                        env._backend,
                        camera,
                        label,
                        frame_time,
                        run_status(
                            method,
                            frame_time,
                            args.crouch_time,
                            args.stand_time,
                        ),
                    )
                )
                next_frame += 1
        if len(frames) != frame_count:
            raise RuntimeError(f"rendered {len(frames)} of {frame_count} frames")
        return frames, initial_state, action_checks
    finally:
        if renderer is not None:
            renderer.close()
        env._backend.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trot_run", default="Nov16_20-08-42_trot_v1")
    parser.add_argument("--height_run", default="Oct16_18-01-11_height_v1")
    parser.add_argument("--checkpoint", type=int, default=500)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--crouch_time", type=float, default=2.0)
    parser.add_argument("--stand_time", type=float, default=4.0)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5110)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/video/assignment_6_frozen_policy_comparison_sequential.mp4"
        ),
    )
    args = parser.parse_args()
    if not 0 < args.crouch_time < args.stand_time < args.duration or args.fps < 1:
        parser.error("require 0 < crouch time < stand time < duration and positive fps")

    import gym.envs  # noqa: F401

    all_frames = []
    initial_states = []
    checks = {}
    for method, label in RUNS:
        frames, initial_state, action_checks = simulate_run(method, label, args)
        all_frames.append(frames)
        initial_states.append(initial_state)
        checks[method] = action_checks
    if not all(torch.equal(initial_states[0], state) for state in initial_states[1:]):
        raise RuntimeError("sequential runs did not start from identical states")

    immediate_error = max(
        row["mean_abs_applied_minus_b"] for row in checks["immediate"]
    )
    blend_error = max(row["mean_abs_applied_minus_b"] for row in checks["blend_025"])
    if immediate_error > 1e-7 or blend_error <= 1e-4:
        raise RuntimeError(
            "action logic check failed: "
            f"immediate_to_B={immediate_error}, blend_to_B={blend_error}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart"],
    ) as writer:
        for index in range(len(all_frames[0])):
            writer.append_data(np.vstack([frames[index] for frames in all_frames]))

    check_path = args.output.with_name(args.output.stem + "_action_check.json")
    summary = {
        "sequential_runs": True,
        "identical_initial_states": True,
        "immediate_max_mean_abs_applied_minus_b": immediate_error,
        "blend_max_mean_abs_applied_minus_b": blend_error,
        "blend_window": checks["blend_025"],
    }
    check_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary | {"video": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
