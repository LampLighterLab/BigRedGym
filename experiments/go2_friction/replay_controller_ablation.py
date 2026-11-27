"""Render synchronized side views from matched controller evaluations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import evaluate_friction as evaluation
import metrics
from controller_telemetry import MIN_LOAD_N


MANEUVERS = ("stand_1_stop", "steady_0p5", "steady_1p0")
FPS = 25
WIDTH, HEIGHT, VIEW_HEIGHT = 576, 432, 320
HEADER, GUTTER = 56, 8
BACKGROUND = "#102439"
PALE = "#dde9f5"
FEET = ("FL", "FR", "RL", "RR")
COORDINATION_TRACE_FIELDS = {
    "actor_actions",
    "distal_calf_contact",
    "distal_calf_ground_force_world_N",
    "distal_leg_ground_force_world_N",
    "other_non_foot_ground_load_N",
}


def _load_evaluation(directory):
    meta = json.loads((directory / "metadata.json").read_text())
    checkpoint = Path(meta["checkpoint"])
    with checkpoint.open("rb") as stream:
        if (
            hashlib.file_digest(stream, "sha256").hexdigest()
            != meta["checkpoint_sha256"]
        ):
            raise ValueError(f"Checkpoint changed after evaluation: {checkpoint}")
    if (
        meta["status"] != "completed"
        or meta["backend"] != "mujoco_cpu"
        or meta["policy_device"] != "cpu"
        or meta["dt_s"] != 0.02
        or meta["frictions"] != [1.25]
        or meta["noise_scales"] != [0.0]
        or meta["foot_order"] != [f"{leg}_foot" for leg in FEET]
        or not meta.get("controller_profile")
    ):
        raise ValueError(f"Expected a completed CPU controller evaluation: {directory}")
    return meta, checkpoint


def _draw_panel(
    renderer, camera, backend, slot, sample, label, fonts, outcome, maneuver
):
    pos = sample["base_position_world_m"][slot]
    frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    if outcome != "numerical_failure":
        camera.lookat[0] = float(pos[0]) + 0.08
        camera.lookat[1] = float(pos[1])
        renderer.update_scene(backend._datas[slot], camera=camera)
        frame.paste(Image.fromarray(renderer.render()), (0, HEADER))
    draw = ImageDraw.Draw(frame)
    title, small, mono = fonts
    draw.text((12, 6), label, fill="white", font=title)
    draw.text((564, 10), maneuver, fill=PALE, font=small, anchor="ra")
    command = sample["command"][slot, 0]
    vx = sample["base_linear_velocity_body_m_s"][slot, 0]
    time = sample["time_s"][slot]
    for x, value in (
        (12, f"Cmd {command:+.2f} m/s"),
        (149, f"Actual {vx:+.2f} m/s"),
        (321, f"Height {pos[2]:.3f} m"),
        (476, f"t {time:.2f} s"),
    ):
        draw.text((x, 34), value, fill=PALE, font=mono)

    for i, (name, forces, color) in enumerate(
        (
            ("Foot Fz", "foot_ground_force_world_N", "#177b6a"),
            ("Distal calf Fz", "distal_calf_ground_force_world_N", "#9e4925"),
        )
    ):
        y = HEADER + VIEW_HEIGHT + 6 + 24 * i
        draw.text((12, y + 2), name, fill=PALE, font=small)
        for j, foot in enumerate(FEET):
            x = 119 + 89 * j
            load = sample[forces][slot, j, 2]
            draw.rounded_rectangle(
                (x, y, x + 79, y + 19),
                radius=4,
                fill=color if load > MIN_LOAD_N else "#293f53",
            )
            draw.text(
                (x + 40, y + 10),
                f"{foot} {load:.0f}",
                fill="white",
                font=small,
                anchor="mm",
            )
    draw.text((477, 396), "N · >5 shaded", fill=PALE, font=small)
    if outcome != "completed":
        draw.rounded_rectangle((12, 67, 238, 94), radius=5, fill="#921d26")
        draw.text(
            (21, 72),
            f"Trial ended · {outcome.replace('_', ' ')}",
            fill="white",
            font=small,
        )
    return frame


def _render_evaluation(directory, meta, checkpoint, maneuvers, seed):
    seeds = tuple(meta["trial_seeds"])
    index = seeds.index(seed)
    offset = index - index % meta["num_envs"]
    slot = index - offset
    batch = seeds[offset : offset + meta["num_envs"]]
    args = SimpleNamespace(
        original_cfg=False,
        num_envs=meta["num_envs"],
        device="cpu",
        gait_frequency=meta["gait_frequency_hz"],
        frictions=(1.25,),
    )
    stages = evaluation.select_maneuvers(",".join(meta["maneuvers"]))
    env, cfg, schedules = evaluation.make_environment(checkpoint, args, seeds, stages)
    backend = env._backend
    try:
        if env.checkpoint_controller_profile != meta["controller_profile"]:
            raise ValueError(f"Saved controller profile changed: {checkpoint}")
        policy = evaluation.load_policy(env, cfg, checkpoint)
        contacts = evaluation.FootGroundContacts(env, record_distal=True)
        with (directory / "trials.csv").open(newline="") as stream:
            rows = {
                row["maneuver"]: row
                for row in csv.DictReader(stream)
                if int(row["eval_seed"]) == seed
            }
        weight = policy.critic_cfg["reward"]["weights"].get("paired_contact")
        reward_label = (
            f"Paired-contact weight {weight:g}"
            if weight is not None
            else "Legacy contact rewards"
        )
        label = f"Iteration {policy.it} · {reward_label}"
        if pilot := (meta.get("training") or {}).get("fresh_pilot"):
            label = (
                f"Iteration {policy.it} · {pilot['arm'].title()} · "
                f"lin_vel_z {pilot['lin_vel_z_weight']:.2f}"
            )
        telemetry = meta["controller_telemetry"]
        unrecorded = set()
        if not telemetry.get("coordination_diagnostics"):
            unrecorded.update(COORDINATION_TRACE_FIELDS)
        if not telemetry.get("non_foot_force_groups"):
            unrecorded.update(("calf_ground_load_N", "head_ground_load_N"))
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation, camera.distance = 90, -13, 1.65
        camera.lookat[2] = 0.23
        fonts = (
            ImageFont.truetype("DejaVuSans-Bold.ttf", 18),
            ImageFont.truetype("DejaVuSans.ttf", 12),
            ImageFont.truetype("DejaVuSansMono.ttf", 14),
        )
        clips = {}
        model = backend._model_for_env(slot)
        with mujoco.Renderer(model, height=VIEW_HEIGHT, width=WIDTH) as renderer:
            for maneuver in maneuvers:
                row = rows[maneuver]
                count = int(row["recorded_steps"])
                path = (directory / row["trace"]).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError(
                        f"Trace is outside the evaluation directory: {path}"
                    )
                frames = []

                def frame_step(step, sample):
                    if step > count or (step % 2 and step != count):
                        return
                    outcome = row["status"] if step == count else "completed"
                    frames.append(
                        _draw_panel(
                            renderer,
                            camera,
                            backend,
                            slot,
                            sample,
                            label,
                            fonts,
                            outcome,
                            maneuver,
                        )
                    )

                schedule = schedules[maneuver]
                if schedule != metrics.positive_schedule(meta["maneuvers"][maneuver]):
                    raise ValueError(
                        f"Recorded and replayed command schedules differ: {maneuver}"
                    )
                traces = list(
                    evaluation.collect_batch(
                        env,
                        policy,
                        contacts,
                        1.25,
                        schedule,
                        batch,
                        None,
                        on_step=frame_step,
                    )
                )
                trace, phase = traces[slot]
                if (
                    phase != float(row["initial_phase_rad"])
                    or evaluation.trial_status(trace) != row["status"]
                ):
                    raise ValueError(f"Replayed trial diverged from {path}")
                saved = metrics.load_series(path)
                if set(trace) - unrecorded != set(saved) or any(
                    not np.array_equal(trace[key], values, equal_nan=True)
                    for key, values in saved.items()
                ):
                    raise ValueError(f"Replayed telemetry diverged from {path}")
                clips[maneuver] = frames
        return clips
    finally:
        backend.close()


def _synchronize(panels):
    """Use the same two-actor-tick timeline and hold a terminated panel."""
    width = len(panels) * (WIDTH + GUTTER) - GUTTER
    frames = []
    for i in range(max(map(len, panels))):
        frame = Image.new("RGB", (width, HEIGHT), "#dbe4ed")
        for slot, panel in enumerate(panels):
            frame.paste(panel[min(i, len(panel) - 1)], (slot * (WIDTH + GUTTER), 0))
        frames.append(frame)
    return frames


def _ffmpeg(explicit):
    if explicit is not None:
        return str(explicit.resolve(strict=True))
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as exc:
        raise ValueError(
            "Set --ffmpeg, place FFmpeg on PATH, or install imageio-ffmpeg"
        ) from exc
    return get_ffmpeg_exe()


def _save(frames, stem, ffmpeg):
    frames[0].save(
        stem.with_suffix(".gif"),
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    with subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{frames[0].width}x{HEIGHT}",
            "-framerate",
            str(FPS),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(stem.with_suffix(".mp4")),
        ],
        stdin=subprocess.PIPE,
    ) as encoder:
        with encoder.stdin as stream:
            for frame in frames:
                stream.write(frame.tobytes())
        if encoder.wait():
            raise RuntimeError("FFmpeg failed to encode the replay")


def replay(
    directories,
    output,
    maneuvers,
    seed,
    ffmpeg,
    *,
    progression=False,
    reference_dir=None,
):
    if len(directories) not in ((2, 3) if progression else (2,)):
        raise ValueError(
            "Supply two evaluation directories, or three with --progression"
        )
    directories = [directory.resolve() for directory in directories]
    datasets = [_load_evaluation(directory) for directory in directories]
    anchor = datasets[0][0]
    iterations = [meta["checkpoint_iteration"] for meta, _ in datasets]
    if progression and iterations != sorted(set(iterations)):
        raise ValueError("Supply progression checkpoints in increasing iteration order")
    if not progression and len(set(iterations)) != 1:
        raise ValueError("For checkpoints at different iterations, use --progression")
    reference_iteration = None
    if reference_dir is not None:
        if progression:
            raise ValueError("Use --reference-evaluation with a same-iteration pair")
        reference_dir = reference_dir.resolve()
        reference = _load_evaluation(reference_dir)
        directories.append(reference_dir)
        datasets.append(reference)
        reference_iteration = reference[0]["checkpoint_iteration"]
        iterations.append(reference_iteration)
    for field in (
        "num_envs",
        "maneuvers",
        "trial_seeds",
        "gait_frequency_hz",
        "mujoco_version",
    ):
        if any(meta[field] != anchor[field] for meta, _ in datasets[1:]):
            raise ValueError(f"Evaluation conditions differ in {field}")
    output.mkdir(parents=True, exist_ok=False)
    clips = {}
    for directory, (meta, checkpoint) in zip(directories, datasets, strict=True):
        if directory not in clips:
            clips[directory] = _render_evaluation(
                directory, meta, checkpoint, maneuvers, seed
            )
    files = {}
    iteration_label = (
        "_".join(map(str, iterations)) if progression else str(iterations[0])
    )
    if reference_iteration is not None:
        iteration_label += f"_vs_{reference_iteration}"
    for maneuver in maneuvers:
        frames = _synchronize([clips[directory][maneuver] for directory in directories])
        stem = output / f"iter_{iteration_label}_{maneuver}_seed_{seed}"
        _save(frames, stem, ffmpeg)
        files[maneuver] = {"file_stem": stem.name, "frames": len(frames)}
    evaluation.write_metadata(
        output / "replay_metadata.json",
        {
            "checkpoints": [str(checkpoint) for _, checkpoint in datasets],
            "checkpoint_sha256": [meta["checkpoint_sha256"] for meta, _ in datasets],
            "iterations": iterations,
            **(
                {"reference_iteration": reference_iteration}
                if reference_iteration is not None
                else {}
            ),
            "eval_seed": seed,
            "friction": 1.25,
            "noise_scale": 0.0,
            "fps": FPS,
            "camera": {"azimuth_deg": 90, "elevation_deg": -13, "distance_m": 1.65},
            "telemetry_matches_saved_traces": True,
            "replays": files,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--progression",
        action="store_true",
        help="Render checkpoints in increasing iteration order",
    )
    parser.add_argument(
        "--reference-evaluation",
        type=Path,
        help="Third panel for a same-iteration comparison",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--maneuvers", default=",".join(MANEUVERS))
    parser.add_argument(
        "--ffmpeg", type=Path, help="Path to an FFmpeg binary with libx264"
    )
    args = parser.parse_args(argv)
    maneuvers = tuple(args.maneuvers.split(","))
    if len(set(maneuvers)) != len(maneuvers) or any(
        m not in MANEUVERS for m in maneuvers
    ):
        parser.error(f"Select a unique subset of {', '.join(MANEUVERS)}")
    try:
        replay(
            args.evaluation_dirs,
            args.output_dir,
            maneuvers,
            args.eval_seed,
            _ffmpeg(args.ffmpeg),
            progression=args.progression,
            reference_dir=args.reference_evaluation,
        )
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved synchronized replays to {args.output_dir}")


if __name__ == "__main__":
    main()
