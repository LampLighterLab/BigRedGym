"""Play a Go2 policy and show its gait in the terminal and MuJoCo viewer."""

import math

import torch

from gym.utils.gait_classifier import GaitClassifier, GaitResult

if __package__:
    from .play import get_play_parser, setup, setup_viewer
else:
    from play import get_play_parser, setup, setup_viewer


FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")  # RF, LF, RH, LH


def get_args(argv=None):
    parser = get_play_parser()
    parser.description = "Play a Go2 policy and classify robot 0's footfalls"
    parser.set_defaults(num_envs=1)
    parser.add_argument("--duration", type=float, default=30, help="Simulation seconds")
    parser.add_argument(
        "--contact_on", type=float, default=5, help="Touchdown cutoff (N; tune to Go2)"
    )
    parser.add_argument(
        "--contact_off", type=float, default=2, help="Liftoff cutoff (N; tune to Go2)"
    )
    args = parser.parse_args(argv)
    if not 0 < args.duration < math.inf or args.num_envs < 1:
        parser.error("duration must be finite and positive; num_envs must be positive")
    if not 0 <= args.contact_off < args.contact_on < math.inf:
        parser.error("contact cutoffs must satisfy 0 <= off < on < infinity")
    return args


def format_result(result):
    lines = [f"Gait: {result.label}"]
    if result.valid:
        lines.append(
            f"Nearest: {result.nearest_gait} | distance: {result.distance:.2f}"
        )
        phases = ", ".join(f"{phase:.2f}" for phase in result.relative_phases)
        lines.append(f"RPD (LF, RH, LH): {phases}")
    elif result.reason:
        lines.append(result.reason)
    return "\n".join(lines)


@torch.no_grad()
def classify(env, runner, args):
    """Sample robot 0 at control frequency; clear history after each reset."""
    feet = torch.tensor(
        env.robot_layout.body_indices(FOOT_NAMES), device=env.device, dtype=torch.long
    )
    classifier = GaitClassifier(
        contact_on=args.contact_on, contact_off=args.contact_off
    )
    start_step = env.common_step_counter
    previous_episode_step = int(env.episode_length_buf[0])
    viewer = getattr(env._backend, "_viewer", None)

    def sample():
        time_s = (env.common_step_counter - start_step) * env.dt
        forces = env.contact_forces[0, feet].norm(dim=-1).cpu().numpy()
        return classifier.update(time_s, forces)

    print(
        f"Robot 0 | sampling {1 / env.dt:g} Hz | "
        f"contact cutoffs: on={args.contact_on:g} N, off={args.contact_off:g} N "
        "(starting values; tune from contact traces)",
        flush=True,
    )
    result = sample()
    last_report_key = None
    while not env.exit:
        time_s = (env.common_step_counter - start_step) * env.dt
        report_key = (
            result.label,
            result.reason,
            result.time_s if result.valid else None,
        )
        if report_key != last_report_key:
            text = format_result(result)
            line = text.replace("\n", " | ")
            print(f"[{time_s:6.2f}s] {line}", flush=True)
            if hasattr(viewer, "set_texts"):
                viewer.set_texts((None, None, text, ""))
            last_report_key = report_key
        if time_s >= args.duration:
            break
        if viewer is not None and not viewer.is_running():
            break

        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        current_episode_step = int(env.episode_length_buf[0])
        if current_episode_step <= previous_episode_step:  # Keyboard reset.
            classifier.reset()
        if bool(env.terminated[0] | env.timed_out[0]):
            classifier.reset()
            result = GaitResult("Resetting", reason="Episode ended")
        else:
            result = sample()
        runner.reset_envs()
        previous_episode_step = int(env.episode_length_buf[0])
    return result


def main(argv=None):
    args = get_args(argv)
    env, runner = setup(args)
    try:
        commands = env.commands.clone()  # Keyboard setup zeroes commands.
        setup_viewer(env, args)
        env.commands.copy_(commands)
        classify(env, runner, args)
    except KeyboardInterrupt:
        pass
    finally:
        viewer = getattr(env._backend, "_viewer", None)
        if viewer is not None:
            viewer.close()
        env._backend.close()


if __name__ == "__main__":
    main()
