import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Deploy a policy to a Unitree Go2.")
    parser.add_argument("interface", help="Robot network interface, e.g. eth0")
    parser.add_argument(
        "--bundle",
        help="Policy bundle directory from scripts/export_go2_policy.py. "
        "Without it, the actor is built from the latest training run in logs/.",
    )
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Go2 deployment requires Linux (timerfd).")

    try:
        from unitree_sdk2py.core import channel
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ModuleNotFoundError as exc:
        if exc.name != "unitree_sdk2py":
            raise
        parser.error(
            "Unitree SDK is not installed. Follow README_DEPLOY.md, then run "
            "uv run --frozen --extra unitree_sdk python -m go2_deploy.deploy "
            "<interface>."
        )

    from go2_deploy.main_controller import MainController
    from go2_deploy.keyboard_handler import KeyboardHandler
    from go2_deploy.rl_controller import RLController

    # Load the policy before touching the robot, so a bad bundle fails here.
    if args.bundle:
        from go2_deploy.policy_bundle import load_bundle

        policy, cfg = load_bundle(args.bundle)
        print(f"Loaded policy bundle {args.bundle}")
    else:
        from go2_deploy.deploy_config import DeployConfig

        policy, cfg = None, DeployConfig()
    rl_controller = RLController(cfg, policy)

    # CycloneDDS 0.10.2 bug workaround
    channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
        "<Verbosity>config</Verbosity>", "<Verbosity>none</Verbosity>"
    )

    ChannelFactoryInitialize(
        0,
        args.interface,
    )
    controller = MainController(cfg, rl_controller)
    keyboard_handler = KeyboardHandler(controller)  # noqa: F841

    print(
        "\nConnection successful! Keep the robot away from obstacles and "
        + "always have access to the remote controller\n"
    )

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
