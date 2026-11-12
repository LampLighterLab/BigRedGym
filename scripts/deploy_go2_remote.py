"""Send the repo and a policy bundle to the Go2's Jetson, then run it in tmux.

    uv run --frozen python scripts/deploy_go2_remote.py \
        --bundle logs/policies/trot_v1 --host unitree@10.42.0.1
    uv run --frozen python scripts/deploy_go2_remote.py \
        --host unitree@10.42.0.1 --pull_logs

The control loop runs entirely on the robot; the RC works whether or not this
SSH session stays attached. Detach with Ctrl+b d, reattach by rerunning.
"""

import argparse
import os
import shlex
import subprocess

TMUX_SESSION = "go2_deploy"
# "/.git/" only: thirdparty/unitree_sdk2_python keeps its own .git for the pin check
RSYNC_EXCLUDES = ["/.venv/", "/logs/", "/wandb/", "__pycache__/", "/.git/"]


def remote_bundle_path(bundle):
    """Where the bundle lands on the robot, relative to the remote repo root."""
    return f"logs/policies/{os.path.basename(os.path.normpath(bundle))}"


def sync_commands(host, remote_dir, bundle):
    remote_bundle = f"{remote_dir}/{remote_bundle_path(bundle)}"
    excludes = [arg for pattern in RSYNC_EXCLUDES for arg in ("--exclude", pattern)]
    return [
        ["rsync", "-az", "--delete", *excludes, "./", f"{host}:{remote_dir}/"],
        ["ssh", host, f"mkdir -p {shlex.quote(remote_bundle)}"],
        ["rsync", "-az", "--delete", f"{bundle}/", f"{host}:{remote_bundle}/"],
    ]


def launch_command(host, remote_dir, bundle, interface, cyclonedds_home):
    deploy = (
        f"cd {shlex.quote(remote_dir)} && "
        # no rc files run (see below), so put uv's install dir on PATH here
        'export PATH="$HOME/.local/bin:$PATH" && '
        f"export CYCLONEDDS_HOME={cyclonedds_home} && "
        "uv run --frozen --extra unitree_sdk python -m go2_deploy.deploy "
        f"{shlex.quote(interface)} --bundle {shlex.quote(remote_bundle_path(bundle))}; "
        # keep the pane open so errors stay readable; --norc skips Unitree's
        # interactive ROS-version prompt in ~/.bashrc
        "exec bash --norc"
    )
    # Run the pane in a bare bash, whatever the login shell is (zsh on some
    # Jetsons) and whatever its rc files do.
    pane = f"bash --noprofile --norc -c {shlex.quote(deploy)}"
    # -A attaches to a session that is already running instead of starting twice
    tmux = f"tmux new-session -A -s {TMUX_SESSION} {shlex.quote(pane)}"
    return ["ssh", "-t", host, tmux]


def pull_logs_command(host, remote_dir):
    os.makedirs("logs/deploy", exist_ok=True)
    return ["rsync", "-az", f"{host}:{remote_dir}/logs/deploy/", "logs/deploy/"]


def run(command, dry_run):
    print("+ " + shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, help="e.g. unitree@10.42.0.1")
    parser.add_argument("--bundle", help="Local bundle directory to deploy")
    parser.add_argument("--remote_dir", default="BigRedGym", help="Relative to ~")
    parser.add_argument("--interface", default="eth0", help="Jetson's robot NIC")
    parser.add_argument(
        "--cyclonedds_home", default="$HOME/.local/opt/cyclonedds-0.10.2"
    )
    parser.add_argument("--pull_logs", action="store_true", help="Only fetch logs")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    if args.pull_logs:
        run(pull_logs_command(args.host, args.remote_dir), args.dry_run)
        return
    if not args.bundle:
        parser.error("--bundle is required unless --pull_logs is given")
    if not os.path.isfile(os.path.join(args.bundle, "deploy_meta.json")):
        parser.error(f"{args.bundle} is not a policy bundle (no deploy_meta.json)")

    for command in sync_commands(args.host, args.remote_dir, args.bundle):
        run(command, args.dry_run)
    run(
        launch_command(
            args.host,
            args.remote_dir,
            args.bundle,
            args.interface,
            args.cyclonedds_home,
        ),
        args.dry_run,
    )


if __name__ == "__main__":
    main()
