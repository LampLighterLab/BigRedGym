"""Collect evaluation traces across Step 1 checkpoints, friction, and noise."""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from evaluate_friction import (
    DEFAULT_FRICTIONS,
    REPO,
    RESULTS,
    add_evaluation_arguments,
    checkpoint_output_path,
    evaluation_settings,
)

EXPERIMENT = re.compile(r"go2(?:friction_lb|noise_scale)_.+_seed_\d+")
MODEL = re.compile(r"model_(\d+)\.pt")


def discover_checkpoints(logs_dir):
    """Choose the newest completed Step 1 run and latest checkpoint per condition."""
    for experiment in sorted(logs_dir.iterdir()):
        if not experiment.is_dir() or not EXPERIMENT.fullmatch(experiment.name):
            continue
        runs = []
        for manifest in experiment.glob("*/friction_experiment.json"):
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("status") != "completed" or "noise_scale" not in data:
                continue
            checkpoints = [
                (int(match[1]), path)
                for path in manifest.parent.glob("model_*.pt")
                if path.is_file() and (match := MODEL.fullmatch(path.name))
            ]
            if checkpoints:
                runs.append((data.get("completed_at", ""), max(checkpoints)[1]))
        if runs:
            yield max(runs)[1].resolve()


def evaluation_command(checkpoint, output, args, seeds):
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_friction.py")),
        "--checkpoint-path",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--frictions",
        ",".join(map(str, args.frictions)),
        "--noise-scales",
        ",".join(map(str, args.noise_scales)),
        "--maneuvers",
        args.maneuvers,
        "--eval-seeds",
        ",".join(map(str, seeds)),
        "--device",
        args.device,
        "--num-envs",
        str(args.num_envs),
        "--gait-frequency",
        str(args.gait_frequency),
    ]
    if args.original_cfg:
        command.append("--original-cfg")
    if args.maneuver_config is not None:
        command.extend(("--maneuver-config", str(args.maneuver_config.resolve())))
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        action="append",
        help="Repeat to pin checkpoints. Defaults to completed Step 1 runs.",
    )
    parser.add_argument("--logs-dir", type=Path, default=REPO / "logs")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "evaluation_sweep")
    parser.add_argument("--dry-run", action="store_true")
    add_evaluation_arguments(
        parser, frictions=DEFAULT_FRICTIONS, noise_scales=(0.0, 1.0)
    )
    args = parser.parse_args(argv)
    seeds, maneuvers = evaluation_settings(parser, args)
    try:
        checkpoints = (
            [path.resolve() for path in args.checkpoint_path]
            if args.checkpoint_path
            else list(discover_checkpoints(args.logs_dir))
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not checkpoints:
        parser.error("No completed Step 1 checkpoints found. Supply --checkpoint-path.")
    if len(set(checkpoints)) != len(checkpoints):
        parser.error("Checkpoint paths must be unique")
    outputs = []
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")
        output = args.output_dir.resolve() / checkpoint_output_path(checkpoint)
        if output.exists() and not args.dry_run:
            parser.error(f"Output directory already exists: {output}")
        outputs.append(output)
    if len(set(outputs)) != len(outputs):
        parser.error("Checkpoints have conflicting experiment/run/model names")

    trials_per_policy = (
        len(args.frictions) * len(args.noise_scales) * len(maneuvers) * len(seeds)
    )
    print(
        f"{len(checkpoints)} checkpoints, {trials_per_policy} trials each", flush=True
    )
    for checkpoint, output in zip(checkpoints, outputs, strict=True):
        command = evaluation_command(checkpoint, output, args, seeds)
        print("$ " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
