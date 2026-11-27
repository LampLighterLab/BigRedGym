"""Collect evaluation traces across policies, test friction, and actor noise."""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from controller_checkpoint import read_controller_profile
from evaluate_friction import (
    DEFAULT_FRICTIONS,
    PRIMARY_MANEUVER,
    REPO,
    RESULTS,
    add_evaluation_arguments,
    checkpoint_output_path,
    evaluation_settings,
)
import final_protocol as final

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


def finalized_checkpoints(logs_dir, run_name, *, dry_run):
    final.validate_run_name(run_name)
    initializations = {}
    template_hashes = set()
    for run in final.final_training_runs():
        directory = (logs_dir / run["experiment_name"] / run_name).resolve()
        checkpoint = directory / f"model_{final.ITERATIONS}.pt"
        if not dry_run:
            data = final.read_completed_run(directory, run)
            details = data.get("final_research", {})
            profile = read_controller_profile(
                checkpoint, expected_iteration=final.ITERATIONS
            )
            if profile is None or profile != data["controller_profile"]:
                raise ValueError(f"Missing or mismatched checkpoint: {checkpoint}")
            template_hashes.add(details["template_checkpoint_sha256"])
            initial = details["initial_actor_sha256"], details["initial_critic_sha256"]
            if initializations.setdefault(run["seed"], initial) != initial:
                raise ValueError(
                    f"Seed {run['seed']} initialization differs: {directory}"
                )
        yield checkpoint
    if not dry_run and len(template_hashes) != 1:
        raise ValueError("The finalized runs use different controller templates")


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
    ]
    if args.gait_frequency is not None:
        command.extend(("--gait-frequency", str(args.gait_frequency)))
    if args.original_cfg:
        command.append("--original-cfg")
    if args.maneuver_config is not None:
        command.extend(("--maneuver-config", str(args.maneuver_config.resolve())))
    if args.record_coordination_diagnostics:
        command.append("--record-coordination-diagnostics")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint-path",
        type=Path,
        action="append",
        help="Repeat to pin checkpoints. Defaults to completed Step 1 runs.",
    )
    source.add_argument(
        "--finalized-run",
        help="Select all 14 finalized iteration-400 policies by their shared run name",
    )
    parser.add_argument("--logs-dir", type=Path, default=REPO / "logs")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "evaluation_sweep")
    parser.add_argument("--dry-run", action="store_true")
    add_evaluation_arguments(
        parser, frictions=None, noise_scales=final.TEST_NOISE_SCALES
    )
    parser.set_defaults(maneuvers=None)
    args = parser.parse_args(argv)
    if args.finalized_run is not None:
        if args.original_cfg:
            parser.error(
                "Finalized policies use the controller profile in the checkpoint"
            )
        args.record_coordination_diagnostics = True
        if args.dry_run:
            print("Previewing finalized paths. Evaluation requires completed runs.")
    if args.maneuvers is None:
        args.maneuvers = (
            ",".join(final.MANEUVERS)
            if args.finalized_run is not None
            else PRIMARY_MANEUVER
        )
    if args.frictions is None:
        args.frictions = (
            final.TEST_FRICTIONS
            if args.finalized_run is not None
            else DEFAULT_FRICTIONS
        )
    seeds, maneuvers = evaluation_settings(parser, args)
    try:
        if args.finalized_run is not None:
            checkpoints = list(
                finalized_checkpoints(
                    args.logs_dir, args.finalized_run, dry_run=args.dry_run
                )
            )
        elif args.checkpoint_path:
            checkpoints = [path.resolve() for path in args.checkpoint_path]
        else:
            checkpoints = list(discover_checkpoints(args.logs_dir))
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    if not checkpoints:
        parser.error("No completed Step 1 checkpoints found. Supply --checkpoint-path.")
    if len(set(checkpoints)) != len(checkpoints):
        parser.error("Checkpoint paths must be unique")
    outputs = []
    for checkpoint in checkpoints:
        if not checkpoint.is_file() and not (
            args.finalized_run is not None and args.dry_run
        ):
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
