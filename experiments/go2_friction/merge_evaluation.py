"""Combine different finalized policies or extend their evaluation conditions."""

import argparse
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from shutil import copyfile, copytree
from tempfile import TemporaryDirectory

import final_protocol as final
from metrics import read_dataset, write_csv


def read_complete_evaluation(dataset):
    meta, _, schedules, frictions, scales, seeds, rows = read_dataset(dataset)
    expected = set(product(schedules, frictions, scales, seeds))
    if (
        meta.get("status") != "completed"
        or meta.get("trials_written") != len(expected)
        or rows.keys() != expected
    ):
        raise ValueError(f"Expected a complete trial grid in {dataset}")

    traces, trial_ids = set(), set()
    for (maneuver, friction, _, _), row in rows.items():
        trace = (dataset / row["trace"]).resolve()
        planned = sum(steps for steps, _ in schedules[maneuver])
        if (
            not trace.is_relative_to(dataset)
            or trace.suffix != ".npz"
            or not trace.is_file()
            or trace in traces
            or not row["trial_id"]
            or row["trial_id"] in trial_ids
            or int(row["planned_steps"]) != planned
            or not 0 < int(row["recorded_steps"]) <= planned
            # The MuJoCo backend stores the coefficient in a float32 tensor.
            or not math.isclose(
                float(row["applied_friction"]), friction, rel_tol=0, abs_tol=1e-7
            )
        ):
            raise ValueError(f"Invalid or repeated trace in {dataset}: {trace}")
        traces.add(trace)
        trial_ids.add(row["trial_id"])
    return meta, rows


def datasets_under(root):
    datasets = {
        path.parent.relative_to(root): path.parent
        for path in sorted(root.rglob("metadata.json"))
    }
    if not datasets:
        raise ValueError(f"No evaluation datasets found under {root}")
    return datasets


def comparable_metadata(metadata):
    comparable = deepcopy(metadata)
    # This startup value enables the setter. Each trial sets its own friction.
    del comparable["env_config"]["domain_randomization"]["startup"][
        "contact_friction_range"
    ]
    for field in ("frictions", "trials_written"):
        del comparable[field]
    return comparable


def different_fields(left, right):
    return [
        key
        for key in sorted(left.keys() | right.keys())
        if key not in left or key not in right or left[key] != right[key]
    ]


def plan_condition_merge(existing, additional):
    plan = []
    checkpoint_hashes = set()
    for name, original in existing.items():
        sources = (original, additional[name])
        (base, old_rows), (extra, new_rows) = [
            read_complete_evaluation(dataset) for dataset in sources
        ]
        left, right = comparable_metadata(base), comparable_metadata(extra)
        mismatched = different_fields(left, right)
        if mismatched:
            raise ValueError(
                f"Evaluation settings differ for {name}: {', '.join(mismatched)}"
            )
        if (
            set(base["frictions"]) & set(extra["frictions"])
            or old_rows.keys() & new_rows.keys()
        ):
            raise ValueError(f"Overlapping trial conditions for {name}")
        if (
            set(base["frictions"] + extra["frictions"]) != set(final.TEST_FRICTIONS)
            or set(base["maneuvers"]) != set(final.MANEUVERS)
            or tuple(base["noise_scales"]) != final.TEST_NOISE_SCALES
            or tuple(base["trial_seeds"]) != final.EVAL_SEEDS
        ):
            raise ValueError(
                f"Combined trials do not match the finalized grid for {name}"
            )
        checkpoint_hash = base["checkpoint_sha256"]
        if not checkpoint_hash or checkpoint_hash in checkpoint_hashes:
            raise ValueError(f"Missing or duplicate checkpoint identity for {name}")
        checkpoint_hashes.add(checkpoint_hash)

        fields = tuple(next(iter(old_rows.values())))
        if fields != tuple(next(iter(new_rows.values()))):
            raise ValueError(f"Trial index schemas differ for {name}")
        metadata = base | {
            "frictions": final.TEST_FRICTIONS,
            "trials_written": len(old_rows) + len(new_rows),
            "merge_sources": [
                {
                    "dataset": str(dataset),
                    "metadata_sha256": hashlib.sha256(
                        (dataset / "metadata.json").read_bytes()
                    ).hexdigest(),
                    "frictions": source["frictions"],
                    "trials_written": source["trials_written"],
                }
                for dataset, source in zip(sources, (base, extra), strict=True)
            ],
        }
        plan.append((name, metadata, fields, sources, (old_rows, new_rows)))
    return plan


def remove_training_condition(config):
    for name in ("experiment_name", "run_name"):
        config["runner"].pop(name)
    for name in ("add_noise", "noise"):
        config["actor"].pop(name)


def comparable_controller(profile):
    profile = deepcopy(profile)
    env, config = profile["env_config"], profile["train_config"]
    env.pop("seed")
    env["domain_randomization"]["startup"].pop("contact_friction_range")
    config.pop("seed")
    config.pop("log_dir")
    remove_training_condition(config)
    return profile


def comparable_protocol(metadata):
    settings = comparable_metadata(metadata)
    for field in (
        "checkpoint",
        "checkpoint_sha256",
        "training",
        "controller_profile",
        "merge_sources",
    ):
        settings.pop(field, None)
    # The evaluator implements test noise separately from these saved actor settings.
    remove_training_condition(settings["runner_config"])
    return settings


def check_training_condition(profile, training, directory):
    env, config = profile["env_config"], profile["train_config"]
    friction = env["domain_randomization"]["startup"]["contact_friction_range"]
    noise = training["observation_noise"]["effective_std"]
    if (
        env["seed"] != training["seed"]
        or config["seed"] != training["seed"]
        or friction != training["friction_range"]
        or config["actor"]["noise"] != noise
        or config["actor"]["add_noise"] != bool(noise)
    ):
        raise ValueError(
            f"Controller profile disagrees with training conditions in {directory}"
        )


def validate_disjoint_policies(sources, *, require_final_matrix):
    from controller_checkpoint import read_controller_profile

    conditions = {run["experiment_name"]: run for run in final.final_training_runs()}
    counts, reference = [], None
    experiments, checkpoint_hashes, templates = set(), set(), set()
    initializations = {}
    for source in sources:
        count = 0
        for name, directory in source.items():
            meta, rows = read_complete_evaluation(directory)
            count += len(rows)
            if (
                set(meta["frictions"]) != set(final.TEST_FRICTIONS)
                or set(meta["maneuvers"]) != set(final.MANEUVERS)
                or tuple(meta["noise_scales"]) != final.TEST_NOISE_SCALES
                or tuple(meta["trial_seeds"]) != final.EVAL_SEEDS
            ):
                raise ValueError(f"Expected the complete finalized grid in {directory}")
            if any(Path(row["trace"]).is_absolute() for row in rows.values()):
                raise ValueError(f"Expected relative trace paths in {directory}")
            if not meta.get("controller_telemetry", {}).get("coordination_diagnostics"):
                raise ValueError(
                    f"Expected finalized contact diagnostics in {directory}"
                )

            checkpoint = Path(meta["checkpoint"])
            experiment = checkpoint.parent.parent.name
            if (
                experiment not in conditions
                or experiment in experiments
                or checkpoint.name != f"model_{final.ITERATIONS}.pt"
                or name != Path(experiment, checkpoint.parent.name, checkpoint.stem)
                or meta["checkpoint_iteration"] != final.ITERATIONS
                or meta.get("training_stage") != "paired-contact"
            ):
                raise ValueError(
                    f"Unexpected or duplicate finalized policy in {directory}"
                )
            experiments.add(experiment)
            with checkpoint.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != meta["checkpoint_sha256"] or digest in checkpoint_hashes:
                raise ValueError(
                    f"Mismatched or duplicate checkpoint hash in {directory}"
                )
            checkpoint_hashes.add(digest)

            training = final.read_completed_run(
                checkpoint.parent, conditions[experiment]
            )
            if training != meta["training"]:
                raise ValueError(
                    f"Training provenance differs from the manifest in {directory}"
                )
            profile = read_controller_profile(
                checkpoint, expected_iteration=final.ITERATIONS
            )
            if (
                profile != training["controller_profile"]
                or profile != meta["controller_profile"]
            ):
                raise ValueError(
                    f"Controller profile differs from the checkpoint in {directory}"
                )
            legacy = final.staged_checkpoint_profile(
                training, final.REWARD_TRANSITION_ITERATION
            )
            for stage_profile in (legacy, profile):
                check_training_condition(stage_profile, training, directory)
            details = training["final_research"]
            templates.add((details["template_checkpoint_sha256"], details["run_name"]))
            init = (details["initial_actor_sha256"], details["initial_critic_sha256"])
            if initializations.setdefault(training["seed"], init) != init:
                raise ValueError(
                    f"Training initialization differs for seed {training['seed']}"
                )

            settings = {
                "evaluation": comparable_protocol(meta),
                "legacy controller": comparable_controller(legacy),
                "paired-contact controller": comparable_controller(profile),
            }
            if reference is None:
                reference = settings
            for kind, fields in settings.items():
                mismatched = different_fields(reference[kind], fields)
                if mismatched:
                    raise ValueError(
                        f"{kind.capitalize()} settings differ for {name}: "
                        f"{', '.join(mismatched)}"
                    )
        counts.append(count)
    if len(templates) != 1:
        raise ValueError(
            "Policies must share the finalized run label and training template"
        )
    if require_final_matrix and experiments != conditions.keys():
        missing = ", ".join(sorted(conditions.keys() - experiments))
        raise ValueError(f"Finalized training conditions are missing: {missing}")
    return tuple(counts)


def write_condition_merge(merged, plan):
    for name, metadata, fields, sources, indexes in plan:
        directory = merged / name
        (directory / "traces").mkdir(parents=True)
        rows = []
        for source, index in zip(sources, indexes, strict=True):
            for original in index.values():
                trial_id = f"trial_{len(rows):06d}"
                trace = f"traces/{trial_id}.npz"
                copyfile(source / original["trace"], directory / trace)
                rows.append(
                    original
                    | {
                        "trial_id": trial_id,
                        "trace": trace,
                        "source_trial_id": original["trial_id"],
                        "source_dataset": str(source),
                    }
                )
        write_csv(
            directory / "trials.csv",
            fields + ("source_trial_id", "source_dataset"),
            rows,
        )
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def merge_evaluations(
    existing, additional, output, *, dry_run=False, require_final_matrix=False
):
    existing, additional, output = (
        Path(path).resolve() for path in (existing, additional, output)
    )
    if existing == additional or any(
        output.is_relative_to(source) for source in (existing, additional)
    ):
        raise ValueError(
            "Use distinct source directories and a separate output directory"
        )
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    original, supplemental = datasets_under(existing), datasets_under(additional)
    disjoint = original.keys().isdisjoint(supplemental)
    if disjoint:
        counts = validate_disjoint_policies(
            (original, supplemental), require_final_matrix=require_final_matrix
        )
    elif original.keys() == supplemental.keys():
        if require_final_matrix:
            raise ValueError(
                "--require-final-matrix requires disjoint policy directories"
            )
        plan = plan_condition_merge(original, supplemental)
        counts = tuple(
            sum(meta["merge_sources"][index]["trials_written"] for _, meta, *_ in plan)
            for index in (0, 1)
        )
    else:
        raise ValueError("Supply identical or disjoint sets of checkpoint directories")
    if dry_run:
        return counts

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".evaluation-merge-") as staging:
        merged = Path(staging) / "evaluation"
        if disjoint:
            for name, source in (original | supplemental).items():
                directory = merged / name
                copytree(source, directory)
                read_complete_evaluation(directory)
        else:
            write_condition_merge(merged, plan)
        if output.exists():
            raise FileExistsError(f"Output directory already exists: {output}")
        merged.rename(output)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("existing", type=Path, help="Original evaluation root")
    parser.add_argument("additional", type=Path, help="Supplemental evaluation root")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--require-final-matrix",
        action="store_true",
        help="Require all 14 finalized policies when combining disjoint datasets",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate both inputs without copying"
    )
    args = parser.parse_args(argv)
    try:
        old, new = merge_evaluations(
            args.existing,
            args.additional,
            args.output_dir,
            dry_run=args.dry_run,
            require_final_matrix=args.require_final_matrix,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    action = "Validated" if args.dry_run else "Merged"
    print(f"{action} {old} original and {new} supplemental trials ({old + new} total)")


if __name__ == "__main__":
    main()
