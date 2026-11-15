"""Aggregate saved simulation results across 7 friction ranges and 3 seeds.

Run from BigRedGym root AFTER evaluation:
    uv run --frozen experiments/go2_friction/analyze_friction.py

Outputs summary CSV and a critical-friction plot. Uses matplotlib only for plot.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


PATTERN = re.compile(r"^go2friction_lb_(\d+p\d+)_seed_(\d+)$")


def read_rows(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("*/summary.csv")):
        match = PATTERN.fullmatch(path.parent.name)
        if not match:
            continue
        lower = float(match.group(1).replace("p", "."))
        seed = int(match.group(2))
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row.update(training_lower_bound=lower, training_seed=seed)
                rows.append(row)
    return rows


def compute_critical(rows: list[dict]) -> list[dict]:
    out = []
    groups = {}
    for row in rows:
        if row["maneuver"] != "stand_2_stop":
            continue
        key = (row["training_lower_bound"], row["training_seed"])
        groups.setdefault(key, []).append(row)
    for (lower, seed), group in sorted(groups.items()):
        # Only the proposal's 9/10 rule qualifies. Fewer trials cannot be used.
        eligible = [float(r["friction"]) for r in group
                    if int(r["trials"]) == 10 and int(r["successes"]) >= 9]
        out.append({"training_lower_bound": lower, "training_seed": seed,
                    "critical_friction": min(eligible) if eligible else ""})
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate Go2 friction results")
    parser.add_argument("--input", default="results")
    parser.add_argument("--output", default="results/aggregate")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent
    input_dir = repo / args.input
    output_dir = repo / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_dir)
    if not rows:
        parser.error(f"No evaluated policies found in {input_dir}")
    critical = compute_critical(rows)
    write_csv(output_dir / "all_conditions.csv", rows)
    write_csv(output_dir / "critical_friction_by_seed.csv", critical)

    groups = {}
    for row in critical:
        if row["critical_friction"] == "":
            continue
        groups.setdefault(row["training_lower_bound"], []).append(float(row["critical_friction"]))
    summary = []
    for lower, vals in sorted(groups.items()):
        summary.append({"training_lower_bound": lower, "seeds_with_pass": len(vals),
                        "critical_mu_mean": statistics.mean(vals),
                        "critical_mu_std": statistics.stdev(vals) if len(vals) >= 2 else ""})
    write_csv(output_dir / "critical_friction_summary.csv", summary)
    if summary:
        import matplotlib.pyplot as plt
        xs = [row["training_lower_bound"] for row in summary]
        ys = [row["critical_mu_mean"] for row in summary]
        errs = [row["critical_mu_std"] or 0 for row in summary]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(xs, ys, yerr=errs, fmt="o-", capsize=4)
        ax.set_xlabel("Minimum training friction coefficient")
        ax.set_ylabel("Lowest successful evaluation friction")
        ax.set_title("Go2 friction robustness across paired seeds")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "critical_friction.png", dpi=180)
        plt.close(fig)
    print(f"Saved aggregate results to {output_dir}")


if __name__ == "__main__":
    main()
