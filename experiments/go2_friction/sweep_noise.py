"""Train Go2trot at the configured actor noise scales with fixed friction."""

import argparse
import math

from sweep_friction import (
    REFERENCE_FRICTION,
    add_sweep_arguments,
    noise_experiment_name,
    parse_seeds,
    parse_values,
    run_sweep,
)

NOISE_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-scales", default=",".join(map(str, NOISE_SCALES)))
    add_sweep_arguments(parser)
    parser.set_defaults(upper=REFERENCE_FRICTION)
    args = parser.parse_args(argv)
    seeds = parse_seeds(parser, args)
    try:
        scales = parse_values(args.noise_scales, float)
    except ValueError as exc:
        parser.error(f"Invalid --noise-scales: {exc}")
    if any(not math.isfinite(scale) or scale < 0 for scale in scales):
        parser.error("--noise-scales must be finite and nonnegative")
    if len(set(scales)) != len(scales):
        parser.error("--noise-scales must be unique")

    print(
        f"{len(scales)} noise conditions x {len(seeds)} seeds "
        f"= {len(scales) * len(seeds)} runs"
    )
    runs = (
        {
            "experiment_name": noise_experiment_name(scale, seed),
            "lower_bound": REFERENCE_FRICTION,
            "noise_scale": scale,
            "seed": seed,
        }
        for scale in scales
        for seed in seeds
    )
    run_sweep(args, runs, "noise_sweep_runs.csv")


if __name__ == "__main__":
    main()
