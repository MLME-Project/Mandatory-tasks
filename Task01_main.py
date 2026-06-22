#Task 1: API connection and experiment execution
#A: Implement a function that connects to the Bioreactor API, logs in, and runs an experiment with specified parameters. The function should return the result of the experiment.
#B: Report: Vizulize the results of the experiment using appropriate plots. 

"""Task01 entrypoint for Latin Hypercube sampling.

This wrapper allows Task01_main.py to be called from the shell and forwards
arguments to the Task01 runner module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from task01_runner import task01_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task01 main wrapper for Latin Hypercube sampling")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--samples-per-scale", "--feintheit", type=int,
                   help="Number of samples per scale (micro/bench/pilot)")
    g.add_argument("--total-samples", type=int,
                   help="Total number of samples across all scales; split evenly")
    p.add_argument("--out", type=str, default=None, help="Output CSV path")
    p.add_argument("--repeats", type=int, default=1,
                   help="Number of repeated measurements per sample point")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save results to disk")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_results = not args.no_save
    # Uncomment the following line to disable saving in the wrapper:
    # save_results = False

    task01_run(
        samples_per_scale=args.samples_per_scale,
        total_samples=args.total_samples,
        out_path=args.out,
        repeats=args.repeats,
        save=save_results,
    )

