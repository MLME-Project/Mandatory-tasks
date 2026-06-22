"""Latin-hypercube sampling runner for Bioreactor API.

This module generates latin hypercube samples for all three reactor scales:
micro, bench and pilot. It executes experiments by calling the Bioreactor API and
returns the collected results. Saving results is handled by a separate module.

Usage examples:
  python latin_hypersample.py --samples-per-scale 30
  python latin_hypersample.py --total-samples 90
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from API_Group8 import BioreactorClient, USER, PASSWORD


def lhs_sample(n: int, dims: int) -> list[list[float]]:
    """Generate an n x dims Latin Hypercube sample in [0,1].

    Each of the `dims` variables is divided into `n` equal-width strata.
    One random point is selected from each stratum for each dimension, and the
    order is shuffled so that every sample contains a unique combination of
    stratum indices.
    """
    # Prepare an empty list to collect sample rows.
    result = [[0.0] * dims for _ in range(n)]

    # Build the sample one dimension at a time.
    for j in range(dims):
        # For dimension j, generate one random point inside each of the n strata.
        coords = [(i + random.random()) / n for i in range(n)]

        # Shuffle the points so that the samples receive a random permutation
        # of the stratum positions in this dimension.
        random.shuffle(coords)

        # Assign the generated coordinate values to the current dimension.
        for i in range(n):
            result[i][j] = coords[i]

    return result


def map_to_bounds(sample: list[float], bounds: list[tuple[float, float]]) -> dict:
    """Convert a normalized sample in [0, 1] into actual reactor parameter values."""
    names = ["T", "pH", "F1", "F2", "F3"]

    # Map each normalized coordinate to the corresponding physical bound.
    return {
        name: low + s * (high - low)
        for s, (low, high), name in zip(sample, bounds, names)
    }


def generate_samples_per_scale(n: int, bounds: list[tuple[float, float]]) -> list[dict]:
    """Generate `n` reactor parameter sets for a single scale."""
    # Generate normalized LHS points in the unit hypercube.
    raw = lhs_sample(n, len(bounds))

    # Map each raw sample to the real parameter ranges.
    mapped = [map_to_bounds(r, bounds) for r in raw]
    return mapped


def build_samples_per_scale(samples_per_scale: int | None, total_samples: int | None) -> dict[str, int]:
    """Build a sample count dictionary for the three reactor scales."""
    scales = ["micro", "bench", "pilot"]
    if samples_per_scale:
        return {s: samples_per_scale for s in scales}
    if total_samples:
        base = total_samples // len(scales)
        rem = total_samples % len(scales)
        return {s: base + (1 if i < rem else 0) for i, s in enumerate(scales)}
    return {"micro": 30, "bench": 30, "pilot": 30}


def run_sampling(client: BioreactorClient, samples_per_scale: dict[str, int], repeats: int = 1) -> list[dict]:
    """Run sampling for each scale and return the collected experiment results."""
    # Define the allowed ranges for each parameter.
    bounds = [(20.0, 60.0), (3.0, 9.5), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
    results: list[dict] = []

    # Iterate through each reactor scale and generate its samples.
    for scale, n in samples_per_scale.items():
        samples = generate_samples_per_scale(n, bounds)

        for idx, sample in enumerate(samples, start=1):
            # Round parameter values for cleaner output.
            base_row = {
                "scale": scale,
                "sample_idx": idx,
                **{k: float(f"{v:.6f}") for k, v in sample.items()},
            }

            for rep in range(1, repeats + 1):
                row = {**base_row, "repeat": rep}

                # Execute the experiment through the API client.
                try:
                    print(f"Running {scale} sample {idx}/{n} repeat {rep}/{repeats} ...", flush=True)
                    res = client.run(scale, T=row["T"], pH=row["pH"], F1=row["F1"], F2=row["F2"], F3=row["F3"])
                    row.update({"status": "ok", "result": json.dumps(res, ensure_ascii=False)})
                except Exception as e:
                    # Capture failures without stopping the full sampling loop.
                    row.update({"status": f"error: {type(e).__name__}", "result": str(e)})

                results.append(row)
                # Pause briefly to reduce risk of triggering API rate limits.
                time.sleep(0.25)

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sampling runner."""
    p = argparse.ArgumentParser(description="Latin Hypercube sampling for Bioreactor API")
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--samples-per-scale",
        "--feintheit",
        type=int,
        help="Number of samples per scale (micro/bench/pilot); controls Latin Hypercube fineness",
    )
    g.add_argument("--total-samples", type=int, help="Total number of samples across all scales; split evenly")
    p.add_argument("--out", type=str, default=None, help="Output CSV path")
    p.add_argument("--repeats", type=int, default=1, help="Number of repeated measurements per sample point")
    return p.parse_args()


def main() -> None:
    """Main entrypoint for script execution."""
    args = parse_args()

    samples_per_scale = build_samples_per_scale(args.samples_per_scale, args.total_samples)
    
    # Generate filename with pattern: Results_DD_MM_HH_MM_SS.xlsx
    now = datetime.now()
    filename = now.strftime("Results_%d_%m_%H_%M_%S.xlsx")
    out = Path(args.out) if args.out else Path(filename)

    # Always create the API client and execute real runs.
    client = BioreactorClient()
    client.login(USER, PASSWORD)

    results = run_sampling(client, samples_per_scale, repeats=args.repeats)
    from save_results import save_results_to_csv

    save_results_to_csv(results, out)
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
