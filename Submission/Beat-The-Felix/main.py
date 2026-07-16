'''
This is just an example for how you submission for Beat-the-Felix should look like.
There should be one file called main.py in the root of your submission folder.
Running this file should execute your algorithm and print the mean perfoemance of
your algorithm over 10 independent trials.
'''

import csv
import glob
import os
import shutil
import subprocess
import sys

N_RUNS = 10  # how many independent bo_loopv6.py runs to execute

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_v2")
BO_SCRIPT = os.path.join(SCRIPT_DIR, "bo_loopv6.py")


def clear_output_dir() -> None:
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_bo_loop_once(run_idx: int) -> str:
    """Run bo_loopv6.py as a subprocess and return the path of the CSV it created."""
    before = set(glob.glob(os.path.join(OUTPUT_DIR, "*.csv")))

    print(f"\n{'#' * 80}\n# RUN {run_idx} / {N_RUNS}\n{'#' * 80}", flush=True)
    subprocess.run([sys.executable, BO_SCRIPT], cwd=SCRIPT_DIR, check=True)

    after = set(glob.glob(os.path.join(OUTPUT_DIR, "*.csv")))
    new_files = after - before
    if len(new_files) != 1:
        raise RuntimeError(
            f"Expected exactly one new CSV file in {OUTPUT_DIR} after run "
            f"{run_idx}, found {len(new_files)}: {new_files}"
        )
    return new_files.pop()


def analyze_run(csv_path: str) -> dict:
    """Extract final yield, cumulative cost, and number of pilot evaluations."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    # bo_loopv6.py logs every experiment in run order; the last row is the
    # final pilot validation run whose yield it also prints at the end.
    final_yield = float(rows[-1]["Yield"])
    cumulative_cost = sum(float(row["Cost"]) for row in rows)
    n_pilot_evals = sum(1 for row in rows if row["Scale"] == "pilot")

    return {
        "csv": os.path.basename(csv_path),
        "final_yield": final_yield,
        "cumulative_cost": cumulative_cost,
        "n_pilot_evals": n_pilot_evals,
    }


if __name__ == "__main__":
    clear_output_dir()

    results = [analyze_run(run_bo_loop_once(run_idx))
               for run_idx in range(1, N_RUNS + 1)]

    print("\n" + "=" * 80)
    print("SUMMARY OVER ALL RUNS")
    print("=" * 80)
    print(f"{'Run':>4} {'Final Yield':>14} {'Cumulative Cost':>18} {'# Pilot Evals':>14}   CSV")
    for i, r in enumerate(results, start=1):
        print(f"{i:>4} {r['final_yield']:>14.6f} {r['cumulative_cost']:>18,.0f} "
              f"{r['n_pilot_evals']:>14d}   {r['csv']}")

    mean_yield = sum(r["final_yield"] for r in results) / len(results)
    mean_cost = sum(r["cumulative_cost"] for r in results) / len(results)
    mean_pilots = sum(r["n_pilot_evals"] for r in results) / len(results)

    print("-" * 80)
    print(f"{'MEAN':>4} {mean_yield:>14.6f} {mean_cost:>18,.0f} {mean_pilots:>14.2f}")
    print("=" * 80)
