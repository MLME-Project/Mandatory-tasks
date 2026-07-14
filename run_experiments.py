"""Run the Bayesian-optimization task repeatedly for a fixed set of init/acquisition combinations.

The script calls the existing Task03 optimization entrypoint as a subprocess, reuses its
per-run CSV outputs, and writes a combined summary CSV after each completed run so the
batch can be resumed safely.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from data_frame_and_csv_manipulation import getDataFrameFromCSV, projectToPilotScale


COMBINATIONS = [
    ("lhc", "ei"),
    ("lhc", "ucb"),
    ("lhc", "pi"),
    ("corners", "ucb"),
    ("random", "ucb"),
]


def build_output_filename(acq_fun, init_method, patience, init_samples, run_id):
    return os.path.join(
        ROOT_DIR,
        "Task03",
        f"data_{acq_fun}_{init_method}_param02_patience{patience}_lhc{init_samples}_final_{run_id}.csv",
    )


def build_log_path(acq_fun, init_method, run_id):
    log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{acq_fun}_{init_method}_run{run_id}.log")


def resolve_script_path():
    preferred = os.path.join(ROOT_DIR, "Task03_Philipp.py")
    if os.path.exists(preferred):
        return preferred

    fallback = os.path.join(ROOT_DIR, "Task03", "Task03_main.py")
    if os.path.exists(fallback):
        return fallback

    raise FileNotFoundError("Could not find Task03_Philipp.py or Task03/Task03_main.py")


def build_command(script_path, acq_fun, init_method, run_id, patience, init_samples, budget):
    return [
        sys.executable,
        script_path,
        "--acq",
        acq_fun,
        "--init-method",
        init_method,
        "--run-id",
        str(run_id),
        "--patience",
        str(patience),
        "--init-samples",
        str(init_samples),
        "--budget",
        str(budget),
    ]


def summarize_result(output_csv, acq_fun, init_method, run_id, log_path, status):
    if not os.path.exists(output_csv):
        return {
            "init_method": init_method,
            "acq_fun": acq_fun,
            "run_id": run_id,
            "status": status,
            "num_experiments": 0,
            "total_cost": np.nan,
            "best_y": np.nan,
            "best_scale": "",
            "best_projected_pilot_y": np.nan,
            "final_pilot_y": np.nan,
            "source_csv": os.path.basename(output_csv),
            "log_file": os.path.basename(log_path),
        }

    df = getDataFrameFromCSV(output_csv)
    num_experiments = int(len(df))
    total_cost = float(np.sum(df["cost_eur"])) if "cost_eur" in df.columns else np.nan

    if num_experiments > 0:
        best_idx = int(np.argmax(df["Y"].values))
        best_y = float(df.iloc[best_idx]["Y"])
        best_scale = str(df.iloc[best_idx]["scale"])
        best_projected_pilot_y = float(projectToPilotScale(best_y, best_scale))
        pilot_rows = df[df["scale"] == "pilot"]
        final_pilot_y = float(pilot_rows.iloc[-1]["Y"]) if not pilot_rows.empty else np.nan
    else:
        best_y = np.nan
        best_scale = ""
        best_projected_pilot_y = np.nan
        final_pilot_y = np.nan

    return {
        "init_method": init_method,
        "acq_fun": acq_fun,
        "run_id": run_id,
        "status": status,
        "num_experiments": num_experiments,
        "total_cost": total_cost,
        "best_y": best_y,
        "best_scale": best_scale,
        "best_projected_pilot_y": best_projected_pilot_y,
        "final_pilot_y": final_pilot_y,
        "source_csv": os.path.basename(output_csv),
        "log_file": os.path.basename(log_path),
    }


def append_summary_row(summary_file, row):
    fieldnames = [
        "init_method",
        "acq_fun",
        "run_id",
        "status",
        "num_experiments",
        "total_cost",
        "best_y",
        "best_scale",
        "best_projected_pilot_y",
        "final_pilot_y",
        "source_csv",
        "log_file",
    ]

    file_exists = os.path.exists(summary_file)
    with open(summary_file, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def run_single_experiment(args, plan, summary_file):
    acq_fun, init_method, run_id, patience, init_samples, budget = plan
    output_csv = build_output_filename(acq_fun, init_method, patience, init_samples, run_id)
    log_path = build_log_path(acq_fun, init_method, run_id)

    if os.path.exists(output_csv):
        print(f"skipping {acq_fun}/{init_method} run {run_id}: {os.path.basename(output_csv)} already exists")
        row = summarize_result(output_csv, acq_fun, init_method, run_id, log_path, "skipped")
        append_summary_row(summary_file, row)
        return row

    print(f"starting {acq_fun}/{init_method} run {run_id}")
    script_path = resolve_script_path()
    cmd = build_command(script_path, acq_fun, init_method, run_id, patience, init_samples, budget)

    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("Command:\n")
        handle.write(" ".join(cmd) + "\n")
        handle.write("\n")
        start = time.time()
        try:
            completed = subprocess.run(
                cmd,
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            output_text = completed.stdout or ""
            handle.write(output_text)
            handle.flush()
            duration = time.time() - start
            if completed.returncode == 0:
                print(f"finished {acq_fun}/{init_method} run {run_id} in {duration:.1f}s, final Y: {summarize_result(output_csv, acq_fun, init_method, run_id, log_path, 'completed')['best_y']}")
                row = summarize_result(output_csv, acq_fun, init_method, run_id, log_path, "completed")
                append_summary_row(summary_file, row)
                return row

            print(f"failed {acq_fun}/{init_method} run {run_id} (exit {completed.returncode})")
            row = summarize_result(output_csv, acq_fun, init_method, run_id, log_path, "failed")
            append_summary_row(summary_file, row)
            return row
        except subprocess.TimeoutExpired as exc:
            handle.write(f"\nTimeout after {args.timeout} seconds\n")
            if exc.stdout:
                handle.write(exc.stdout)
            if exc.stderr:
                handle.write(exc.stderr)
            handle.flush()
            print(f"timed out {acq_fun}/{init_method} run {run_id}")
            row = summarize_result(output_csv, acq_fun, init_method, run_id, log_path, "failed")
            append_summary_row(summary_file, row)
            return row
        except Exception as exc:  # pragma: no cover - defensive fallback
            handle.write(f"\nUnexpected error: {exc}\n")
            handle.flush()
            print(f"error {acq_fun}/{init_method} run {run_id}: {exc}")
            row = summarize_result(output_csv, acq_fun, init_method, run_id, log_path, "failed")
            append_summary_row(summary_file, row)
            return row


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bayesian-optimization experiments repeatedly")
    parser.add_argument("--runs-per-combo", type=int, default=10, help="Number of runs per init/acq combination")
    parser.add_argument("--summary-file", type=str, default=os.path.join("Task03", "experiments_summary.csv"), help="Path to the combined summary CSV")
    parser.add_argument("--timeout", type=int, default=None, help="Optional subprocess timeout in seconds")
    parser.add_argument("--parallel", action="store_true", help="Execute runs in parallel when possible")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned subprocess calls without executing them")
    parser.add_argument("--patience", type=int, default=30, help="Patience passed to the underlying task script")
    parser.add_argument("--init-samples", type=int, default=20, help="Initial sampling size passed to the underlying task script")
    parser.add_argument("--budget", type=float, default=1e6, help="Budget passed to the underlying task script")
    return parser.parse_args()


def build_run_plan(args):
    plans = []
    for init_method, acq_fun in COMBINATIONS:
        for run_id in range(1, args.runs_per_combo + 1):
            plans.append((acq_fun, init_method, run_id, args.patience, args.init_samples, args.budget))
    return plans


def main():
    args = parse_args()
    plan = build_run_plan(args)

    summary_file = os.path.join(ROOT_DIR, args.summary_file) if not os.path.isabs(args.summary_file) else args.summary_file
    summary_dir = os.path.dirname(summary_file)
    os.makedirs(summary_dir, exist_ok=True)

    if args.dry_run:
        print("Dry run - no subprocesses will be executed.")
        for acq_fun, init_method, run_id, patience, init_samples, budget in plan:
            output_csv = build_output_filename(acq_fun, init_method, patience, init_samples, run_id)
            print(f"Would run: {acq_fun}/{init_method} run {run_id} -> {os.path.relpath(output_csv, ROOT_DIR)}")
        return

    if args.parallel:
        with ThreadPoolExecutor(max_workers=min(4, len(plan))) as executor:
            futures = [executor.submit(run_single_experiment, args, item, summary_file) for item in plan]
            for future in as_completed(futures):
                future.result()
    else:
        for item in plan:
            run_single_experiment(args, item, summary_file)

    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
