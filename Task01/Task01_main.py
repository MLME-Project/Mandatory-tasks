#Task 1: API connection and experiment execution
#A: Implement a function that connects to the Bioreactor API, logs in, and runs an experiment with specified parameters. The function should return the result of the experiment.
#B: Report: Vizulize the results of the experiment using appropriate plots. 

"""Task01 entrypoint for Latin Hypercube sampling.

This wrapper allows Task01_main.py to be called from the shell and forwards
arguments to the Task01 runner module.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from latin_hypersample import build_samples_per_scale
from task01_runner import task01_run
from task01_summary import summarize_task01_results
from task01_visualize import visualize_task01_results
from task01_diagnostics import analyze_tp_distribution, print_tp_report, save_tp_heatmap


@dataclass(frozen=True)
class Task01Config:
    samples_per_scale: int | None
    repeats: int
    sampling_mode: str
    t_min: float
    t_max: float
    ph_min: float
    ph_max: float
    f1_min: float
    f1_max: float
    f2_min: float
    f2_max: float
    f3_min: float
    f3_max: float
    visualize: bool
    t_bins: int
    ph_bins: int
    out_path: str | None
    save: bool

    def bounds(self) -> list[tuple[float, float]]:
        return [
            (self.t_min, self.t_max),
            (self.ph_min, self.ph_max),
            (self.f1_min, self.f1_max),
            (self.f2_min, self.f2_max),
            (self.f3_min, self.f3_max),
        ]


# ---------------------------------------------------------------------------
# Benutzerkonfiguration
# Bearbeite die Werte unten direkt in dieser Datei und speichere sie. Wenn das
# Skript ohne Kommandozeilenargumente gestartet wird (z.B. über "Run Python
# File"), werden diese Werte verwendet.
#
# Wichtige Einstellungen:
# - `samples_per_scale`: Anzahl der Messpunkte pro Skala (micro/bench/pilot).
# - `repeats`: Anzahl der Wiederholungen pro Messpunkt.
#
# Bearbeite hier nur `samples_per_scale` und `repeats`.
# ---------------------------------------------------------------------------
USER_CONFIG = Task01Config(
    samples_per_scale=1000,  # int or None
    repeats=2,
    sampling_mode="random",  # "lhs" or "random"
    # Temperature bounds
    t_min=20.0,
    t_max=60.0,
    # pH bounds
    ph_min=3.0,
    ph_max=9.5,
    # Feed fractions bounds
    f1_min=0.0,
    f1_max=2.0,
    f2_min=0.0,
    f2_max=2.0,
    f3_min=0.0,
    f3_max=2.0,
    # Visualization and bucketing
    visualize=True,
    t_bins=4,
    ph_bins=4,
    out_path=None,
    save=True,
)

# Ende Benutzerkonfiguration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task01 main wrapper for Latin Hypercube sampling")
    g = p.add_mutually_exclusive_group()
    p.add_argument("--samples-per-scale", "--feintheit", type=int,
                   help="Number of samples per scale (micro/bench/pilot). Gesamtpunkte = 3 * value.")
    p.add_argument("--repeats", type=int, default=1,
                   help="Anzahl der Wiederholungen je Stichprobe. Gesamt-API-Aufrufe = Stichproben * repeats.")
    p.add_argument("--out", type=str, default=None,
                   help="Output CSV path or file name")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save experiment results to disk")
    p.add_argument("--no-visualization", action="store_true",
                   help="Do not generate triangle plots and heatmap")
    p.add_argument("--t-bins", type=int, default=8,
                   help="Number of buckets in T direction for the triangle overview")
    p.add_argument("--ph-bins", type=int, default=8,
                   help="Number of buckets in pH direction for the triangle overview")
    p.add_argument("--t-min", type=float, default=20.0,
                   help="Minimum temperature bound for sampling")
    p.add_argument("--t-max", type=float, default=60.0,
                   help="Maximum temperature bound for sampling")
    p.add_argument("--ph-min", type=float, default=3.0,
                   help="Minimum pH bound for sampling")
    p.add_argument("--ph-max", type=float, default=9.5,
                   help="Maximum pH bound for sampling")
    p.add_argument("--f1-min", type=float, default=0.0,
                   help="Minimum F1 bound for sampling")
    p.add_argument("--f1-max", type=float, default=2.0,
                   help="Maximum F1 bound for sampling")
    p.add_argument("--f2-min", type=float, default=0.0,
                   help="Minimum F2 bound for sampling")
    p.add_argument("--f2-max", type=float, default=2.0,
                   help="Maximum F2 bound for sampling")
    p.add_argument("--f3-min", type=float, default=0.0,
                   help="Minimum F3 bound for sampling")
    p.add_argument("--f3-max", type=float, default=2.0,
                   help="Maximum F3 bound for sampling")
    p.add_argument("--mode", choices=["lhs", "random"], default="lhs",
                   help="Sampling mode: lhs for Latin Hypercube, random for purely random sampling")
    return p.parse_args()


def main() -> None:
    # Wenn das Skript ohne CLI-Argumente gestartet wird, verwende den
    # editierbaren `USER_CONFIG`-Block oben. Ansonsten werden die CLI-Optionen
    # wie zuvor geparst und verwendet.
    if len(sys.argv) == 1:
        config = USER_CONFIG
    else:
        args = parse_args()
        config = Task01Config(
            samples_per_scale=args.samples_per_scale,
            repeats=args.repeats,
            sampling_mode=args.mode,
            t_min=args.t_min,
            t_max=args.t_max,
            ph_min=args.ph_min,
            ph_max=args.ph_max,
            f1_min=args.f1_min,
            f1_max=args.f1_max,
            f2_min=args.f2_min,
            f2_max=args.f2_max,
            f3_min=args.f3_min,
            f3_max=args.f3_max,
            visualize=not args.no_visualization,
            t_bins=args.t_bins,
            ph_bins=args.ph_bins,
            out_path=args.out,
            save=not args.no_save,
        )

    print("Task01 Konfiguration:")
    print(f"  Samples pro Scale: {config.samples_per_scale}")
    print(f"  Repeats: {config.repeats}")
    print(f"  Grenzen: T=[{config.t_min}, {config.t_max}], pH=[{config.ph_min}, {config.ph_max}], F1=[{config.f1_min}, {config.f1_max}], F2=[{config.f2_min}, {config.f2_max}], F3=[{config.f3_min}, {config.f3_max}]")
    print(f"  Sampling mode: {config.sampling_mode}")
    print(f"  Visualisierung: {config.visualize}")
    print(f"  T-Buckets: {config.t_bins}, pH-Buckets: {config.ph_bins}")
    print(f"  Save results: {config.save}")
    print(f"  Output path: {config.out_path or 'automatisch generiert'}")

    scale_counts = build_samples_per_scale(config.samples_per_scale, None)
    print(f"  Ergebnis der Sample-Aufteilung: {scale_counts} (gesamt {sum(scale_counts.values())})")

    results = task01_run(
        samples_per_scale=config.samples_per_scale,
        out_path=config.out_path,
        repeats=config.repeats,
        save=config.save,
        bounds=config.bounds(),
        mode=config.sampling_mode,
    )

    summarize_task01_results(results)

    if config.visualize:
        visualize_task01_results(results, t_bins=config.t_bins, ph_bins=config.ph_bins)
    else:
        print("Visualisierung wurde deaktiviert.")

    hist = analyze_tp_distribution(results, t_bins=config.t_bins, ph_bins=config.ph_bins)
    print_tp_report(hist, highlight_frac=0.5)
    if config.visualize:
        save_tp_heatmap(hist, output_path="Task01_TP_heatmap.png")
    else:
        print("Heatmap-Ausgabe wurde deaktiviert.")


if __name__ == "__main__":
    main()

