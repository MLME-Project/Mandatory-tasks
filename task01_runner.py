from __future__ import annotations

from datetime import datetime
from pathlib import Path

from API_Group8 import BioreactorClient, USER, PASSWORD
from latin_hypersample import build_samples_per_scale, run_sampling as run_lhs_sampling
from save_results import save_results_to_csv


def task01_run(samples_per_scale: int | None = None, total_samples: int | None = None,
               out_path: str | None = None, repeats: int = 1, save: bool = True,
               bounds: list[tuple[float, float]] | None = None,
               mode: str = "lhs") -> list[dict]:
    """Run Task01 sampling and optionally save the results.

    The optional `bounds` parameter allows overriding the default sampling ranges
    for T, pH, F1, F2 and F3. The `mode` parameter selects the sampling method.
    """
    samples = build_samples_per_scale(samples_per_scale, total_samples)
    total_points = sum(samples.values())
    if total_samples is not None:
        print(f"Using total_samples={total_samples} across all scales.")
    elif samples_per_scale is not None:
        print(f"Using samples_per_scale={samples_per_scale} for each scale.")
    else:
        print("Using default sample count: 30 per scale.")
    print(f"Sample counts per scale: {samples} (total sample points = {total_points})")
    print(f"Repeats per point: {repeats}; total API calls = {total_points * repeats}")
    client = BioreactorClient()
    client.login(USER, PASSWORD)

    if mode == "lhs":
        results = run_lhs_sampling(client, samples, repeats=repeats, bounds=bounds)
    elif mode == "random":
        from random_sampling import run_random_sampling
        results = run_random_sampling(client, samples, repeats=repeats, bounds=bounds)
    else:
        raise ValueError(f"Unsupported sampling mode: {mode}")

    if save:
        if out_path is None:
            # Generate filename with pattern: Results_DD_MM_HH_MM_SS.xlsx
            now = datetime.now()
            filename = now.strftime("Results_%d_%m_%H_%M_%S.xlsx")
            out = Path(filename)
        else:
            out = Path(out_path)
        save_results_to_csv(results, out)
    return results
