from __future__ import annotations

from datetime import datetime
from pathlib import Path

from API_Group8 import BioreactorClient, USER, PASSWORD
from latin_hypersample import build_samples_per_scale, run_sampling
from save_results import save_results_to_csv


def task01_run(samples_per_scale: int | None = None, total_samples: int | None = None,
               out_path: str | None = None, repeats: int = 1, save: bool = True) -> list[dict]:
    """Run Task01 Latin Hypercube sampling and optionally save the results."""
    samples = build_samples_per_scale(samples_per_scale, total_samples)
    client = BioreactorClient()
    client.login(USER, PASSWORD)

    results = run_sampling(client, samples, repeats=repeats)

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
