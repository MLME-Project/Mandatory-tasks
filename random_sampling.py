from __future__ import annotations

import json
import random
import time
from typing import Protocol

from API_Group8 import BioreactorClient


class RandomClient(Protocol):
    def run(self, scale: str, T: float, pH: float, F1: float, F2: float, F3: float) -> object:
        ...


def generate_random_samples(n: int, bounds: list[tuple[float, float]]) -> list[dict]:
    """Generate `n` reactor parameter sets using uniform random sampling."""
    names = ["T", "pH", "F1", "F2", "F3"]
    return [
        {
            name: random.uniform(low, high)
            for name, (low, high) in zip(names, bounds)
        }
        for _ in range(n)
    ]


def run_random_sampling(client: BioreactorClient, samples_per_scale: dict[str, int], repeats: int = 1,
                        bounds: list[tuple[float, float]] | None = None) -> list[dict]:
    """Run random sampling for each scale and return the collected experiment results."""
    if bounds is None:
        bounds = [(20.0, 60.0), (3.0, 9.5), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
    results: list[dict] = []

    for scale, n in samples_per_scale.items():
        samples = generate_random_samples(n, bounds)

        for idx, sample in enumerate(samples, start=1):
            base_row = {
                "scale": scale,
                "sample_idx": idx,
                **{k: float(f"{v:.6f}") for k, v in sample.items()},
            }

            for rep in range(1, repeats + 1):
                row = {**base_row, "repeat": rep}
                try:
                    print(f"Random sampling {scale} sample {idx}/{n} repeat {rep}/{repeats} ...", flush=True)
                    res = client.run(scale, T=row["T"], pH=row["pH"], F1=row["F1"], F2=row["F2"], F3=row["F3"])
                    row.update({"status": "ok", "result": json.dumps(res, ensure_ascii=False)})
                except Exception as e:
                    row.update({"status": f"error: {type(e).__name__}", "result": str(e)})

                results.append(row)
                time.sleep(0.25)

    return results
