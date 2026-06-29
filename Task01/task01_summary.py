from __future__ import annotations

import json


def _parse_result_dict(result_value: object) -> dict | None:
    """Parse the stored result value into a dictionary if possible."""
    if isinstance(result_value, dict):
        return result_value
    if not isinstance(result_value, str):
        return None
    try:
        parsed = json.loads(result_value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def summarize_task01_results(results: list[dict]) -> dict[str, object]:
    """Summarize total cost and best Y values per scale, then print to shell."""
    cost_map = {"micro": 5.0, "bench": 200.0, "pilot": 2000.0}
    total_cost = 0.0
    best_yield_per_scale: dict[str, dict[str, object]] = {}
    trial_counts: dict[str, int] = {"micro": 0, "bench": 0, "pilot": 0}
    successful_trials = 0
    plottable_samples = 0
    plottable_counts: dict[str, int] = {"micro": 0, "bench": 0, "pilot": 0}

    for row in results:
        scale = row.get("scale")
        if isinstance(scale, str) and scale in cost_map:
            total_cost += cost_map[scale]
            trial_counts[scale] += 1

        if row.get("status") != "ok":
            continue

        successful_trials += 1
        parsed = _parse_result_dict(row.get("result"))
        if parsed is None:
            continue

        y_value = parsed.get("Y")
        if y_value is None:
            y_value = parsed.get("y")
        if not isinstance(y_value, (int, float)):
            continue

        if not isinstance(scale, str):
            continue

        t = row.get("T")
        ph = row.get("pH")
        f1 = row.get("F1")
        f2 = row.get("F2")
        f3 = row.get("F3")
        try:
            t = float(t)
            ph = float(ph)
            f1 = float(f1)
            f2 = float(f2)
            f3 = float(f3)
        except (TypeError, ValueError):
            continue

        plottable_samples += 1
        if scale in plottable_counts:
            plottable_counts[scale] += 1

        existing = best_yield_per_scale.get(scale)
        if existing is None or float(y_value) > float(existing["yield"]):
            best_yield_per_scale[scale] = {
                "yield": float(y_value),
                "sample_idx": row.get("sample_idx"),
                "parameters": {
                    "T": row.get("T"),
                    "pH": row.get("pH"),
                    "F1": row.get("F1"),
                    "F2": row.get("F2"),
                    "F3": row.get("F3"),
                },
                "result": parsed,
            }

    summary = {
        "total_cost": total_cost,
        "best_yield_per_scale": best_yield_per_scale,
        "total_trials": len(results),
        "successful_trials": successful_trials,
        "trial_counts": trial_counts,
        "plottable_samples": plottable_samples,
        "plottable_counts": plottable_counts,
    }
    _print_task01_summary(summary)
    return summary


def _print_task01_summary(summary: dict[str, object]) -> None:
    """Print the experiment summary to the shell."""
    total_cost = summary["total_cost"]
    best_yield_per_scale = summary["best_yield_per_scale"]
    total_trials = summary["total_trials"]
    successful_trials = summary["successful_trials"]
    trial_counts = summary["trial_counts"]
    plottable_samples = summary["plottable_samples"]
    plottable_counts = summary["plottable_counts"]

    print("\nExperimentzusammenfassung:")
    print(f"  Gesamtkosten der Experimentreihe: {total_cost:.6f}")
    print(f"  Gesamtversuche: {total_trials}")
    print(f"  Erfolgreiche Versuche: {successful_trials}")
    print(f"  Plottbare Datenpunkte: {plottable_samples}")
    print(f"  Versuche pro Scale: micro={trial_counts['micro']}, bench={trial_counts['bench']}, pilot={trial_counts['pilot']}")
    print(f"  Plottbare Datenpunkte pro Scale: micro={plottable_counts['micro']}, bench={plottable_counts['bench']}, pilot={plottable_counts['pilot']}")
    print("  Beste Yields pro Scale:")

    for scale in ["micro", "bench", "pilot"]:
        best = best_yield_per_scale.get(scale)
        if best is None:
            print(f"    - {scale}: keine gültigen Y-Werte gefunden")
            continue

        params = best["parameters"]
        print(
            f"    - {scale}: Y={best['yield']:.6f}, sample_idx={best['sample_idx']}, "
            f"T={params['T']}, pH={params['pH']}, F1={params['F1']}, F2={params['F2']}, F3={params['F3']}"
        )
