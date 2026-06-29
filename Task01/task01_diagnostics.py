from __future__ import annotations

from typing import Tuple
import math


def analyze_tp_distribution(results: list[dict], t_bins: int = 10, ph_bins: int = 10) -> dict:
    """Compute a 2D histogram of T vs pH and return counts and percentages.

    Returns a dict with keys: counts (2D list), t_edges, ph_edges, total
    """
    t_vals = []
    ph_vals = []
    for r in results:
        if r.get("status") != "ok":
            continue
        try:
            t = float(r["T"])
            ph = float(r["pH"])
        except Exception:
            continue
        t_vals.append(t)
        ph_vals.append(ph)

    if not t_vals:
        return {"total": 0, "counts": [], "t_edges": [], "ph_edges": []}

    t_min, t_max = min(t_vals), max(t_vals)
    ph_min, ph_max = min(ph_vals), max(ph_vals)
    if t_min == t_max:
        t_max = t_min + 1.0
    if ph_min == ph_max:
        ph_max = ph_min + 1.0

    t_edges = [t_min + i * (t_max - t_min) / t_bins for i in range(t_bins + 1)]
    ph_edges = [ph_min + i * (ph_max - ph_min) / ph_bins for i in range(ph_bins + 1)]

    counts = [[0 for _ in range(t_bins)] for _ in range(ph_bins)]
    total = 0
    for t, ph in zip(t_vals, ph_vals):
        # find bin indices
        tx = min(int((t - t_min) / (t_max - t_min) * t_bins), t_bins - 1)
        px = min(int((ph - ph_min) / (ph_max - ph_min) * ph_bins), ph_bins - 1)
        counts[px][tx] += 1
        total += 1

    return {"total": total, "counts": counts, "t_edges": t_edges, "ph_edges": ph_edges}


def print_tp_report(hist: dict, highlight_frac: float = 0.5) -> None:
    total = hist.get("total", 0)
    if total == 0:
        print("Keine Daten für T/pH-Analyse.")
        return
    counts = hist["counts"]
    # Find top cell count and its fraction
    flat = [c for row in counts for c in row]
    sorted_flat = sorted(flat, reverse=True)
    top = sorted_flat[0]
    top_frac = top / total
    print(f"Gesamtmessungen: {total}")
    print(f"Maximale Zellanzahl: {top} ({top_frac:.2%} aller Messungen)")
    if top_frac >= highlight_frac:
        print("Hinweis: Starke Konzentration in einer Zelle festgestellt (Bias).")
    else:
        print("Keine extreme Konzentration in einer einzelnen Zelle.")


if __name__ == "__main__":
    print("Dieses Modul stellt Funktionen zur Diagnose von T/pH-Verteilungen bereit.")


def save_tp_heatmap(hist: dict, output_path: str = "Task01_TP_heatmap.png") -> None:
    """Save a heatmap image of the T vs pH counts.

    Expects the `hist` dict returned from `analyze_tp_distribution`.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("Matplotlib oder numpy nicht installiert; Heatmap kann nicht erstellt werden.")
        return

    total = hist.get("total", 0)
    if total == 0:
        print("Keine Daten für Heatmap.")
        return

    counts = hist.get("counts", [])
    t_edges = hist.get("t_edges", [])
    ph_edges = hist.get("ph_edges", [])
    if not counts or not t_edges or not ph_edges:
        print("Unvollständige Histogrammdaten; Heatmap übersprungen.")
        return

    arr = np.array(counts)
    # arr shape is (ph_bins, t_bins) — display with origin='lower'
    t_min, t_max = t_edges[0], t_edges[-1]
    ph_min, ph_max = ph_edges[0], ph_edges[-1]
    extent = [t_min, t_max, ph_min, ph_max]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap="Reds")
    plt.colorbar(im, label="Anzahl Messungen")
    plt.xlabel("T")
    plt.ylabel("pH")
    plt.title("T/pH Verteilung")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heatmap gespeichert in: {output_path}")
