from __future__ import annotations

import json


def _parse_result_dict(result_value: object) -> dict | None:
    if isinstance(result_value, dict):
        return result_value
    if not isinstance(result_value, str):
        return None
    try:
        parsed = json.loads(result_value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_quality(parsed: dict) -> float | None:
    for key in ("Y", "y", "yield", "quality", "score"):
        value = parsed.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _barycentric_coords(f1: float, f2: float, f3: float) -> tuple[float, float] | None:
    total = f1 + f2 + f3
    if total <= 0:
        return None
    w1 = f1 / total
    w3 = f3 / total
    x = 0.5 * w1 + w3
    y = w1
    return x, y


def visualize_task01_results(results: list[dict], t_bins: int = 8, ph_bins: int = 8) -> None:
    """Visualize Task01 samples as a matrix of triangular diagrams.

    The visualization is created in two steps:
    1. Partition the valid T/pH data points into a grid of bins.
    2. For each bin, draw a barycentric triangle for the F1/F2/F3 ratios.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors  # Hinzugefügt für die Skalierung
    except ImportError:
        print("Matplotlib ist nicht installiert. Installiere es mit: pip install matplotlib")
        return

    valid_samples: list[dict] = []
    for row in results:
        if row.get("status") != "ok":
            continue
        parsed = _parse_result_dict(row.get("result"))
        if parsed is None:
            continue
        quality = _extract_quality(parsed)
        if quality is None:
            continue
        try:
            t = float(row["T"])
            ph = float(row["pH"])
            f1 = float(row["F1"])
            f2 = float(row["F2"])
            f3 = float(row["F3"])
        except (TypeError, ValueError, KeyError):
            continue
        valid_samples.append({
            "scale": row.get("scale", ""),
            "T": t,
            "pH": ph,
            "F1": f1,
            "F2": f2,
            "F3": f3,
            "quality": quality,
        })

    if not valid_samples:
        print("Keine gültigen Messdaten zum Visualisieren gefunden.")
        return

    print(f"Visualisierung: {len(valid_samples)} valide Datenpunkte gefunden.")
    scale_counts = {
        scale: sum(1 for s in valid_samples if s["scale"] == scale)
        for scale in ["micro", "bench", "pilot"]
    }
    print(f"  Punkte pro Scale: micro={scale_counts['micro']}, bench={scale_counts['bench']}, pilot={scale_counts['pilot']}")

    # Bin edges for T and pH
    t_values = [s["T"] for s in valid_samples]
    ph_values = [s["pH"] for s in valid_samples]
    t_min, t_max = min(t_values), max(t_values)
    ph_min, ph_max = min(ph_values), max(ph_values)
    if t_min == t_max:
        t_max = t_min + 1.0
    if ph_min == ph_max:
        ph_max = ph_min + 1.0

    t_edges = [t_min + i * (t_max - t_min) / t_bins for i in range(t_bins + 1)]
    ph_edges = [ph_min + i * (ph_max - ph_min) / ph_bins for i in range(ph_bins + 1)]

    def find_bin_index(value: float, edges: list[float]) -> int:
        for idx in range(len(edges) - 1):
            if edges[idx] <= value < edges[idx + 1] or (idx == len(edges) - 2 and value == edges[-1]):
                return idx
        return len(edges) - 2

    # Determine quality ranges for each scale separately so each plot gets its own color scale.
    scale_quality_ranges: dict[str, tuple[float, float] | None] = {}
    for scale in ["micro", "bench", "pilot"]:
        quality_values = [s["quality"] for s in valid_samples if s["scale"] == scale]
        if quality_values:
            scale_quality_ranges[scale] = (min(quality_values), max(quality_values))
        else:
            scale_quality_ranges[scale] = None

    # Create the bin structure once for all scales
    binned_samples: dict[str, dict[tuple[int, int], list[dict]]] = {"micro": {}, "bench": {}, "pilot": {}}
    for sample in valid_samples:
        scale = sample["scale"]
        if scale not in binned_samples:
            continue
        t_idx = find_bin_index(sample["T"], t_edges)
        ph_idx = find_bin_index(sample["pH"], ph_edges)
        bin_key = (t_idx, ph_idx)
        binned_samples[scale].setdefault(bin_key, []).append(sample)

    for scale in ["micro", "bench", "pilot"]:
        fig, axes = plt.subplots(ph_bins, t_bins, figsize=(12, 12), squeeze=False)
        fig.suptitle(f"Task01: {scale.capitalize()}-Scale Dreiecksdiagramme", fontsize=16)
        figure_scatter = None
        scale_range = scale_quality_ranges[scale]
        if scale_range is not None:
            scale_vmin, scale_vmax = scale_range
        else:
            scale_vmin, scale_vmax = 0.0, 1.0

        # Hier definieren wir die PowerNorm. Gamma=0.5 zieht die Farbskala 
        # so auseinander, dass Unterschiede im unteren Bereich deutlicher werden.
        # Wenn nötig, kannst du gamma auf 0.3 oder 0.4 anpassen, wenn es noch extremer sein soll.
        norm = mcolors.PowerNorm(gamma=0.01, vmin=scale_vmin, vmax=scale_vmax)

        for ph_idx in range(ph_bins):
            for t_idx in range(t_bins):
                ax = axes[ph_bins - 1 - ph_idx][t_idx]
                bin_key = (t_idx, ph_idx)
                bin_samples = binned_samples[scale].get(bin_key, [])

                triangle_x = [0.5, 0.0, 1.0, 0.5]
                triangle_y = [1.0, 0.0, 0.0, 1.0]
                ax.plot(triangle_x, triangle_y, color="black", linewidth=1.2)
                ax.text(0.5, 1.02, "F1", ha="center", va="bottom", fontsize=7)
                ax.text(-0.05, -0.05, "F2", ha="right", va="top", fontsize=7)
                ax.text(1.05, -0.05, "F3", ha="left", va="top", fontsize=7)

                if bin_samples:
                    coords = [
                        (_barycentric_coords(s["F1"], s["F2"], s["F3"]), s["quality"]) for s in bin_samples
                    ]
                    coords = [(c, q) for c, q in coords if c is not None]
                    if coords:
                        xs, ys = zip(*[c for c, _ in coords])
                        qualities = [q for _, q in coords]
                        if figure_scatter is None:
                            figure_scatter = ax.scatter(
                                xs,
                                ys,
                                c=qualities,
                                cmap="rainbow",  # Der echte Regenbogen von Lila bis Rot
                                norm=norm,       # Verwendet unsere gestreckte Skalierung anstatt vmin/vmax
                                s=25,
                                edgecolors="black",
                                linewidths=0.2,
                            )
                        else:
                            ax.scatter(
                                xs,
                                ys,
                                c=qualities,
                                cmap="rainbow",
                                norm=norm,
                                s=25,
                                edgecolors="black",
                                linewidths=0.2,
                            )
                        ax.text(0.5, 0.05, f"n={len(coords)}", ha="center", va="bottom", fontsize=6, color="black")
                else:
                    ax.text(0.5, 0.5, "leer", ha="center", va="center", fontsize=7, color="gray")

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")

                t_lo = t_edges[t_idx]
                t_hi = t_edges[t_idx + 1]
                ph_lo = ph_edges[ph_idx]
                ph_hi = ph_edges[ph_idx + 1]
                ax.set_title(f"T {t_lo:.0f}-{t_hi:.0f}\npH {ph_lo:.1f}-{ph_hi:.1f}", fontsize=6)

        fig.subplots_adjust(left=0.03, right=0.92, top=0.94, bottom=0.03, hspace=0.4, wspace=0.4)

        if figure_scatter is not None:
            cax = fig.add_axes([0.94, 0.1, 0.02, 0.8])
            fig.colorbar(figure_scatter, cax=cax, label="Yield")
        output_path = f"Task01_visualization_{scale}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Visualisierung gespeichert in: {output_path}")