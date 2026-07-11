from __future__ import annotations

import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Benoetigt: pip install pandas openpyxl numpy matplotlib")
    sys.exit(1)


def _barycentric_coords(f1: float, f2: float, f3: float) -> tuple[float, float] | None:
    total = f1 + f2 + f3
    if total <= 0:
        return None
    w1 = f1 / total
    w3 = f3 / total
    x = 0.5 * w1 + w3
    y = w1
    return x, y


def read_and_compute_stddev(path: str) -> list[dict]:
    """Read CSV or Excel, group by identical conditions, compute std of Y per group."""
    path_obj = Path(path)
    required = {"scale", "T", "pH", "F1", "F2", "F3", "Y"}

    if path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(path_obj)
        if not required.issubset(set(df.columns)):
            print("Keine gueltigen Spalten in der CSV gefunden.")
            return []
        all_dfs = [df[list(required)]]
    else:
        xls = pd.ExcelFile(path_obj)
        all_dfs = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, header=0)
            if not required.issubset(set(df.columns)):
                continue
            all_dfs.append(df[list(required)])

    if not all_dfs:
        print("Keine gueltigen Sheets gefunden.")
        return []

    data = pd.concat(all_dfs, ignore_index=True)
    data = data.dropna(subset=list(required))

    group_cols = ["scale", "T", "pH", "F1", "F2", "F3"]
    grouped = data.groupby(group_cols)["Y"].agg(
        count="count",
        std_y="std",
        mean_y="mean",
    ).reset_index()

    grouped = grouped[grouped["count"] >= 2].copy()
    grouped["std_y"] = grouped["std_y"].fillna(0.0)
    grouped["std_y_relative"] = grouped["std_y"] / grouped["mean_y"].replace(0, np.nan)
    grouped["std_y_relative"] = grouped["std_y_relative"].fillna(0.0)

    rows = []
    for _, row in grouped.iterrows():
        coords = _barycentric_coords(row["F1"], row["F2"], row["F3"])
        if coords is None:
            continue
        rows.append({
            "scale":  str(row["scale"]).strip(),
            "T":      float(row["T"]),
            "pH":     float(row["pH"]),
            "F1":     float(row["F1"]),
            "F2":     float(row["F2"]),
            "F3":     float(row["F3"]),
            "std_y":  float(row["std_y"]),
            "std_y_relative": float(row["std_y_relative"]),
            "mean_y": float(row["mean_y"]),
            "count":  int(row["count"]),
        })

    print(f"Gefundene Gruppen mit >=2 Wiederholungen: {len(rows)}")
    for scale in ["micro", "bench", "pilot"]:
        scale_rows = [r for r in rows if r["scale"] == scale]
        n = len(scale_rows)
        if n > 0:
            avg_std = sum(r["std_y"] for r in scale_rows) / n
            print(f"  {scale}: {n} Gruppen, Avg Std = {avg_std:.6f}")
        else:
            print(f"  {scale}: 0 Gruppen")
    return rows


def visualize_stddev(rows: list[dict], t_bins: int = 8, ph_bins: int = 8, value_key: str = "std_y", suffix: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("Matplotlib nicht installiert: pip install matplotlib")
        return

    if not rows:
        print("Keine Daten zum Visualisieren.")
        return

    t_values  = [r["T"]  for r in rows]
    ph_values = [r["pH"] for r in rows]
    t_min, t_max   = min(t_values),  max(t_values)
    ph_min, ph_max = min(ph_values), max(ph_values)
    if t_min  == t_max:  t_max  = t_min  + 1.0
    if ph_min == ph_max: ph_max = ph_min + 1.0

    t_edges  = [t_min  + i * (t_max  - t_min)  / t_bins  for i in range(t_bins  + 1)]
    ph_edges = [ph_min + i * (ph_max - ph_min) / ph_bins for i in range(ph_bins + 1)]

    def find_bin(value: float, edges: list[float]) -> int:
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1] or (i == len(edges) - 2 and value == edges[-1]):
                return i
        return len(edges) - 2

    scale_std_ranges: dict[str, tuple[float, float] | None] = {}
    for scale in ["micro", "bench", "pilot"]:
        vals = [r[value_key] for r in rows if r["scale"] == scale]
        scale_std_ranges[scale] = (min(vals), max(vals)) if vals else None

    binned: dict[str, dict[tuple[int, int], list[dict]]] = {
        "micro": {}, "bench": {}, "pilot": {}
    }
    for r in rows:
        scale = r["scale"]
        if scale not in binned:
            continue
        key = (find_bin(r["T"], t_edges), find_bin(r["pH"], ph_edges))
        binned[scale].setdefault(key, []).append(r)

    for scale in ["micro", "bench", "pilot"]:
        scale_rows = [r for r in rows if r["scale"] == scale]
        if not scale_rows:
            print(f"Keine Daten fuer Scale '{scale}', wird uebersprungen.")
            continue

        fig, axes = plt.subplots(ph_bins, t_bins, figsize=(12, 12), squeeze=False)
        title_suffix = "(relativ zum Mittelwert)" if value_key == "std_y_relative" else ""
        fig.suptitle(
            f"Task01: {scale.capitalize()}-Scale - Standardabweichung des Yields {title_suffix}".strip(),
            fontsize=16,
        )

        std_range = scale_std_ranges[scale]
        vmin, vmax = (std_range if std_range else (0.0, 1.0))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        figure_scatter = None

        for ph_idx in range(ph_bins):
            for t_idx in range(t_bins):
                ax = axes[ph_bins - 1 - ph_idx][t_idx]
                bin_samples = binned[scale].get((t_idx, ph_idx), [])

                ax.plot([0.5, 0.0, 1.0, 0.5], [1.0, 0.0, 0.0, 1.0],
                        color="black", linewidth=1.2)
                ax.text(0.5,   1.02,  "F1", ha="center", va="bottom", fontsize=7)
                ax.text(-0.05, -0.05, "F2", ha="right",  va="top",    fontsize=7)
                ax.text(1.05,  -0.05, "F3", ha="left",   va="top",    fontsize=7)

                if bin_samples:
                    bin_samples = sorted(bin_samples, key=lambda r: r["std_y"])
                    coords = [_barycentric_coords(r["F1"], r["F2"], r["F3"])
                              for r in bin_samples]
                    valid = [(c, r[value_key], r["count"])
                             for c, r in zip(coords, bin_samples) if c is not None]

                    if valid:
                        xs       = [c[0] for c, _, _ in valid]
                        ys       = [c[1] for c, _, _ in valid]
                        std_vals = [s    for _, s, _ in valid]

                        scatter_kwargs = dict(
                            c=std_vals,
                            cmap="YlOrRd",
                            norm=norm,
                            s=25,
                            edgecolors="black",
                            linewidths=0.2,
                        )
                        if figure_scatter is None:
                            figure_scatter = ax.scatter(xs, ys, **scatter_kwargs)
                        else:
                            ax.scatter(xs, ys, **scatter_kwargs)

                        ax.text(0.5, 0.05, f"n={len(valid)}",
                                ha="center", va="bottom", fontsize=6, color="black")
                else:
                    ax.text(0.5, 0.5, "leer",
                            ha="center", va="center", fontsize=7, color="gray")

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
                ax.set_title(
                    f"T {t_edges[t_idx]:.0f}-{t_edges[t_idx+1]:.0f}\n"
                    f"pH {ph_edges[ph_idx]:.1f}-{ph_edges[ph_idx+1]:.1f}",
                    fontsize=6,
                )

        fig.subplots_adjust(
            left=0.03, right=0.92, top=0.94, bottom=0.03, hspace=0.4, wspace=0.4
        )
        if figure_scatter is not None:
            cax = fig.add_axes((0.94, 0.1, 0.02, 0.8))
            color_label = "Std.-Abweichung Yield / Mean" if value_key == "std_y_relative" else "Std.-Abweichung Yield"
            fig.colorbar(figure_scatter, cax=cax, label=color_label)

        output_path = f"Task01_stddev_{scale}{suffix}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Gespeichert: {output_path}")


def visualize_histograms(rows: list[dict], bins: int = 15, min_count: int = 4, value_key: str = "std_y", suffix: str = "") -> None:
    """Creates and saves a histogram of std_y for each scale, filtering for groups with >= min_count."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for scale in ["micro", "bench", "pilot"]:
        # Hier filtern wir zusätzlich nach der Anzahl der Wiederholungen (count)
        scale_rows = [r for r in rows if r["scale"] == scale and r["count"] >= min_count]
        
        if not scale_rows:
            print(f"Keine Daten für Scale '{scale}' mit n >= {min_count} vorhanden. Überspringe Histogramm.")
            continue

        std_vals = [r[value_key] for r in scale_rows]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        # Histogramm zeichnen
        ax.hist(std_vals, bins=bins, color="#f15a24", edgecolor="black", alpha=0.8, rwidth=0.9)
        
        title_prefix = "Verteilung der relativen Std.-Abweichung" if value_key == "std_y_relative" else "Verteilung der Std.-Abweichung"
        ax.set_title(f"{title_prefix} ({scale.capitalize()}-Scale)\nNur Gruppen mit n ≥ {min_count}", 
                     fontsize=12, fontweight='bold', pad=12)
        x_label = "Relative Standardabweichung des Yields (std_y / mean_y)" if value_key == "std_y_relative" else "Standardabweichung des Yields (std_y)"
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("Anzahl der Gruppen", fontsize=10)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # Statistische Kennzahlen für die gefilterten Daten berechnen
        mean_std = np.mean(std_vals)
        median_std = np.median(std_vals)
        textstr = f"Gruppen (n≥{min_count}): {len(std_vals)}\nSchnitt: {mean_std:.4f}\nMedian: {median_std:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        output_path = f"Task01_hist_{scale}{suffix}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Gespeichert: {output_path}")


if __name__ == "__main__":
    path = "Summery_Versuche2.csv"
    if len(sys.argv) > 1:
        path = " ".join(sys.argv[1:]).strip('"')

    file_path = Path(path)
    if not file_path.exists():
        script_dir = Path(__file__).resolve().parent
        candidates = list(script_dir.glob("*.csv")) + list(script_dir.glob("*.xlsx"))
        for c in candidates:
            name_norm = "".join(ch for ch in c.name.lower() if ch.isalnum())
            req_norm  = "".join(ch for ch in file_path.name.lower() if ch.isalnum())
            if name_norm == req_norm:
                file_path = c
                break
        else:
            print(f"Datei nicht gefunden: {path}")
            for c in candidates:
                print(f"  {c.name}")
            sys.exit(1)

    print(f"Verwende: {file_path}")
    rows = read_and_compute_stddev(str(file_path))
    if rows:
        visualize_stddev(rows)
        visualize_stddev(rows, value_key="std_y_relative", suffix="_relative")
        visualize_histograms(rows, min_count=4)
        visualize_histograms(rows, min_count=4, value_key="std_y_relative", suffix="_relative")