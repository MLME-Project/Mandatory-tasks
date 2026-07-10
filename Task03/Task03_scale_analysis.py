"""
Scale-Comparison Analysis for Multi-Scale Bioreactor
===================================================
Strategy:
  - Initialization : Latin Hypercube Sampling with 100 initial points
  - Measurement    : Evaluate each recipe on micro, bench and pilot scales
  - Optimization   : 100 BO iterations to find promising recipes
  - Output         : CSV file with all measurements and a plot of the
                     relationship between Y_micro, Y_bench and Y_pilot
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from API_Group8 import BioreactorClient, USER, PASSWORD


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOUNDS = np.array([
    [35.0, 40.0],
    [3.0, 9.5],
    [0.0, 2.0],
    [0.0, 2.0],
    [0.0, 2.0],
])

SCALES = ["micro", "bench", "pilot"]
ACTIVE_SCALES = ["micro", "bench"]
SCALE_ENCODE = {"micro": 0, "bench": 1, "pilot": 2}

N_INIT = 100
N_CANDIDATES = 2000
TARGET_Y = 14.0
MAX_ITER = 100

OUTPUT_CSV = Path(__file__).with_name("scale_comparison_results.csv")
OUTPUT_SORTED_CSV = Path(__file__).with_name("scale_comparison_results_sorted.csv")
OUTPUT_PLOT = Path(__file__).with_name("scale_comparison_plot.png")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def lhs_init(n: int) -> tuple[np.ndarray, list[str]]:
    sampler = qmc.LatinHypercube(d=5, seed=42)
    unit_cube = sampler.random(n)
    recipes = qmc.scale(unit_cube, BOUNDS[:, 0], BOUNDS[:, 1])
    scales = ["micro"] * n
    return recipes, scales


def build_X(recipes: np.ndarray, scales: list[str]) -> np.ndarray:
    s_enc = np.array([SCALE_ENCODE[s] for s in scales], dtype=float).reshape(-1, 1)
    return np.hstack([recipes, s_enc])


def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                          f_best: float, xi: float = 0.01) -> np.ndarray:
    improvement = mu - f_best - xi
    Z = improvement / (sigma + 1e-9)
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray,
                            beta: float) -> np.ndarray:
    return mu + beta * sigma


def adaptive_acquisition(mu: np.ndarray, sigma: np.ndarray,
                          f_best: float, iteration: int,
                          max_iter: int, xi: float = 0.01) -> np.ndarray:
    ei = expected_improvement(mu, sigma, f_best, xi=xi)
    progress = (iteration - 1) / max(1, max_iter - 1)
    exploitation_weight = progress ** 1.5
    exploration_weight = 1.0 - exploitation_weight

    beta = 1.5 * exploration_weight + 0.2 * exploitation_weight
    ucb = upper_confidence_bound(mu, sigma, beta)

    ei_scaled = ei / (np.max(ei) + 1e-12)
    ucb_scaled = ucb / (np.max(ucb) + 1e-12)
    return exploration_weight * ucb_scaled + exploitation_weight * ei_scaled


def fit_gp(X_train: np.ndarray, y_train: np.ndarray,
           scaler: StandardScaler) -> GaussianProcessRegressor:
    X_scaled = scaler.transform(X_train)
    kernel = Matern(nu=2.5, length_scale=np.ones(X_train.shape[1]),
                    length_scale_bounds=(1e-2, 1e2)) \
             + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
    )
    gp.fit(X_scaled, y_train)
    return gp


def sample_candidates(n: int, seed: int) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    unit = rng.uniform(size=(n, 5))
    recipes = BOUNDS[:, 0] + unit * (BOUNDS[:, 1] - BOUNDS[:, 0])
    scale_idx = rng.integers(0, len(ACTIVE_SCALES), size=n)
    scales = [ACTIVE_SCALES[i] for i in scale_idx]
    return recipes, scales


def parse_result(result: dict) -> tuple[float, float]:
    y = float(result["Y"])
    cost = result.get("cost", result.get("cost_eur", 0.0))
    return y, float(cost)


def run_with_retry(client, scale: str, T: float, pH: float,
                   F1: float, F2: float, F3: float) -> tuple[float, float] | None:
    try:
        result = client.run(scale, T, pH, F1, F2, F3)
        return parse_result(result)
    except Exception as exc:
        print(f"\nAPI request failed for scale='{scale}': {exc}")
        return None


def evaluate_recipe(client, recipe: np.ndarray) -> dict:
    T, pH, F1, F2, F3 = recipe
    row = {
        "T": float(T),
        "pH": float(pH),
        "F1": float(F1),
        "F2": float(F2),
        "F3": float(F3),
        "Y_micro": None,
        "Y_bench": None,
        "Y_pilot": None,
        "cost_micro": None,
        "cost_bench": None,
        "cost_pilot": None,
    }

    for scale in SCALES:
        outcome = run_with_retry(client, scale, T, pH, F1, F2, F3)
        if outcome is None:
            return {}
        y, cost = outcome
        row[f"Y_{scale}"] = y
        row[f"cost_{scale}"] = cost

    row["delta_micro_vs_pilot"] = row["Y_micro"] - row["Y_pilot"]
    row["delta_bench_vs_pilot"] = row["Y_bench"] - row["Y_pilot"]
    return row


def report_recipe(label: str, recipe: np.ndarray, row: dict) -> None:
    T, pH, F1, F2, F3 = recipe
    print(
        f"[{label}] recipe -> T={T:.2f}, pH={pH:.2f}, F1={F1:.2f}, F2={F2:.2f}, F3={F3:.2f} | "
        f"Y_micro={row.get('Y_micro', float('nan')):.3f}, "
        f"Y_bench={row.get('Y_bench', float('nan')):.3f}, "
        f"Y_pilot={row.get('Y_pilot', float('nan')):.3f}"
    )


def write_rows(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "T", "pH", "F1", "F2", "F3",
        "Y_micro", "Y_bench", "Y_pilot",
        "cost_micro", "cost_bench", "cost_pilot",
        "delta_micro_vs_pilot", "delta_bench_vs_pilot",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sxx = np.sum((x - x_mean) ** 2)
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")

    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    residuals = y - (intercept + slope * x)
    dof = len(x) - 2
    if dof <= 0:
        return slope, intercept, float("nan")

    sigma2 = np.sum(residuals ** 2) / dof
    slope_se = np.sqrt(sigma2 / sxx)
    return slope, intercept, slope_se


def plot_results(rows: list[dict], path: Path) -> None:
    x_pilot = np.array([row["Y_pilot"] for row in rows], dtype=float)
    y_micro = np.array([row["Y_micro"] for row in rows], dtype=float)
    y_bench = np.array([row["Y_bench"] for row in rows], dtype=float)
    y_pilot = np.array([row["Y_pilot"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        ("micro", y_micro, "tab:blue"),
        ("bench", y_bench, "tab:orange"),
        ("pilot", y_pilot, "tab:green"),
    ]

    for idx, (label, y_vals, color) in enumerate(series):
        slope, intercept, slope_se = fit_linear_regression(x_pilot, y_vals)
        ax.scatter(x_pilot, y_vals, color=color, alpha=0.7, s=30, label=label)

        x_fit = np.linspace(np.nanmin(x_pilot), np.nanmax(x_pilot), 100)
        if np.isfinite(slope) and np.isfinite(intercept):
            y_fit = intercept + slope * x_fit
            ax.plot(x_fit, y_fit, color=color, linewidth=1.8)

            ax.text(
                0.02,
                0.95 - 0.10 * (idx + 1),
                f"{label}: slope = {slope:.3f} ± {slope_se:.3f}",
                transform=ax.transAxes,
                color=color,
                fontsize=9,
                va="top",
            )

    ax.set_xlabel("Pilot yield")
    ax.set_ylabel("Yield")
    ax.set_title("Relationship between pilot yield and scale yields")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_scale_analysis() -> None:
    client = BioreactorClient()
    client.login(USER, PASSWORD)
    print("Logged in.\n")

    rows: list[dict] = []
    all_recipes: list[np.ndarray] = []
    all_scales: list[str] = []
    all_y_pilot: list[float] = []

    best_observed_y = -np.inf
    best_recipe: np.ndarray | None = None

    print("Starting initialization with 100 Latin Hypercube samples...")
    init_recipes, init_scales = lhs_init(N_INIT)

    for idx, recipe in enumerate(init_recipes, start=1):
        row = evaluate_recipe(client, recipe)
        if not row:
            print("Initialization stopped because the API failed.")
            return

        report_recipe(f"init {idx:03d}", recipe, row)
        rows.append(row)
        all_recipes.append(recipe)
        all_scales.append("pilot")
        all_y_pilot.append(row["Y_pilot"])

        if row["Y_pilot"] > best_observed_y:
            best_observed_y = row["Y_pilot"]
            best_recipe = recipe.copy()

    print(f"Initialization complete. Best pilot Y so far: {best_observed_y:.3f} g/L")

    scaler = StandardScaler()
    for iteration in range(1, MAX_ITER + 1):
        X_train = build_X(np.array(all_recipes), all_scales)
        y_train = np.array(all_y_pilot)
        scaler.fit(X_train)
        gp = fit_gp(X_train, y_train, scaler)

        cand_recipes, cand_scales = sample_candidates(N_CANDIDATES, seed=iteration)
        X_cand = build_X(cand_recipes, cand_scales)
        X_cand_scaled = scaler.transform(X_cand)

        mu, sigma = gp.predict(X_cand_scaled, return_std=True)
        acq = adaptive_acquisition(mu, sigma, best_observed_y, iteration, MAX_ITER)

        best_idx = int(np.argmax(acq))
        next_recipe = cand_recipes[best_idx]
        T, pH, F1, F2, F3 = next_recipe

        row = evaluate_recipe(client, next_recipe)
        if not row:
            print("Optimization stopped because the API failed.")
            break

        report_recipe(f"iter {iteration:03d}", next_recipe, row)
        rows.append(row)
        all_recipes.append(next_recipe)
        all_scales.append("pilot")
        all_y_pilot.append(row["Y_pilot"])

        if row["Y_pilot"] > best_observed_y:
            best_observed_y = row["Y_pilot"]
            best_recipe = next_recipe.copy()

    print(f"\nBest recipe found: T={best_recipe[0]:.2f} pH={best_recipe[1]:.2f} "
          f"F1={best_recipe[2]:.2f} F2={best_recipe[3]:.2f} F3={best_recipe[4]:.2f}")

    rows_sorted = sorted(rows, key=lambda r: r["Y_pilot"])
    write_rows(rows, OUTPUT_CSV)
    write_rows(rows_sorted, OUTPUT_SORTED_CSV)
    plot_results(rows_sorted, OUTPUT_PLOT)

    print(f"Saved raw measurements to: {OUTPUT_CSV}")
    print(f"Saved sorted measurements to: {OUTPUT_SORTED_CSV}")
    print(f"Saved plot to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    run_scale_analysis()
