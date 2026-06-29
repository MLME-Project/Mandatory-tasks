"""
Bayesian Optimization Loop – Multi-Scale Bioreactor
====================================================
Strategy:
  - Initialization : Latin Hypercube Sampling across all scales
  - Surrogate      : Single GP with scale as ordinal input feature (6D)
  - Acquisition    : Expected Improvement normalized by experiment cost
  - Stop criterion : best observed pilot-Y >= 14 g/L
"""

import sys
import os
import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# ---------------------------------------------------------------------------
# Import API client from parent directory
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from API_Group8 import BioreactorClient, BASE_URL, USER, PASSWORD


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Parameter bounds: [T, pH, F1, F2, F3]
BOUNDS = np.array([
    [20.0, 60.0],   # T   [°C]
    [ 3.0,  9.5],   # pH
    [ 0.0,  2.0],   # F1  [g/L/h]
    [ 0.0,  2.0],   # F2  [g/L/h]
    [ 0.0,  2.0],   # F3  [g/L/h]
])

SCALES        = ["micro", "bench", "pilot"]
SCALE_ENCODE  = {"micro": 0, "bench": 1, "pilot": 2}   # ordinal encoding

N_INIT        = 15      # LHS initialization experiments
N_CANDIDATES  = 2000    # random candidates evaluated per BO iteration
TARGET_Y      = 14.0    # stop when best pilot-Y exceeds this [g/L]
MAX_ITER      = 200     # hard safety cap


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def lhs_init(n: int) -> tuple[np.ndarray, list[str]]:
    """
    Draw n points via Latin Hypercube Sampling.
    All initialization experiments run on micro scale (cheapest).
    Returns recipe array (n×5) and list of scale strings.
    """
    sampler   = qmc.LatinHypercube(d=5, seed=42)
    unit_cube = sampler.random(n)
    recipes   = qmc.scale(unit_cube, BOUNDS[:, 0], BOUNDS[:, 1])
    scales    = ["micro"] * n
    return recipes, scales

def build_X(recipes: np.ndarray, scales: list[str]) -> np.ndarray:
    """Concatenate recipe matrix with ordinal scale encoding → shape (n, 6)."""
    s_enc = np.array([SCALE_ENCODE[s] for s in scales], dtype=float).reshape(-1, 1)
    return np.hstack([recipes, s_enc])


def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                          f_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Standard Expected Improvement.
    xi: small exploration bonus (jitter).
    """
    improvement = mu - f_best - xi
    Z           = improvement / (sigma + 1e-9)
    ei          = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei


def scale_cost(scale: str, mu: float, sigma: float) -> float:
    """
    Return the cost (EUR) for a given scale.
    Values are the real costs as provided by the simulator specification.
    """
    COST = {"micro": 10.0, "bench": 500.0, "pilot": 2000.0}
    return COST[scale]


def fit_gp(X_train: np.ndarray, y_train: np.ndarray,
           scaler: StandardScaler) -> GaussianProcessRegressor:
    """
    Fit a GP with Matérn-5/2 kernel + white noise on standardized inputs.
    Hier werden die Daten auch skaliert, um die Optimierung zu erleichtern. Die Länge der Skalen wird auf 1 gesetzt, um eine gleichmäßige Behandlung aller Dimensionen zu gewährleisten.
    """
    X_scaled = scaler.transform(X_train)
    kernel   = Matern(nu=2.5, length_scale=np.ones(X_train.shape[1]),
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
    """
    Draw n random candidates uniformly from the recipe space × all scales.
    """
    rng       = np.random.default_rng(seed)
    unit      = rng.uniform(size=(n, 5))
    recipes   = BOUNDS[:, 0] + unit * (BOUNDS[:, 1] - BOUNDS[:, 0])
    scale_idx = rng.integers(0, len(SCALES), size=n)
    scales    = [SCALES[i] for i in scale_idx]
    return recipes, scales


def parse_result(result: dict) -> tuple[float, float]:
    """Extract Y and cost from the API response in a backward-compatible way."""
    y = float(result["Y"])
    cost = result.get("cost", result.get("cost_eur", 0.0))
    return y, float(cost)


# ---------------------------------------------------------------------------
# Main BO loop
# ---------------------------------------------------------------------------

def run_bo():
    client = BioreactorClient()
    client.login(USER, PASSWORD)
    print("Logged in.\n")

    # --- Storage ---
    all_recipes: list[np.ndarray] = []
    all_scales:  list[str]        = []
    all_Y:       list[float]      = []
    all_costs:   list[float]      = []

    best_pilot_Y    = -np.inf
    cumulative_cost = 0.0
    trajectory      = []   # (iteration, best_pilot_Y, cumulative_cost)

    # -----------------------------------------------------------------------
    # 1. INITIALIZATION via LHS
    # -----------------------------------------------------------------------
    print(f"=== Initialization: {N_INIT} LHS experiments ===")
    init_recipes, init_scales = lhs_init(N_INIT)

    for i, (recipe, scale) in enumerate(zip(init_recipes, init_scales)):
        T, pH, F1, F2, F3 = recipe
        print(f"  [{i+1}/{N_INIT}] scale={scale:6s}  "
              f"T={T:.1f} pH={pH:.2f} F1={F1:.2f} F2={F2:.2f} F3={F3:.2f}",
              end="  →  ", flush=True)
        result = client.run(scale, T, pH, F1, F2, F3)
        y, cost = parse_result(result)
        print(f"Y={y:.3f} g/L  cost={cost:.1f} EUR")

        all_recipes.append(recipe)
        all_scales.append(scale)
        all_Y.append(y)
        all_costs.append(cost)
        cumulative_cost += cost

        if scale == "pilot":
            best_pilot_Y = max(best_pilot_Y, y)

    print(f"\nInit done. Best pilot-Y so far: {best_pilot_Y:.3f} g/L | "
          f"Total cost: {cumulative_cost:.1f} EUR\n")

    # -----------------------------------------------------------------------
    # 2. BO LOOP
    # -----------------------------------------------------------------------
    scaler = StandardScaler()

    for iteration in range(1, MAX_ITER + 1):

        # --- Stop criterion ---
        if best_pilot_Y >= TARGET_Y:
            print(f"\n✓ Target reached: best pilot-Y = {best_pilot_Y:.3f} g/L >= {TARGET_Y} g/L")
            break

        # --- Fit GP ---
        X_train = build_X(np.array(all_recipes), all_scales)
        y_train = np.array(all_Y)

        scaler.fit(X_train)
        gp = fit_gp(X_train, y_train, scaler)

        # f_best: best observed pilot-Y (use global best if no pilot run yet)
        pilot_Y_vals = [y for y, s in zip(all_Y, all_scales) if s == "pilot"]
        f_best       = max(pilot_Y_vals) if pilot_Y_vals else max(all_Y)

        # --- Sample candidates and evaluate acquisition ---
        cand_recipes, cand_scales = sample_candidates(N_CANDIDATES, seed=iteration)
        X_cand = build_X(cand_recipes, cand_scales)
        X_cand_scaled = scaler.transform(X_cand)

        mu, sigma = gp.predict(X_cand_scaled, return_std=True)
        ei        = expected_improvement(mu, sigma, f_best)

        # Normalize EI by expected cost of that scale
        costs_cand = np.array([scale_cost(s, 0, 0) for s in cand_scales])
        ei_norm    = ei / costs_cand

        # Pick best candidate
        best_idx   = int(np.argmax(ei_norm))
        next_recipe = cand_recipes[best_idx]
        next_scale  = cand_scales[best_idx]
        T, pH, F1, F2, F3 = next_recipe

        print(f"[Iter {iteration:03d}] scale={next_scale:6s}  "
              f"T={T:.1f} pH={pH:.2f} F1={F1:.2f} F2={F2:.2f} F3={F3:.2f}  "
              f"EI_norm={ei_norm[best_idx]:.4f}",
              end="  →  ", flush=True)

        # --- Run experiment ---
        result = client.run(next_scale, T, pH, F1, F2, F3)
        y, cost = parse_result(result)
        print(f"Y={y:.3f} g/L  cost={cost:.1f} EUR")

        all_recipes.append(next_recipe)
        all_scales.append(next_scale)
        all_Y.append(y)
        all_costs.append(cost)
        cumulative_cost += cost

        if next_scale == "pilot":
            best_pilot_Y = max(best_pilot_Y, y)

        trajectory.append((iteration, best_pilot_Y, cumulative_cost))
        print(f"          best pilot-Y: {best_pilot_Y:.3f} g/L | "
              f"total cost: {cumulative_cost:.1f} EUR")

    # -----------------------------------------------------------------------
    # 3. RESULTS
    # -----------------------------------------------------------------------
    print("\n=== FINAL RESULTS ===")
    print(f"Best pilot-Y    : {best_pilot_Y:.3f} g/L")
    print(f"Total cost      : {cumulative_cost:.1f} EUR")
    print(f"Total experiments: {len(all_Y)}")

    # Best recipe found on pilot
    pilot_indices = [i for i, s in enumerate(all_scales) if s == "pilot"]
    if pilot_indices:
        best_pilot_idx = pilot_indices[int(np.argmax([all_Y[i] for i in pilot_indices]))]
        r = all_recipes[best_pilot_idx]
        print(f"Best recipe     : T={r[0]:.2f} pH={r[1]:.2f} "
              f"F1={r[2]:.2f} F2={r[3]:.2f} F3={r[4]:.2f}")

    return trajectory, all_recipes, all_scales, all_Y, all_costs


if __name__ == "__main__":
    run_bo()