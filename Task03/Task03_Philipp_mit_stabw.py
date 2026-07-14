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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ACTIVE_SCALES = ["micro" , "bench"]
SCALE_ENCODE  = {"micro": 0, "bench": 1, "pilot": 2}   # ordinal encoding
SCALE_STD     = {"micro": 0.1225, "bench": 0.0160, "pilot": 0.0028}

N_INIT        = 15      # LHS initialization experiments
N_CANDIDATES  = 2000    # random candidates evaluated per BO iteration
TARGET_Y      = 40.0    # stop when best pilot-Y exceeds this [g/L]
MAX_ITER      = 100     # hard safety cap
MAX_COST      = 15000.0 # hard safety cap on total cost [EUR]


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


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray,
                            beta: float) -> np.ndarray:
    """Simple UCB acquisition: favor points with high mean and high uncertainty."""
    return mu + beta * sigma


def predicted_improvement(mu: np.ndarray, f_best: float) -> np.ndarray:
    """Simple predicted improvement over the current best value."""
    return np.maximum(mu - f_best, 0.0)


def adaptive_acquisition(mu: np.ndarray, sigma: np.ndarray,
                          f_best: float, iteration: int,
                          max_iter: int, xi: float = 0.01) -> np.ndarray:
    """
    Blend UCB and EI smoothly over time:
    - early iterations: more exploration via UCB
    - later iterations: more exploitation via EI
    """
    ei = expected_improvement(mu, sigma, f_best, xi=xi)

    # Smoothly increase exploitation from 0 to 1 over the run.
    progress = (iteration - 1) / max(1, max_iter - 1)
    exploitation_weight = progress**1.5
    exploration_weight = 1.0 - exploitation_weight

    # UCB becomes less dominant as exploitation grows.
    beta = 1.5 * exploration_weight + 0.2 * exploitation_weight
    ucb = upper_confidence_bound(mu, sigma, beta)

    ei_scaled = ei / (np.max(ei) + 1e-12)
    ucb_scaled = ucb / (np.max(ucb) + 1e-12)
    return exploration_weight * ucb_scaled + exploitation_weight * ei_scaled


def scale_cost(scale: str, mu: float, sigma: float) -> float:
    """
    Return the cost (EUR) for a given scale.
    Values are the real costs as provided by the simulator specification.
    """
    COST = {"micro": 10.0, "bench": 500.0, "pilot": 2000.0}
    return COST[scale]


def scale_reliability(scale: str) -> float:
    """Return a reliability factor based on the provided per-scale standard deviation."""
    return 1.0 / SCALE_STD.get(scale, 1.0)


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
    Draw n random candidates uniformly from the recipe space and only use
    the cheaper scales micro/bench during BO.
    """
    rng       = np.random.default_rng(seed)
    unit      = rng.uniform(size=(n, 5))
    recipes   = BOUNDS[:, 0] + unit * (BOUNDS[:, 1] - BOUNDS[:, 0])
    scale_idx = rng.integers(0, len(ACTIVE_SCALES), size=n)
    scales    = [ACTIVE_SCALES[i] for i in scale_idx]
    return recipes, scales


def parse_result(result: dict) -> tuple[float, float]:
    """Extract Y and cost from the API response in a backward-compatible way."""
    y = float(result["Y"])
    cost = result.get("cost", result.get("cost_eur", 0.0))
    return y, float(cost)


def estimate_pilot_yield(scale: str, observed_y: float) -> float:
    """Map observed scale yield to an expected pilot yield using the fitted relations."""
    if scale == "micro":
        return observed_y / 0.040
    if scale == "bench":
        return observed_y / 1.496
    if scale == "pilot":
        return observed_y
    return float("nan")


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
    yield_history: list[float]    = []

    best_expected_y = -np.inf
    best_recipe: np.ndarray | None = None
    cumulative_cost = 0.0
    trajectory      = []   # (iteration, best_expected_y, cumulative_cost)

    # -----------------------------------------------------------------------
    # 1. INITIALIZATION via LHS
    # -----------------------------------------------------------------------
    init_recipes, init_scales = lhs_init(N_INIT)
    total_micro = 0
    total_bench = 0

    for i, (recipe, scale) in enumerate(zip(init_recipes, init_scales)):
        T, pH, F1, F2, F3 = recipe
        result = client.run(scale, T, pH, F1, F2, F3)
        y, cost = parse_result(result)

        if scale == "micro":
            total_micro += 1
        elif scale == "bench":
            total_bench += 1

        expected_pilot_y = estimate_pilot_yield(scale, y)
        cumulative_cost += cost
        print(
            f"[iter {i+1:03d}: \"{scale}\"] {T:.2f},{pH:.2f},{F1:.2f},{F2:.2f},{F3:.2f}"
            f"Y_obs={y:.4f}, expY_p={expected_pilot_y:.4f}"
        )
        print(f"          total cost={cumulative_cost:.1f}, total micro_ex={total_micro} total bench_ex={total_bench}")

        all_recipes.append(recipe)
        all_scales.append(scale)
        all_Y.append(expected_pilot_y)
        all_costs.append(cost)
        yield_history.append(y)

        if expected_pilot_y > best_expected_y:
            best_expected_y = expected_pilot_y
            best_recipe = recipe.copy()

    print(f"\nInit done. Best expected Y so far: {best_expected_y:.3f} g/L | "
          f"Total cost: {cumulative_cost:.1f} EUR\n")

    # -----------------------------------------------------------------------
    # 2. BO LOOP
    # -----------------------------------------------------------------------
    scaler = StandardScaler()

    for iteration in range(1, MAX_ITER + 1):

        # --- Stop criterion ---
        if best_expected_y >= TARGET_Y:
            print(f"\n✓ Target reached: best expected Y = {best_expected_y:.3f} g/L >= {TARGET_Y} g/L")
            break

        # --- Fit GP ---
        X_train = build_X(np.array(all_recipes), all_scales)
        y_train = np.array(all_Y)

        scaler.fit(X_train)
        gp = fit_gp(X_train, y_train, scaler)

        # f_best: best expected Y from the BO runs so far
        f_best = best_expected_y

        # --- Sample candidates and evaluate acquisition ---
        cand_recipes, cand_scales = sample_candidates(N_CANDIDATES, seed=iteration)
        X_cand = build_X(cand_recipes, cand_scales)
        X_cand_scaled = scaler.transform(X_cand)

        mu, sigma = gp.predict(X_cand_scaled, return_std=True)
        acq       = adaptive_acquisition(mu, sigma, f_best, iteration, MAX_ITER)
        pred_imp  = predicted_improvement(mu, f_best)

        # Normalize acquisition score by expected cost of that scale and
        # include the per-scale standard deviation as a reliability-weighting term.
        costs_cand = np.array([scale_cost(s, 0, 0) for s in cand_scales])
        acq_norm   = acq / costs_cand
        pred_imp_norm = pred_imp / (np.max(pred_imp) + 1e-12)
        rel_cand   = np.array([scale_reliability(s) for s in cand_scales])
        acq_final  = 0.5 * (acq_norm * rel_cand) + 0.5 * (pred_imp_norm * rel_cand)

        # Pick best candidate
        best_idx   = int(np.argmax(acq_final))
        next_recipe = cand_recipes[best_idx]
        next_scale  = cand_scales[best_idx]
        T, pH, F1, F2, F3 = next_recipe

        # --- Run experiment ---
        if cumulative_cost + scale_cost(next_scale, 0, 0) < MAX_COST - scale_cost("pilot", 0, 0):
            result = client.run(next_scale, T, pH, F1, F2, F3)
            y, cost = parse_result(result)
        else:
            print(f"\nBudget limits reached: no more experiments allowed. Stopping BO.")
            break

        if next_scale == "micro":
            total_micro += 1
        elif next_scale == "bench":
            total_bench += 1

        expected_pilot_y = estimate_pilot_yield(next_scale, y)
        cumulative_cost += cost
        print(
            f"[iter {iteration:03d}]\"{next_scale}\" {T:.2f},{pH:.2f},{F1:.2f},{F2:.2f},{F3:.2f} "
            f"Y_obs={y:.4f}, expY_p={expected_pilot_y:.4f},best_expY_p={best_expected_y:.4f}, std={SCALE_STD[next_scale]:.4f} "
            f"total cost={cumulative_cost:.1f} total micro_ex={total_micro} total bench_ex={total_bench}"
        )

        all_recipes.append(next_recipe)
        all_scales.append(next_scale)
        all_Y.append(expected_pilot_y)
        all_costs.append(cost)
        yield_history.append(expected_pilot_y)

        if expected_pilot_y > best_expected_y:
            best_expected_y = expected_pilot_y
            best_recipe = next_recipe.copy()

        trajectory.append((iteration, best_expected_y, cumulative_cost))

    # -----------------------------------------------------------------------
    # 3. RESULTS
    # -----------------------------------------------------------------------
    if best_recipe is not None:
        r = best_recipe
        print(f"Best recipe     : T={r[0]:.2f} pH={r[1]:.2f} "
              f"F1={r[2]:.2f} F2={r[3]:.2f} F3={r[4]:.2f}")

        print("\n=== FINAL PILOT VALIDATION ===")
        T, pH, F1, F2, F3 = r
        print(f"Running pilot-scale validation for recipe: T={T:.2f} pH={pH:.2f} "
              f"F1={F1:.2f} F2={F2:.2f} F3={F3:.2f}")
        result = client.run("pilot", T, pH, F1, F2, F3)
        pilot_y, pilot_cost = parse_result(result)
        cumulative_cost += pilot_cost

        print(f"Pilot cost      : {pilot_cost:.1f} EUR")
        print("\n=== FINAL RESULTS ===")
        print(f"Pilot Y         : {pilot_y:.3f} g/L")
        print(f"Total cost      : {cumulative_cost:.1f} EUR")
        print(f"Total experiments: {len(all_Y)}")

    if yield_history:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(1, len(yield_history) + 1), yield_history, marker="o", linestyle="-", color="tab:blue")
        ax.axhline(TARGET_Y, color="tab:red", linestyle="--", label=f"Target {TARGET_Y:.1f} g/L")
        ax.set_title("Yield over iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Yield Y [g/L]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        output_path = os.path.join(os.path.dirname(__file__), "yield_history.png")
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved yield history plot to: {output_path}")

    return trajectory, all_recipes, all_scales, all_Y, all_costs


if __name__ == "__main__":
    run_bo()