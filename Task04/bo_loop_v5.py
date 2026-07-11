from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency fallback
    plt = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from API_Group8 import BioreactorClient


# -----------------------------
# Configuration
# -----------------------------

MICRO_SCALE = "micro"
MICRO_EXPERIMENT_COST_EUR = 10.0

# -----------------------------------------------------------------------------
# BO loop configuration for the combined T/pH loop and the separate feed loop.
# -----------------------------------------------------------------------------

BO_CONFIG = {
    "temperature_ph": {
        "budget_eur": 800.0,
        "initial_points": 20,
        "uncertainty_threshold": 0.01,
        "ei_threshold": 1e-4,
        "patience": 20,
    },
    "feed": {
        "budget_eur": 1000.0,
        "initial_points": 30,
        "uncertainty_threshold": 0.02,
        "ei_threshold": 1e-4,
        "patience": 6,
    },
}

TEMPERATURE_BOUNDS = (20.0, 60.0)
PH_BOUNDS = (3.0, 9.5)

FIXED_FEED_VALUES = {"F1": 1.0, "F2": 1.0, "F3": 1.0}


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Evaluation:
    """Single experiment result for one BO step."""

    variable_name: str
    value: float
    y: float
    cost: float
    iteration: int


@dataclass
class LoopResult:
    """Result container for the combined T/pH BO loop."""

    best_values: dict[str, float] | None = None
    best_y: float = -np.inf
    best_cost: float = np.inf
    total_cost: float = 0.0
    history: list[Evaluation] = field(default_factory=list)


@dataclass
class FeedEvaluation:
    """Single experiment result for one feed-factor BO step."""

    values: dict[str, float]
    y: float
    cost: float
    iteration: int


@dataclass
class FeedLoopResult:
    """Result container for the F1/F2/F3 BO loop."""

    best_values: dict[str, float] | None = None
    best_y: float = -np.inf
    best_cost: float = np.inf
    total_cost: float = 0.0
    history: list[FeedEvaluation] = field(default_factory=list)


# -----------------------------
# Bayesian optimizer
# -----------------------------

class TwoVariableBO:
    """Two-dimensional Bayesian optimizer for T and pH with expected improvement."""

    def __init__(self, bounds: dict[str, tuple[float, float]], seed: int = 0):
        self.names = list(bounds.keys())
        self.bounds = {name: (float(lo), float(hi)) for name, (lo, hi) in bounds.items()}
        self.rng = np.random.default_rng(seed)

        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.array([1.0] * len(self.names)), nu=2.5)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-2))
        )
        self.gp: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=seed,
        )
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def _to_array(self, values: dict[str, float]) -> np.ndarray:
        return np.array([float(values[name]) for name in self.names], dtype=float).reshape(1, -1)

    def add(self, values: dict[str, float], y: float) -> None:
        self.X.append(self._to_array(values))
        self.y.append(float(y))

    def fit(self) -> None:
        if len(self.X) >= 2:
            self.gp.fit(np.vstack(self.X), np.array(self.y, dtype=float))

    def predict(self, values: dict[str, float]) -> tuple[float, float]:
        prediction = self.gp.predict(self._to_array(values), return_std=True)
        mu = float(np.asarray(prediction[0]).reshape(-1)[0])
        sigma = float(np.asarray(prediction[1]).reshape(-1)[0])
        return mu, sigma

    def expected_improvement(self, values: dict[str, float]) -> float:
        if len(self.y) == 0:
            return 0.0

        best_y = max(self.y)
        mu, sigma = self.predict(values)
        if sigma <= 1e-12:
            return 0.0

        improvement = mu - best_y
        if improvement <= 0.0:
            return 0.0

        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return max(0.0, float(ei))

    def propose(self, n_candidates: int = 2048) -> dict[str, float]:
        candidates = np.empty((n_candidates, len(self.names)), dtype=float)
        for idx, name in enumerate(self.names):
            lo, hi = self.bounds[name]
            candidates[:, idx] = self.rng.uniform(lo, hi, size=n_candidates)

        best_values: dict[str, float] | None = None
        best_score = -np.inf
        for candidate in candidates:
            values = {name: float(candidate[idx]) for idx, name in enumerate(self.names)}
            score = self.expected_improvement(values)
            if score > best_score:
                best_score = score
                best_values = values

        if best_values is None:
            return {name: self.bounds[name][0] for name in self.names}
        return best_values

    def initial_points(self, n: int, seed: int, mode: str = "lhs") -> list[dict[str, float]]:
        """Generate initial points with Latin Hypercube Sampling."""

        if mode != "lhs":
            rng = np.random.default_rng(seed)
            return [
                {
                    name: float(rng.uniform(self.bounds[name][0], self.bounds[name][1]))
                    for name in self.names
                }
                for _ in range(n)
            ]

        rng = np.random.default_rng(seed)
        points = np.empty((n, len(self.names)), dtype=float)
        for idx, name in enumerate(self.names):
            lo, hi = self.bounds[name]
            bins = (rng.permutation(n) + rng.random(n)) / n
            points[:, idx] = lo + bins * (hi - lo)

        return [{name: float(points[row, col]) for col, name in enumerate(self.names)} for row in range(n)]


class FeedBO:
    """Three-dimensional Bayesian optimizer for F1/F2/F3 with expected improvement."""

    def __init__(self, bounds: dict[str, tuple[float, float]], seed: int = 0):
        self.names = list(bounds.keys())
        self.bounds = {name: (float(lo), float(hi)) for name, (lo, hi) in bounds.items()}
        self.rng = np.random.default_rng(seed)

        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.array([1.0] * len(self.names)), nu=2.5)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-2))
        )
        self.gp: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=seed,
        )
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def _to_array(self, values: dict[str, float]) -> np.ndarray:
        return np.array([float(values[name]) for name in self.names], dtype=float).reshape(1, -1)

    def add(self, values: dict[str, float], y: float) -> None:
        self.X.append(self._to_array(values))
        self.y.append(float(y))

    def fit(self) -> None:
        if len(self.X) >= 2:
            self.gp.fit(np.vstack(self.X), np.array(self.y, dtype=float))

    def predict(self, values: dict[str, float]) -> tuple[float, float]:
        prediction = self.gp.predict(self._to_array(values), return_std=True)
        mu = float(np.asarray(prediction[0]).reshape(-1)[0])
        sigma = float(np.asarray(prediction[1]).reshape(-1)[0])
        return mu, sigma

    def expected_improvement(self, values: dict[str, float]) -> float:
        if len(self.y) == 0:
            return 0.0

        best_y = max(self.y)
        mu, sigma = self.predict(values)
        if sigma <= 1e-12:
            return 0.0

        improvement = mu - best_y
        if improvement <= 0.0:
            return 0.0

        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return max(0.0, float(ei))

    def propose(self, n_candidates: int = 2048) -> dict[str, float]:
        candidates = np.empty((n_candidates, len(self.names)), dtype=float)
        for idx, name in enumerate(self.names):
            lo, hi = self.bounds[name]
            candidates[:, idx] = self.rng.uniform(lo, hi, size=n_candidates)

        best_values: dict[str, float] | None = None
        best_score = -np.inf
        for candidate in candidates:
            values = {name: float(candidate[idx]) for idx, name in enumerate(self.names)}
            score = self.expected_improvement(values)
            if score > best_score:
                best_score = score
                best_values = values

        if best_values is None:
            return {name: self.bounds[name][0] for name in self.names}
        return best_values

    def initial_points(self, n: int, seed: int, mode: str = "lhs") -> list[dict[str, float]]:
        """Generate initial points with Latin Hypercube Sampling."""

        if mode != "lhs":
            rng = np.random.default_rng(seed)
            return [
                {
                    name: float(rng.uniform(self.bounds[name][0], self.bounds[name][1]))
                    for name in self.names
                }
                for _ in range(n)
            ]

        rng = np.random.default_rng(seed)
        points = np.empty((n, len(self.names)), dtype=float)
        for idx, name in enumerate(self.names):
            lo, hi = self.bounds[name]
            bins = (rng.permutation(n) + rng.random(n)) / n
            points[:, idx] = lo + bins * (hi - lo)

        return [{name: float(points[row, col]) for col, name in enumerate(self.names)} for row in range(n)]


# -----------------------------
# Helpers
# -----------------------------

def _extract_y_cost(response: dict[str, Any]) -> tuple[float, float]:
    """Extract the objective value and cost from the API response."""

    y = response.get("Y", response.get("yield", response.get("y")))
    if y is None:
        raise KeyError("Response does not contain yield/Y/y.")
    cost = response.get("cost", response.get("C", 0.0))
    return float(y), float(cost)


def _run_temperature_ph_loop(
    client: BioreactorClient,
    budget: float,
    n_init: int,
    uncertainty_threshold: float,
    ei_threshold: float,
    patience: int,
    seed: int = 0,
) -> LoopResult:
    """Optimize T and pH jointly in one BO loop while keeping feed values fixed."""

    bounds = {"T": TEMPERATURE_BOUNDS, "pH": PH_BOUNDS}
    bo = TwoVariableBO(bounds=bounds, seed=seed)
    result = LoopResult()

    initial_values = bo.initial_points(n=n_init, seed=seed, mode="lhs")

    stagnation_count = 0

    for iteration, values in enumerate(initial_values, start=1):
        if result.total_cost + MICRO_EXPERIMENT_COST_EUR > budget:
            break

        payload = {
            "T": float(values["T"]),
            "pH": float(values["pH"]),
            "F1": float(FIXED_FEED_VALUES["F1"]),
            "F2": float(FIXED_FEED_VALUES["F2"]),
            "F3": float(FIXED_FEED_VALUES["F3"]),
        }
        response = client.run(scale=MICRO_SCALE, **payload)
        y, _ = _extract_y_cost(response)

        bo.add(values, y)
        bo.fit()

        result.total_cost += MICRO_EXPERIMENT_COST_EUR
        result.history.append(Evaluation(variable_name="T/pH", value=float(values["T"]), y=y, cost=MICRO_EXPERIMENT_COST_EUR, iteration=iteration))

        if y > result.best_y:
            result.best_y = y
            result.best_values = {name: float(values[name]) for name in values}
            result.best_cost = MICRO_EXPERIMENT_COST_EUR

        print(f"T/pH optimization ({iteration}/{n_init}) | experiments: {len(result.history)}, total cost: {result.total_cost:.2f}/{budget:.2f}")

    while result.total_cost < budget:
        if len(result.history) == 0:
            break
        if result.total_cost + MICRO_EXPERIMENT_COST_EUR > budget:
            break

        next_values = bo.propose()
        _, sigma = bo.predict(next_values)
        ei = bo.expected_improvement(next_values)

        if len(result.history) >= n_init:
            if sigma <= uncertainty_threshold and ei <= ei_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if stagnation_count >= patience:
                break

        payload = {
            "T": float(next_values["T"]),
            "pH": float(next_values["pH"]),
            "F1": float(FIXED_FEED_VALUES["F1"]),
            "F2": float(FIXED_FEED_VALUES["F2"]),
            "F3": float(FIXED_FEED_VALUES["F3"]),
        }
        response = client.run(scale=MICRO_SCALE, **payload)
        y, _ = _extract_y_cost(response)

        bo.add(next_values, y)
        bo.fit()

        result.total_cost += MICRO_EXPERIMENT_COST_EUR
        result.history.append(Evaluation(variable_name="T/pH", value=float(next_values["T"]), y=y, cost=MICRO_EXPERIMENT_COST_EUR, iteration=len(result.history) + 1))

        if y > result.best_y:
            result.best_y = y
            result.best_values = {name: float(next_values[name]) for name in next_values}
            result.best_cost = MICRO_EXPERIMENT_COST_EUR

        print(
            f"T/pH optimization ({len(result.history)}/{int(budget / MICRO_EXPERIMENT_COST_EUR)}) | "
            f"experiments: {len(result.history)}, total cost: {result.total_cost:.2f}/{budget:.2f}"
        )

    return result


def _run_feed_loop(
    client: BioreactorClient,
    fixed_T: float,
    fixed_pH: float,
    budget: float,
    n_init: int,
    uncertainty_threshold: float,
    ei_threshold: float,
    patience: int,
    seed: int = 0,
) -> FeedLoopResult:
    """Optimize F1/F2/F3 with EI and Latin Hypercube Sampling while T and pH are fixed."""

    bounds = {"F1": (0.1, 2.0), "F2": (0.1, 2.0), "F3": (0.1, 2.0)}
    bo = FeedBO(bounds=bounds, seed=seed)
    result = FeedLoopResult()

    initial_values = bo.initial_points(n=n_init, seed=seed, mode="lhs")

    stagnation_count = 0

    for iteration, values in enumerate(initial_values, start=1):
        if result.total_cost + MICRO_EXPERIMENT_COST_EUR > budget:
            break

        payload = {
            "T": float(fixed_T),
            "pH": float(fixed_pH),
            "F1": float(values["F1"]),
            "F2": float(values["F2"]),
            "F3": float(values["F3"]),
        }
        response = client.run(scale=MICRO_SCALE, **payload)
        y, _ = _extract_y_cost(response)

        bo.add(values, y)
        bo.fit()

        result.total_cost += MICRO_EXPERIMENT_COST_EUR
        result.history.append(FeedEvaluation(values={name: float(values[name]) for name in values}, y=y, cost=MICRO_EXPERIMENT_COST_EUR, iteration=iteration))

        if y > result.best_y:
            result.best_y = y
            result.best_values = {name: float(values[name]) for name in values}
            result.best_cost = MICRO_EXPERIMENT_COST_EUR

        print(f"Feed optimization ({iteration}/{n_init}) | experiments: {len(result.history)}, total cost: {result.total_cost:.2f}/{budget:.2f}")

    while result.total_cost < budget:
        if len(result.history) == 0:
            break
        if result.total_cost + MICRO_EXPERIMENT_COST_EUR > budget:
            break

        next_values = bo.propose()
        _, sigma = bo.predict(next_values)
        ei = bo.expected_improvement(next_values)

        if len(result.history) >= n_init:
            if sigma <= uncertainty_threshold and ei <= ei_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if stagnation_count >= patience:
                break

        payload = {
            "T": float(fixed_T),
            "pH": float(fixed_pH),
            "F1": float(next_values["F1"]),
            "F2": float(next_values["F2"]),
            "F3": float(next_values["F3"]),
        }
        response = client.run(scale=MICRO_SCALE, **payload)
        y, _ = _extract_y_cost(response)

        bo.add(next_values, y)
        bo.fit()

        result.total_cost += MICRO_EXPERIMENT_COST_EUR
        result.history.append(FeedEvaluation(values={name: float(next_values[name]) for name in next_values}, y=y, cost=MICRO_EXPERIMENT_COST_EUR, iteration=len(result.history) + 1))

        if y > result.best_y:
            result.best_y = y
            result.best_values = {name: float(next_values[name]) for name in next_values}
            result.best_cost = MICRO_EXPERIMENT_COST_EUR

        print(
            f"Feed optimization ({len(result.history)}/{int(budget / MICRO_EXPERIMENT_COST_EUR)}) | "
            f"experiments: {len(result.history)}, total cost: {result.total_cost:.2f}/{budget:.2f}"
        )

    return result


def _plot_history(history: list[Evaluation], variable_name: str, output_dir: Path) -> Path | None:
    """Create a scatter plot of the variable value against the observed yield."""

    if plt is None:
        return None

    values = [entry.value for entry in history if entry.variable_name == variable_name]
    ys = [entry.y for entry in history if entry.variable_name == variable_name]
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(values, ys, color="tab:blue", s=50, alpha=0.8)
    ax.set_title(f"{variable_name} vs. Yield")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / f"{variable_name.lower()}_vs_yield.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main(user: str | None = None, password: str | None = None, base_url: str = "https://mlme26biosim.org") -> dict[str, Any]:
    """Run the combined T/pH BO loop and the feed BO loop, then perform a final pilot experiment."""

    from API_Group8 import USER as DEFAULT_USER, PASSWORD as DEFAULT_PASSWORD

    client = BioreactorClient(base_url=base_url)
    client.login(user or DEFAULT_USER, password or DEFAULT_PASSWORD)

    temperature_ph_config = BO_CONFIG["temperature_ph"]
    temperature_ph_result = _run_temperature_ph_loop(
        client,
        budget=temperature_ph_config["budget_eur"],
        n_init=temperature_ph_config["initial_points"],
        uncertainty_threshold=temperature_ph_config["uncertainty_threshold"],
        ei_threshold=temperature_ph_config["ei_threshold"],
        patience=temperature_ph_config["patience"],
    )

    fixed_T = temperature_ph_result.best_values["T"] if temperature_ph_result.best_values is not None else 37.5
    fixed_pH = temperature_ph_result.best_values["pH"] if temperature_ph_result.best_values is not None else 6.0

    feed_config = BO_CONFIG["feed"]
    feed_result = _run_feed_loop(
        client,
        fixed_T=float(fixed_T),
        fixed_pH=float(fixed_pH),
        budget=feed_config["budget_eur"],
        n_init=feed_config["initial_points"],
        uncertainty_threshold=feed_config["uncertainty_threshold"],
        ei_threshold=feed_config["ei_threshold"],
        patience=feed_config["patience"],
    )

    final_recipe = {
        "T": float(fixed_T),
        "pH": float(fixed_pH),
        "F1": float(feed_result.best_values["F1"]) if feed_result.best_values is not None else 1.0,
        "F2": float(feed_result.best_values["F2"]) if feed_result.best_values is not None else 1.0,
        "F3": float(feed_result.best_values["F3"]) if feed_result.best_values is not None else 1.0,
    }

    pilot_response = client.run(
        scale=MICRO_SCALE,
        T=final_recipe["T"],
        pH=final_recipe["pH"],
        F1=final_recipe["F1"],
        F2=final_recipe["F2"],
        F3=final_recipe["F3"],
    )
    pilot_y, pilot_cost = _extract_y_cost(pilot_response)

    output_dir = ROOT / "output_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_plot = _plot_history(temperature_ph_result.history, "T", output_dir)
    ph_plot = _plot_history(temperature_ph_result.history, "pH", output_dir)

    if temp_plot is not None:
        print(f"Saved temperature plot to {temp_plot}")
    if ph_plot is not None:
        print(f"Saved pH plot to {ph_plot}")

    summary = {
        "optimal_T": fixed_T,
        "optimal_pH": fixed_pH,
        "temperature_ph_result": temperature_ph_result,
        "feed_result": feed_result,
        "final_recipe": final_recipe,
        "pilot_result": {
            "T": final_recipe["T"],
            "pH": final_recipe["pH"],
            "F1": final_recipe["F1"],
            "F2": final_recipe["F2"],
            "F3": final_recipe["F3"],
            "Y": pilot_y,
            "cost": pilot_cost,
        },
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
