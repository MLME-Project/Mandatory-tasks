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
MICRO_EXPERIMENT_COST_EUR = 10.0  # Cost per single experiment; change here if the budget model changes

# -----------------------------------------------------------------------------
# BO loop configuration: each loop has its own independent parameters.
# Change these values here if you want to adjust stopping behavior, budget, or
# the number of initial points for one specific optimization loop.
# -----------------------------------------------------------------------------

BO_CONFIG = {
    "temperature": {
        "budget_eur": 500.0,  # Maximum budget for the temperature BO loop
        "initial_points": 20,  # Number of initial experiments before BO starts
        "uncertainty_threshold": 0.01,  # Stop when model uncertainty is very low
        "ei_threshold": 1e-4,  # Stop when expected improvement is very small
        "patience": 10,  # Consecutive low-improvement iterations before stopping
    },
    "ph": {
        "budget_eur": 300.0,  # Maximum budget for the pH BO loop
        "initial_points": 15,  # Number of initial experiments before BO starts
        "uncertainty_threshold": 0.01,  # Stop when model uncertainty is very low
        "ei_threshold": 1e-4,  # Stop when expected improvement is very small
        "patience": 3,  # Consecutive low-improvement iterations before stopping
    },
    "feed": {
        "budget_eur": 1000.0,  # Maximum budget for the F1/F2/F3 BO loop
        "initial_points": 15,  # Number of initial experiments before BO starts
        "uncertainty_threshold": 0.02,  # Stop when model uncertainty is very low
        "ei_threshold": 1e-4,  # Stop when expected improvement is very small
        "patience": 5,  # Consecutive low-improvement iterations before stopping
    },
}

TEMPERATURE_BOUNDS = (30.0, 40.0)
PH_BOUNDS = (3.0, 9.5)

TEMPERATURE_FIXED_VALUES = {"pH": 7.0, "F1": 1.0, "F2": 1.0, "F3": 1.0}
PH_FIXED_VALUES = {"T": 37.5, "F1": 1.0, "F2": 1.0, "F3": 1.0}


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
    """Result container for one single-variable BO loop."""

    variable_name: str
    best_value: float | None = None
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

class SingleVariableBO:
    """One-dimensional Bayesian optimizer with expected improvement."""

    def __init__(self, lower: float, upper: float, seed: int = 0):
        self.lower = float(lower)
        self.upper = float(upper)
        self.rng = np.random.default_rng(seed)

        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.array([1.0]), nu=2.5)
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

    def _clip(self, value: float) -> float:
        return float(np.clip(value, self.lower, self.upper))

    def add(self, value: float, y: float) -> None:
        self.X.append(np.array([[value]], dtype=float))
        self.y.append(float(y))

    def fit(self) -> None:
        if len(self.X) >= 2:
            self.gp.fit(np.vstack(self.X), np.array(self.y, dtype=float))

    def predict(self, value: float) -> tuple[float, float]:
        prediction = self.gp.predict(np.array([[value]], dtype=float), return_std=True)
        mu = float(np.asarray(prediction[0]).reshape(-1)[0])
        sigma = float(np.asarray(prediction[1]).reshape(-1)[0])
        return mu, sigma

    def expected_improvement(self, value: float) -> float:
        if len(self.y) == 0:
            return 0.0

        best_y = max(self.y)
        mu, sigma = self.predict(value)
        if sigma <= 1e-12:
            return 0.0

        improvement = mu - best_y
        if improvement <= 0.0:
            return 0.0

        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return max(0.0, float(ei))

    def propose(self, n_candidates: int = 2048) -> float:
        candidates = self.rng.uniform(self.lower, self.upper, size=n_candidates)
        best_value = self.lower
        best_score = -np.inf

        for candidate in candidates:
            score = self.expected_improvement(candidate)
            if score > best_score:
                best_score = score
                best_value = candidate

        return self._clip(best_value)

    def initial_points(self, n: int, seed: int, mode: str = "beta") -> list[float]:
        """Create initial points with either a center-heavy or evenly spaced distribution."""

        if mode == "uniform":
            if n <= 1:
                return [self.lower + 0.5 * (self.upper - self.lower)]
            return [float(value) for value in np.linspace(self.lower, self.upper, n)]

        rng = np.random.default_rng(seed)
        samples = rng.beta(4.0, 4.0, size=n)
        return [self.lower + sample * (self.upper - self.lower) for sample in samples]


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


def _run_single_variable_loop(
    client: BioreactorClient,
    variable_name: str,
    bounds: tuple[float, float],
    fixed_values: dict[str, float],
    budget: float,
    n_init: int,
    uncertainty_threshold: float,
    ei_threshold: float,
    patience: int,
    seed: int = 0,
    initial_points_mode: str = "beta",
) -> LoopResult:
    """Run one independent BO loop until the budget is exhausted."""

    bo = SingleVariableBO(lower=bounds[0], upper=bounds[1], seed=seed)
    result = LoopResult(variable_name=variable_name)

    initial_values = bo.initial_points(n=n_init, seed=seed, mode=initial_points_mode)

    # Stopping criteria for the single-variable BO loop.
    # Change these values if you want the optimizer to stop earlier or continue longer.
    # The stopping thresholds are now taken from the central BO configuration.
    stagnation_count = 0

    for iteration, value in enumerate(initial_values, start=1):
        if result.total_cost + MICRO_EXPERIMENT_COST_EUR > budget:
            break

        payload = dict(fixed_values)
        payload[variable_name] = float(value)
        response = client.run(
            scale=MICRO_SCALE,
            T=payload.get("T", TEMPERATURE_FIXED_VALUES.get("T", 37.5)),
            pH=payload.get("pH", PH_FIXED_VALUES.get("pH", 6.0)),
            F1=payload.get("F1", 1.0),
            F2=payload.get("F2", 1.0),
            F3=payload.get("F3", 1.0),
        )
        y, _ = _extract_y_cost(response)

        bo.add(value, y)
        bo.fit()

        result.total_cost += MICRO_EXPERIMENT_COST_EUR
        result.history.append(
            Evaluation(variable_name=variable_name, value=float(value), y=y, cost=MICRO_EXPERIMENT_COST_EUR, iteration=iteration)
        )

        if y > result.best_y:
            result.best_y = y
            result.best_value = float(value)
            result.best_cost = MICRO_EXPERIMENT_COST_EUR

        print(f"{variable_name} optimization ({iteration}/{n_init}) | experiments: {len(result.history)}, total cost: {result.total_cost:.2f}/{budget:.2f}")

    while result.total_cost < budget:
        if len(result.history) == 0:
            break
        if result.total_cost + MICRO_EXPERIMENT_COST_EUR > budget:
            break

        next_value = bo.propose()
        _, sigma = bo.predict(next_value)
        ei = bo.expected_improvement(next_value)

        if len(result.history) >= n_init:
            if sigma <= uncertainty_threshold and ei <= ei_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if stagnation_count >= patience:
                break

        payload = dict(fixed_values)
        payload[variable_name] = float(next_value)
        response = client.run(
            scale=MICRO_SCALE,
            T=payload.get("T", TEMPERATURE_FIXED_VALUES.get("T", 37.5)),
            pH=payload.get("pH", PH_FIXED_VALUES.get("pH", 6.0)),
            F1=payload.get("F1", 1.0),
            F2=payload.get("F2", 1.0),
            F3=payload.get("F3", 1.0),
        )
        y, _ = _extract_y_cost(response)

        bo.add(next_value, y)
        bo.fit()

        result.total_cost += MICRO_EXPERIMENT_COST_EUR
        result.history.append(
            Evaluation(variable_name=variable_name, value=float(next_value), y=y, cost=MICRO_EXPERIMENT_COST_EUR, iteration=len(result.history) + 1)
        )

        if y > result.best_y:
            result.best_y = y
            result.best_value = float(next_value)
            result.best_cost = MICRO_EXPERIMENT_COST_EUR

        print(
            f"{variable_name} optimization ({len(result.history)}/{int(budget / MICRO_EXPERIMENT_COST_EUR)}) | "
            f"experiments: {len(result.history)}, total cost: {result.total_cost:.2f}/{budget:.2f}"
        )

    return result


def run_temperature_loop(client: BioreactorClient) -> LoopResult:
    """Optimize temperature with pH and the feed variables fixed."""

    config = BO_CONFIG["temperature"]
    return _run_single_variable_loop(
        client=client,
        variable_name="T",
        bounds=TEMPERATURE_BOUNDS,
        fixed_values=TEMPERATURE_FIXED_VALUES,
        budget=config["budget_eur"],
        n_init=config["initial_points"],
        uncertainty_threshold=config["uncertainty_threshold"] * (TEMPERATURE_BOUNDS[1] - TEMPERATURE_BOUNDS[0]),
        ei_threshold=config["ei_threshold"],
        patience=config["patience"],
    )


def run_ph_loop(client: BioreactorClient, fixed_T: float | None = None) -> LoopResult:
    """Optimize pH while keeping the temperature fixed at the chosen value."""

    config = BO_CONFIG["ph"]
    fixed_values = dict(PH_FIXED_VALUES)
    fixed_values["T"] = float(fixed_T if fixed_T is not None else TEMPERATURE_FIXED_VALUES.get("T", 37.5))

    return _run_single_variable_loop(
        client=client,
        variable_name="pH",
        bounds=PH_BOUNDS,
        fixed_values=fixed_values,
        budget=config["budget_eur"],
        n_init=config["initial_points"],
        uncertainty_threshold=config["uncertainty_threshold"] * (PH_BOUNDS[1] - PH_BOUNDS[0]),
        ei_threshold=config["ei_threshold"],
        patience=config["patience"],
        initial_points_mode="uniform",
    )


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

    # Stopping criteria for the feed-factor BO loop (F1/F2/F3).
    # These are separate from the T/pH loop and can be tuned independently.
    # The feed-loop stopping thresholds are also taken from the central BO configuration.
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
    """Run both BO loops and return the selected optimum values."""

    from API_Group8 import USER as DEFAULT_USER, PASSWORD as DEFAULT_PASSWORD

    client = BioreactorClient(base_url=base_url)
    client.login(user or DEFAULT_USER, password or DEFAULT_PASSWORD)

    temperature_result = run_temperature_loop(client)
    fixed_T = temperature_result.best_value if temperature_result.best_value is not None else TEMPERATURE_FIXED_VALUES.get("T", 37.5)
    ph_result = run_ph_loop(client, fixed_T=fixed_T)
    fixed_pH = ph_result.best_value if ph_result.best_value is not None else PH_FIXED_VALUES.get("pH", 6.0)
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

    temp_plot = _plot_history(temperature_result.history, "T", output_dir)
    ph_plot = _plot_history(ph_result.history, "pH", output_dir)

    if temp_plot is not None:
        print(f"Saved temperature plot to {temp_plot}")
    if ph_plot is not None:
        print(f"Saved pH plot to {ph_plot}")

    summary = {
        "optimal_T": temperature_result.best_value,
        "optimal_pH": ph_result.best_value,
        "temperature_result": temperature_result,
        "ph_result": ph_result,
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
        "plot_paths": {
            "T_vs_Y": str(temp_plot) if temp_plot is not None else None,
            "pH_vs_Y": str(ph_plot) if ph_plot is not None else None,
        },
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
