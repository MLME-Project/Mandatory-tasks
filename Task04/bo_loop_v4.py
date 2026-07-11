from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from API_Group8 import BioreactorClient


# -----------------------------
# Configuration
# -----------------------------

MICRO_SCALE = "micro"
BUDGET_EUR = 200.0
INITIAL_POINTS = 5
MICRO_EXPERIMENT_COST_EUR = 10.0

TEMPERATURE_BOUNDS = (25.0, 50.0)
PH_BOUNDS = (4.0, 8.0)

TEMPERATURE_FIXED_VALUES = {"pH": 6.0, "F1": 1.0, "F2": 1.0, "F3": 1.0}
PH_FIXED_VALUES = {"T": 37.5, "F1": 1.0, "F2": 1.0, "F3": 1.0}


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Evaluation:
    """Single experiment result for one BO loop."""

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
        mu, sigma = self.gp.predict(np.array([[value]], dtype=float), return_std=True)
        return float(mu[0]), float(sigma[0])

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

    def initial_points(self, n: int, seed: int) -> list[float]:
        """Create initial points with a center-heavy distribution."""

        rng = np.random.default_rng(seed)
        samples = rng.beta(4.0, 4.0, size=n)
        return [self.lower + sample * (self.upper - self.lower) for sample in samples]


# -----------------------------
# Helpers
# -----------------------------

def _extract_y_cost(response: dict[str, Any]) -> tuple[float, float]:
    """Extract the objective value and cost from the API response."""

    y = response.get("Y", response.get("yield", response.get("y")))
    if y is None:
        raise KeyError("Response does not contain yield/Y/y.")
    cost = response.get("cost", resp.get("C", 0.0))
    return float(y), float(cost)


def _run_single_variable_loop(
    client: BioreactorClient,
    variable_name: str,
    bounds: tuple[float, float],
    fixed_values: dict[str, float],
    budget: float = BUDGET_EUR,
    n_init: int = INITIAL_POINTS,
    seed: int = 0,
) -> LoopResult:
    """Run one independent BO loop until the budget is exhausted."""

    bo = SingleVariableBO(lower=bounds[0], upper=bounds[1], seed=seed)
    result = LoopResult(variable_name=variable_name)

    initial_values = bo.initial_points(n=n_init, seed=seed)

    uncertainty_threshold = 0.05 * (bounds[1] - bounds[0])
    ei_threshold = 1e-3
    patience = 3
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
        mu, sigma = bo.predict(next_value)
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

    return _run_single_variable_loop(
        client=client,
        variable_name="T",
        bounds=TEMPERATURE_BOUNDS,
        fixed_values=TEMPERATURE_FIXED_VALUES,
    )


def run_ph_loop(client: BioreactorClient) -> LoopResult:
    """Optimize pH with temperature and the feed variables fixed."""

    return _run_single_variable_loop(
        client=client,
        variable_name="pH",
        bounds=PH_BOUNDS,
        fixed_values=PH_FIXED_VALUES,
    )


def main(user: str | None = None, password: str | None = None, base_url: str = "https://mlme26biosim.org") -> dict[str, Any]:
    """Run both BO loops and return the selected optimum values."""

    from API_Group8 import USER as DEFAULT_USER, PASSWORD as DEFAULT_PASSWORD

    client = BioreactorClient(base_url=base_url)
    client.login(user or DEFAULT_USER, password or DEFAULT_PASSWORD)

    temperature_result = run_temperature_loop(client)
    ph_result = run_ph_loop(client)

    summary = {
        "optimal_T": temperature_result.best_value,
        "optimal_pH": ph_result.best_value,
        "temperature_result": temperature_result,
        "ph_result": ph_result,
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
