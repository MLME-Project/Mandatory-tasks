from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

from API_Group8 import BioreactorClient


# -----------------------------
# Config
# -----------------------------

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

SCALE_ORDER = ["micro", "bench"]
FINAL_SCALE = "pilot"

DEFAULT_BOUNDS = {
    "T": (25.0, 40.0),
    "pH": (5.5, 7.5),
    "F1": (0.1, 1.0),
    "F2": (0.1, 1.0),
    "F3": (0.1, 1.0),
}


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Observation:
    x: dict[str, float]
    y: float
    cost: float
    kind: str
    iteration: int


@dataclass
class RunResult:
    scale: str
    best_x: dict[str, float] | None = None
    best_y: float = -np.inf
    best_cost: float = np.inf
    history: list[Observation] = field(default_factory=list)
    pilot_checks: list[Observation] = field(default_factory=list)


# -----------------------------
# Search space
# -----------------------------

class SearchSpace:
    def __init__(self, bounds: dict[str, tuple[float, float]]):
        self.bounds = bounds
        self.keys = list(bounds.keys())
        self.dim = len(self.keys)

    def clip(self, x: dict[str, float]) -> dict[str, float]:
        return {
            k: float(np.clip(v, self.bounds[k][0], self.bounds[k][1]))
            for k, v in x.items()
        }

    def to_vector(self, x: dict[str, float]) -> np.ndarray:
        return np.array([x[k] for k in self.keys], dtype=float)

    def from_vector(self, x: np.ndarray) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.keys, x)}

    def lhs(self, n: int, seed: int | None = None) -> list[dict[str, float]]:
        sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
        u = sampler.random(n=n)
        points = qmc.scale(
            u,
            l_bounds=[self.bounds[k][0] for k in self.keys],
            u_bounds=[self.bounds[k][1] for k in self.keys],
        )
        return [self.from_vector(row) for row in points]

    def random(self, n: int, seed: int | None = None) -> list[dict[str, float]]:
        rng = np.random.default_rng(seed)
        pts = []
        for _ in range(n):
            pts.append({
                k: float(rng.uniform(lo, hi))
                for k, (lo, hi) in self.bounds.items()
            })
        return pts


# -----------------------------
# Bayesian optimizer
# -----------------------------

class BayesianOptimizer:
    def __init__(
        self,
        space: SearchSpace,
        beta: float = 2.0,
        seed: int = 0,
        n_restarts: int = 5,
    ):
        self.space = space
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(space.dim), nu=2.5)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-2))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=seed,
        )
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.best_y_trace: list[float] = []

    def add(self, x: dict[str, float], y: float):
        self.X.append(self.space.to_vector(x))
        self.y.append(float(y))
        self.best_y_trace.append(float(np.max(self.y)))

    def fit(self):
        if len(self.X) >= 2:
            self.gp.fit(np.vstack(self.X), np.array(self.y, dtype=float))

    def ucb(self, x_vec: np.ndarray) -> float:
        mu, sigma = self.gp.predict(x_vec.reshape(1, -1), return_std=True)
        return float(mu[0] + self.beta * sigma[0])

    def propose(self, n_candidates: int = 4096) -> dict[str, float]:
        cand = self.space.lhs(n_candidates, seed=int(self.rng.integers(0, 1_000_000)))
        best_x = cand[0]
        best_val = -np.inf
        for x in cand:
            val = self.ucb(self.space.to_vector(x))
            if val > best_val:
                best_val = val
                best_x = x
        return self.space.clip(best_x)

    def mean_expected_improvement_pct_last5(self) -> float | None:
        if len(self.best_y_trace) < 6:
            return None
        diffs = []
        tail = self.best_y_trace[-6:]
        for prev, cur in zip(tail[:-1], tail[1:]):
            if abs(prev) < 1e-12:
                continue
            diffs.append(100.0 * (cur - prev) / abs(prev))
        if not diffs:
            return 0.0
        return float(np.mean(diffs[-5:]))

    def should_stop(self, threshold_pct: float) -> bool:
        score = self.mean_expected_improvement_pct_last5()
        return score is not None and score < threshold_pct


# -----------------------------
# Helpers
# -----------------------------

def _extract_y_cost(resp: dict[str, Any]) -> tuple[float, float]:
    y = resp.get("Y", resp.get("yield", resp.get("y")))
    if y is None:
        raise KeyError("Response does not contain yield/Y/y.")
    cost = resp.get("cost", resp.get("C", 0.0))
    return float(y), float(cost)


def _pilot_payload(scale: str, x: dict[str, float]) -> dict[str, Any]:
    payload = {
        "scale": scale,
        "recipe": {
            "T": float(x["T"]),
            "pH": float(x["pH"]),
            "F1": float(x["F1"]),
            "F2": float(x["F2"]),
            "F3": float(x["F3"]),
        },
    }
    return payload


def _plot_scale_history(result: RunResult, title: str):
    iters = [o.iteration for o in result.history]
    ys = [o.y for o in result.history]
    costs = [o.cost for o in result.history]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(iters, ys, marker="o", label="BO yield")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Yield Y")
    ax1.grid(True, alpha=0.3)

    if result.pilot_checks:
        pilot_iters = [o.iteration for o in result.pilot_checks]
        pilot_ys = [o.y for o in result.pilot_checks]
        ax1.plot(
            pilot_iters,
            pilot_ys,
            linestyle="None",
            marker="x",
            markersize=10,
            color="tab:green",
            label="Pilot checks",
            zorder=5,
        )

    ax2 = ax1.twinx()
    ax2.plot(iters, costs, marker="s", color="tab:red", label="Cost")
    ax2.set_ylabel("Cost")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    plt.title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{title.replace(' ', '_').lower()}.png", dpi=160)
    plt.close(fig)


def _plot_testmode_checks(result: RunResult, title: str):
    if not result.pilot_checks:
        return
    iters = [o.iteration for o in result.pilot_checks]
    ys = [o.y for o in result.pilot_checks]

    fig = plt.figure(figsize=(9, 4))
    plt.plot(iters, ys, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Pilot check yield")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{title.replace(' ', '_').lower()}.png", dpi=160)
    plt.close(fig)


# -----------------------------
# Main optimization loop
# -----------------------------

def run_scale_loop(
    client: BioreactorClient,
    scale: str,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_init: int = 8,
    max_iter: int = 40,
    beta: float = 2.0,
    stop_threshold_pct: float = 0.5,
    test_mode: bool = False,
    pilot_every: int = 5,
) -> RunResult:
    space = SearchSpace(bounds or DEFAULT_BOUNDS)
    bo = BayesianOptimizer(space=space, beta=beta, seed=42)
    result = RunResult(scale=scale)

    init_points = space.lhs(n_init, seed=42)
    for i, x in enumerate(init_points, start=1):
        resp = client.run(scale=scale, T=x["T"], pH=x["pH"], F1=x["F1"], F2=x["F2"], F3=x["F3"])
        y, cost = _extract_y_cost(resp)
        bo.add(x, y)
        bo.fit()

        obs = Observation(x=x, y=y, cost=cost, kind="init", iteration=i)
        result.history.append(obs)

        if y > result.best_y:
            result.best_y = y
            result.best_x = x
            result.best_cost = cost

    for i in range(n_init + 1, max_iter + 1):
        if bo.should_stop(stop_threshold_pct):
            break

        x_next = bo.propose()
        resp = client.run(scale=scale, T=x_next["T"], pH=x_next["pH"], F1=x_next["F1"], F2=x_next["F2"], F3=x_next["F3"])
        y, cost = _extract_y_cost(resp)

        bo.add(x_next, y)
        bo.fit()

        obs = Observation(x=x_next, y=y, cost=cost, kind="bo", iteration=i)
        result.history.append(obs)

        if y > result.best_y:
            result.best_y = y
            result.best_x = x_next
            result.best_cost = cost

        if test_mode and (i % pilot_every == 0) and result.best_x is not None:
            pilot_resp = client.run(
                scale=FINAL_SCALE,
                T=result.best_x["T"],
                pH=result.best_x["pH"],
                F1=result.best_x["F1"],
                F2=result.best_x["F2"],
                F3=result.best_x["F3"],
            )
            py, pcost = _extract_y_cost(pilot_resp)
            result.pilot_checks.append(
                Observation(
                    x=dict(result.best_x),
                    y=py,
                    cost=pcost,
                    kind="pilot_check",
                    iteration=i,
                )
            )

    _plot_scale_history(result, title=f"{scale} history")
    if test_mode:
        _plot_testmode_checks(result, title=f"{scale} pilot checks")

    return result


# -----------------------------
# Campaign orchestration
# -----------------------------

def main(
    user: str | None = None,
    password: str | None = None,
    base_url: str = "https://mlme26biosim.org",
    test_mode: bool = False,
):
    from API_Group8 import USER as DEFAULT_USER, PASSWORD as DEFAULT_PASSWORD

    client = BioreactorClient(base_url=base_url)
    client.login(user or DEFAULT_USER, password or DEFAULT_PASSWORD)

    micro_result = run_scale_loop(
        client=client,
        scale="micro",
        n_init=8,
        max_iter=40,
        beta=2.0,
        stop_threshold_pct=0.5,
        test_mode=test_mode,
        pilot_every=5,
    )

    bench_result = run_scale_loop(
        client=client,
        scale="bench",
        n_init=8,
        max_iter=40,
        beta=2.0,
        stop_threshold_pct=0.5,
        test_mode=test_mode,
        pilot_every=5,
    )

    final_x = bench_result.best_x or micro_result.best_x
    if final_x is None:
        raise RuntimeError("No valid optimum found in micro/bench loops.")

    final_pilot = client.run(
        scale=FINAL_SCALE,
        T=final_x["T"],
        pH=final_x["pH"],
        F1=final_x["F1"],
        F2=final_x["F2"],
        F3=final_x["F3"],
    )
    final_y, final_cost = _extract_y_cost(final_pilot)

    summary = {
        "micro_best_y": micro_result.best_y,
        "bench_best_y": bench_result.best_y,
        "final_pilot_y": final_y,
        "final_pilot_cost": final_cost,
        "micro_best_x": micro_result.best_x,
        "bench_best_x": bench_result.best_x,
        "final_x": final_x,
    }

    print(summary)
    return summary