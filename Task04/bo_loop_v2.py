from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

from API_Group8 import BioreactorClient

OUTPUT_DIR = Path("output_v2")
OUTPUT_DIR.mkdir(exist_ok=True)


def _resolve_plot_output(plot_out: Path | None) -> tuple[Path, str | None]:
    if plot_out is None:
        return OUTPUT_DIR, None
    if plot_out.exists() and plot_out.is_dir():
        return plot_out, None

    output_dir = plot_out.parent if plot_out.parent != Path("") else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_out.suffix:
        return output_dir, plot_out.stem
    return output_dir, None


SCALE_ORDER = ["micro", "bench"]
FINAL_SCALE = "pilot"

DEFAULT_BOUNDS = {
    "T": (25.0, 40.0),
    "pH": (5.5, 7.5),
    "F1": (0.1, 1.0),
    "F2": (0.1, 1.0),
    "F3": (0.1, 1.0),
}


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
    final_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)


class SearchSpace:
    def __init__(self, bounds: dict[str, tuple[float, float]]):
        self.bounds = {k: tuple(v) for k, v in bounds.items()}
        self.keys = list(self.bounds.keys())
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
        rng = np.random.default_rng(seed)
        result: list[list[float]] = [[0.0] * self.dim for _ in range(n)]
        for j in range(self.dim):
            coords = [(i + rng.random()) / n for i in range(n)]
            rng.shuffle(coords)
            for i in range(n):
                result[i][j] = coords[i]

        points = []
        for row in result:
            scaled = [
                self.bounds[k][0] + val * (self.bounds[k][1] - self.bounds[k][0])
                for val, k in zip(row, self.keys)
            ]
            points.append(self.from_vector(np.array(scaled, dtype=float)))
        return points

    def shrink_around(self, center: dict[str, float], factor: float = 0.95) -> None:
        new_bounds: dict[str, tuple[float, float]] = {}
        for key, (low, high) in self.bounds.items():
            width = high - low
            target_width = width * factor
            center_val = center[key]
            half = target_width / 2.0
            new_low = center_val - half
            new_high = center_val + half
            if new_low < low:
                new_low = low
                new_high = low + target_width
            if new_high > high:
                new_high = high
                new_low = high - target_width
            new_bounds[key] = (new_low, new_high)
        self.bounds = new_bounds


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

    def add(self, x: dict[str, float], y: float) -> None:
        self.X.append(self.space.to_vector(x))
        self.y.append(float(y))
        self.best_y_trace.append(float(np.max(self.y)))

    def fit(self) -> None:
        if len(self.X) >= 2:
            self.gp.fit(np.vstack(self.X), np.array(self.y, dtype=float))

    def ucb(self, x_vec: np.ndarray) -> float:
        mu, sigma = self.gp.predict(x_vec.reshape(1, -1), return_std=True)
        return float(mu[0] + self.beta * sigma[0])

    def propose(self, n_candidates: int = 4096) -> dict[str, float]:
        candidates = self.space.lhs(n_candidates, seed=int(self.rng.integers(0, 1_000_000)))
        best_x = candidates[0]
        best_val = -np.inf
        for x in candidates:
            val = self.ucb(self.space.to_vector(x))
            if val > best_val:
                best_val = val
                best_x = x
        return self.space.clip(best_x)

    def mean_improvement_pct(self) -> float | None:
        if len(self.best_y_trace) < 6:
            return None
        diffs = []
        tail = self.best_y_trace[-6:]
        for prev, cur in zip(tail[:-1], tail[1:]):
            if abs(prev) < 1e-12:
                continue
            diffs.append(100.0 * (cur - prev) / abs(prev))
        return float(np.mean(diffs)) if diffs else 0.0

    def should_stop(self, threshold_pct: float) -> bool:
        score = self.mean_improvement_pct()
        return score is not None and score < threshold_pct


def _extract_y_cost(resp: dict[str, Any]) -> tuple[float, float]:
    y = resp.get("Y", resp.get("yield", resp.get("y")))
    if y is None:
        raise KeyError("Response does not contain yield/Y/y.")
    cost = resp.get("cost", resp.get("C", 0.0))
    return float(y), float(cost)


def _save_plot(fig: plt.Figure, filename: str, output_dir: Path, prefix: str | None = None) -> None:
    if prefix:
        filename = f"{prefix}_{filename}"
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def _plot_scale_history(
    result: RunResult,
    title: str,
    output_dir: Path,
    prefix: str | None = None,
) -> None:
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
    _save_plot(fig, f"{title.replace(' ', '_').lower()}.png", output_dir, prefix)


def _plot_testmode_checks(
    result: RunResult,
    title: str,
    output_dir: Path,
    prefix: str | None = None,
) -> None:
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
    _save_plot(fig, f"{title.replace(' ', '_').lower()}.png", output_dir, prefix)


def run_scale_loop_v2(
    client: BioreactorClient,
    scale: str,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_init: int = 30,
    max_iter: int = 40,
    beta: float = 2.0,
    stop_threshold_pct: float = 0.5,
    test_mode: bool = False,
    pilot_every: int = 5,
    shrink_factor: float = 0.95,
    seed: int = 42,
    plot_out: Path | None = None,
) -> RunResult:
    current_bounds = bounds or DEFAULT_BOUNDS.copy()
    space = SearchSpace(current_bounds)
    bo = BayesianOptimizer(space=space, beta=beta, seed=seed)
    result = RunResult(scale=scale)

    init_points = space.lhs(n_init, seed=seed)
    for i, x in enumerate(init_points, start=1):
        resp = client.run(scale=scale, T=x["T"], pH=x["pH"], F1=x["F1"], F2=x["F2"], F3=x["F3"])
        y, cost = _extract_y_cost(resp)
        bo.add(x, y)
        bo.fit()

        if y > result.best_y:
            result.best_y = y
            result.best_x = x
            result.best_cost = cost

        result.history.append(Observation(x=x, y=y, cost=cost, kind="init", iteration=i))
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
                Observation(x=dict(result.best_x), y=py, cost=pcost, kind="pilot_check", iteration=i)
            )

    for i in range(n_init + 1, max_iter + 1):
        if bo.should_stop(stop_threshold_pct):
            break

        x_next = bo.propose()
        resp = client.run(scale=scale, T=x_next["T"], pH=x_next["pH"], F1=x_next["F1"], F2=x_next["F2"], F3=x_next["F3"])
        y, cost = _extract_y_cost(resp)

        bo.add(x_next, y)
        bo.fit()

        if y > result.best_y:
            result.best_y = y
            result.best_x = x_next
            result.best_cost = cost

        result.history.append(Observation(x=x_next, y=y, cost=cost, kind="bo", iteration=i))

        if result.best_x is not None:
            space.shrink_around(result.best_x, factor=shrink_factor)
            bo.space = space

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
                Observation(x=dict(result.best_x), y=py, cost=pcost, kind="pilot_check", iteration=i)
            )

    result.final_bounds = dict(space.bounds)
    plot_output_dir, plot_output_prefix = _resolve_plot_output(plot_out)
    _plot_scale_history(
        result,
        title=f"{scale} v2 history",
        output_dir=plot_output_dir,
        prefix=plot_output_prefix,
    )
    if test_mode:
        _plot_testmode_checks(
            result,
            title=f"{scale} v2 pilot checks",
            output_dir=plot_output_dir,
            prefix=plot_output_prefix,
        )

    return result


def main(user: str | None = None, password: str | None = None, base_url: str = "https://mlme26biosim.org") -> None:
    parser = argparse.ArgumentParser(description="Bayesian Optimization loop v2 with iterative search-space shrinkage.")
    parser.add_argument("--test-mode", action="store_true", help="Activate pilot analysis mode every 5 iterations.")
    parser.add_argument("--max-iter", type=int, default=40, help="Maximum total iterations including LHS seed.")
    parser.add_argument("--n-init", type=int, default=30, help="Latin Hypercube initial seed points.")
    parser.add_argument("--beta", type=float, default=2.0, help="UCB exploration parameter.")
    parser.add_argument("--stop-threshold-pct", type=float, default=0.5, help="Stop when mean improvement percent falls below threshold.")
    parser.add_argument("--pilot-every", type=int, default=5, help="Perform pilot checks every N iterations in test mode.")
    parser.add_argument("--shrink-factor", type=float, default=0.95, help="Relative search-space shrink factor after each BO iteration.")
    parser.add_argument("--plot-out", type=Path, default=None, help="Optional output directory or filename prefix for generated plots.")
    parser.add_argument("--user", type=str, default=None, help="API user name.")
    parser.add_argument("--password", type=str, default=None, help="API password.")
    parser.add_argument("--base-url", type=str, default=base_url, help="API base URL.")
    args = parser.parse_args()

    client = BioreactorClient(base_url=args.base_url)
    from API_Group8 import USER as DEFAULT_USER, PASSWORD as DEFAULT_PASSWORD
    client.login(args.user or DEFAULT_USER, args.password or DEFAULT_PASSWORD)

    micro_result = run_scale_loop_v2(
        client=client,
        scale="micro",
        n_init=args.n_init,
        max_iter=args.max_iter,
        beta=args.beta,
        stop_threshold_pct=args.stop_threshold_pct,
        test_mode=args.test_mode,
        pilot_every=args.pilot_every,
        shrink_factor=args.shrink_factor,
        plot_out=args.plot_out,
    )

    bench_result = run_scale_loop_v2(
        client=client,
        scale="bench",
        bounds=micro_result.final_bounds,
        n_init=args.n_init,
        max_iter=args.max_iter,
        beta=args.beta,
        stop_threshold_pct=args.stop_threshold_pct,
        test_mode=args.test_mode,
        pilot_every=args.pilot_every,
        shrink_factor=args.shrink_factor,
        plot_out=args.plot_out,
    )

    final_x = bench_result.best_x or micro_result.best_x
    if final_x is None:
        raise RuntimeError("No valid optimum found for the final pilot run.")

    final_resp = client.run(
        scale=FINAL_SCALE,
        T=final_x["T"],
        pH=final_x["pH"],
        F1=final_x["F1"],
        F2=final_x["F2"],
        F3=final_x["F3"],
    )
    final_y, final_cost = _extract_y_cost(final_resp)

    summary = {
        "micro_best_y": micro_result.best_y,
        "bench_best_y": bench_result.best_y,
        "final_pilot_y": final_y,
        "final_pilot_cost": final_cost,
        "micro_final_bounds": micro_result.final_bounds,
        "bench_final_bounds": bench_result.final_bounds,
    }
    print(summary)


if __name__ == "__main__":
    main()
