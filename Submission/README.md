# Submission

Bayesian optimization (`main.py`) and a Sobol-sequence baseline (`Sobol.py`) for tuning bioreactor recipes.

## Setup

Requires Python 3.12.

## Files

- `main.py` — Bayesian optimization loop: fits a Gaussian process (scikit-learn) over the recipe space, picks the next point via an acquisition function (EI/UCB/PI), and adaptively chooses which scale to run it at. Results are appended to a run-specific CSV.
- `Sobol.py` — Non-adaptive baseline that samples a fixed Sobol sequence across micro/bench/pilot, then validates the best recipe on pilot scale. Saves `sobol_run_log.csv` and comparison plots (`sobol_plot*.png`) to this directory.
- `API_Group8.py` — REST client (`BioreactorClient`) for the bioreactor simulation API, including login/CSRF handling and retry/backoff logic.
- `data_frame_and_csv_manipulation.py` — Helpers for reading/writing the experiment CSV and converting between DataFrame and (X, y) arrays.
- `global_constants.py` — Shared bounds, per-scale costs, and noise variances.

## Running

```bash
python main.py
python Sobol.py
```

Both scripts log in to the API on startup and will incur cost against the group's budget — check the constants at the top of each file (e.g. `BUDGET`, `N_MICRO`/`N_BENCH`/`N_PILOT`) before running.

## Beat-The-Felix subfolder

`Beat-The-Felix/` is the standalone submission for the "Beat the Felix" challenge and uses a **different, separate environment** from the rest of this repo — it depends on `torch`, `botorch`, and `gpytorch` instead of scikit-learn, so don't install it into the same venv as above.

## Files

- `main.py` — entry point required by the challenge: runs `bo_loopv6.py` 10 times as independent subprocesses (writing each run's CSV to `output_v2/`) and prints the mean performance across trials.
- `bo_loopv6.py` — the actual BoTorch/GPyTorch-based Bayesian optimization loop for a single trial.
- `API_Group8.py` — a local copy of the API client (kept in sync manually with the top-level `API_Group8.py`).
- `requirements.txt` — dependencies for this subfolder only.
