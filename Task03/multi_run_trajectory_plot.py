import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import glob
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from data_frame_and_csv_manipulation import getDataFrameFromCSV, projectToPilotScale

# Picks up every seeded run of the finalized BO design, e.g.
# "data_ucb_param02_patience30_lhc20_final_06.csv", but not the
# "..._not_final_.." trial runs used while tuning the algorithm.
TASK03_DIR = os.path.dirname(__file__)
RUN_FILES = sorted(
    f for f in glob.glob(os.path.join(TASK03_DIR, '*_final_*.csv'))
    if re.search(r'_final_\d+\.csv$', f) and '_not_final_' not in f
)

BLUE = '#2a78d6'      # mean +- std across all runs
ORANGE = '#eb6834'     # single highlighted (best) run - far-apart hue, plus
                       # dashed linestyle + markers so it doesn't rely on color alone


def run_trajectories(df):
    """Per-iteration running-best projected-pilot-Y and cumulative cost for one run."""
    projected_y = projectToPilotScale(df['Y'].values, df['scale'].values)
    running_best = np.maximum.accumulate(projected_y)
    cum_cost = df['cost_eur'].values.cumsum()
    return running_best, cum_cost


runs = []
summary_rows = []
for f in RUN_FILES:
    df = getDataFrameFromCSV(f)
    running_best, cum_cost = run_trajectories(df)
    runs.append({'file': os.path.basename(f), 'best': running_best, 'cost': cum_cost})

    # every run ends on an actual pilot-scale evaluation, so the last row's Y
    # is the run's real (non-projected) final result - use this to compare
    # runs, not the running-best *projected* Y (which can be a noisy inflated
    # micro-scale point, see run_trajectories/best_matrix below)
    final_evaluation_Y = df['Y'].values[-1]
    pilot_mask = (df['scale'] == 'pilot').values
    best_pilot_titer = df['Y'].values[pilot_mask].max() if pilot_mask.any() else np.nan
    summary_rows.append({
        'run': os.path.basename(f),
        'n_iterations': len(df),
        'final_evaluation_Y': final_evaluation_Y,
        'best_pilot_titer': best_pilot_titer,
        'total_cost_eur': cum_cost[-1],
    })

n_runs = len(runs)
max_len = max(len(r['best']) for r in runs)

# Runs that stop early (patience-triggered) have no more spending and no more
# improvement afterwards, so their trajectories are held flat (forward-filled)
# out to the longest run - this is what lets every run contribute a value at
# every iteration on the shared x-axis below.
best_matrix = np.full((n_runs, max_len), np.nan)
cost_matrix = np.full((n_runs, max_len), np.nan)
for i, r in enumerate(runs):
    n = len(r['best'])
    best_matrix[i, :n] = r['best']
    best_matrix[i, n:] = r['best'][-1]
    cost_matrix[i, :n] = r['cost']
    cost_matrix[i, n:] = r['cost'][-1]

best_mean, best_std = best_matrix.mean(axis=0), best_matrix.std(axis=0)
cost_mean, cost_std = cost_matrix.mean(axis=0), cost_matrix.std(axis=0)

# hardcoded to run 6
best_run_idx = next(i for i, r in enumerate(runs) if re.search(r'final_06\.csv$', r['file']))
best_run_label = re.search(r'final_(\d+)', runs[best_run_idx]['file']).group(0)
iterations = np.arange(1, max_len + 1)

# --- summary table (per-run final evaluation & total cost) ---
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(TASK03_DIR, 'final_runs_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(summary_df.sort_values('final_evaluation_Y', ascending=False).to_string(index=False))
print(f"\nmean final evaluation Y = {summary_df['final_evaluation_Y'].mean():.3f} "
      f"+- {summary_df['final_evaluation_Y'].std():.3f}")
print(f"mean total cost         = {summary_df['total_cost_eur'].mean():.1f} "
      f"+- {summary_df['total_cost_eur'].std():.1f}")

# --- plot ---
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axes[0].plot(iterations, best_mean, color=BLUE, linewidth=2, label='mean best projected pilot Y')
axes[0].fill_between(iterations, best_mean - best_std, best_mean + best_std,
                      color=BLUE, alpha=0.2, label='+- 1 std across runs')
axes[0].plot(iterations, best_matrix[best_run_idx], color=ORANGE, linestyle='--',
             marker='o', markevery=max(1, max_len // 20), markersize=4,
             label=f'best run')
axes[0].set_ylabel('best projected\npilot Y [g/L]')
axes[0].set_title(f'Bayesian optimization trajectory across {n_runs} seeded runs (mean ± std)')
axes[0].legend(fontsize=8)

axes[1].plot(iterations, cost_mean, color=BLUE, linewidth=2, label='mean cumulative cost')
axes[1].fill_between(iterations, cost_mean - cost_std, cost_mean + cost_std,
                      color=BLUE, alpha=0.2, label='+- 1 std across runs')
axes[1].plot(iterations, cost_matrix[best_run_idx], color=ORANGE, linestyle='--',
             marker='o', markevery=max(1, max_len // 20), markersize=4,
             label='best run')
axes[1].set_ylabel('cumulative\ncost [€]')
axes[1].set_xlabel('iteration')
axes[1].legend(fontsize=8)

fig.tight_layout()
fig_path = os.path.join(TASK03_DIR, 'multi_run_trajectory.pdf')
fig.savefig(fig_path)
print(f"\nsaved plot to {fig_path}")
plt.show()
