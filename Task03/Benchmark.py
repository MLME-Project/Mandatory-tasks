import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from API_Group8 import BioreactorClient
from data_frame_and_csv_manipulation import getDataFrameFromCSV, getXyFromDataFrame, projectToPilotScale, INPUT_COLS


CSV_FILE = 'Task03/data_ucb_param02_patience30_lhc20_final_06.csv'
N_EVALS = 3
INCLUDE_INPUTS = False
EVALUATE_PILOT = False


df = getDataFrameFromCSV(CSV_FILE)

X_all, y_all = getXyFromDataFrame(df)
n_total = len(X_all)

df['projected_pilot_Y'] = projectToPilotScale(df['Y'].values, df['scale'].values)

# measured pilot Y - comment out this block to skip the expensive client calls while iterating on the rest
if EVALUATE_PILOT:
    client = BioreactorClient()
    client.login()

    df['measured_pilot_Y'] = np.nan
    for i, (row_id, row) in enumerate(df.iterrows()):
        T, pH, F1, F2, F3 = row[INPUT_COLS]
        Y_iter = [client.run('pilot', T, pH, F1, F2, F3)['Y'] for _ in range(N_EVALS)]
        Y_mean = float(np.mean(Y_iter))
        df.loc[row_id, 'measured_pilot_Y'] = Y_mean
        print(f"iteration {i+1}/{n_total} | X = [T={T:.3f} pH={pH:.3f} F1={F1:.3f} F2={F2:.3f} F3={F3:.3f}] | measured Y = {Y_mean:.3f} ± {np.std(Y_iter):.3f}")

# recipe that achieved the best projected pilot Y so far (running argmax)
Y_projected = df['projected_pilot_Y'].to_numpy()
running_best_idx = np.zeros(len(Y_projected), dtype=int)
best_i = 0
for i in range(len(Y_projected)):
    if Y_projected[i] > Y_projected[best_i]:
        best_i = i
    running_best_idx[i] = best_i

df['best_projected_pilot_Y'] = Y_projected[running_best_idx]
for col in INPUT_COLS:
    df[f'best_{col}'] = df[col].to_numpy()[running_best_idx]

df.to_csv(CSV_FILE)

# plot

height_ratios = [2.2] + [1] * (7 if INCLUDE_INPUTS else 2)
fig, axes = plt.subplots((8 if INCLUDE_INPUTS else 3), 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': height_ratios})

# axes[0].plot(df['measured_pilot_Y'].values, color='tomato', linestyle='--', linewidth=1, label='measured pilot Y')
axes[0].plot(df['projected_pilot_Y'].values, color='tomato', linestyle='-', linewidth=1, label='projected pilot Y')
axes[0].plot(df['best_projected_pilot_Y'].values, color='steelblue', linestyle='-', linewidth=2, label='best projected pilot Y')
axes[0].set_ylabel('Y [g/L]')
axes[0].set_title('Evaluation per iteration')
axes[0].legend()

axes[1].plot(df['cost_eur'].cumsum().values, color='darkslategray')
axes[1].set_ylabel('cumulative\ncost [€]')

scales = ['micro', 'bench', 'pilot']
for i, scale in enumerate(scales):
    mask = (df['scale'] == scale).values
    axes[2].scatter(np.where(mask)[0], [i] * mask.sum(), color='darkslategray', label=scale, marker='.')
axes[2].set_yticks(range(3))
axes[2].set_yticklabels(scales)
axes[2].set_ylim(-0.5, 2.5)
axes[2].set_ylabel('scale')

if INCLUDE_INPUTS:
    for ax, col in zip(axes[3:], INPUT_COLS):
        ax.plot(df[col].values, color='darkslategray', label=col)
        ax.plot(df[f'best_{col}'].values, color='darkslategray', linestyle='--', linewidth=1, label='cumulative best')
        ax.set_ylabel(col)

axes[-1].set_xlabel('iteration')
fig.tight_layout()
fig.savefig(CSV_FILE.removesuffix('.csv') + '_benchmark.pdf')
plt.show()
