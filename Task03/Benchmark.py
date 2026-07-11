import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from API_Group8 import BioreactorClient
from data_frame_and_csv_manipulation import getDataFrameFromCSV, getXyFromDataFrame, INPUT_COLS
from global_constants import SCALE_MULTIPLIERS


CSV_FILE = 'Task03/bench_data_ei(0.01)_01.csv'
N_EVALS = 3


df = getDataFrameFromCSV(CSV_FILE)
for col in ['projected_pilot_Y','measured_pilot_Y']:
    df[col] = np.nan

X_all, y_all = getXyFromDataFrame(df)
n_total = len(X_all)

client = BioreactorClient()
client.login()

for row_id, row in df.iterrows():
    Y_projected = row['Y'] * SCALE_MULTIPLIERS[row['scale']]
    df.loc[row_id, 'projected_pilot_Y'] = Y_projected

    T, pH, F1, F2, F3 = row[INPUT_COLS]
    Y_iter = [client.run('pilot', T, pH, F1, F2, F3)['Y'] for _ in range(N_EVALS)]
    Y_mean = float(np.mean(Y_iter))
    df.loc[row_id, 'measured_pilot_Y'] = Y_mean
    print(f"row {row_id} | X = [T={T:.3f} pH={pH:.3f} F1={F1:.3f} F2={F2:.3f} F3={F3:.3f}] | measured Y = {Y_mean:.3f} ± {np.std(Y_iter):.3f}")

df.to_csv(CSV_FILE)

# plot

height_ratios = [2.2] + [1] * 7
fig, axes = plt.subplots(8, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': height_ratios})

axes[0].plot(df['measured_pilot_Y'].values, color='tomato', label='measured pilot Y')
axes[0].plot(df['projected_pilot_Y'].values, color='steelblue', label='projected pilot Y')
axes[0].set_ylabel('Y')
axes[0].set_title('Evaluation per iteration')
axes[0].legend()

axes[1].plot(df['cost_eur'].cumsum().values, color='darkslategray')
axes[1].set_ylabel('cumulative\ncost')

scales = ['micro', 'bench', 'pilot']
for i, scale in enumerate(scales):
    mask = (df['scale'] == scale).values
    axes[2].scatter(np.where(mask)[0], [i] * mask.sum(), color='darkslategray', label=scale, marker='.')
axes[2].set_yticks(range(3))
axes[2].set_yticklabels(scales)
axes[2].set_ylim(-0.5, 2.5)
axes[2].set_ylabel('scale')

for ax, col in zip(axes[3:], INPUT_COLS):
    ax.plot(df[col].values, color='darkslategray')
    ax.set_ylabel(col)

axes[-1].set_xlabel('iteration')
fig.tight_layout()
fig.savefig(CSV_FILE.removesuffix('.csv') + '_benchmark.pdf')
plt.show()
