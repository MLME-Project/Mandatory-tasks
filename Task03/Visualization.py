import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from matplotlib import pyplot as plt
from Task03_main import setupPipeline, T_BOUNDS, pH_BOUNDS, F_BOUNDS
from data_frame_and_csv_manipulation import getDataFrameFromCSV, getXyFromDataFrame, INPUT_COLS


MAX_GP_POINTS = 300
CSV_FILE = 'mlme26_group08_2026-06-29T10-47-17.csv'
# CSV_FILE = 'Task03/micro_data_ei(0.01)_05.csv'

df_all = getDataFrameFromCSV(CSV_FILE)
scales = df_all['scale'].unique()

n = 3000

for scale in scales:
    df = df_all[df_all['scale'] == scale]
    X_measured, y_measured = getXyFromDataFrame(df)
    X_fit, y_fit = X_measured, y_measured

    if len(X_fit) > MAX_GP_POINTS:
        n_best = MAX_GP_POINTS // 3
        n_rand = MAX_GP_POINTS - n_best
        best_idx = np.argsort(y_fit)[-n_best:]
        remaining = np.setdiff1d(np.arange(len(X_fit)), best_idx)
        rand_idx = np.random.choice(remaining, n_rand, replace=False)
        idx = np.concatenate([best_idx, rand_idx])
        X_fit, y_fit = X_fit[idx], y_fit[idx]

    pipe = setupPipeline()
    pipe.named_steps['gp'].n_restarts_optimizer = 10
    pipe.fit(X_fit, y_fit)

    # sample random points across full input space
    X = np.column_stack([
        np.random.uniform(*T_BOUNDS, n),
        np.random.uniform(*pH_BOUNDS, n),
        np.random.uniform(*F_BOUNDS, n),
        np.random.uniform(*F_BOUNDS, n),
        np.random.uniform(*F_BOUNDS, n),
    ])
    mu, sigma = pipe.predict(X, return_std=True)

    # marker size encodes sigma; color (light = high sigma) does too
    sigma_norm = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-12)
    sizes = 3 + 60 * sigma_norm

    # draw largest circles first so smaller ones end up in front
    draw_order = np.argsort(-sizes)
    X_draw, mu_draw, sigma_draw, sizes_draw = X[draw_order], mu[draw_order], sigma[draw_order], sizes[draw_order]

    # figure: top row = GP predictions (size/color encode sigma), bottom row = measured data
    ncols = len(INPUT_COLS)
    side = 2.8
    fig, axes = plt.subplots(2, ncols, figsize=(side * ncols + 1, side * 2 + 1.5), sharex='col', sharey='row')
    fig.suptitle(f'[{scale}]')

    for col, col_name in enumerate(INPUT_COLS):
        col_idx = INPUT_COLS.index(col_name)
        sc = axes[0, col].scatter(
            X_draw[:, col_idx], mu_draw,
            c=sigma_draw, cmap='Blues_r',
            s=sizes_draw,
            edgecolors='none',
        )
        axes[0, col].set_xlabel(col_name)
        axes[0, col].tick_params(labelbottom=True)
        axes[1, col].scatter(
            X_measured[:, col_idx], y_measured,
            s=10, color='tab:orange', edgecolors='none',
        )
        axes[1, col].set_xlabel(col_name)
    axes[0, 0].set_ylabel('mu')
    axes[1, 0].set_ylabel('Y')

    fig.tight_layout(rect=[0, 0, 0.9, 0.94])
    fig.subplots_adjust(hspace=0.5)

    pos_top_left = axes[0, 0].get_position()
    pos_top_right = axes[0, -1].get_position()
    pos_bottom_left = axes[1, 0].get_position()
    center_x = (pos_top_left.x0 + pos_top_right.x1) / 2

    fig.text(center_x, pos_top_left.y1 + 0.02, 'GP predictions (size/color = sigma)',
              ha='center', fontsize=11)
    fig.text(center_x, pos_bottom_left.y1 + 0.02, 'Measured data',
              ha='center', fontsize=11)

    cbar_ax = fig.add_axes([pos_top_right.x1 + 0.02, pos_top_left.y0, 0.02, pos_top_left.height])
    fig.colorbar(sc, cax=cbar_ax, label='sigma')

plt.show()
