import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from matplotlib import pyplot as plt
from Task03_main import setupPipeline, expectedImprovement, T_BOUNDS, pH_BOUNDS, F_BOUNDS
from data_frame_and_csv_manipulation import getDataFrameFromCSV, getXyFromDataFrame, INPUT_COLS


MAX_GP_POINTS = 300
CSV_FILE = 'mlme26_group08_2026-06-29T10-47-17.csv'

df_all = getDataFrameFromCSV(CSV_FILE)
scales = df_all['scale'].unique()

n = 3000

for scale in scales:
    df = df_all[df_all['scale'] == scale]
    X_train, y_train = getXyFromDataFrame(df)

    if len(X_train) > MAX_GP_POINTS:
        n_best = MAX_GP_POINTS // 3
        n_rand = MAX_GP_POINTS - n_best
        best_idx = np.argsort(y_train)[-n_best:]
        remaining = np.setdiff1d(np.arange(len(X_train)), best_idx)
        rand_idx = np.random.choice(remaining, n_rand, replace=False)
        idx = np.concatenate([best_idx, rand_idx])
        X_train, y_train = X_train[idx], y_train[idx]

    pipe = setupPipeline()
    pipe.named_steps['gp'].n_restarts_optimizer = 10
    pipe.fit(X_train, y_train)

    # sample random points across full input space
    X = np.column_stack([
        np.random.uniform(*T_BOUNDS, n),
        np.random.uniform(*pH_BOUNDS, n),
        np.random.uniform(*F_BOUNDS, n),
        np.random.uniform(*F_BOUNDS, n),
        np.random.uniform(*F_BOUNDS, n),
    ])
    y_best = np.max(y_train)
    ei = expectedImprovement(X=X, pipeline=pipe, y_best=y_best, xi=0.1)
    mu, sigma = pipe.predict(X, return_std=True)

    # sample with T and pH fixed
    X_fixed_T_ph = np.column_stack([
        np.full(n, fill_value=36),
        np.full(n, fill_value=8),
        np.random.uniform(*F_BOUNDS, n),
        np.random.uniform(*F_BOUNDS, n),
        np.random.uniform(*F_BOUNDS, n),
    ])
    ei_fixed_T_ph = expectedImprovement(X=X_fixed_T_ph, pipeline=pipe, y_best=y_best, xi=0.1)
    mu_fixed_T_ph, sigma_fixed_T_ph = pipe.predict(X_fixed_T_ph, return_std=True)

    # figure 1: all inputs × EI / mu / sigma
    fig1, axes1 = plt.subplots(len(INPUT_COLS), 3, figsize=(9, 1.4 * len(INPUT_COLS)))
    fig1.suptitle(f'GP predictions over all inputs  [{scale}]')

    for row, col_name in enumerate(INPUT_COLS):
        col_idx = INPUT_COLS.index(col_name)
        axes1[row, 0].scatter(X[:, col_idx], ei, s=0.5)
        axes1[row, 0].set_xlabel(col_name)
        axes1[row, 0].set_ylabel('EI')
        axes1[row, 1].scatter(X[:, col_idx], mu, s=0.5)
        axes1[row, 1].set_xlabel(col_name)
        axes1[row, 1].set_ylabel('mu')
        axes1[row, 2].scatter(X[:, col_idx], sigma, s=0.5)
        axes1[row, 2].set_xlabel(col_name)
        axes1[row, 2].set_ylabel('sigma')
        if row == 0:
            axes1[row, 0].set_title('EI')
            axes1[row, 1].set_title('mu')
            axes1[row, 2].set_title('sigma')

    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # figure 2: feed rate inputs only, T=36 pH=8
    inputs_feed = INPUT_COLS[2:]
    fig2, axes2 = plt.subplots(len(inputs_feed), 3, figsize=(9, 2 * len(inputs_feed)))
    fig2.suptitle(f'GP predictions over feed rate inputs (T=36°, pH=8)  [{scale}]')

    for row, col_name in enumerate(inputs_feed):
        col_idx = INPUT_COLS.index(col_name)
        axes2[row, 0].scatter(X_fixed_T_ph[:, col_idx], ei_fixed_T_ph, s=0.5)
        axes2[row, 0].set_xlabel(col_name)
        axes2[row, 0].set_ylabel('EI')
        axes2[row, 1].scatter(X_fixed_T_ph[:, col_idx], mu_fixed_T_ph, s=0.5)
        axes2[row, 1].set_xlabel(col_name)
        axes2[row, 1].set_ylabel('mu')
        axes2[row, 2].scatter(X_fixed_T_ph[:, col_idx], sigma_fixed_T_ph, s=0.5)
        axes2[row, 2].set_xlabel(col_name)
        axes2[row, 2].set_ylabel('sigma')
        if row == 0:
            axes2[row, 0].set_title('EI')
            axes2[row, 1].set_title('mu')
            axes2[row, 2].set_title('sigma')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
