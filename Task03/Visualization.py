import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from matplotlib import pyplot as plt
from Task03_main import setupPipeline, expectedImprovement, T_BOUNDS, pH_BOUNDS, F_BOUNDS
from data_frame_and_csv_manipulation import getXyFromCSV, INPUT_COLS



# fit gp
X_train, y_train = getXyFromCSV('Task03/pilot_data.csv')
pipe = setupPipeline()
pipe.fit(X_train, y_train)


# sample gaussian process and acquisition function
n = 10000
X = np.column_stack([
    np.random.uniform(*T_BOUNDS, n),
    np.random.uniform(*pH_BOUNDS, n),
    np.random.uniform(*F_BOUNDS, n),
    np.random.uniform(*F_BOUNDS, n),
    np.random.uniform(*F_BOUNDS, n),
])
y_best = np.max(y_train)
ei = expectedImprovement(
    X=X, 
    pipeline=pipe, 
    y_best=y_best,
    xi=0.1)
mu, sigma = pipe.predict(X, return_std=True)

X_fixed_T_ph = np.column_stack([
    np.full((n), fill_value=36),
    np.full((n), fill_value=8),
    np.random.uniform(*F_BOUNDS, n),
    np.random.uniform(*F_BOUNDS, n),
    np.random.uniform(*F_BOUNDS, n),
])
ei_fixed_T_ph = expectedImprovement(
    X=X_fixed_T_ph, 
    pipeline=pipe, 
    y_best=y_best,
    xi=0.1)
mu_fixed_T_ph, sigma_fixed_T_ph = pipe.predict(X_fixed_T_ph, return_std=True)

# plotting — block 1: all inputs (rows) × EI / mu / sigma (cols)
fig1, axes1 = plt.subplots(len(INPUT_COLS), 3, figsize=(9, 1.4 * len(INPUT_COLS)))
fig1.suptitle('GP predictions over all inputs')

for row, inputOfInterest in enumerate(INPUT_COLS):
    col_idx = INPUT_COLS.index(inputOfInterest)

    axes1[row, 0].scatter(X[:, col_idx], ei, s=0.5)
    axes1[row, 0].set_xlabel(inputOfInterest)
    axes1[row, 0].set_ylabel('EI')

    axes1[row, 1].scatter(X[:, col_idx], mu, s=0.5)
    axes1[row, 1].set_xlabel(inputOfInterest)
    axes1[row, 1].set_ylabel('mu')

    axes1[row, 2].scatter(X[:, col_idx], sigma, s=0.5)
    axes1[row, 2].set_xlabel(inputOfInterest)
    axes1[row, 2].set_ylabel('sigma')

    if row == 0:
        axes1[row, 0].set_title('EI')
        axes1[row, 1].set_title('mu')
        axes1[row, 2].set_title('sigma')

fig1.tight_layout(rect=[0, 0, 1, 0.96])

# block 2: inputs fixed at T=36°, pH=8
inputs_block2 = INPUT_COLS[2:]
fig2, axes2 = plt.subplots(len(inputs_block2), 3, figsize=(9, 2 * len(inputs_block2)))
fig2.suptitle('GP predictions over feed rate inputs (T=36°, pH=8)')

for row, inputOfInterest in enumerate(inputs_block2):
    col_idx = INPUT_COLS.index(inputOfInterest)

    axes2[row, 0].scatter(X_fixed_T_ph[:, col_idx], ei_fixed_T_ph, s=0.5)
    axes2[row, 0].set_xlabel(inputOfInterest)
    axes2[row, 0].set_ylabel('EI')

    axes2[row, 1].scatter(X_fixed_T_ph[:, col_idx], mu_fixed_T_ph, s=0.5)
    axes2[row, 1].set_xlabel(inputOfInterest)
    axes2[row, 1].set_ylabel('mu')

    axes2[row, 2].scatter(X_fixed_T_ph[:, col_idx], sigma_fixed_T_ph, s=0.5)
    axes2[row, 2].set_xlabel(inputOfInterest)
    axes2[row, 2].set_ylabel('sigma')

    if row == 0:
        axes2[row, 0].set_title('EI')
        axes2[row, 1].set_title('mu')
        axes2[row, 2].set_title('sigma')

fig2.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()