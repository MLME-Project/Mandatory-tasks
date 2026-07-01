import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from matplotlib import pyplot as plt
from API_Group8 import BioreactorClient
from Task03_main import setupPipeline, T_BOUNDS, pH_BOUNDS, F_BOUNDS, N_INITIAL_SAMPLES
from data_frame_and_csv_manipulation import getXyFromCSV


CSV_FILE = 'Task03/micro_data_ei(0.01)_05.csv'
N_EVALS = 3
SCALE = 'pilot'

X_all, y_all = getXyFromCSV(CSV_FILE)
n_total = len(X_all)

client = BioreactorClient()
client.login()

n_search = 100000
Y_measured = []

for n_train in range(N_INITIAL_SAMPLES, n_total + 1):
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_best = X_train[np.argmax(y_train)]

    T, pH, F1, F2, F3 = X_best

    iteration = n_train - N_INITIAL_SAMPLES
    Y_iter = []
    for e in range(N_EVALS):
        result = client.run(SCALE, T, pH, F1, F2, F3)
        Y_iter.append(result['Y'])
    Y_measured.append(Y_iter)
    print(f"iteration {iteration:3d} | X_best = [T={T:.1f} pH={pH:.2f} F1={F1:.2f} F2={F2:.2f} F3={F3:.2f}] | measured Y = {np.mean(Y_iter):.3f} ± {np.std(Y_iter):.3f}")

# plot
iterations = np.arange(len(Y_measured))
Y_means = np.array([np.mean(y) for y in Y_measured])

fig, ax = plt.subplots(figsize=(8, 4))
for i, Y_iter in enumerate(Y_measured):
    ax.scatter(np.full(len(Y_iter), i), Y_iter, color='steelblue', alpha=0.6, s=20, zorder=3)
ax.plot(iterations, np.maximum.accumulate(Y_means), color='tomato', linewidth=1.5, label='best mean so far')
ax.set_xlabel('BO iteration')
ax.set_ylabel(f'Y  [{SCALE}]')
ax.set_title('Evaluation at best measurement per BO iteration')
ax.legend()
fig.tight_layout()
fig.savefig(CSV_FILE.removesuffix('.csv') + f'_{SCALE}_benchmark.pdf')
plt.show()
