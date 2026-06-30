import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from matplotlib import pyplot as plt
import time
from Task03_main import setupPipeline, expectedImprovement, T_BOUNDS, pH_BOUNDS, F_BOUNDS
from data_frame_and_csv_manipulation import getXyFromCSV

# fit gp
print("fitting", end=' ')
t0 = time.time()
X_train, y_train = getXyFromCSV('Task03/micro_data.csv')
pipe = setupPipeline()
pipe.fit(X_train, y_train)

# sample acquisition function
t1 = time.time()
print(t1-t0, "s")
print("sampling", end=' ')
n_test = 100000
X_test = np.column_stack([
    np.random.uniform(*T_BOUNDS, n_test),
    np.random.uniform(*pH_BOUNDS, n_test),
    np.random.uniform(*F_BOUNDS, n_test),
    np.random.uniform(*F_BOUNDS, n_test),
    np.random.uniform(*F_BOUNDS, n_test),
])
y_best = np.max(y_train)
acq_test = expectedImprovement(
    X_candidates=X_test, 
    pipeline=pipe, 
    y_best=y_best,
    xi=0.1)

# plotting
t2 = time.time()
print(t2-t1, "s")
print("plotting")
plt.scatter(X_test[:,0], acq_test, s=1)
plt.xlabel('T')
plt.ylabel('EI')
plt.show()