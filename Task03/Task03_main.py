import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from API_Group8 import BioreactorClient
from data_frame_and_csv_manipulation import appendToCSV, getXyFromCSV, getDataFrameFromCSV, getXyFromDataFrame


# setup bounds
T_BOUNDS = [20, 60]
pH_BOUNDS = [3, 9.5]
F_BOUNDS = [0, 2]

# setup csv
FILENAME = 'Task03/micro_data.csv'

# setup client
client = BioreactorClient()
client.login()



def sampleHypercubeCorners(scale, T_Bounds, pH_Bounds, F1_Bounds, F2_Bounds, F3_Bounds, client, fileName):
    for T in T_Bounds:
        for pH in pH_Bounds:
            for F1 in F1_Bounds:
                for F2 in F2_Bounds:
                    for F3 in F3_Bounds:
                        print(f"running experiment [{scale}, {T}, {pH}, {F1}, {F2}, {F3}]")
                        result = client.run(scale, T, pH, F1, F2, F3)
                        appendToCSV(fileName, scale, T, pH, F1, F2, F3, result)
N_INITIAL_SAMPLES = 2**5


def setupPipeline():
    rbf = ConstantKernel() * RBF() + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=50)
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('gp', gp)
    ])
    return pipe


# acquisition function
def expectedImprovement(X_candidates, pipeline, y_best, xi=0.01):
    mu, sigma = pipeline.predict(X_candidates, return_std=True)
    Z = (mu - y_best - xi) / (sigma + 1e-9)
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei



# seed rng
np.random.seed(67)

# get initial data on micro scale
# sampleHypercubeCorners(
#     scale='micro', 
#     T_Bounds=T_BOUNDS, 
#     pH_Bounds=pH_BOUNDS, 
#     F1_Bounds=F_BOUNDS, 
#     F2_Bounds=F_BOUNDS, 
#     F3_Bounds=F_BOUNDS, 
#     client=client, 
#     fileName=FILENAME)

# main training loop on micro scale
for i in range(100):
    # get data so far
    df = getDataFrameFromCSV(fileName=FILENAME)
    cummulativeCost = np.sum(df["cost_eur"])
    X, y = getXyFromDataFrame(df)
    y_best = np.max(y)

    # fit gp
    pipe = setupPipeline()
    pipe.fit(X, y)

    # sample acquisition function
    n_test = 100000
    X_test = np.column_stack([
        np.random.uniform(*T_BOUNDS, n_test),
        np.random.uniform(*pH_BOUNDS, n_test),
        np.random.uniform(*F_BOUNDS, n_test),
        np.random.uniform(*F_BOUNDS, n_test),
        np.random.uniform(*F_BOUNDS, n_test),
    ])
    acq_test = expectedImprovement(
        X_candidates=X_test, 
        pipeline=pipe, 
        y_best=y_best)
    acq_max = np.max(acq_test)
    X_opt = X_test[np.argmax(acq_test)]

    # extract variables
    scale = "micro"
    T, pH, F1, F2, F3 = X_opt

    # prediction on the optimum
    y_mu_max, y_sigma_max = pipe.predict(
        np.atleast_2d(X_opt), 
        return_std=True)
    
    # run new experiment
    result = client.run(scale, T, pH, F1, F2, F3)
    appendToCSV(FILENAME, scale, T, pH, F1, F2, F3, result)

    # print progress
    print(f"best Y = {y_best:.3f} | cost so far = {cummulativeCost} €")
    print(f"next point: T={T:.1f} pH={pH:.2f} F1={F1:.2f} F2={F2:.2f} F3={F3:.2f} | acq={acq_max:.3e} | GP: mu={y_mu_max[0]:.3f} sigma={y_sigma_max[0]:.3f}")
    print(f"measurement: Y = {result['Y']:.3f}\n")
    