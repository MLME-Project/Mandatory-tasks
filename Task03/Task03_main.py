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
import warnings
from sklearn.exceptions import ConvergenceWarning
from API_Group8 import BioreactorClient
from data_frame_and_csv_manipulation import appendToCSV, getXyFromCSV, getDataFrameFromCSV, getXyFromDataFrame, INPUT_COLS
from global_constants import T_BOUNDS, pH_BOUNDS, F_BOUNDS



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
    # σ_signal² · exp(-‖x−x'‖²/2ℓ²) + σ_noise²·δ(x,x')
    # 5 length scales for automatic relevance determination (ARD)
    rbf = ConstantKernel() * RBF(length_scale=[1.0]*5, length_scale_bounds=[(1e-5, 3.0)]*5) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=50)
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('gp', gp)
    ])
    return pipe


# acquisition functions
def expectedImprovement(X, pipeline, y_best, xi=0.01):
    """
    EI(X) = (mu(X) - y_best) * Phi(Z) + sigma(X) * phi(Z)                       \\
    Z = (mu(X) - y_best) / sigma(X)                                             \\
    Phi = cumulative distribution function of standard normal distribution      \\
    phi = probability density function of standard normal distribution          \\
    mu(X), sigma(X) = pipeline.predict(X, return_std=True)

    Args:
        X (array): Inputs of shape (n_samles x n_inputs)
        pipeline (Pipeline): Object on which predict(X) is called; includes the gp
        y_best (float): Maximum y which improvements are relative to
        xi (float, optional): Exploration factor. Defaults to 0.01.

    Returns:
        array: Outputs of shape (n_samples)
    """
    mu, sigma = pipeline.predict(X, return_std=True)
    Z = (mu - y_best - xi) / (sigma + 1e-9)
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei


def upperConfidenceBound(X, pipeline, kappa=2.0):
    """
    UCB(X) = mu(X) + kappa * sigma(X)

    Args:
        X (array): Inputs of shape (n_samples x n_inputs)
        pipeline (Pipeline): Object on which predict(X) is called; includes the gp
        kappa (float, optional): Exploration-exploitation trade-off. Defaults to 2.0.

    Returns:
        array: Outputs of shape (n_samples)
    """
    mu, sigma = pipeline.predict(X, return_std=True)
    return mu + kappa * sigma



if __name__ == "__main__":
    # np.random.seed(67)

    # setup experiment
    SCALE = 'pilot'
    ACQ_FUN = 'ei'
    ACQ_FUN_VAR = 0.01
    RUN_ID = '03'
    FILENAME = f'Task03/{SCALE}_data_{ACQ_FUN}({ACQ_FUN_VAR})_{RUN_ID}.csv'
    
    # setup client
    client = BioreactorClient()
    client.login()

    #get initial data
    if not os.path.exists(FILENAME):
        sampleHypercubeCorners(
            scale=SCALE, 
            T_Bounds=T_BOUNDS, 
            pH_Bounds=pH_BOUNDS, 
            F1_Bounds=F_BOUNDS, 
            F2_Bounds=F_BOUNDS, 
            F3_Bounds=F_BOUNDS, 
            client=client, 
            fileName=FILENAME)

    # main training loop
    for i in range(100):
        # get data so far
        df = getDataFrameFromCSV(fileName=FILENAME)
        cummulativeCost = np.sum(df["cost_eur"])
        X, y = getXyFromDataFrame(df)
        y_best = np.max(y)
        print(f"\niteration {i} - so far: best Y = {y_best:.3f} | cost = {cummulativeCost} €")

        # fit gp
        pipe = setupPipeline()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.fit(X, y)
        for warning in w:
            if isinstance(warning.message, ConvergenceWarning):
                message = str(warning).split(' ')
                dimension = int(message[8])
                variable = INPUT_COLS[dimension]
                upperLower = message[17]
                print("length_scale of", variable, "is at", upperLower, "bound")

        # sample acquisition function
        n_test = 100000
        X_test = np.column_stack([
            np.random.uniform(*T_BOUNDS, n_test),
            np.random.uniform(*pH_BOUNDS, n_test),
            np.random.uniform(*F_BOUNDS, n_test),
            np.random.uniform(*F_BOUNDS, n_test),
            np.random.uniform(*F_BOUNDS, n_test),
        ])
        if ACQ_FUN == 'ei':
            acq_test = expectedImprovement(
                X=X_test, 
                pipeline=pipe, 
                y_best=y_best,
                xi=ACQ_FUN_VAR)
        elif ACQ_FUN == 'ucb':
            acq_test = upperConfidenceBound(
                X=X_test, 
                pipeline=pipe,
                kappa=ACQ_FUN_VAR)
        else:
            raise NotImplementedError()
        acq_max = np.max(acq_test)
        X_opt = X_test[np.argmax(acq_test)]

        # extract variables
        T, pH, F1, F2, F3 = X_opt

        # prediction on the optimum
        y_mu_max, y_sigma_max = pipe.predict(
            np.atleast_2d(X_opt), 
            return_std=True)
        print(f"next point: T={T:.1f} pH={pH:.2f} F1={F1:.2f} F2={F2:.2f} F3={F3:.2f} | acq={acq_max:.3e} | GP: mu={y_mu_max[0]:.3f} sigma={y_sigma_max[0]:.3f}")
        
        # run new experiment
        result = client.run(SCALE, T, pH, F1, F2, F3)
        appendToCSV(FILENAME, SCALE, T, pH, F1, F2, F3, result)
        print(f"measurement: Y = {result['Y']:.3f}")

    # evaluate best point on pilot scale
    X, y = getXyFromCSV(FILENAME)
    X_best = X[np.argmax(y)]
    T, pH, F1, F2, F3 = X_best
    result = client.run('pilot', T, pH, F1, F2, F3)
    print(f"Best point on pilot scale: T={T:.1f} pH={pH:.2f} F1={F1:.2f} F2={F2:.2f} F3={F3:.2f} | Y = {result['Y']:.3f}")
