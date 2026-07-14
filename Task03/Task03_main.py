import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import argparse
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
from sklearn.exceptions import ConvergenceWarning
from API_Group8 import BioreactorClient
from data_frame_and_csv_manipulation import appendToCSV, getXyFromCSV, getDataFrameFromCSV, getXyFromDataFrame, getVarFromDataFrame, INPUT_COLS, projectToPilotScale
from global_constants import T_BOUNDS, pH_BOUNDS, F_BOUNDS, COST, SCALED_STDS



def sampleHypercubeCorners(scale, T_Bounds, pH_Bounds, F1_Bounds, F2_Bounds, F3_Bounds, client, fileName):
    for T in T_Bounds:
        for pH in pH_Bounds:
            for F1 in F1_Bounds:
                for F2 in F2_Bounds:
                    for F3 in F3_Bounds:
                        print(f"running experiment [{scale}, {T}, {pH}, {F1}, {F2}, {F3}]")
                        result = client.run(scale, T, pH, F1, F2, F3)
                        appendToCSV(fileName, scale, T, pH, F1, F2, F3, result)



def sampleLatinHyperCube(n_samples, scale, T_Bounds, pH_Bounds, F1_Bounds, F2_Bounds, F3_Bounds, client, fileName):
    lhc_sample = qmc.LatinHypercube(d=5).random(n=n_samples)
    lower_bounds = [T_Bounds[0], pH_Bounds[0], F1_Bounds[0], F2_Bounds[0], F3_Bounds[0]]
    upper_bounds = [T_Bounds[1], pH_Bounds[1], F1_Bounds[1], F2_Bounds[1], F3_Bounds[1]]
    scaled_lhc_sample = qmc.scale(lhc_sample, l_bounds=lower_bounds, u_bounds=upper_bounds)
    for recipe in scaled_lhc_sample:
        T, pH, F1, F2, F3 = recipe
        print(f"running experiment [{scale}, {T}, {pH}, {F1}, {F2}, {F3}]")
        result = client.run(scale, T, pH, F1, F2, F3)
        appendToCSV(fileName, scale, T, pH, F1, F2, F3, result)


def sampleRandomSampling(n_samples, scale, T_Bounds, pH_Bounds, F1_Bounds, F2_Bounds, F3_Bounds, client, fileName):
    for _ in range(n_samples):
        T = np.random.uniform(*T_Bounds)
        pH = np.random.uniform(*pH_Bounds)
        F1 = np.random.uniform(*F1_Bounds)
        F2 = np.random.uniform(*F2_Bounds)
        F3 = np.random.uniform(*F3_Bounds)
        print(f"running experiment [{scale}, {T}, {pH}, {F1}, {F2}, {F3}]")
        result = client.run(scale, T, pH, F1, F2, F3)
        appendToCSV(fileName, scale, T, pH, F1, F2, F3, result)


def setupAndFitPipeline(X, y, includeWhiteKernel=True, var=None):
    # σ_signal² · exp(-‖x−x'‖²/2ℓ²) + σ_noise²·δ(x,x')
    # 5 length scales for automatic relevance determination (ARD)
    rbf = ConstantKernel() * RBF(length_scale=[1.0]*5, length_scale_bounds=[(1e-5, 3.0)]*5)
    if includeWhiteKernel:
        rbf += WhiteKernel()
    if var is not None:
        gp = GaussianProcessRegressor(
            kernel=rbf, 
            normalize_y=True, 
            n_restarts_optimizer=50, 
            alpha=var)
    else:
        gp = GaussianProcessRegressor(
            kernel=rbf, 
            normalize_y=True, 
            n_restarts_optimizer=50)
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('gp', gp)
    ])
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
            else:
                warnings.warn(warning.message)
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


def predictedImprovement(X, pipeline, y_best):
    """PI(X) = max(mu(X) - y_best, 0)."""
    mu, _ = pipeline.predict(X, return_std=True)
    return np.maximum(mu - y_best, 0.0)




if __name__ == "__main__":
    # np.random.seed(67)
    parser = argparse.ArgumentParser(
        description='Run Task03 Bayesian optimization with selectable acquisition functions.'
    )
    parser.add_argument('--acq', choices=['ei', 'ucb', 'pi'], default='ucb',
                        help="Acquisition function: 'ei', 'ucb', or 'pi' (predicted improvement)")
    parser.add_argument('--run-id', default='10', help='Run identifier for output filename')
    parser.add_argument('--patience', type=int, default=30,
                        help='Number of iterations without improvement to stop')
    parser.add_argument('--init-method', choices=['corners', 'lhc', 'random'], default='lhc',
                        help='Initial sampling method for warm-start data')
    parser.add_argument('--init-samples', type=int, default=20,
                        help='Number of initial samples for Latin Hypercube or random sampling')
    parser.add_argument('--budget', type=float, default=1e6,
                        help='Total budget in EUR')
    parser.add_argument('--filename', type=str, default=None,
                        help='Optional explicit output CSV filename')
    args = parser.parse_args()

    # setup experiment
    BUDGET = args.budget
    ACQ_FUN = args.acq
    INIT_METHOD = args.init_method
    if ACQ_FUN == 'ucb':
        ACQ_FUN_VAR_MAX = 2.0
        ACQ_FUN_VAR_MIN = -0.1
        ACQ_FUN_VAR_DECAY = 30
    elif ACQ_FUN == 'ei':
        ACQ_FUN_VAR_MAX = 0.1
        ACQ_FUN_VAR_MIN = 0.01
        ACQ_FUN_VAR_DECAY = 30
    elif ACQ_FUN == 'pi':
        ACQ_FUN_VAR_MAX = 0.1
        ACQ_FUN_VAR_MIN = 0.01
        ACQ_FUN_VAR_DECAY = 30
    else:
        raise NotImplementedError(f"Unknown acquisition function: {ACQ_FUN}")
    PATIENCE = args.patience
    LHC_SAMPLES = args.init_samples
    RUN_ID = args.run_id
    FILENAME = args.filename or f'Task03/data_{ACQ_FUN}_{INIT_METHOD}_param02_patience{PATIENCE}_lhc{LHC_SAMPLES}_final_{RUN_ID}.csv'
    
    # setup client
    client = BioreactorClient()
    client.login()

    #get initial data
    if not os.path.exists(FILENAME):
        if INIT_METHOD == 'corners':
            sampleHypercubeCorners(
                scale='micro', 
                T_Bounds=T_BOUNDS, 
                pH_Bounds=pH_BOUNDS, 
                F1_Bounds=F_BOUNDS, 
                F2_Bounds=F_BOUNDS, 
                F3_Bounds=F_BOUNDS, 
                client=client, 
                fileName=FILENAME
            )
        elif INIT_METHOD == 'lhc':
            sampleLatinHyperCube(
                n_samples=LHC_SAMPLES,
                scale='micro', 
                T_Bounds=T_BOUNDS, 
                pH_Bounds=pH_BOUNDS, 
                F1_Bounds=F_BOUNDS, 
                F2_Bounds=F_BOUNDS, 
                F3_Bounds=F_BOUNDS, 
                client=client, 
                fileName=FILENAME
            )
        elif INIT_METHOD == 'random':
            sampleRandomSampling(
                n_samples=LHC_SAMPLES,
                scale='micro', 
                T_Bounds=T_BOUNDS, 
                pH_Bounds=pH_BOUNDS, 
                F1_Bounds=F_BOUNDS, 
                F2_Bounds=F_BOUNDS, 
                F3_Bounds=F_BOUNDS, 
                client=client, 
                fileName=FILENAME
            )
        else:
            raise NotImplementedError(f"Unknown init method: {INIT_METHOD}")

    # main training loop
    iterations_with_unchanged_best = 0
    y_best = 0
    BUDGET -= COST['pilot']
    for i in range(10000):
        # get data so far
        df = getDataFrameFromCSV(fileName=FILENAME)
        cummulativeCost = np.sum(df["cost_eur"])
        X, y = getXyFromDataFrame(df, scaleY=True)
        var = getVarFromDataFrame(df, scaleY=True)
        var = var / np.var(y) # scaling to counteract normalize_y=True
        if np.max(y) > y_best:
            iterations_with_unchanged_best = 0
        else:
            iterations_with_unchanged_best += 1
        y_best = np.max(y)
        X_best = X[np.argmax(y)]
        print(f"\niteration {i} - best Y = {y_best:.3f} (for {iterations_with_unchanged_best} iterations) | cost = {cummulativeCost} €")

        # fit gp
        pipe = setupAndFitPipeline(X, y, includeWhiteKernel=False, var=var)

        # sample acquisition function
        n_test = 100000
        X_test = np.column_stack([
            np.random.uniform(*T_BOUNDS, n_test),
            np.random.uniform(*pH_BOUNDS, n_test),
            np.random.uniform(*F_BOUNDS, n_test),
            np.random.uniform(*F_BOUNDS, n_test),
            np.random.uniform(*F_BOUNDS, n_test),
        ])
        ACQ_FUN_VAR = ACQ_FUN_VAR_MAX + (ACQ_FUN_VAR_MIN - ACQ_FUN_VAR_MAX) * min(i / ACQ_FUN_VAR_DECAY, 1.0)
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
        elif ACQ_FUN == 'pi':
            acq_test = predictedImprovement(
                X=X_test,
                pipeline=pipe,
                y_best=y_best)
        else:
            raise NotImplementedError(f"Unknown acquisition function: {ACQ_FUN}")
        acq_max = np.max(acq_test)
        X_opt = X_test[np.argmax(acq_test)]

        # extract variables
        T, pH, F1, F2, F3 = X_opt

        # prediction on the optimum
        y_mu_opt, y_sigma_opt = pipe.predict(
            np.atleast_2d(X_opt), 
            return_std=True)
        print(f"next point: T={T:.1f} pH={pH:.2f} F1={F1:.2f} F2={F2:.2f} F3={F3:.2f} | acq={acq_max:.3e} | GP: mu={y_mu_opt[0]:.3f} sigma={y_sigma_opt[0]:.3f}")

        # choose scale
        if y_sigma_opt < 2 * SCALED_STDS['bench']:
            scale = 'pilot'
        elif y_sigma_opt < 1.5 * SCALED_STDS['micro']:
            scale = 'bench'
        else:
            scale = 'micro'

        # stopping criterions
        if iterations_with_unchanged_best >= PATIENCE:
            break
        if cummulativeCost + COST[scale] > BUDGET:
            break
        
        # run new experiment
        result = client.run(scale, T, pH, F1, F2, F3)
        appendToCSV(FILENAME, scale, T, pH, F1, F2, F3, result)
        print(f"measurement: Y = {result['Y']:.3f} -> projected: Y = {projectToPilotScale(result['Y'], scale):.3f}")

    # evaluate best point on pilot scale
    df = getDataFrameFromCSV(fileName=FILENAME)
    cummulativeCost = np.sum(df["cost_eur"])
    X, y = getXyFromDataFrame(df, scaleY=True)
    X_best = X[np.argmax(y)]
    T, pH, F1, F2, F3 = X_best
    result = client.run('pilot', T, pH, F1, F2, F3)
    appendToCSV(FILENAME, 'pilot', T, pH, F1, F2, F3, result)
    print(f"\nBest point on pilot scale: T={T:.1f} pH={pH:.2f} F1={F1:.2f} F2={F2:.2f} F3={F3:.2f} | Y = {result['Y']:.3f} | cost = {cummulativeCost}")
