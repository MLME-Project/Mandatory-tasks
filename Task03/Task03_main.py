import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from API_Group8 import BioreactorClient
from data_frame_and_csv_manipulation import appendToCSV, getDataFrameFromCSV, getXyFromDataFrame



T_BOUNDS = [20, 60]
pH_BOUNDS = [3, 9.5]
F_BOUNDS = [0, 2]

# setup client
client = BioreactorClient()
client.login()

# get initial data
scale = 'micro'
fileName = 'micro_data.csv'
if False:
    for T in T_BOUNDS:
        for pH in pH_BOUNDS:
            for F1 in F_BOUNDS:
                for F2 in F_BOUNDS:
                    for F3 in F_BOUNDS:
                        print(f"running experiment [{scale}, {T}, {pH}, {F1}, {F2}, {F3}]")
                        result = client.run(scale, T, pH, F1, F2, F3)
                        appendToCSV(fileName, scale, T, pH, F1, F2, F3, result)

# assemble data
X, y = getXyFromDataFrame(getDataFrameFromCSV(fileName))

# set up gp
rbf = ConstantKernel() * RBF() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=50)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('gp', gp)
    ])

# initial fit
pipe.fit(X, y)