import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from API_Group8 import BioreactorClient
from data_frame_and_csv_manipulation import getXyFromCSV



CSV_PATHS = [
    "Task03/micro_data_ei(0.01)_03.csv",
    "Task03/micro_data_ei(0.01)_04.csv",
    "Task03/micro_data_ei(0.01)_05.csv",
    "Task03/micro_data_ei(0.01)_06.csv",
    "Task03/micro_data_ei(0.01)_07.csv",
    "Task03/bench_data_ei(0.01)_01.csv",
    "Task03/bench_data_ei(0.01)_02.csv",
    "Task03/bench_data_ei(0.01)_03.csv",
    "Task03/pilot_data_ei(0.01)_01.csv",
    "Task03/pilot_data_ei(0.01)_02.csv",
    "Task03/pilot_data_ei(0.01)_03.csv",
]

OUTPUT_PATH = "Task03/meta_analysis.csv"



client = BioreactorClient()
client.login()

rows = []



for csv_path in CSV_PATHS:
    X, y = getXyFromCSV(csv_path)
    X_opt = X[np.argmax(y)]
    result_pilot = client.run('pilot', *X_opt)
    rows.append({'csv_path': csv_path, 'X_opt': X_opt, 'result_pilot': result_pilot['Y']})

df = pd.DataFrame(rows, columns=['csv_path', 'X_opt', 'result_pilot'])

df.to_csv(OUTPUT_PATH, index=False)