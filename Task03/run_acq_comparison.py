import subprocess
import os
import sys

SCRIPT = os.path.join(os.path.dirname(__file__), 'Task03_main.py')
ACQ_FUNS = ['ei', 'ucb', 'pi']
INIT_METHODS = ['corners', 'lhc', 'random']
RUN_ID = 'compare'
PATIENCE = 30
INIT_SAMPLES = 20
BUDGET = 30000

for acq in ACQ_FUNS:
    for repeat in range(10):
        cmd = [
            sys.executable, SCRIPT,
            '--acq', acq,
            '--run-id', f'{acq}_{RUN_ID}_{repeat+1}',
            '--patience', str(PATIENCE),
            '--init-samples', str(INIT_SAMPLES),
            '--budget', str(BUDGET),
        ]
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)

for init_method in INIT_METHODS:
    for repeat in range(10):
        cmd = [
            sys.executable, SCRIPT,
            '--init-method', init_method,
            '--run-id', f'{init_method}_{RUN_ID}_{repeat+1}',
            '--patience', str(PATIENCE),
            '--init-samples', str(INIT_SAMPLES),
            '--budget', str(BUDGET),
        ]
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
