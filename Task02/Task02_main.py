#%% Task 2
#A: Quantify per scale measurment noise and bias for each scale and compare
#B: Report: Discuss how these properties influence the optimization process 

import sys
import os
import pandas as pd
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scipy.stats import qmc
from API_Group8 import BioreactorClient
from global_constants import T_BOUNDS, pH_BOUNDS, F_BOUNDS
from data_frame_and_csv_manipulation import INPUT_COLS

FILENAME = 'Task02/lhc_samples_and_analysis.csv'
N_LHC_SAMPLES = 20
N_REPEATS = 20



#%% lhc sampling
lhc_sample = qmc.LatinHypercube(d=5).random(n=N_LHC_SAMPLES)
l_bounds = [T_BOUNDS[0], pH_BOUNDS[0], F_BOUNDS[0], F_BOUNDS[0], F_BOUNDS[0]]
u_bounds = [T_BOUNDS[1], pH_BOUNDS[1], F_BOUNDS[1], F_BOUNDS[1], F_BOUNDS[1]]
scaled_lhc_sample = qmc.scale(lhc_sample, l_bounds=l_bounds, u_bounds=u_bounds)
print("LHC samples:")
print(scaled_lhc_sample)


# evaluations
client = BioreactorClient()
client.login()

rows = []
for scale in ['micro', 'bench', 'pilot']:
    for recipe in scaled_lhc_sample:
        for i in range(N_REPEATS):
            T, pH, F1, F2, F3 = recipe
            print(f"running experiment [{scale}, {T:.3f}, {pH:.3f}, {F1:.3f}, {F2:.3f}, {F3:.3f}]")
            result = client.run(scale, T, pH, F1, F2, F3)
            row = {'scale': scale, 'T': T, 'pH': pH, 'F1': F1, 'F2': F2, 'F3': F3, 'Y': result['Y']}
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(FILENAME, index=False)



#%% statistical analysis
df = pd.read_csv(FILENAME)

for scale in df['scale'].unique():
    for T in df['T'].unique(): # iterates over different recipes
        mask = (df['scale'] == scale) & (df['T'] == T) 
        mask_pilot = (df['scale'] == 'pilot') & (df['T'] == T)
        subset = df[mask]
        subset_pilot = df[mask_pilot]

        per_recipe_mean_Y = subset['Y'].mean()
        per_recipe_std_Y = subset['Y'].std()
        per_recipe_mean_Y_pilot = subset_pilot['Y'].mean()

        scale_multiplier = per_recipe_mean_Y_pilot / per_recipe_mean_Y
        std_norm = per_recipe_std_Y * scale_multiplier

        df.loc[mask, 'mean Y per recipe'] = per_recipe_mean_Y
        df.loc[mask, 'std Y per recipe'] = per_recipe_std_Y
        df.loc[mask, 'scale multiplier'] = scale_multiplier
        df.loc[mask, 'normalized std'] = std_norm

    mask = (df['scale'] == scale)
    subset = df[mask]
    df.loc[mask, 'mean scale multiplier'] = subset['scale multiplier'].mean()
    df.loc[mask, 'mean std'] = subset['std Y per recipe'].mean()
    df.loc[mask, 'mean normalized std'] = subset['normalized std'].mean()

df.to_csv(FILENAME, index=False)



#%% plotting
df = pd.read_csv(FILENAME)

METRICS = ['mean Y per recipe', 'Y / Y_pilot', 'std Y per recipe', 'normalized std']
SCALE_STYLES = {
    'micro': {'color': '#0072B2', 'marker': 'o'},
    'bench': {'color': '#E69F00', 'marker': 's'},
    'pilot': {'color': '#D55E00', 'marker': '^'},
}


for scale, style in SCALE_STYLES.items():
    subset = df[df['scale'] == scale]

    fig, axes = plt.subplots(
        len(METRICS), len(INPUT_COLS),
        figsize=(2.6 * len(INPUT_COLS), 2.2 * len(METRICS)),
        sharex='col', sharey='row',
    )

    for row, metric in enumerate(METRICS):
        for col, input_col in enumerate(INPUT_COLS):
            axes[row, col].scatter(
                subset[input_col], subset[metric],
                s=16, alpha=0.7, color=style['color'], marker=style['marker'],
                edgecolors='none',
            )

    for row, metric in enumerate(METRICS):
        axes[row, 0].set_ylabel(metric)
    for col, input_col in enumerate(INPUT_COLS):
        axes[-1, col].set_xlabel(input_col)

    fig.suptitle(scale)
    fig.tight_layout()
    fig.savefig(FILENAME.removesuffix('.csv') + f'_{scale}_stats.pdf', bbox_inches='tight')

plt.show()


# %%
