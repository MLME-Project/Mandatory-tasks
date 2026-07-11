import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import plotly.express as px
from data_frame_and_csv_manipulation import getDataFrameFromCSV

# CSV_PATH = 'mlme26_group08_2026-07-01T16-59-36.csv'
CSV_PATH = 'Task03/micro_data_ei(0.01)_04.csv'

df = getDataFrameFromCSV(CSV_PATH)

y_min, y_max = df['Y'].min(), df['Y'].max()

# Gamma < 1 stretches color resolution near the optimum (y_max) and
# compresses it further away, so near-optimal points are easier to tell apart.
# GAMMA = 0.4
GAMMA = 0.2

for scale in df['scale'].unique():
    subset = df[df['scale'] == scale]

    subset = subset.sort_values('Y', ascending=True).reset_index()  # 'id' becomes a column, before reordering
    subset['color_val'] = y_max - (y_max - subset['Y']) ** GAMMA

    fig = px.parallel_coordinates(
        subset,
        dimensions=['id', 'T', 'pH', 'F1', 'F2', 'F3', 'Y'],
        color='color_val',
        color_continuous_scale=px.colors.sequential.Turbo,
        title=f'Parallel Coordinates — {scale} scale',
    )
    for dim in fig.data[0].dimensions:
        if dim.label == 'Y':
            dim.range = [y_min, y_max]

    # Colorbar ticks are in transformed units by default; relabel them back to Y.
    # Use this subset's own Y range (color autoscales per subset), not the
    # global range, otherwise ticks for narrow subsets fall outside the
    # colorbar's actual span and get clipped down to a single visible tick.
    tick_y = np.linspace(subset['Y'].min(), subset['Y'].max(), 8)
    fig.update_coloraxes(
        colorbar_title='Y',
        colorbar_tickvals=y_max - (y_max - tick_y) ** GAMMA,
        colorbar_ticktext=[f'{v:.3g}' for v in tick_y],
    )
    fig.show(renderer='browser')
