import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import plotly.express as px
from data_frame_and_csv_manipulation import getDataFrameFromCSV

CSV_PATH = 'mlme26_group08_2026-07-13T12-46-31.csv'
# CSV_PATH = 'Task03/pilot_data_ei(0.01)_01.csv'

df = getDataFrameFromCSV(CSV_PATH)

y_min, y_max = df['Y'].min(), df['Y'].max()

# Gamma < 1 stretches color resolution near the optimum (y_max) and
# compresses it further away, so near-optimal points are easier to tell apart.
GAMMA = 0.4
# GAMMA = 0.2

FONT_SIZE = 28


def make_parallel_coords_fig(subset, scale, y_min, y_max, gamma, font_size):
    subset = subset.sort_values('Y', ascending=True).reset_index()  # 'id' becomes a column, before reordering
    # Use this subset's own max (not the global y_max) so the GAMMA compression
    # highlights near-optimal points within THIS subset. With the global y_max,
    # subsets far from the overall optimum (e.g. pilot, micro) get almost all
    # of their points crushed into a sliver near the bottom of the color range.
    subset_y_max = subset['Y'].max()
    subset['color_val'] = subset_y_max - (subset_y_max - subset['Y']) ** gamma

    fig = px.parallel_coordinates(
        subset,
        # dimensions=['id', 'T', 'pH', 'F1', 'F2', 'F3', 'Y'],
        dimensions=['T', 'pH', 'F1', 'F2', 'F3', 'Y'],
        color='color_val',
        color_continuous_scale=px.colors.sequential.Turbo,
        # title=f'Parallel Coordinates - {scale} scale',
        # title=f'{scale} scale',
        template=dict(layout=dict(font=dict(size=font_size))),
    )
    for dim in fig.data[0].dimensions:
        if dim.label == 'Y':
            dim.range = [y_min, y_max]
        # dim.range is only non-None once explicitly set (as done above for
        # 'Y'); otherwise it defaults to this dimension's own column range.
        dim_range = dim.range if dim.range is not None else (subset[dim.label].min(), subset[dim.label].max())
        # Explicit, evenly-spaced ticks that land exactly on the axis bounds.
        # Plotly's auto ticks are "nice round numbers" that usually DON'T land
        # on the true min/max, so the auto tick nearest an edge (e.g. pH's "9")
        # ends up crowded right next to the bold min/max label (e.g. "9.5"),
        # reading as a doubled tick.
        dim_ticks = np.linspace(*dim_range, 6)
        dim.tickvals = dim_ticks
        dim.ticktext = [f'{v:.3g}' for v in dim_ticks]

    # Colorbar ticks are in transformed units by default; relabel them back to Y.
    # Use this subset's own Y range (color autoscales per subset), not the
    # global range, otherwise ticks for narrow subsets fall outside the
    # colorbar's actual span and get clipped down to a single visible tick.
    tick_y = np.linspace(subset['Y'].min(), subset['Y'].max(), 8)
    fig.update_coloraxes(
        colorbar_title='Y',
        colorbar_tickvals=subset_y_max - (subset_y_max - tick_y) ** gamma,
        colorbar_ticktext=[f'{v:.3g}' for v in tick_y],
    )
    return fig


for scale in df['scale'].unique():
    subset = df[df['scale'] == scale]
    fig = make_parallel_coords_fig(subset, scale, y_min, y_max, GAMMA, FONT_SIZE)
    fig.show(renderer='browser')
