import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import plotly.express as px
from data_frame_and_csv_manipulation import getDataFrameFromCSV

# CSV_PATH = 'mlme26_group08_2026-07-01T16-59-36.csv'
CSV_PATH = 'Task03/micro_data_ei(0.01)_04.csv'

df = getDataFrameFromCSV(CSV_PATH)

y_min, y_max = df['Y'].min(), df['Y'].max()

for scale in df['scale'].unique():
    subset = df[df['scale'] == scale]

    subset = subset.sort_values('Y', ascending=True).reset_index()  # 'id' becomes a column, before reordering

    fig = px.parallel_coordinates(
        subset,
        dimensions=['id', 'T', 'pH', 'F1', 'F2', 'F3', 'Y'],
        color='Y',
        color_continuous_scale=px.colors.sequential.Turbo,
        title=f'Parallel Coordinates — {scale} scale',
    )
    for dim in fig.data[0].dimensions:
        if dim.label == 'Y':
            dim.range = [y_min, y_max]
    fig.show(renderer='browser')
