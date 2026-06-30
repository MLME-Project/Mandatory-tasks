import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import plotly.express as px
from data_frame_and_csv_manipulation import getDataFrameFromCSV

CSV_PATH = 'mlme26_group08_2026-06-29T10-47-17.csv'
# CSV_PATH = 'Task03/micro_data.csv'

df = getDataFrameFromCSV(CSV_PATH)

y_min, y_max = df['Y'].min(), df['Y'].max()

for scale in df['scale'].unique():
    fig = px.parallel_coordinates(
        df[df['scale'] == scale].sort_values('Y', ascending=True),
        dimensions=['T', 'pH', 'F1', 'F2', 'F3', 'Y'],
        color='Y',
        color_continuous_scale=px.colors.sequential.Turbo,
        title=f'Parallel Coordinates — {scale} scale',
    )
    for dim in fig.data[0].dimensions:
        if dim.label == 'Y':
            dim.range = [y_min, y_max]
    fig.show(renderer='browser')
