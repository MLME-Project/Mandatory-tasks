import pandas as pd

df = pd.read_csv('mlme26_group08_2026-07-10T13-42-11.csv')

max_Y_per_scale = df.groupby('scale')['Y'].max()
print("max Y per scale:")
print(max_Y_per_scale)

scales_from_maxima = max_Y_per_scale['pilot'] / max_Y_per_scale
print("\nscales computed from per-scale maxima (relative to pilot):")
print(scales_from_maxima)
