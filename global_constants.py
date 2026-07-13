T_BOUNDS = [20, 60]
pH_BOUNDS = [3, 9.5]
F_BOUNDS = [0, 2]

SCALE_MULTIPLIERS = {
    'micro': 18.302811,
    'bench': 0.667888,
    'pilot': 1.0,
}

COST = {
    'micro': 10,
    'bench': 500,
    'pilot': 2000,
}

STDS = {
    'micro': 0.019977964846977798,
    'bench': 0.0567652114182699,
    'pilot': 0.004981763078098352,
}

VARS = {key: value**2 for key, value in STDS.items()}

SCALED_STDS = {key: STDS[key] * SCALE_MULTIPLIERS[key] for key in STDS.keys()}

SCALED_VARS = {key: value**2 for key, value in SCALED_STDS.items()}

# print(SCALED_STDS)