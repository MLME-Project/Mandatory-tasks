'''
This is just an example for how you submission for Beat-the-Felix should look like.
There should be one file called main.py in the root of your submission folder.
Running this file should execute your algorithm and print the mean perfoemance of
your algorithm over 10 independent trials.
'''


from _helper import BioreactorClient
import numpy as np


# --- Configuration ---------------------------------------------------------
# Replace these with your own group's credentials.

USER = "your_group_name"
PASSWORD = "your_group_password"

N_TRIALS = 10            # how many independent trials to run
N_EXPERIMENTS = 5       # random experiments per trial
SEED = 0                 # change this to get a different random sequence

# The five recipe parameters and their allowed ranges [low, high].
# A submitted recipe must stay inside these bounds.
RECIPE_KEYS = ("T", "pH", "F1", "F2", "F3")
RECIPE_LOW = np.array([20.0, 3.0, 0.0, 0.0, 0.0])
RECIPE_HIGH = np.array([42.0, 9.5, 2.0, 2.0, 2.0])

# The three scales you can submit to. Bigger scales cost more but are closer
# to the real target process.
SCALES = ("micro", "bench", "pilot")


def random_recipe(rng: np.random.Generator) -> list[float]:
    """Draw one recipe uniformly at random inside the allowed bounds."""
    return rng.uniform(RECIPE_LOW, RECIPE_HIGH).tolist()


def run_trial(client: BioreactorClient, rng: np.random.Generator) -> list[float]:
    """Run N_EXPERIMENTS random experiments and return the list of yields."""
    yields: list[float] = []
    for i in range(N_EXPERIMENTS):
        scale = rng.choice(SCALES)
        recipe = random_recipe(rng)
        res = client.run(scale, *recipe)
        y = float(res["Y"])
        yields.append(y)
        print(f"  exp {i + 1:2d}  scale={scale:<5s}  Y={y:.4f}  "
              f"cost={res['cost_eur']}€", flush=True)
    return yields


if __name__ == "__main__":
    client = BioreactorClient()
    client.login(USER, PASSWORD)

    rng = np.random.default_rng(SEED)
    trial_means: list[float] = []

    for trial in range(N_TRIALS):
        print(f"\n=== TRIAL {trial + 1} / {N_TRIALS} ===")
        yields = run_trial(client, rng)
        mean = float(np.mean(yields))
        var = float(np.var(yields))
        trial_means.append(mean)
        print(f"  -> mean Y = {mean:.4f}   variance = {var:.4f}")

    print("\n=== SUMMARY OVER ALL TRIALS ===")
    print(f"  mean of trial means = {np.mean(trial_means):.4f}")
    print(f"  variance of trial means = {np.var(trial_means):.4f}")
