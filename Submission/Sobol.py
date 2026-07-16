"""

  - Die ersten N_MICRO Punkte werden auf micro ausgeführt
  - Die nächsten N_BENCH Punkte werden auf bench ausgeführt
  - Die nächsten N_PILOT Punkte werden auf pilot ausgeführt
  - Anschliessend bestes gefundenes Rezept (höchstes normiertes Y aus micro und bench) einmalig auf Pilot validieren
  - Alle Skalen erkunden denselben Recipe-Raum (gleiche Sobol-Punkte, nur die Skala der Ausf+hrung unterscheidet sich)

English:
    - The first N_MICRO points are executed at the micro scale
    - The next N_BENCH points are executed at the bench scale
    - The next N_PILOT points are executed at the pilot scale
    - Subsequently, validate the best recipe found (highest normalized Y from micro and bench) once at the pilot scale
    - All scales explore the same recipe space (identical Sobol points, only the execution scale differs)
"""

import time
import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.stats import qmc

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


USER = "group08"
PASSWORD = "4dv1m-vppbi-fa17y"
BASE_URL = "https://mlme26biosim.org"

# Matplotlib global settings
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})


# hier Werte eintragen

# Anzahl Experimente pro Skala
N_MICRO = 134
N_BENCH = 11
N_PILOT = 0  #  + 1 final

# Ergebnisse des BO-Loops für die Vergleichsplots
BO_Y_PILOT  = 23.41359242     # bestes Y_pilot aus dem BO-Loop [g/L]
BO_COST     = 8840.0          # Gesamtkosten des BO-Loops [EUR]

# Felix-Baseline (aus Aufgabenstellung):
FELIX_Y_PILOT = 14.0   # g/L
FELIX_COST    = 15000.0


COST_PER_RUN = {"micro": 10.0, "bench": 500.0, "pilot": 2000.0}


# API-Client

class BioreactorClient:
    def __init__(self, base_url=BASE_URL):
        self.s = requests.Session()
        self.base = base_url.rstrip("/")

    def login(self, user, password):
        r = self.s.post(
            f"{self.base}/api/login",
            json={"user": user, "password": password},
            timeout=15,
        )
        r.raise_for_status()

    def _csrf(self):
        token = self.s.cookies.get("mlme26_csrf")
        if not token:
            raise RuntimeError("no CSRF cookie - call login() first")
        return token

    def run(self, scale, T, pH, F1, F2, F3):
        payload = {
            "scale": scale,
            "recipe": {"T": T, "pH": pH, "F1": F1, "F2": F2, "F3": F3},
        }
        last_err = None
        for attempt in range(8):
            try:
                r = self.s.post(
                    f"{self.base}/api/run", json=payload,
                    headers={"X-CSRF-Token": self._csrf()},
                    timeout=60,
                )
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = e
                wait = 2.5 ** attempt
                print(f"  [network error: {type(e).__name__}, sleeping {wait:.1f}s]")
                time.sleep(wait)
                continue
            if r.status_code == 429:
                wait = 2.5 ** attempt
                print(f"  [rate-limited, sleeping {wait:.1f}s]")
                time.sleep(wait)
                continue
            if r.status_code == 402:
                raise RuntimeError(f"budget exhausted: {r.json().get('detail')}")
            if r.status_code >= 500:
                wait = 2.5 ** attempt
                print(f"  [server {r.status_code}, sleeping {wait:.1f}s]")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"too many failed attempts (last: {last_err!r})")

    def history(self):
        r = self.s.get(f"{self.base}/api/history", timeout=30)
        r.raise_for_status()
        return r.json()["items"]


client = BioreactorClient()
client.login(USER, PASSWORD)





BOUNDS = {
    "T":   (20.0, 60.0),
    "pH":  (3.0, 9.5),
    "F1":  (0.0, 2.0),
    "F2":  (0.0, 2.0),
    "F3":  (0.0, 2.0),
}
VAR_NAMES = ["T", "pH", "F1", "F2", "F3"]


N_TOTAL = N_MICRO + N_BENCH + N_PILOT
expected_cost = (N_MICRO * COST_PER_RUN["micro"]
               + N_BENCH * COST_PER_RUN["bench"]
               + N_PILOT * COST_PER_RUN["pilot"]
               + COST_PER_RUN["pilot"])  # + finale Validierung auf pilot


print("Sobol-Baseline Konfiguration")
print(f"  micro-Runs:               {N_MICRO}  x {COST_PER_RUN['micro']:.0f} EUR"
      f" = {N_MICRO * COST_PER_RUN['micro']:.0f} EUR")
print(f"  bench-Runs:               {N_BENCH}  x {COST_PER_RUN['bench']:.0f} EUR"
      f" = {N_BENCH * COST_PER_RUN['bench']:.0f} EUR")
print(f"  pilot-Runs (im Loop):     {N_PILOT}  x {COST_PER_RUN['pilot']:.0f} EUR"
      f" = {N_PILOT * COST_PER_RUN['pilot']:.0f} EUR")
print(f"  finaler Pilot-Call:       1  x {COST_PER_RUN['pilot']:.0f} EUR"
      f" = {COST_PER_RUN['pilot']:.0f} EUR")
print(f"  Gesamt-Sobol-Punkte:      {N_TOTAL}")
print(f"  Erwartete Gesamtkosten:   {expected_cost:.0f} EUR  (PLANUNGSWERT, siehe unten)")
print(f"  BO-Loop zum Vergleich:    {BO_Y_PILOT} g/L bei {BO_COST:.0f} EUR")





m = int(np.ceil(np.log2(max(N_TOTAL, 2))))  # kleinste Potenz >= N_TOTAL
N_SOBOL_GEN = 2 ** m                         # generierte Punkte (>= N_TOTAL)

sampler    = qmc.Sobol(d=5, scramble=True, seed=42)
sobol_unit = sampler.random_base2(m=m)       
sobol_unit = sobol_unit[:N_TOTAL]            # auf exakt N_TOTAL gekürzt

lows  = np.array([BOUNDS[v][0] for v in VAR_NAMES])
highs = np.array([BOUNDS[v][1] for v in VAR_NAMES])
sobol_scaled = qmc.scale(sobol_unit, lows, highs)  

print(f"Sobol-Sequenz: {N_SOBOL_GEN} Punkte generiert (2^{m}), "
      f"auf {N_TOTAL} gekuerzt (Praefix, siehe Kommentar oben).\n")

# Aufteilung

scale_assignments = (["micro"] * N_MICRO
                   + ["bench"] * N_BENCH
                   + ["pilot"] * N_PILOT)


# Alle Sobol-Punkte ausführen

cost_spent = 0.0
run_log    = []

scale_colors = {"micro": "tab:blue", "bench": "tab:orange", "pilot": "tab:green"}

print(f" Starte Sobol-Sampling ({N_TOTAL} Runs: "
      f"{N_MICRO} micro / {N_BENCH} bench / {N_PILOT} pilot) ")

for i, (point, scale) in enumerate(zip(sobol_scaled, scale_assignments)):
    recipe = {v: float(point[j]) for j, v in enumerate(VAR_NAMES)}

    res  = client.run(scale=scale, T=recipe["T"], pH=recipe["pH"],
                      F1=recipe["F1"], F2=recipe["F2"], F3=recipe["F3"])
    y    = res["Y"]
    cost = res["cost_eur"]


    cost_spent += cost

    run_log.append({
        "run":         i + 1,
        "scale":       scale,
        "Y":           y,
        "cost":        cost,
        "cum_cost":    cost_spent,
        **recipe,
    })

    if (i + 1) % 50 == 0 or i == 0 or (i + 1) == N_TOTAL:
        print(f"  Run {i+1:5d}/{N_TOTAL} | scale={scale:6s} | Y={y:.3f} | "
              f"cum_cost={cost_spent:.1f} EUR")

print(f"\n=== Sobol-Sampling abgeschlossen ===")
print(f"Kosten Sampling-Phase: {cost_spent:.1f} EUR ({N_TOTAL} Runs)\n")

df_runs = pd.DataFrame(run_log)

# Normierung: Z-Score 

pool_mask = df_runs["scale"].isin(["micro", "bench"])

df_runs["Y_norm"] = np.nan
df_runs.loc[pool_mask, "Y_norm"] = (
    df_runs.loc[pool_mask]
           .groupby("scale")["Y"]
           .transform(lambda x: (x - x.mean()) / x.std(ddof=0))
)

# bestes Rezept
pool_df     = df_runs.loc[pool_mask]
best_run    = pool_df.loc[pool_df["Y_norm"].idxmax()]
best_recipe = {v: best_run[v] for v in VAR_NAMES}

print(f"Bestes Rezept (Run #{int(best_run['run'])}, Skala: {best_run['scale']}, "
      f"Y_norm={best_run['Y_norm']:.3f}):")
for v in VAR_NAMES:
    print(f"  {v} = {best_recipe[v]:.4f}")
print(f"  Y (raw, {best_run['scale']}) = {best_run['Y']:.3f} g/L\n")


# Finaler Pilot-Call mit bestem Rezept

print("Finale Pilot-Validierung des besten Sobol-Rezepts")
res_pilot  = client.run(scale="pilot", T=best_recipe["T"], pH=best_recipe["pH"],
                        F1=best_recipe["F1"], F2=best_recipe["F2"],
                        F3=best_recipe["F3"])
y_pilot    = res_pilot["Y"]
cost_pilot = res_pilot["cost_eur"]
cost_spent += cost_pilot

print(f"Y_pilot (Sobol) = {y_pilot:.3f} g/L  (Pilot-Kosten: {cost_pilot:.1f} EUR)")
print(f"Gesamtkosten Sobol: {cost_spent:.1f} EUR")
print(f"\nVergleich:")
print(f"  Felix-Baseline: Y_pilot = {FELIX_Y_PILOT:.1f} g/L  bei {FELIX_COST:.0f} EUR")
print(f"  Sobol-Baseline: Y_pilot = {y_pilot:.3f} g/L  bei {cost_spent:.1f} EUR")
print(f"  BO-Loop:        Y_pilot = {BO_Y_PILOT:.1f} g/L  bei {BO_COST:.0f} EUR\n")

# CSV speichern
df_runs.to_csv(os.path.join(OUTPUT_DIR, "sobol_run_log.csv"), index=False)
print("CSV gespeichert: sobol_run_log.csv\n")


# Plots 

# 1
pool_traj = pool_df.copy().sort_values("run")
pool_traj["best_norm_so_far"] = pool_traj["Y_norm"].cummax()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pool_traj["run"], pool_traj["best_norm_so_far"],
        color="tab:blue", lw=1.8, label="Sobol: best normalized Y so far")
if N_MICRO > 0 and N_BENCH > 0:
    ax.axvline(N_MICRO, color="gray", linestyle=":", lw=1)
    ax.text(N_MICRO + 1, ax.get_ylim()[0], "micro | bench", fontsize=11, color="gray")
ax.set_xlabel("Sobol run (non-adaptive, chronological)")
ax.set_ylabel("Best normalized objective found so far (z-score)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sobol_plot1_trajectory.png"), dpi=150)
plt.close(fig)


# 2
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_runs["run"], df_runs["cum_cost"],
        color="tab:red", lw=1.8, label="Sobol: cumulative cost")
ax.axhline(BO_COST, color="tab:orange", linestyle="--",
           label=f"BO loop cost ({BO_COST:.0f} EUR)")
ax.axhline(FELIX_COST, color="black", linestyle="--",
           label=f"Felix budget ({FELIX_COST:.0f} EUR)")
ax.set_xlabel("Sobol run")
ax.set_ylabel("Cumulative cost [EUR]")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sobol_plot2_cost.png"), dpi=150)
plt.close(fig)

# 3
fig, ax = plt.subplots(figsize=(8, 5))
for scale in ["micro", "bench", "pilot"]:
    mask = df_runs["scale"] == scale
    if mask.sum() > 0:
        ax.scatter(df_runs.loc[mask, "run"], df_runs.loc[mask, "Y"],
                   color=scale_colors[scale], label=scale, s=14, alpha=0.65)
ax.set_xlabel("Sobol run")
ax.set_ylabel("Y [g/L]")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sobol_plot3_Y_by_scale.png"), dpi=150)
plt.close(fig)

# 4
fig, ax = plt.subplots(figsize=(7.5, 5))
for scale in ["micro", "bench", "pilot"]:
    mask = df_runs["scale"] == scale
    if mask.sum() > 0:
        ax.hist(df_runs.loc[mask, "Y"], bins=30,
                color=scale_colors[scale], alpha=0.6,
                label=scale, edgecolor="white")
ax.axvline(y_pilot, color="black", linestyle=":",
           label=f"final Y_pilot ({y_pilot:.2f} g/L)")
ax.set_xlabel("Y [g/L]")
ax.set_ylabel("Number of runs")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sobol_plot4_Y_distribution.png"), dpi=150)
plt.close(fig)

# 5
fig, ax = plt.subplots(figsize=(6.5, 5))
methods = ["Felix\nbaseline", "Sobol\nbaseline", "BO loop\n(our approach)"]
y_vals  = [FELIX_Y_PILOT, y_pilot, BO_Y_PILOT]
colors  = ["tab:gray", "tab:blue", "tab:orange"]
bars    = ax.bar(methods, y_vals, color=colors, width=0.5)
for bar, val in zip(bars, y_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
            f"{val:.1f} g/L", ha="center", va="bottom", fontsize=13)
ax.set_ylabel("Best Y_pilot [g/L]")
ax.set_ylim(0, max(y_vals) * 1.2)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sobol_plot5_comparison.png"), dpi=150)
plt.close(fig)

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 11,
})

lows  = np.array([BOUNDS[v][0] for v in VAR_NAMES])
highs = np.array([BOUNDS[v][1] for v in VAR_NAMES])
sobol_01 = (sobol_scaled - lows) / (highs - lows)

# 6
fig, ax = plt.subplots(figsize=(9, 5))
x_axes = np.arange(len(VAR_NAMES))

for row in sobol_01:
    ax.plot(x_axes, row, color="tab:blue", alpha=0.15, lw=0.8)

ax.set_xticks(x_axes)
ax.set_xticklabels(VAR_NAMES)
ax.set_ylabel("Normalized parameter value [0, 1]")
for x in x_axes:
    ax.axvline(x, color="gray", lw=0.8, zorder=0)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sobol_coverage_parallel.png"), dpi=150)
plt.close(fig)

print(f"Alle Plots gespeichert in {OUTPUT_DIR}:")
print("  sobol_plot1_trajectory.png")
print("  sobol_plot2_cost.png")
print("  sobol_plot3_Y_by_scale.png")
print("  sobol_plot4_Y_distribution.png")
print("  sobol_plot5_comparison.png")
