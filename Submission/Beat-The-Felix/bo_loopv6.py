"""
====================================================================================
MULTI-FIDELITY BAYESIAN OPTIMIZATION FOR BIOREACTOR YIELD MAXIMIZATION
bo_loopv6.py - Sequential 3-Step Multi-Fidelity BO with Cost Awareness
====================================================================================
"""

import os
import csv
import json
import warnings
import numpy as np
import torch
from typing import Tuple, List, Dict
from datetime import datetime
import sys

# Add parent directory to path to import API_Group8
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# BoTorch imports
import botorch
from botorch.models import MultiTaskGP 
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.objective import ScalarizedPosteriorTransform
import gpytorch

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the API module
from API_Group8 import BioreactorClient

##====================================================================================
## CENTRALIZED HYPERPARAMETER DEFINITIONS
##====================================================================================
print("\n" + "="*80)
print("HYPERPARAMETER CONFIGURATION")
print("="*80)

# ==================== EXPERIMENT SCALE CONFIGURATION ====================
SCALES = ['micro', 'bench', 'pilot']
SCALE_TO_FIDELITY = {'micro': 0.0, 'bench': 1.0, 'pilot': 2.0}
SCALE_COSTS = {'micro': 10, 'bench': 200, 'pilot': 2000}

# ==================== BUDGET CONSTRAINTS ====================
TOTAL_BUDGET = 14800                    # Total EUR budget
RESERVED_PILOT_BUDGET = 2200            # Reserve for final validation run
EXPLORATION_BUDGET = TOTAL_BUDGET - RESERVED_PILOT_BUDGET  # 12000 EUR for exploration

# ==================== PARAMETER BOUNDS ====================
# Temperature (T), pH, Feed Rate 1 (F1), Feed Rate 2 (F2), Feed Rate 3 (F3)
BOUNDS = {
    'T': (20.0, 60.0),          # Temperature: 20-60 °C
    'pH': (3.0, 9.5),            # pH: 3-9.5
    'F1': (0.0, 2.0),            # Feed Rate 1: 0-2
    'F2': (0.0, 2.0),            # Feed Rate 2: 0-2
    'F3': (0.0, 2.0),            # Feed Rate 3: 0-2
}

# ==================== BASELINE PARAMETERS ====================
BASELINE_PARAMS = {
    'T': 35.0,                   # Baseline Temperature
    'pH': 7.0,                   # Baseline pH
    'F1': 1.0,                   # Baseline Feed Rate 1
    'F2': 1.0,                   # Baseline Feed Rate 2
    'F3': 1.0,                   # Baseline Feed Rate 3
}

# ==================== BO LOOP HYPERPARAMETERS ====================
# Initial samples per fidelity level (Step 1, 2, 3)
INITIAL_SAMPLES = {
    'micro': 3,                   # Initial micro scale samples
    'bench': 3,                   # NEU: Zwingt das Modell, Bench-Daten zu sammeln (vorher 0)
    'pilot': 0,                   # Initial pilot scale samples (use for validation only)
}

# Number of optimization iterations per step
BO_ITERATIONS = {
    'step1': 35,                  # Step 1 (T optimization) iterations
    'step2': 35,                  # Step 2 (pH optimization) iterations
    'step3': 35,                  # NEU: Reduziert von 30 auf 20, um Budget für Bench-Runs freizumache
}
    
# ==================== ABBRUCHKRITERIEN (EARLY STOPPING) ====================
EARLY_STOP_WINDOW = 7              # Anzahl Iterationen für die Konvergenzprüfung
                                    # (5 statt 3, um weniger anfällig für Messrauschen zu sein)
EARLY_STOP_REL_THRESHOLD = 0.002   # 0.2% relative Änderung -> für T, pH (1D)
EARLY_STOP_DIST_THRESHOLD = 0.04   # normierte euklidische Distanz -> für F1,F2,F3 (3D)
EXTRAPOLATION_FACTOR = 1.1         # Multiplikator auf die mittlere Steigung

# Candidates evaluated per acquisition function call
N_CANDIDATES = 10000                # NEU: Erhöht von 4000 auf 10000 für feinere interne Suche
# ==================== ACQUISITION FUNCTION SETTINGS ====================
ACQUISITION_BETA = 0.5              # Temperature for expected improvement (higher = more explorative)

# ==================== COST-AWARE SAMPLING STRATEGY ====================
COST_SCALING_FACTOR = 2.1       # Scaling factor for cost-aware weighting

print(f"\n[BUDGET]")
print(f"  Total Budget:           {TOTAL_BUDGET:,} EUR")
print(f"  Reserved for Validation: {RESERVED_PILOT_BUDGET:,} EUR")
print(f"  Exploration Budget:     {EXPLORATION_BUDGET:,} EUR")
print(f"\n[SCALES & COSTS]")
for scale in SCALES:
    print(f"  {scale.capitalize():10s}: Fidelity={SCALE_TO_FIDELITY[scale]:.1f}, Cost={SCALE_COSTS[scale]:4d} EUR")
print(f"\n[INITIAL SAMPLING]")
for scale in SCALES:
    print(f"  {scale.capitalize():10s}: {INITIAL_SAMPLES[scale]:d} samples")
print(f"\n[BO ITERATIONS]")
for step, iters in BO_ITERATIONS.items():
    print(f"  {step.upper():10s}: {iters:d} iterations")
print(f"\n[OTHER]")
print(f"  N_CANDIDATES:           {N_CANDIDATES:d}")
print(f"  ACQUISITION_BETA:       {ACQUISITION_BETA:.4f}")
print("="*80 + "\n")

##====================================================================================
## LOGGING AND TRACKING
##====================================================================================

class ExperimentLogger:
    """Manages CSV logging and run statistics."""
    
    def __init__(self, output_dir: str = 'output_v2'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(output_dir, f'bo_loopv6_results_{timestamp}.csv')
        
        # Initialize CSV with headers
        self.headers = ['Run_Number', 'Scale', 'T', 'pH', 'F1', 'F2', 'F3', 'Cost', 'Yield']
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        # Tracking variables
        self.run_count = 0
        self.scale_counts = {'micro': 0, 'bench': 0, 'pilot': 0}
        self.total_cost = 0
        self.all_results = []
    
    def log_experiment(self, scale: str, recipe: List[float], cost: float, yield_val: float):
        """Log a single experiment to CSV and update tracking."""
        self.run_count += 1
        self.scale_counts[scale] += 1
        self.total_cost += cost
        
        T, pH, F1, F2, F3 = recipe
        row = [self.run_count, scale, T, pH, F1, F2, F3, cost, yield_val]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        self.all_results.append({
            'run': self.run_count,
            'scale': scale,
            'recipe': recipe,
            'cost': cost,
            'yield': yield_val
        })
        
        # Print live counter
        print(f"[RUN {self.run_count:3d}] Scale: {scale:6s} | "
              f"Micro: {self.scale_counts['micro']:2d}, "
              f"Bench: {self.scale_counts['bench']:2d}, "
              f"Pilot: {self.scale_counts['pilot']:2d} | "
              f"Budget: {self.total_cost:7,.0f} / {EXPLORATION_BUDGET:,} EUR")
    
    def print_final_report(self, final_recipe: List[float], final_yield: float):
        """Print formatted final report."""
        T, pH, F1, F2, F3 = final_recipe
        print("\n" + "="*80)
        print("FINAL BAYESIAN OPTIMIZATION REPORT")
        print("="*80)
        print(f"\nFinal Optimal Recipe:")
        print(f"  Temperature (T):    {T:.4f} °C")
        print(f"  pH:                 {pH:.4f}")
        print(f"  Feed Rate 1 (F1):   {F1:.4f}")
        print(f"  Feed Rate 2 (F2):   {F2:.4f}")
        print(f"  Feed Rate 3 (F3):   {F3:.4f}")
        print(f"\nTotal Consumed Cost:   {self.total_cost:,.0f} EUR")
        print(f"Final Pilot Yield:     {final_yield:.6f}")
        print(f"\nCSV Log:               {self.csv_path}")
        print("="*80 + "\n")

logger = ExperimentLogger()

##====================================================================================
## BIOREACTOR API CLIENT
##====================================================================================

print("\n[INITIALIZING] Connecting to Bioreactor API...")
bio_client = BioreactorClient()
bio_client.login()
print("[SUCCESS] Connected and authenticated to Bioreactor API.\n")

##====================================================================================
## UTILITY FUNCTIONS
##====================================================================================

def run_single_experiment(recipe: np.ndarray, scale: str) -> Tuple[float, float]:
    """
    Run a single experiment via the API and log results.
    Returns: (yield_value, cost)
    """
    T, pH, F1, F2, F3 = recipe
    result = bio_client.run(scale, T=T, pH=pH, F1=F1, F2=F2, F3=F3)
    #print(f"\n[DEBUG] API Response für {scale}: {result}\n")
    # Extract yield from the API response
    yield_val = result.get('Y', 0.0)
    cost = SCALE_COSTS[scale]
    
    logger.log_experiment(scale, recipe.tolist() if isinstance(recipe, np.ndarray) else recipe, cost, yield_val)
    
    return yield_val, cost

def tensor_to_recipe(X_tensor: torch.Tensor, step: int, fixed_params: Dict[str, float]) -> np.ndarray:
    """
    Convert a tensor of optimized parameters to a full recipe array.
    X_tensor: tensor of optimized parameters (depends on step)
    step: 1, 2, or 3
    fixed_params: dict with previously fixed parameters (T, pH)
    Returns: [T, pH, F1, F2, F3] numpy array
    """
    recipe = np.array([
        fixed_params.get('T', BASELINE_PARAMS['T']),
        fixed_params.get('pH', BASELINE_PARAMS['pH']),
        fixed_params.get('F1', BASELINE_PARAMS['F1']),
        fixed_params.get('F2', BASELINE_PARAMS['F2']),
        fixed_params.get('F3', BASELINE_PARAMS['F3']),
    ])
    
    # Handle both 1D and 0D tensors
    if X_tensor.dim() == 0:
        # Scalar tensor
        if step == 1:
            recipe[0] = X_tensor.item()  # Set T
        elif step == 2:
            recipe[1] = X_tensor.item()  # Set pH
    else:
        # 1D tensor
        if step == 1:
            recipe[0] = X_tensor[0].item()  # Set T
        elif step == 2:
            recipe[1] = X_tensor[0].item()  # Set pH
        elif step == 3:
            recipe[2] = X_tensor[0].item()  # Set F1
            recipe[3] = X_tensor[1].item()  # Set F2
            recipe[4] = X_tensor[2].item()  # Set F3
    
    return recipe

def get_step_bounds(step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get parameter bounds for a specific optimization step."""
    if step == 1:
        # Optimize T only
        bounds = torch.tensor([
            [BOUNDS['T'][0]],
            [BOUNDS['T'][1]],
        ])
    elif step == 2:
        # Optimize pH only
        bounds = torch.tensor([
            [BOUNDS['pH'][0]],
            [BOUNDS['pH'][1]],
        ])
    elif step == 3:
        # Optimize F1, F2, F3
        bounds = torch.tensor([
            [BOUNDS['F1'][0], BOUNDS['F2'][0], BOUNDS['F3'][0]],
            [BOUNDS['F1'][1], BOUNDS['F2'][1], BOUNDS['F3'][1]],
        ])
    else:
        raise ValueError(f"Invalid step: {step}")
    
    return bounds

def run_initial_sampling(step: int, fixed_params: Dict[str, float],
                          center: torch.Tensor = None,
                          spread_fraction: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run initial random sampling for training BO model.
    Falls 'center' gesetzt ist, wird um dieses Zentrum herum gesampelt
    (Breite = spread_fraction * volle Bound-Breite), statt über den vollen Bereich.
    Returns: (X, Y) tensors where X is [N, D] and Y is [N, 1]
    """
    print(f"\n>>> Initial Sampling for Step {step}...")
    
    X_list = []
    Y_list = []
    
    bounds = get_step_bounds(step)
    dim = bounds.shape[1]
    
    # Sampling-Bereich bestimmen: entweder voller Bound-Bereich, oder eingeengt um 'center'
    if center is not None:
        half_width = spread_fraction * (bounds[1] - bounds[0])
        sample_lower = torch.clamp(center - half_width, min=bounds[0])
        sample_upper = torch.clamp(center + half_width, max=bounds[1])
        print(f"    Sampling um Zentrum {center.tolist()} mit Breite {spread_fraction}")
    else:
        sample_lower = bounds[0]
        sample_upper = bounds[1]
    
    # Sample from each scale sequentially
    for scale in ['micro', 'bench']:
        n_samples = INITIAL_SAMPLES[scale]
        if n_samples == 0:
            continue
        
        print(f"    Sampling {n_samples} experiments at {scale} scale...")
        
# Generate random samples (innerhalb von run_initial_sampling)
        for _ in range(n_samples):
            x_opt = torch.rand(1, dim) * (sample_upper - sample_lower) + sample_lower
            x_opt = x_opt.squeeze(0)  # [dim]
            recipe = tensor_to_recipe(x_opt, step, fixed_params)
            
            # Run experiment
            yield_val, _ = run_single_experiment(recipe, scale)
            
            # NEU: Fidelity anfügen
            fidelity_val = SCALE_TO_FIDELITY[scale]
            x_with_fidelity = torch.cat([x_opt, torch.tensor([fidelity_val])])
            
            # Store data
            X_list.append(x_with_fidelity)
            Y_list.append(torch.tensor(yield_val))
    
    if len(X_list) == 0:
        raise ValueError(f"No samples collected in initial sampling for step {step}")
    
    # Stack into tensors with proper dimensions
    X = torch.stack(X_list)  # [N, dim]
    Y = torch.stack(Y_list).unsqueeze(-1)  # [N, 1]
    
    print(f"    Initial sampling complete. Collected {len(X_list)} samples.")
    
    return X, Y

def fit_multi_fidelity_model(X: torch.Tensor, Y: torch.Tensor) -> MultiTaskGP:
    """
    Fit a Multi-Task Gaussian Process model.
    X: [N, D+1] tensor of parameter values (letzte Spalte ist Fidelity: 0.0, 1.0, oder 2.0)
    Y: [N, 1] tensor of observations
    """
    if X.dim() == 1:
        X = X.unsqueeze(-1)
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
    
    Y_mean = Y.mean()
    Y_std = Y.std()
    if Y_std < 1e-6:
        Y_std = torch.tensor(1.0)
    Y_norm = (Y - Y_mean) / Y_std
    
    # Nutze MultiTaskGP, bei dem die letzte Spalte die Skala definiert
    model = MultiTaskGP(X, Y_norm, task_feature=-1, all_tasks=[0, 1, 2]) 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    
    fit_gpytorch_mll(mll)
    
    model.Y_mean = Y_mean
    model.Y_std = Y_std
    
    return model

def get_next_candidates(model: MultiTaskGP, 
                       bounds: torch.Tensor, 
                       X: torch.Tensor,
                       Y: torch.Tensor,
                       step: int) -> Tuple[torch.Tensor, str]:
    device = X.device
    dim = bounds.shape[1]
    
# Wir definieren eine Transformation, die den Output (Yield) extrahiert
    obj = ScalarizedPosteriorTransform(weights=torch.tensor([1.0], device=X.device))
    
    # Wir übergeben dieses Objekt an die Acquisition Function
    acq_func = qUpperConfidenceBound(model, beta=ACQUISITION_BETA, posterior_transform=obj) 
    
    best_acq_value = -float('inf')
    best_candidate = None
    best_scale = None
    
    for scale in ['micro', 'bench', 'pilot']:
        fidelity_val = SCALE_TO_FIDELITY[scale]
        
        # NEU: Bounds temporär um die Fidelity-Dimension erweitern (fester Wert für diese Skala)
        scale_bounds = torch.cat([bounds, torch.tensor([[fidelity_val], [fidelity_val]])], dim=1)
        
        try:
            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=scale_bounds,
                q=1,
                num_restarts=5,
                raw_samples=N_CANDIDATES,
                options={"seed": np.random.randint(0, 1000000)},
            )
            
            acq_value = acq_values.max().item()
            
            # NEU: Angepasste Kostenbestrafung durch die Quadratwurzel
            cost_factor = 1.0 / (SCALE_COSTS[scale] ** 0.5)
            weighted_acq = acq_value * cost_factor
            
            if weighted_acq > best_acq_value:
                best_acq_value = weighted_acq
                # NEU: Fidelity am Ende abschneiden, damit nur die echten Parameter zurückgegeben werden
                best_candidate = candidates.squeeze(0)[:dim]
                best_scale = scale
        except Exception as e:
            print(f"      Warning: Acquisition optimization failed for {scale}: {e}")
            continue
    
    if best_candidate is None:
        best_candidate = torch.rand(dim, device=device) * (bounds[1] - bounds[0]) + bounds[0]
        best_scale = 'micro'
    
    return best_candidate, best_scale
def check_convergence_1d(best_so_far_history: List[float], window: int, 
                          rel_threshold: float) -> Tuple[bool, float]:
    """
    Prüft für 1D-Parameter (T, pH), ob sich der BISHER BESTE Parameterwert
    (nicht der zuletzt vorgeschlagene Kandidat!) über die letzten 'window'
    Iterationen um weniger als 'rel_threshold' verändert hat.
    Grund für 'bisher bester' statt 'letzter Kandidat': Der zuletzt vorgeschlagene
    Punkt springt durch die UCB-Exploration naturgemäß umher, auch wenn das
    eigentliche Optimum längst stabil ist. Der bisher beste Wert ist dagegen
    monoton und damit ein verlässlicheres Konvergenzsignal.
    Returns: (is_converged, extrapolated_value)
    """
    if len(best_so_far_history) < window:
        return False, None

    recent = best_so_far_history[-window:]
    start_val, end_val = recent[0], recent[-1]

    # Schutz gegen Division durch (nahe) 0
    denom = max(abs(start_val), 1e-6)
    rel_change = abs(end_val - start_val) / denom

    if rel_change >= rel_threshold:
        return False, None

    # Konvergiert -> mittlere Steigung im Fenster berechnen und extrapolieren
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    avg_slope = sum(diffs) / len(diffs)
    extrapolated_value = end_val + avg_slope * EXTRAPOLATION_FACTOR

    return True, extrapolated_value


def validate_extrapolated_point(model, extrapolated_value: float, bounds: torch.Tensor,
                                 best_observed_yield: float) -> float:
    """
    Validiert den extrapolierten Wert über die GP-Posterior-Mean-Vorhersage,
    bevor er als Optimum übernommen wird. Das verhindert, dass eine lineare
    Extrapolation (die die Nichtlinearität der echten Zielfunktion ignoriert)
    ein eigentlich gutes Ergebnis verschlechtert.
    Clippt zusätzlich auf die gültigen Bounds.
    Gibt den validierten Wert zurück, oder None, falls die Extrapolation
    verworfen werden soll (-> Fallback auf den bisher besten beobachteten Punkt).
    """
    low, high = bounds[0, 0].item(), bounds[1, 0].item()
    clipped_value = min(max(extrapolated_value, low), high)

    # Vorhersage auf PILOT-Fidelity, da diese Skala für das finale Ergebnis zählt
    x_test = torch.tensor([[clipped_value, SCALE_TO_FIDELITY['pilot']]])
    with torch.no_grad():
        posterior = model.posterior(x_test)
        pred_mean_norm = posterior.mean.item()
    pred_mean = pred_mean_norm * model.Y_std.item() + model.Y_mean.item()

    if pred_mean >= best_observed_yield:
        return clipped_value
    return None


def check_convergence_nd(best_so_far_history: List[torch.Tensor], window: int,
                          bounds: torch.Tensor, dist_threshold: float) -> bool:
    """
    Prüft für mehrdimensionale Parameter (F1, F2, F3), ob sich der bisher beste
    Punkt über die letzten 'window' Iterationen kaum noch bewegt hat.
    Alle Dimensionen werden zunächst per Min-Max auf [0,1] normiert (anhand der
    Bounds), damit F-Werte nahe 0 keine Division-durch-0-Probleme verursachen
    (im Gegensatz zu einer relativen Prozent-Änderung pro Parameter).
    Anschließend wird die euklidische Distanz zwischen Fensteranfang und
    Fensterende im normierten Raum berechnet.
    Es gibt bewusst KEINE Extrapolation hier: Eine unabhängige lineare
    Extrapolation pro Feed-Rate würde die Wechselwirkungen zwischen F1, F2, F3
    ignorieren und das Risiko einer Verschlechterung nur vergrößern.
    Bei Konvergenz wird stattdessen einfach der bisher beste beobachtete
    Punkt als Optimum übernommen.
    """
    if len(best_so_far_history) < window:
        return False

    recent = best_so_far_history[-window:]
    low, high = bounds[0], bounds[1]
    range_ = (high - low).clamp(min=1e-6)

    start_norm = (recent[0] - low) / range_
    end_norm = (recent[-1] - low) / range_

    dist = torch.norm(end_norm - start_norm).item()
    return dist < dist_threshold

def run_bo_loop(step: int, fixed_params: Dict[str, float],
                 n_iterations: int = None,
                 init_center: torch.Tensor = None,
                 init_spread_fraction: float = None,
                 previous_stage_yield: float = None) -> Tuple[Dict[str, float], np.ndarray, float]:
    """
    Execute one complete BO optimization loop.
    n_iterations: überschreibt die Standard-Iterationsanzahl aus BO_ITERATIONS, falls gesetzt.
    init_center / init_spread_fraction: falls gesetzt, wird die initiale Stichprobe
    um dieses Zentrum herum gezogen statt über den vollen Suchraum.
    previous_stage_yield: bester finaler Yield der vorherigen Stufe (Abbruchkriterium).
    Falls None (z.B. Step 1a ohne Vorgänger), wird diese Bedingung übersprungen.
    Returns: (fixed_params_updated, best_recipe, best_yield)
    """
    print(f"\n" + "="*80)
    print(f"STEP {step}: BAYESIAN OPTIMIZATION LOOP")
    print("="*80)
    
    bounds = get_step_bounds(step)
    
    # NEU: Höchstwert aller bisherigen Experimente über alle Steps hinweg aus dem Logger auslesen
    highest_prior_yield = max([r['yield'] for r in logger.all_results]) if logger.all_results else -float('inf')
    
# Initial sampling
    X, Y = run_initial_sampling(step, fixed_params, center=init_center, spread_fraction=init_spread_fraction)
    
    # NEU: Nutze nur Bench-Läufe für den Durchschnitt, falls vorhanden
    bench_fidelity = SCALE_TO_FIDELITY['bench']
    bench_mask = (X[:, -1] == bench_fidelity)
    
    if bench_mask.any():
        avg_init_yield = Y[bench_mask].mean().item()
        print(f"    Durchschnittlicher Init-Yield (nur Bench-Läufe als Referenz): {avg_init_yield:.6f}")
    else:
        avg_init_yield = Y.mean().item()
        print(f"    Durchschnittlicher Init-Yield (Fallback auf alle Läufe, kein Bench vorhanden): {avg_init_yield:.6f}")
    
    # BO iterations
    if n_iterations is None:
        n_iterations = BO_ITERATIONS[f'step{step}']
    
    print(f"\n>>> Starting BO iterations (max {n_iterations})...")
    
    best_so_far_history = []      # Verlauf des bisher besten Parameterwerts/-punkts
    early_stop_override_x = None  # Bei Abbruch gesetzter finaler X-Wert
    
    for iteration in range(n_iterations):
        # Check budget
        if logger.total_cost >= EXPLORATION_BUDGET:
            print(f"    Budget exhausted. Stopping optimization.")
            break
        
        # Fit model
        model = fit_multi_fidelity_model(X, Y)
        
        # Get next candidate
        x_next, suggested_scale = get_next_candidates(model, bounds, X, Y, step)
        
        # Construct full recipe
        recipe = tensor_to_recipe(x_next, step, fixed_params)
        
        print(f"  Iteration {iteration + 1}/{n_iterations}: Suggesting {suggested_scale}...")
        yield_val, cost = run_single_experiment(recipe, suggested_scale)
        
        fidelity_val = SCALE_TO_FIDELITY[suggested_scale]
        x_next_with_fidelity = torch.cat([x_next, torch.tensor([fidelity_val])])
        
        x_next_2d = x_next_with_fidelity.unsqueeze(0)
        y_next_2d = torch.tensor([[yield_val]])
        
        X = torch.cat([X, x_next_2d])
        Y = torch.cat([Y, y_next_2d])
        
# ==================== ABBRUCHKRITERIUM ====================
        current_best_idx = Y.argmax(dim=0).item()
        current_best_x = X[current_best_idx, :-1]  # ohne Fidelity-Spalte
        current_best_yield = Y[current_best_idx].item()
        
        # Speichert die tatsächlich vorgeschlagene Reaktionsvariable (Parameter-Wert)
        best_so_far_history.append(x_next.clone())
        
        # NEU: Dynamische Bestimmung der Ertrags-Hürde
        if step == 1:
            # Im ersten Run muss der beste Yield über dem Durchschnitt der Bench-Initialisierung liegen
            yield_condition = current_best_yield > avg_init_yield
        else:
            # Ab dem zweiten Run (Step >= 2) muss der beste Yield über dem absolut höchsten jemals gemessenen Yield liegen
            yield_condition = current_best_yield > highest_prior_yield
        
        if yield_condition and len(best_so_far_history) >= EARLY_STOP_WINDOW:
            if step in (1, 2):
                # Extrahiere die 1D-Reaktionsvariable (z.B. pH oder T) aus der Historie
                history_1d = [x[0].item() for x in best_so_far_history]
                converged, extrapolated_value = check_convergence_1d(
                    history_1d, EARLY_STOP_WINDOW, EARLY_STOP_REL_THRESHOLD
                )
                if converged:
                    print(f"    [ABBRUCH] Konvergenz der Reaktionsvariablen erkannt (Änderung < {EARLY_STOP_REL_THRESHOLD*100:.2f}% "
                          f"über die letzten {EARLY_STOP_WINDOW} Iterationen).")
                    validated_value = validate_extrapolated_point(
                        model, extrapolated_value, bounds, current_best_yield
                    )
                    if validated_value is not None:
                        print(f"    Extrapolation auf {extrapolated_value:.4f} vom GP bestätigt -> übernommen.")
                        early_stop_override_x = torch.tensor([validated_value])
                    else:
                        print(f"    Extrapolation vom GP NICHT bestätigt -> bisher bester "
                              f"beobachteter Punkt wird stattdessen verwendet.")
                        early_stop_override_x = current_best_x.clone()
                    break
            elif step == 3:
                # Prüfe die normierte euklidische Distanz der 3D-Reaktionsvariablen (F1, F2, F3)
                converged = check_convergence_nd(
                    best_so_far_history, EARLY_STOP_WINDOW, bounds, EARLY_STOP_DIST_THRESHOLD
                )
                if converged:
                    print(f"    [ABBRUCH] Konvergenz der Reaktionsvariablen erkannt (normierte Distanz < {EARLY_STOP_DIST_THRESHOLD} "
                          f"über die letzten {EARLY_STOP_WINDOW} Iterationen).")
                    early_stop_override_x = current_best_x.clone()
                    break
        # ==================== ENDE ABBRUCHKRITERIUM ====================

    # Find best sample from current step (ggf. durch Abbruch-Ergebnis überschrieben)
    if early_stop_override_x is not None:
        best_x = early_stop_override_x
        best_yield = Y[Y.argmax(dim=0).item()].item()
    else:
        best_idx = Y.argmax(dim=0).item()
        best_x = X[best_idx, :-1]
        best_yield = Y[best_idx].item()
    
    # Update fixed parameters
    fixed_params_new = fixed_params.copy()
    if step == 1:
        fixed_params_new['T'] = best_x[0].item()
        param_name = 'T'
        param_value = fixed_params_new['T']
    elif step == 2:
        fixed_params_new['pH'] = best_x[0].item()
        param_name = 'pH'
        param_value = fixed_params_new['pH']
    elif step == 3:
        fixed_params_new['F1'] = best_x[0].item()
        fixed_params_new['F2'] = best_x[1].item()
        fixed_params_new['F3'] = best_x[2].item()
        param_name = 'F1, F2, F3'
        param_value = f"({best_x[0].item():.4f}, {best_x[1].item():.4f}, {best_x[2].item():.4f})"
    
    best_recipe = tensor_to_recipe(best_x, step, fixed_params_new)
    
    print(f"\n>>> Step {step} Complete!")
    print(f"    Best {param_name} optimized: {param_value}")
    print(f"    Best yield found: {best_yield:.6f}")
    
    return fixed_params_new, best_recipe, best_yield

##====================================================================================
## MAIN EXECUTION
##====================================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("MULTI-FIDELITY BAYESIAN OPTIMIZATION FOR BIOREACTOR YIELD MAXIMIZATION")
    print("="*80)
    
    fixed_params = BASELINE_PARAMS.copy()
    
    # 1. KLARE, LINEARE REIHENFOLGE:

    
    # Step 1: Optimize Temperature
    fixed_params, best_recipe_step1, yield_step1 = run_bo_loop(step=1, fixed_params=fixed_params)
    
        # Step 2: Optimize pH
    fixed_params, best_recipe_step2, yield_step2 = run_bo_loop(
        step=2, fixed_params=fixed_params, previous_stage_yield=yield_step1
    )
   
    # Step 3: Optimize Feed Rates
    fixed_params, best_recipe_step3, yield_step3 = run_bo_loop(
        step=3, fixed_params=fixed_params, previous_stage_yield=yield_step2
    )
    
    
    # Final Step: Execute one pilot run for validation
    print(f"\n" + "="*80)
    print("FINAL VALIDATION PILOT RUN")
    print("="*80)
    
    final_recipe = np.array([
        fixed_params['T'],
        fixed_params['pH'],
        fixed_params['F1'],
        fixed_params['F2'],
        fixed_params['F3'],
    ])
    
    print(f"\nExecuting final validation run at PILOT scale...")
    final_yield, final_cost = run_single_experiment(final_recipe, 'pilot')
    
    # Print final report
    logger.print_final_report(final_recipe.tolist(), final_yield)
    
    return fixed_params, final_yield, logger.total_cost

if __name__ == '__main__':
    try:
        fixed_params, final_yield, total_cost = main()
        print("\n[SUCCESS] Bayesian Optimization completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Optimization failed: {e}")
        import traceback
        traceback.print_exc()