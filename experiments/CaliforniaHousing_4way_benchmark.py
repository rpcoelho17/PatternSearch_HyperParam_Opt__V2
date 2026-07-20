"""
California Housing — 4-way comparison: Optuna GPSampler vs BayesHalvingSearchCV
vs PatternSearchCV(eager) vs PatternSearchCV(patient).

Search space and CV protocol match the "California Housing task" section of
EXPERIMENTS.md (2026-07-19): max_features in {2,4,6,8}, n_estimators
10..260 step 10, max_depth in {5,8,11,14,17,20,25,30}; KFold(5, shuffle,
random_state=0); subsample="random" (no time ordering); verbose=2.
"""
import json
import time
from collections import Counter

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score

from bayes_halving_search_cv import BayesHalvingSearchCV, PatternSearchCV

df = pd.read_csv(r"C:\FILES\Code\Benchmarking\Working_on_Train_Set\V2025\pattern-search-cv\Data\california_housing.csv")
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
N = len(y)
print(f"X: {X.shape}")

param_grid = {
    "max_features": [2, 4, 6, 8],
    "n_estimators": list(range(10, 261, 10)),
    "max_depth": [5, 8, 11, 14, 17, 20, 25, 30],
}
cv = KFold(n_splits=5, shuffle=True, random_state=0)


def make_clf():
    return ExtraTreesRegressor(n_jobs=1, random_state=0)


def _summarize(arm, search, wall):
    res = search.cv_results_
    n_res = np.asarray(res["n_resources"])
    tiers = Counter(int(v) for v in n_res)
    out = {
        "arm": arm, "wall": wall,
        "n_fits": len(res["params"]),
        "tiers": {int(k): int(v) for k, v in tiers.items()},
        "equiv": float(np.sum(n_res / N)),
        "best": search.best_params_, "best_mae": float(-search.best_score_),
    }
    print(f"\n{arm}: {out['n_fits']} evals, {out['equiv']:.3f} equiv, "
          f"{wall:.1f}s wall, best {out['best']} MAE {out['best_mae']:.4f}")
    return out


results = {}

# --- Optuna GPSampler leg: run separately (see CH_optuna_gp_leg.py, executed
# with the psc-opt environment's Python, which has torch) and merged here ---
with open("california_housing_optuna_gp_result.json") as f:
    optuna_result = json.load(f)
optuna_result["tiers"] = {int(k): v for k, v in optuna_result["tiers"].items()}
results["Optuna GPSampler"] = optuna_result

# --- BayesHalvingSearchCV ---
search_bhs = BayesHalvingSearchCV(
    make_clf(), param_grid, scoring="neg_mean_absolute_error", cv=cv,
    n_jobs=-1, subsample="random", random_state=0, verbose=2,
)
t0 = time.time()
search_bhs.fit(X, y)
results["BayesHalvingSearchCV"] = _summarize("BayesHalvingSearchCV", search_bhs, time.time() - t0)

# --- PatternSearchCV, eager ---
search_psc_eager = PatternSearchCV(
    make_clf(), param_grid, scoring="neg_mean_absolute_error", cv=cv,
    n_jobs=-1, subsample="random", contraction="eager", random_state=0, verbose=2,
)
t0 = time.time()
search_psc_eager.fit(X, y)
results["PatternSearchCV (eager)"] = _summarize("PatternSearchCV (eager)", search_psc_eager, time.time() - t0)

# --- PatternSearchCV, patient ---
search_psc_patient = PatternSearchCV(
    make_clf(), param_grid, scoring="neg_mean_absolute_error", cv=cv,
    n_jobs=-1, subsample="random", contraction="patient", random_state=0, verbose=2,
)
t0 = time.time()
search_psc_patient.fit(X, y)
results["PatternSearchCV (patient)"] = _summarize("PatternSearchCV (patient)", search_psc_patient, time.time() - t0)

with open("california_housing_4way_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# --- comparison table ---
arms = list(results.values())
all_tiers = sorted(set().union(*(a["tiers"].keys() for a in arms)))
cols = [a["arm"] for a in arms]
print("\n" + "=" * 130)
print(f"{'':24s}" + "".join(f"{c:>26s}" for c in cols))
print(f"{'total evaluations':24s}" + "".join(f"{a['n_fits']:>26d}" for a in arms))
for n_rows in all_tiers:
    frac = n_rows / N
    print(f"{'fits @ ' + f'{frac:.4%}':24s}" + "".join(f"{a['tiers'].get(n_rows, 0):>26d}" for a in arms))
print(f"{'full-fit equivalents':24s}" + "".join(f"{a['equiv']:>26.3f}" for a in arms))
print(f"{'wall-clock (s)':24s}" + "".join(f"{a['wall']:>26.1f}" for a in arms))
print(f"{'best point':24s}" + "".join(
    f"{str((a['best']['max_features'], a['best']['n_estimators'], a['best']['max_depth'])):>26s}" for a in arms))
print(f"{'best CV MAE':24s}" + "".join(f"{a['best_mae']:>26.4f}" for a in arms))
