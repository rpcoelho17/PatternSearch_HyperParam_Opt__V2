"""
California Housing — Optuna GPSampler leg only.
Run with the `psc-opt` environment's Python (has torch installed), since this
project's own .venv can't install torch due to a Windows MAX_PATH limit on
its deeply nested project path.
"""
import json
import time

import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score

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


def objective(trial):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", param_grid["n_estimators"]),
        "max_features": trial.suggest_categorical("max_features", param_grid["max_features"]),
        "max_depth": trial.suggest_categorical("max_depth", param_grid["max_depth"]),
    }
    clf = ExtraTreesRegressor(n_jobs=1, random_state=0, **params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    return -scores.mean()


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GPSampler(seed=0))
t0 = time.time()
study.optimize(objective, n_trials=15)
wall = time.time() - t0

result = {
    "arm": "Optuna GPSampler", "wall": wall, "n_fits": 15,
    "tiers": {N: 15}, "equiv": 15.0,
    "best": study.best_params, "best_mae": float(study.best_value),
}
print(f"\nOptuna GPSampler: 15 evals, 15.000 equiv, {wall:.1f}s wall, "
      f"best {study.best_params} MAE {study.best_value:.4f}")

with open("california_housing_optuna_gp_result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
