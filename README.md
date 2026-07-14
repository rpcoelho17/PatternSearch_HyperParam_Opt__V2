# pattern-search-cv

Hooke-Jeeves pattern search hyperparameter optimization for scikit-learn, with
**bullseye multi-fidelity data growth** (searches start on a small,
representative subsample and buy more data as their own movement shows them
approaching an optimum) and **scatter-search multi-start** (independent
climbers from maximally-spread start points, all run to completion, best
full-data optimum wins).

```python
from pattern_search_cv import PatternSearchCV

search = PatternSearchCV(
    estimator,
    {"max_depth": [3, 5, 7, 9, 12, 16], "min_samples_leaf": [1, 2, 4, 8]},
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error",
    n_starts=4,          # scatter-search multi-start
    subsample="stratified",  # transition sampling for time-series data
    random_state=0,
)
search.fit(X, y)
search.best_params_      # chosen ONLY from full-data evaluations
search.local_optima_     # the map: every distinct optimum found
```

Design specification: see `PatternSearchCV_SPEC.md`.

## Development

```
python -m venv .venv
.venv/Scripts/pip install -e .[test]
.venv/Scripts/python -m pytest
```

Logging: the package logs every algorithmic decision (moves, contractions,
ring calibrations and crossings, data climbs, merges, cache statistics) to the
`pattern_search_cv` logger. `verbose=1` attaches a stream handler at INFO,
`verbose=2` at DEBUG.
