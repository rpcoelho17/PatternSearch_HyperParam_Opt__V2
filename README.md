# BayesHalvingSearchCV & PatternSearchCV

During my post-graduate studies in Data Science, I had to hyperparameter-tune
models on genuinely large datasets, and I went looking for a way to find good
hyperparameters quickly instead of paying full price for every trial.
Inspired by MATLAB's Global Optimization Toolbox pattern search for discrete
problems, I built a first version of what became `PatternSearchCV` — and even
that first version already beat Bayesian search on real benchmarks, simply
because it deduplicated repeated evaluations, something the Bayesian search
tools I compared it against didn't do; that alone saved real fits on large
datasets. I later added a "halving" mechanism on top of that base pattern
search — growing how much data each evaluation sees as the search's own
trajectory shows it converging (measured by the shrinking distance between
successive improving moves), instead of committing to full data from the
start — and that further improved the gains over the deduplication-only
version. When I eventually benchmarked properly against Optuna, the picture
became precise: Optuna's Bayesian search found the optimum in fewer raw
trials, but every one of those trials evaluated 100% of the data — my
advantage was coming from deduplication, the multi-fidelity data growth, and
the halving logic together, not from having fewer trials. So I adapted that
same multi-fidelity infrastructure onto a from-scratch Bayesian search of my
own, giving `BayesHalvingSearchCV` Optuna-quality trial efficiency without
paying full price for every trial.

**Both estimators implement scikit-learn's standard search-CV interface and
work with any scikit-learn-compatible estimator** — they drop into existing
pipelines exactly the way `GridSearchCV` or `RandomizedSearchCV` do. Our own
benchmarks use `ExtraTreesRegressor` specifically because it's fast and
performs well on the dataset we tested against, not because either algorithm
is limited to it.

**Best measured results** (523K-row retail regression benchmark; full
methodology and every run logged in [`EXPERIMENTS.md`](EXPERIMENTS.md)):
`BayesHalvingSearchCV` reached the benchmark's historical optimum at
**3.125 full-fit equivalents** (Experiment 17); `PatternSearchCV` reached the
same optimum at **5.04 full-fit equivalents** (Experiment 11). For
comparison, Optuna's own GP-based sampler needed **15.00 full-fit
equivalents** — every one of its trials at 100% of the data — to reach a
comparable answer. That's a **4.8× reduction** in compute for
`BayesHalvingSearchCV` and a **2.98× reduction** for `PatternSearchCV`,
relative to Optuna, for the same answer.

GPU acceleration is available the same way it is for any scikit-learn
estimator: enable it in `estimator`'s own settings. See
[`API_REFERENCE.md`](API_REFERENCE.md) for the full explanation.

Both estimators share one **multi-fidelity "bullseye" data-growth mechanism**
and one **scatter-search multi-start layer**:

- `PatternSearchCV` — classic Hooke-Jeeves pattern search (1961), adapted to
  grow its data budget as it converges.
- `BayesHalvingSearchCV` — a from-scratch Gaussian Process + Expected
  Improvement Bayesian search, on the exact same multi-fidelity
  infrastructure, with **zero additional dependencies** (no Optuna, no
  torch — just `numpy`, `scipy`, `scikit-learn`, already required by
  `PatternSearchCV`).

Both estimators start every search on a small, representative subsample of
the training data and only pay full price once their own search trajectory
shows them converging on an optimum — every reported result is still
confirmed on 100% of the data before it's trusted.

## Quick start

```python
from pattern_search_cv import PatternSearchCV, BayesHalvingSearchCV
from sklearn.model_selection import TimeSeriesSplit

param_grid = {"max_depth": [3, 5, 7, 9, 12, 16], "min_samples_leaf": [1, 2, 4, 8]}

search = PatternSearchCV(
    estimator, param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error",
    n_starts=4,               # scatter-search multi-start
    subsample="stratified",   # transition sampling for time-series data
    random_state=0,
)
search.fit(X, y)
search.best_params_       # chosen ONLY from full-data evaluations
search.local_optima_      # the map: every distinct optimum found

# or the Bayesian search, on the exact same multi-fidelity infrastructure:
search = BayesHalvingSearchCV(estimator, param_grid, cv=TimeSeriesSplit(5),
                              scoring="neg_mean_absolute_error", random_state=0)
search.fit(X, y)
```

For the full parameter reference and worked examples — how to specify a
search space, the data ladder, `contraction="eager"`'s cost/risk trade-off,
and why `subsample="stratified"` matters for time series — see
[`API_REFERENCE.md`](API_REFERENCE.md). For the full design rationale behind
every default, see [`PatternSearchCV_SPEC.md`](PatternSearchCV_SPEC.md) and
[`BAYESHALVINGSearchCV_SPEC.md`](BAYESHALVINGSearchCV_SPEC.md).

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
