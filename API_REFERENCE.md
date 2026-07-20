# API Reference

Two scikit-learn-compatible hyperparameter search estimators, sharing one
**multi-fidelity "bullseye" data-growth mechanism** and one **scatter-search
multi-start layer**:

- [`PatternSearchCV`](#patternsearchcv) — classic Hooke-Jeeves pattern
  search (1961), adapted to grow its data budget as it converges.
- [`BayesHalvingSearchCV`](#bayeshalvingsearchcv) — a from-scratch Gaussian
  Process + Expected Improvement Bayesian search, on the exact same
  multi-fidelity infrastructure, with **zero additional dependencies**
  (no Optuna, no torch — just `numpy`, `scipy`, `scikit-learn`, already
  required by `PatternSearchCV`).

Both estimators start every search on a small, representative subsample of
the training data and only pay full price once their own search trajectory
shows them converging on an optimum — every reported result is still
confirmed on 100% of the data before it's trusted.

For the full design rationale behind every default, see
[`PatternSearchCV_SPEC.md`](PatternSearchCV_SPEC.md) and
[`BAYESHALVINGSearchCV_SPEC.md`](BAYESHALVINGSearchCV_SPEC.md). For a
generated, browsable version of this same reference (once the package is
ready to publish), see the Sphinx site in [`doc/`](doc/).

---

## Quick Start

```bash
pip install -e .
```

```python
from bayes_halving_search_cv import PatternSearchCV
from sklearn.model_selection import TimeSeriesSplit

search = PatternSearchCV(
    estimator,
    {"max_depth": [3, 5, 7, 9, 12, 16], "min_samples_leaf": [1, 2, 4, 8]},
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_mean_absolute_error",
    random_state=0,
)
search.fit(X, y)
search.best_params_
```

The examples below cover the parts of the API that aren't self-explanatory
from a single call — how to specify a search space, how to specify a data
ladder, and the two most consequential knobs each estimator exposes
(`contraction` and `subsample`).

### 1. Specifying `param_grid` — explicit lists vs. `(low, high, num)` tuples

`param_grid` is a plain dict. Each value can be **either** an explicit list
of values, **or** a `(low, high, num)` tuple that gets expanded into an
evenly-spaced grid (like `numpy.linspace`) — and you can freely mix both
forms in the same grid:

```python
# Form 1: explicit lists - use when you know exactly which values matter
param_grid = {
    "max_features": [2, 3, 4],
    "criterion": ["squared_error", "absolute_error"],
}

# Form 2: (low, high, num) tuples - use for a regular sweep across a range
param_grid = {
    "n_estimators": (10, 260, 26),   # -> 10, 20, 30, ..., 260 (26 values)
    "max_depth": (5, 17, 13),        # -> 5, 6, 7, ..., 17 (13 values)
}

# Both forms together, in one grid:
param_grid = {
    "max_features": [2, 3, 4],        # explicit list
    "n_estimators": (10, 260, 26),    # tuple spec
    "max_depth": (5, 17, 13),         # tuple spec
}

search = PatternSearchCV(estimator, param_grid, cv=cv, scoring=scoring)
```

Integer-endpoint tuples (like the two above) produce an integer grid
automatically; float endpoints produce a float grid. Both estimators build
their search space from `param_grid` in exactly the same way — this is not
something you configure per-estimator.

### 2. Specifying the data ladder (`data_zones`) — a list vs. a single number

`data_zones` controls the multi-fidelity ladder: how many rows each rung of
the search sees before the final, mandatory full-data confirmation.

```python
# Form 1: explicit ascending fractions, ending at 1.0 (both estimators' default)
search = PatternSearchCV(estimator, param_grid, cv=cv, scoring=scoring,
                         data_zones=(0.005, 0.01, 0.1, 1.0))   # 0.5%, 1%, 10%, 100%

# Form 2: a single int n -> n evenly divided levels
search = PatternSearchCV(estimator, param_grid, cv=cv, scoring=scoring,
                         data_zones=4)   # -> [0.25, 0.5, 0.75, 1.0]

# data_zones=1 disables multi-fidelity entirely - every evaluation uses 100% of the data
search = PatternSearchCV(estimator, param_grid, cv=cv, scoring=scoring,
                         data_zones=1)
```

In practice, the intermediate rungs are a **safety net**, not the primary
cost saver — on well-behaved searches the algorithm typically jumps straight
from the cheapest rung to the full-data confirmation once its mesh has
converged, skipping the middle entirely (this is measured behavior, not a
guess — see `EXPERIMENTS.md`, Experiments 13-17). The bulk of the savings
comes from how cheap the *first* rung is, not from how many rungs exist
after it.

### 3. `contraction="patient"` vs. `contraction="eager"` (`PatternSearchCV` only)

`contraction` controls *when* the search mesh shrinks — i.e. how quickly the
algorithm narrows in on a candidate before either finding a better one or
giving up and demanding more data.

```python
# default: "patient" - classic Hooke-Jeeves. The mesh only contracts after
# a full exploratory sweep fails to find anything better.
search = PatternSearchCV(estimator, param_grid, cv=cv, scoring=scoring)

# "eager" - prototype-faithful. A failed PATTERN move ALSO contracts the
# mesh (not just a failed sweep), so step resolution gets spent faster.
search = PatternSearchCV(estimator, param_grid, cv=cv, scoring=scoring,
                         contraction="eager", n_starts=4)
```

**The theoretical trade-off**: `"eager"` spends its step resolution faster,
which *in principle* could mean fewer evaluations to converge — but shrinking
the mesh faster also means committing to a smaller neighborhood sooner,
which is the classic premature-convergence risk on landscapes with more than
one local optimum (the mesh can narrow past a promising region before ever
sampling it).

**What we actually measured**: across five controlled head-to-head rounds
on this project's benchmark (`EXPERIMENTS.md`, Experiments 7-11), `"patient"`
and `"eager"` were tied on every cost metric that matters — identical
evaluation counts, identical full-fit equivalents, identical best point and
score, every single round. Wall-clock bounced both directions within this
machine's noise floor with no consistent winner. **There has never been a
measured advantage to `"eager"` in this project.** That's why `"patient"` is
the default — it carries eager's theoretical risk at zero measured benefit,
so far. If you use `"eager"` anyway (e.g. because your landscape might
behave differently from ours), pair it with `n_starts > 1` to hedge the
premature-convergence risk with independent starts.

### 4. `subsample="random"` vs. `subsample="stratified"` — and why this matters for time series

This is the parameter most likely to surprise you, so read this one
carefully.

`subsample` controls **which rows** get selected for each cheap-data rung of
the ladder — it operates entirely on the **feature matrix `X`**, and has
**nothing to do with the target `y`**. This is a different kind of
"stratified" than scikit-learn's `StratifiedKFold`, which balances class
labels in `y`; here, "stratified" means the row-selection order is built by
watching for **transitions in feature values** — the opposite of scikit-learn's usual meaning, so don't assume the two behave alike.

```python
from bayes_halving_search_cv import PatternSearchCV
from sklearn.model_selection import TimeSeriesSplit

# subsample="random": a uniform random sample of rows, independent of any
# feature pattern.
search_random = PatternSearchCV(
    estimator, param_grid, cv=TimeSeriesSplit(5), scoring=scoring,
    subsample="random", random_state=0,
)

# subsample="stratified": rows are prioritized by detecting genuine
# transitions in a small set of low-cardinality, behaviorally meaningful
# feature columns - e.g. for the retail dataset this project was built on,
# watching just the daily categorical flags:
search_stratified = PatternSearchCV(
    estimator, param_grid, cv=TimeSeriesSplit(5), scoring=scoring,
    subsample="stratified",
    subsample_columns=[3, 4, 5],   # column indices into X, e.g. HasPromotions/IsHoliday/IsOpen
    random_state=0,
)
```

**Why `"random"` is risky on a time-series dataset**: even though the rows
`ZoneSplitter` selects get sorted back into their original time order before
`TimeSeriesSplit` ever sees them (so fold boundaries stay temporally valid),
the *subset itself* is a random scatter across the whole timeline rather
than a realistic "what if we only had early data" sample. That means a
cheap, small-fraction evaluation under `"random"` doesn't faithfully
simulate limited historical data the way a genuinely time-ordered or
transition-aware sample does — it's thinning the *full* timeline, which is a
meaningfully different (and, in the worst case, misleading) condition from
what an early-stage deployment would actually see.

**Why we built `"stratified"` instead, and what it guarantees**: on the
dataset this project was benchmarked against, a side-analysis measured that
at very small fractions (0.1% of rows), a random sample only covered ~302 of
601 distinct stores on average — while the stratified sampler's
low-discrepancy design *guaranteed* 415 of 601, by construction, not by
luck. That gap is the entire reason this sampler exists: at small enough
fractions, random sampling can miss whole categories of your data purely by
chance, and a search that only sees a fraction of the real behavioral
diversity in the data can converge on hyperparameters that look great on
that narrow slice and don't hold up once the full data confirms them (see
`EXPERIMENTS.md`'s "Why stratified sampling has actually been winning" and
Experiment 15 for a real, measured case of exactly this happening).

**NOTE**: the transition-detection logic has a different behavior if `subsample_columns` points at repeating, low-cardinality columns or at columns that have changing values in every row. 
If your data has any near-continuous columns (like raw sensor or weather readings) that make
almost every row look unique, watching *every* column (the default when
`subsample_columns` is left unset) will change the behavior of the sampler
to selecting evenly-spread data points over the entire dataset — this will
create a "mini" dataset that is evenly spread over the entire population. That fallback is still a
principled, low-discrepancy sample (not a broken one), but it isn't the
transition-aware sample the sampler is capable of — see Experiment 17 for a
real before/after case where narrowing `subsample_columns` to just the
meaningful flags slightly improved search outcomes for the Bayesian search. `subsample_columns`
is provided specifically so you can point the sampler at the columns that
actually carry repeatable structure and unlock the feature value aware transition behavior
(picks the row when the value of a feature changes from let's say 'A' to 'B').

---

## `PatternSearchCV`

```
class bayes_halving_search_cv.PatternSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, random_state=None, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False, poll='auto', mesh_expansion=1.0, contraction='patient', data_zones=(0.005, 0.01, 0.1, 1.0), warmup=3, subsample='auto', subsample_columns=None, n_starts=1, start_points=None)
```

Hooke-Jeeves pattern search over a discrete hyperparameter grid.

### Parameters

**estimator** : *estimator object*
> The estimator to tune.

**param_grid** : *dict*
> Maps parameter names to either an explicit list of values or a
> `(low, high, num)` tuple expanded to a linspace grid. See
> [example 1](#1-specifying-param_grid--explicit-lists-vs-low-high-num-tuples)
> above.

**scoring** : *str, callable, or None, default=None*
> Passed straight through to `BaseSearchCV`; same semantics as
> `GridSearchCV`.

**n_jobs** : *int, default=None*
> Passed straight through to `BaseSearchCV`; same semantics as
> `GridSearchCV`. If you want to speed up execution, the best place to do
> that is your estimator's own options: set `n_jobs=1` here and enable GPU
> (or its own internal parallelism) in the estimator's settings instead. We
> looked at building a GPU-enabled version of the search itself, but since
> the search logic runs so fast on its own, there was no benefit to it — the
> time that matters is inside your estimator's `fit()`, so that's where
> acceleration belongs.

**refit** : *bool or str, default=True*
> Passed straight through to `BaseSearchCV`; same semantics as
> `GridSearchCV`.

**cv** : *cross-validation generator, default=None*
> Passed straight through to `BaseSearchCV`; same semantics as
> `GridSearchCV`. Determines whether `subsample="auto"` resolves to
> `"stratified"` (time-ordered splitters, e.g. `TimeSeriesSplit`) or
> `"random"` (everything else).

**verbose** : *int, default=0*
> `0`: silent. `1`: attaches a `StreamHandler` at `INFO` and narrates every
> search decision as it happens (moves, contractions, ring crossings, data
> climbs, merges); also logs a full `cross_validate` pass on the winning
> parameters at the end of `fit` (adds `n_splits` extra fits — skipped
> entirely at `verbose=0`, so it never costs anything unless requested).
> `2`: additionally logs per-probe debug detail, and cascades into
> scikit-learn's own native per-fold `[CV] END ...` printing (since
> `verbose` is passed through to `BaseSearchCV`).

**random_state** : *int, RandomState instance, or None, default=None*
> Controls every source of randomness in one place: the scatter-search
> start selection and (for `subsample="random"`) the row sample. Two fits
> with the same `random_state` produce identical `cv_results_`.

**pre_dispatch, error_score, return_train_score**
> Passed straight through to `BaseSearchCV`; same semantics as
> `GridSearchCV`.

**poll** : *{"auto", "complete", "opportunistic"}, default="auto"*
> Exploratory sweep mode. `"complete"` evaluates all +/-delta probes around
> the fixed center in one parallel batch (MATLAB `UseCompletePoll`) plus the
> composite of improving dimensions; `"opportunistic"` is the classic 1961
> sequential sweep with immediate acceptance. `"auto"` picks `"complete"`
> when `n_jobs / n_splits >= 2`, else `"opportunistic"` — on every machine
> this package has been benchmarked on so far (5-fold CV, ≤8 cores) that
> resolves to `"opportunistic"`, so `"complete"` poll has not actually been
> measured, but `"auto"` keeps the adaptivity for users with many more
> cores than CV folds instead of hardcoding a choice that was never tested
> against.

**mesh_expansion** : *float, default=1.0*
> Step-size multiplier applied after a successful sweep. `1.0` (default) is
> classic Hooke-Jeeves (contraction only); `2.0` matches MATLAB GPS. Raise
> it on fine, continuous-like grids.

**contraction** : *{"patient", "eager"}, default="patient"*
> When the mesh contracts. See
> [example 3](#3-contractionpatient-vs-contractioneager-patternsearchcv-only)
> above for the full trade-off. Summary: tied on every measured cost metric
> across five controlled rounds; `"patient"` is default because it carries
> none of `"eager"`'s theoretical premature-convergence risk for zero
> measured cost. Pair `"eager"` with `n_starts > 1` if you use it.

**data_zones** : *int or sequence of float, default=(0.005, 0.01, 0.1, 1.0)*
> The data ladder. See
> [example 2](#2-specifying-the-data-ladder-data_zones--a-list-vs-a-single-number)
> above. This aggressive 0.5%-start ladder was set as the default after five
> successive halvings of the starting zone, each matching or beating the one
> before it on this project's benchmark. **Caveat**: evidence is from one
> dataset/grid; this dataset's favorable behavior at small fractions is
> partly attributable to its own row structure, not necessarily a universal
> property of aggressive subsampling. The resource floor
> (`min_rows = max(2*(n_splits+1), 8)`) protects small datasets from an
> unreasonably tiny first rung regardless.

**warmup** : *int, default=3*
> Number of positions (starting point included) before the bullseye rings
> self-calibrate. The patience dial: higher = data is added closer to the
> optimum. Minimum 3 (two displacement readings).

**subsample** : *{"auto", "expanding", "stratified", "random"}, default="auto"*
> How the data-zone priority ordering is built. See
> [example 4](#4-subsamplerandom-vs-subsamplestratified--and-why-this-matters-for-time-series)
> above for the full explanation. `"auto"` picks `"stratified"` for
> time-ordered splitters, else `"random"`. `"expanding"` (oldest rows
> first) remains available explicitly but is no longer the time-series
> default — `"stratified"` measurably beat it (lower MAE, less compute,
> faster wall-clock) and degrades safely to systematic sampling in the
> worst case.

**subsample_columns** : *sequence of int, optional*
> Column subset watched by the `"stratified"` transition sampler. See the
> "one more thing worth knowing" note under
> [example 4](#4-subsamplerandom-vs-subsamplestratified--and-why-this-matters-for-time-series).
> Leaving this unset watches every column, which can silently degrade the
> sampler if any columns are near-continuous.

**n_starts** : *int, default=1*
> Independent climbers. Starts are chosen by scatter search (QMC pool +
> greedy maximin); every climber runs to completion (no elimination) and
> the best full-data optimum wins.

**start_points** : *list of dict, optional*
> Explicit start points (parameter dicts); they take seats before
> scatter-search generation fills the rest.

### Attributes

Standard `GridSearchCV`-style attributes (`best_params_`, `best_score_`,
`best_estimator_`, `cv_results_`, `best_index_`, `n_splits_`, `refit_time_`,
`scorer_`) plus:

**cv_results_["n_resources"]**
> How many rows each row of `cv_results_` was actually evaluated on —
> `best_*` is always chosen only from rows where this equals the full
> dataset size.

**local_optima_** : *list of dict*
> One entry per distinct converged start (deduped by final position, best
> score first): `{"params", "score", "n_starts_converged", "start_points"}`.

**search_history_** : *list of dict*
> Every confirmed-improving move across every start: climber index, params,
> data fraction, score, and (for the final position) convergence status.

**n_cache_hits_** : *int*
> Count of proposals served from the shared dedup cache instead of a
> genuine model fit — this is where the multi-start layer's savings
> actually come from.

---

## `BayesHalvingSearchCV`

```
class bayes_halving_search_cv.BayesHalvingSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, random_state=None, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False, n_iter=25, promote_k=3, data_zones=(0.005, 0.01, 0.1, 1.0), warmup=3, subsample='auto', subsample_columns=None, n_starts=1, start_points=None)
```

Gaussian-Process Bayesian search over a discrete hyperparameter grid, with
bullseye multi-fidelity data growth and scatter-search multi-start.

Despite the name, the fidelity mechanism is this package's self-calibrating
bullseye-ring data growth (see `PatternSearchCV` above), **not** classic
successive halving — the name was chosen by the project's author and kept
as-is.

`param_grid` and multi-start (`n_starts`/`start_points`) follow the *exact
same standard* as `PatternSearchCV` — see its section above for the
search-space and scatter-search conventions; they are not duplicated here.
This estimator has **zero additional dependencies** beyond
`PatternSearchCV`'s own (`numpy`, `scipy`, `scikit-learn` — the GP surrogate
is `sklearn.gaussian_process.GaussianProcessRegressor` plus a hand-rolled
Expected Improvement acquisition; no Optuna, no torch).

### Parameters

**estimator** : *estimator object*
> The estimator to tune.

**param_grid** : *dict*
> Identical standard to `PatternSearchCV.param_grid` — see
> [example 1](#1-specifying-param_grid--explicit-lists-vs-low-high-num-tuples)
> above.

**scoring, n_jobs, refit, cv, pre_dispatch, error_score, return_train_score**
> Passed straight through to `BaseSearchCV`; same semantics as
> `GridSearchCV` and identical to `PatternSearchCV`'s own — see
> `PatternSearchCV`'s `n_jobs` entry above for where to look for speedups
> (your estimator's own settings).

**verbose** : *int, default=0*
> `0`: silent. `1`: narrates every search decision as it happens
> (proposals, ring crossings, data climbs, final polish), and logs the same
> end-of-fit `cross_validate` summary as `PatternSearchCV`. `2`:
> additionally logs per-proposal debug detail, and cascades into
> scikit-learn's own native per-fold printing, same as `PatternSearchCV`.

**random_state** : *int, RandomState instance, or None, default=None*
> Controls every source of randomness in one place, in a fixed, documented
> order: the scatter-search start selection, then a per-`(start, zone)` GP
> seed drawn for every fresh Gaussian Process, in start-then-zone order.
> Two fits with the same `random_state` (including with `n_starts > 1`)
> produce identical `cv_results_`.

**n_iter** : *int, default=25*
> Budget of genuine (non-cache-served) model evaluations **per start**,
> across all data zones combined, excluding the bounded final-polish
> re-scores (at most `promote_k + 1` extra evaluations per start). This is
> the direct analog of "how long do we search" — there's no early-stopping
> criterion the way `PatternSearchCV`'s mesh convergence gives it one, so
> this budget is what bounds the search instead.

**promote_k** : *int, default=3*
> Number of top-scored configurations re-scored (and used to seed a fresh
> Gaussian Process) whenever the bullseye rings climb to a new data zone,
> and again for the mandatory final polish at full data. This is a hard
> ceiling on how much full-data cost the final confirmation step can ever
> cost — unlike `PatternSearchCV`, which keeps re-sweeping at whatever
> fraction it's on until 3 consecutive sweeps fail, potentially costing more
> full-price evaluations if it hasn't fully converged by the time it
> reaches 100% data (see `EXPERIMENTS.md`, Experiment 13's follow-up, for a
> real measured comparison of this exact difference).

**data_zones** : *int or sequence of float, default=(0.005, 0.01, 0.1, 1.0)*
> The data ladder. Identical semantics to `PatternSearchCV.data_zones` — see
> [example 2](#2-specifying-the-data-ladder-data_zones--a-list-vs-a-single-number)
> above.

**warmup** : *int, default=3*
> Positions (starting point included) before the bullseye rings
> self-calibrate. Identical semantics to `PatternSearchCV.warmup`. Note:
> this is a *different* readiness concept from the GP's own cold-start rule
> (which always needs exactly 2 scored observations before trusting the
> surrogate, regardless of `warmup`) — the two are deliberately decoupled.

**subsample** : *{"auto", "expanding", "stratified", "random"}, default="auto"*
> Identical semantics to `PatternSearchCV.subsample` — see
> [example 4](#4-subsamplerandom-vs-subsamplestratified--and-why-this-matters-for-time-series)
> above.

**subsample_columns** : *sequence of int, optional*
> Column subset watched by the `"stratified"` transition sampler. Identical
> semantics to `PatternSearchCV.subsample_columns`.

**n_starts** : *int, default=1*
> Independent Bayesian searches. Starts are chosen by the *same*
> scatter-search mechanism as `PatternSearchCV` (QMC pool + greedy maximin);
> every start runs to completion (no elimination between starts) and the
> best full-data optimum wins. **Unlike `PatternSearchCV`**, there is no
> state-match merging between starts (no clean analog for a stochastic
> search) — the shared dedup cache is the cost-saving mechanism instead.

**start_points** : *list of dict, optional*
> Explicit start points (parameter dicts); they take seats before
> scatter-search generation fills the rest. Identical semantics to
> `PatternSearchCV.start_points`.

### Attributes

Same standard `GridSearchCV`-style surface as `PatternSearchCV`
(`best_params_`, `best_score_`, `best_estimator_`, `cv_results_` with an
`"n_resources"` key, `best_index_`, `n_splits_`, `refit_time_`,
`scorer_`), plus the identically-shaped `local_optima_`, `search_history_`,
and `n_cache_hits_` (though `search_history_`'s entries additionally carry
a `"trial"` number and an `"event"` tag — `"trial"`, `"climb-rescore"`, or
`"final-polish"` — since every genuine evaluation is logged, not just
confirmed-improving ones).

---

## What's shared vs. what's different

| | `PatternSearchCV` | `BayesHalvingSearchCV` |
|---|---|---|
| Search space (`param_grid`) | Same standard | Same standard |
| Multi-start mechanism | Same scatter-search | Same scatter-search |
| Data ladder (`data_zones`) | Same semantics | Same semantics |
| Row-sampling (`subsample`) | Same semantics | Same semantics |
| Search algorithm | Hooke-Jeeves pattern search | Gaussian Process + Expected Improvement |
| When to stop | Mesh contracts to floor + 3 consecutive failed sweeps | Fixed `n_iter` budget per start |
| Cost of final confirmation | Unbounded (re-sweeps until convergence, at full data price) | Bounded (`promote_k` + 1 evaluations, always) |
| Extra dependencies | None (`numpy`, `scipy`, `scikit-learn`) | None (same three) |
| `contraction` knob | Yes | N/A (no mesh) |
| Merging between starts | State-match merging | None (shared cache only) |
