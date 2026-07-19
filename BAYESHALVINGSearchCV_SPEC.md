# BayesHalvingSearchCV — Spec / Architectural Document

**Audience:** Claude Sonnet 5, implementing and testing this estimator inside the
existing `pattern-search-cv` repository.
**Status:** agreed design, ready for implementation. Revision 3 (2026-07-15):
dropped Optuna and torch as runtime dependencies entirely — the GP surrogate is
built on `sklearn.gaussian_process.GaussianProcessRegressor` (already a required
dependency) plus a hand-rolled Expected Improvement acquisition function. Added a
dev-time-only validation step comparing our GP proposer's paths and results
against Optuna's `GPSampler` on both a synthetic function and the real benchmark
grid. Multi-start, shared search-space standard, and same-package reuse from
Revision 2 are unchanged.
**Author of record:** design session 2026-07-15 (user + Claude Fable 5).

---

## 0. Mission

`Optuna GPSampler` (all trials at 100% data) found this project's benchmark optimum
in 15 trials — fewer *evaluations* than `PatternSearchCV` — but paid full price for
every one of them: **15.00 full-fit equivalents vs PatternSearchCV's ~5.04–5.85**
(see `EXPERIMENTS.md`, Experiment 4 vs Experiments 7–11). Its weakness is not the
search logic; it is that it has no multi-fidelity machinery.

**Build `BayesHalvingSearchCV`: a scikit-learn-compatible estimator, shipped in the
SAME pip package as `PatternSearchCV`, using the SAME environment (no new runtime
dependencies), that keeps this package's entire multi-fidelity infrastructure —
stratified priority ordering, data zones, the bullseye ring methodology, the shared
dedup cache, the full-data-only best selection, the scatter-search multi-start
layer — but replaces the Hooke-Jeeves search logic with Bayesian optimization: a
Gaussian Process surrogate (via scikit-learn) driven by Expected Improvement.** It
must use the identical `param_grid` search-space standard as `PatternSearchCV`
(same `Space` class, no parallel space abstraction) and the identical multi-start
mechanism (`n_starts`, `start_points`, scatter-search selection).

Before benchmarking it against `PatternSearchCV`, **validate the from-scratch GP
proposer's correctness against Optuna's `GPSampler`** (a trusted, widely-used
reference implementation) — on a synthetic function (isolated optimizer logic, no
model fits) and on the real benchmark grid at a fixed cheap data fraction (0.25%,
for a fast, apples-to-apples proposal-path comparison — not a rerun of Experiment
4's 100%-data configuration). This validation is dev-time-only: it needs Optuna
and torch as external reference tools, but neither becomes a dependency of the
shipped estimator. See §8.

Reference numbers the new estimator is trying to beat (official space:
`max_features` {2,3,4} × `n_estimators` {10..260 step 10} × `max_depth` {5..17},
523K-row retail dataset, `TimeSeriesSplit(5)`, MAE):

| | evaluations | full-fit equiv | best CV MAE | wall-clock |
|---|---|---|---|---|
| Optuna GP, all-100%-data (Exp. 4) | 15 | 15.00 | 805.730 at (4,150,17) | 964.6 s |
| PatternSearchCV, current defaults (Exp. 10) | 22 | 5.09 | 805.038 at (4,130,17) | 443.8 s |

Success looks like: GP-quality answers (≈805) at well under 15 equivalents, at
`n_starts=1` (the directly comparable configuration to the reference rows above).

**Naming note:** the user chose the name `BayesHalvingSearchCV`. The fidelity
mechanism is the bullseye-zones design, *not* classic successive halving — keep the
user's name; add one line in the docstring clarifying the mechanism.

---

## 1. Packaging and environment — ONE pip library, NO new runtime dependencies

**Hard requirement: both estimators ship in the same package, `pattern_search_cv`,
from the same `pyproject.toml`, installed by the same `pip install pattern-search-cv`.**
There is no separate package, no separate repo, no separate `src/` tree.

- New modules: `src/pattern_search_cv/_bayes.py` (the estimator) and
  `src/pattern_search_cv/_gp.py` (the `GPProposer` class, §5) — alongside the
  existing `_search.py`, `_climber.py`, `_engine.py`, `_sampling.py`, `_space.py`,
  `_fidelity.py`-to-be-added (§6).
- `src/pattern_search_cv/__init__.py`: add `BayesHalvingSearchCV` to the imports
  and `__all__`, next to `PatternSearchCV`, `Space`, `Dimension`.
- **`pyproject.toml`'s `dependencies` list does not change at all**: `numpy`,
  `scipy`, `scikit-learn`. `BayesHalvingSearchCV`'s GP mode is fully covered by
  these — `GaussianProcessRegressor` lives in `sklearn.gaussian_process`, the EI
  formula needs only `scipy.stats.norm`. **Do not add `optuna` or `torch` to
  `dependencies` or to any `[project.optional-dependencies]` group.** There is no
  `bayes` extras group in this revision — none is needed.
- Consequence: `BayesHalvingSearchCV` runs in the **exact same `.venv`** as
  `PatternSearchCV`, always, with no exceptions and no extras flags. Every test in
  §9 runs there, unconditionally — no `skipif torch missing` markers anywhere in
  the shipped test suite (a real simplification vs. Revision 1/2's design).

- Repo root: `C:\FILES\Code\Benchmarking\Working_on_Train_Set\V2025\pattern-search-cv`
  (git: `https://github.com/rpcoelho17/PatternSearch_HyperParam_Opt__V2`, branch `main`).
- Package venv: `.venv` inside the repo — Python 3.11, sklearn 1.9.0, numpy 2.4.6.
  This is the only environment needed to build, test, and use both estimators.
- Jupyter kernel `psc-venv` is registered for the package venv; benchmark notebooks
  are executed headlessly with
  `.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=psc-venv <nb>`.

### 1.1 The one place torch/Optuna are still used: dev-time validation only

`C:\FILES\Code\Benchmarking\psc-opt` is a **pre-existing, separate venv** (torch
2.5.1+cpu, optuna 4.9.0, sklearn 1.9.0 — already verified working) used **only**
to run the §8 validation script, which imports Optuna's `GPSampler` purely as an
external reference to compare our implementation against. `pattern_search_cv`
itself must be installed there too (`pip install -e .` from the repo, using that
venv's pip) so the validation script can import `GPProposer`/`BayesHalvingSearchCV`
normally — but this is a dev convenience for running the comparison, not a second
supported runtime path. **`BayesHalvingSearchCV` itself never requires `psc-opt`
or torch for anything** — every real use of it, including the main benchmark in
§10, runs in the plain package `.venv`.

If torch is ever needed again on this Windows box: the verified recipe is
`pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu`
from a SHORT filesystem path (two prior failures: nested license paths exceeding
Windows' 260-char limit under a deep venv path, and torch 2.13's `c10.dll` failing
to initialize regardless of path).

---

## 2. What to reuse (exact inventory — do not reimplement these)

All in `src/pattern_search_cv/`:

| Component | Location | What it gives you |
|---|---|---|
| `Space` | `_space.py` | grid dims from `param_grid` (lists or `(low,high,num)` tuples); integer index tuples; `params(idx)`/`indices(params)`; `distance(a,b)` = Euclidean over normalized numeric coords + Hamming for categoricals; `min_step`; `midpoint()` |
| `stratified_order(X, columns=None)` | `_sampling.py` | the priority ordering (novel-first + boundary/midpoint + bit-reversed thinning); falls back safely to Even Sampling (a bit-reversed full-timeline permutation) when every row is unique |
| `expanding_order`, `random_order` | `_sampling.py` | the other two orderings |
| `ZoneSplitter(base_cv, subset)` | `_sampling.py` | CV splitter over a sorted subset of rows, mapped back to original indices; same `n_splits` as base (required by `BaseSearchCV`) |
| `PatternSearchCV` | `_search.py` | the *reference implementation* for every sklearn-integration pattern below. Read it before writing code. |

From `PatternSearchCV` specifically, replicate these patterns (copy the approach,
not necessarily the code):

1. **`BaseSearchCV` subclass with `_run_search(self, evaluate_candidates, **kwargs)`.**
   `evaluate_candidates(params_list, cv=<ZoneSplitter or None>, more_results={...})`
   is the only way models get fitted. Its returned results dict is **cumulative**
   across calls — slice `results[score_key][-len(batch):]` for the new scores.
2. **`more_results={"n_resources": [rows]*len(batch)}` on EVERY call** (full-data
   calls included), or the `cv_results_` columns misalign.
3. **`_select_best_index` static-method override**: mask to rows where
   `n_resources == max(n_resources)`, nan-safe argmax of `mean_test_<metric>`.
   `best_*` must come ONLY from full-data evaluations.
4. **Validation/floor block in `_prepare_run`**: zones validation (int → even
   levels; list → ascending, in (0,1], ends at 1.0; `1` disables), resource floor
   `min_rows = max(2*(n_splits_guess+1), 8)` with rung merging + truncation log line.
   Copy this logic verbatim (or extract to a shared private helper both estimators
   call — preferred, same spirit as the `_select_starts` extraction in §3.1).
5. **Pickling rule (CRITICAL, cost us a real bug):** with `n_jobs>1`, sklearn
   pickles `self` for every parallel task. NOTHING unpicklable may sit on the
   instance during `fit`: no live `GaussianProcessRegressor` fit-in-progress state
   that isn't plain data, no logging handlers, no generators. (A *fitted*
   `GaussianProcessRegressor` object itself is picklable — sklearn estimators are
   — but keep it in local frames of `_run_search` regardless; don't stash it on
   `self`.) Stash only plain-data results into `self._ctx["results"]`;
   `self._ctx = None` in `fit`'s `finally`.
6. **Tag delegation** (`__sklearn_tags__`): copy PatternSearchCV's — `allow_nan`
   and `target_tags` from the sub-estimator (BaseSearchCV misses these).
7. **`y is None` guard and `n_samples < 1` guard** with the exact error-message
   styles used there (estimator checks grep for them).
8. **Verbose conventions**: logger `pattern_search_cv` with NullHandler default;
   `verbose>=1` attaches a StreamHandler and MUST print the header (optimizing
   metric via `_scoring_label()`-style resolution, cv class name, every dimension's
   values, the estimator's own knobs — for this estimator: `n_iter`, `promote_k`,
   `warmup`, zones, `n_starts`) — and log every fidelity decision and every
   start's activity as it happens (mirror `Climber`'s per-decision logging, one
   line per event, prefixed with the start index), plus the end-of-run
   `_log_cv_summary` (per-fold EV/MAE/MSE/RMSE/R2 + means, fit/score times — copy
   or share PatternSearchCV's implementation). `verbose=0` = silent, no extra fits.

### 2.1 The one permitted change to `PatternSearchCV`: extract `_select_starts`

The user requires *identical* multi-start methodology in both estimators. The
only correct way to guarantee that is one shared implementation, not two
independently-written copies that could silently drift.

**Required refactor (small, behavior-preserving, zero test changes):**

1. Move the logic currently in `PatternSearchCV._select_starts` (QMC candidate
   pool + greedy maximin selection + `start_points` seat priority + midpoint
   fallback — read the existing method in `_search.py` in full before touching
   anything) into a new **free function** in a new file, `src/pattern_search_cv/_starts.py`:
   ```python
   def select_starts(space, n_starts, start_points, rng):
       """Scatter-search start selection (MATLAB MultiStart-style): explicit
       start_points take seats first, then the grid midpoint, then QMC pool +
       greedy maximin fill. Returns a list of index tuples, length <= n_starts."""
   ```
   Same signature shape as the current method minus `self` (pass `space`,
   `self.n_starts`, `self.start_points`, `rng` explicitly).
2. `PatternSearchCV._select_starts` becomes a **one-line delegating wrapper**:
   `return select_starts(space, self.n_starts, self.start_points, rng)`. Do not
   remove or rename the method — existing call sites in `_search.py` and any
   test that might reference it stay untouched.
3. `BayesHalvingSearchCV` calls `select_starts(...)` directly (no wrapper needed
   since it's a new class).
4. **Gate**: run the full existing suite (`pytest tests -q`) immediately after
   this refactor, before writing one line of `BayesHalvingSearchCV`. All 105
   tests must still pass, unmodified. If anything changes, the refactor was not
   behavior-preserving — fix it, don't touch the tests.

---

## 3. Public API

```python
class BayesHalvingSearchCV(BaseSearchCV):
    _required_parameters = ["estimator", "param_grid"]

    def __init__(self, estimator, param_grid, *, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0, random_state=None,
                 pre_dispatch="2*n_jobs", error_score=np.nan,
                 return_train_score=False,
                 # --- Bayesian search ---
                 n_iter=25,               # PER-START budget of NEW model evaluations
                                          # (cache hits free) - see multi-start section
                 promote_k=3,             # top-k configs re-scored when a zone climbs
                 # --- multi-fidelity (identical semantics to PatternSearchCV) ---
                 data_zones=(0.005, 0.01, 0.1, 1.0),
                 warmup=3,
                 subsample="auto",        # "auto"|"expanding"|"stratified"|"random"
                 subsample_columns=None,
                 # --- multi-start (identical semantics to PatternSearchCV) ---
                 n_starts=1,
                 start_points=None):
```

Note vs. earlier revisions: **no `sampler` parameter.** Revision 1/2 exposed
`"gp"|"tpe"` specifically so the estimator stayed usable without torch. Since the
GP path itself no longer needs torch (§0, §5), there is nothing left for a "tpe
fallback" to be a fallback *from* — one search algorithm, one fewer parameter to
test, document, and explain.

**Hard rule, verified against `PatternSearchCV.__init__` (`_search.py` line 127):
`__init__` does NOTHING but call `super().__init__(...)` and assign every
remaining argument to `self.<name>` verbatim.** No validation, no `Space`
construction, no defaults resolution. All of that happens in `_prepare_run`,
called from `fit`, exactly where `PatternSearchCV` does it. This is required for
`clone()`, `get_params()`/`set_params()`, and `parametrize_with_checks` (test #16)
to pass — it is not a style preference.

### 3.1 Search space — mandatory: identical standard to `PatternSearchCV`

- `param_grid`: **exactly** the same accepted forms as `PatternSearchCV.param_grid`
  — dict mapping name → explicit list of values, or → `(low, high, num)` tuple
  expanded to a linspace. Both estimators build their space via the same call:
  `space = Space(self.param_grid)`. **No parallel search-space abstraction is
  permitted** — do not expose ConfigSpace, do not invent a second
  `Dimension`-like class. `Space`/`Dimension` (`_space.py`, unmodified) is the
  single search-space standard for the whole package, and it is also the ONLY
  geometry the GP proposer's kernel sees (§5) — reused, not paralleled.
- `space.distance(a, b)` (Euclidean over normalized numeric coords + Hamming for
  categoricals) is reused unmodified for the multi-start selection (§3.2) and the
  `BullseyeController`'s displacement readings (§6) — same geometry, same code,
  both estimators, and the same code that builds the GP's feature vectors (§5.1).

### 3.2 Multi-start — mandatory: identical mechanism and philosophy to `PatternSearchCV`

- `n_starts`, `start_points`: same parameters, same defaults, same semantics as
  `PatternSearchCV`. Starts are selected via the shared `select_starts(space,
  n_starts, start_points, rng)` function from §2.1 — **the exact same call
  PatternSearchCV makes**, not a reimplementation.
- **MultiStart philosophy carries over unchanged: no elimination.** Every one of
  the `n_starts` independent Bayesian searches runs to its own completion (its own
  `n_iter` budget — see §4.2), exactly mirroring "every climber runs to
  completion; best full-data optimum wins" (spec section 6.2 of
  `PatternSearchCV_SPEC.md`). Do not add ASHA-style culling between starts.
- **What does NOT carry over, and why**: `PatternSearchCV`'s state-match merging
  has no clean analog for a stochastic Bayesian search — there is no notion of two
  GP optimizers being in an "identical state." **The shared dedup cache is the
  cost-saving mechanism instead** (§4.2): redundant `(params, fraction)` proposals
  across different starts get served from cache for free. Do not invent a merging
  heuristic — explicitly out of scope.
- Fitted attribute `local_optima_`: list of dicts, one per **distinct** converged
  start result (dedup by final incumbent index tuple, best score first) — same
  shape/spirit as `PatternSearchCV.local_optima_`: `{"params", "score",
  "n_starts_converged", "start_points"}`. `n_starts=1` still populates this with
  one entry.

### 3.3 Everything else

- No new exposed hyperparameters beyond `n_iter`, `promote_k` for the search
  itself — the GP's kernel and acquisition settings are deliberate internal
  constants, not user-facing knobs (§5), matching this package's established
  policy of only promoting a setting to a public parameter once evidence
  justifies it (see `mesh_expansion`, `poll` in `PatternSearchCV_SPEC.md` for the
  precedent).
- `n_iter`: budget of *genuine fits* **per start**, across all zones combined for
  that start, excluding the final-polish re-scores (which are ≤ `promote_k`+1
  extra evaluations per start, reported in `cv_results_` like everything else).
  Cache-served proposals (whether from this start's own earlier history or from
  another start's history) don't consume budget; guard with `max_asks = 10 *
  n_iter` total proposal calls **per start** to make infinite dedup loops
  impossible (if hit, log a warning and proceed to that start's final polish).
- Fitted attributes: standard SearchCV surface + `n_resources` key in
  `cv_results_` + `local_optima_` (§3.2) + `search_history_` (list of plain
  dicts: **start index**, trial number, params, fraction, score, event tag e.g.
  "trial"/"climb-rescore"/"final-polish") + `n_cache_hits_`.
- Add the class to `__init__.py`'s `__all__` and exports (§1).
- Docstring: numpydoc, same caveat-forward style as `PatternSearchCV`'s (evidence
  provenance, name-vs-mechanism note from §0, and a note that `param_grid` and
  multi-start are governed by the same standard as `PatternSearchCV` — point the
  reader there rather than duplicating prose). Mention explicitly that this
  estimator has **zero additional dependencies** beyond `PatternSearchCV`'s own.

---

## 4. Search algorithm (normative)

### 4.1 GP proposal mode

No external library, no ask/tell session object, no nameservers. One
`GPProposer` instance (§5) per **(start, data zone)** pair — a fresh surrogate
model per fidelity level *within* each independent start, so scores from
different fractions are never mixed in one surrogate (the same
never-compare-across-fractions rule `PatternSearchCV` enforces), and each start's
search is fully independent of every other start's (per §3.2).

Suggestion space = **index space**, via `Space` (§3.1) — identical for every
start; only the *seeding* differs per start.

### 4.2 Outer loop: multi-start

```
space          = Space(param_grid)                              (§3.1)
rng            = check_random_state(random_state)
starts         = select_starts(space, n_starts, start_points, rng)   (§2.1, §3.2)
zones, sizes   = validated ladder + resource floor                (§2 item 4)
order          = priority ordering per `subsample`                (reuse; "auto":
                 stratified for TimeSeries* cv, else random — same rule as
                 PatternSearchCV)
splitters      = {frac: ZoneSplitter(cv, order[:size]) or None for full data}
cache          = {}   # (idx_tuple, frac) -> score   SHARED ACROSS ALL STARTS
n_cache_hits   = 0
per_start_results = []

for start_i, start_point in enumerate(starts):
    result = run_one_start(start_i, start_point, space, zones, sizes, splitters,
                            cache, rng, ...)   # §4.3, appends its cache hits to
                                               # n_cache_hits, uses/extends `cache`
    per_start_results.append(result)

local_optima_ = dedup(per_start_results by final incumbent idx, best score first)
best_start    = argmax(per_start_results, key=score at fraction 1.0)
# best_* attributes populated by the standard _select_best_index path (§2 item 3)
# operating over the UNION of all starts' cv_results_ rows, restricted to
# fraction==1.0 rows, exactly as it already does for a single search.
```

All of this lives in local frames of `_run_search` (§2 item 5) — the per-start
loop, the proposers, everything. Only plain-data results end up in anything
durable.

### 4.3 Inner loop: one start (per-start GP-EI search with bullseye fidelity)

```
controller = BullseyeController(space.min_step, n_boundaries=len(zones)-1,
                                warmup=warmup)      (§6; ONE controller per start,
                                                     independent state, mirrors
                                                     each Climber calibrating on
                                                     its own trajectory)
zone_i     = 0
incumbent, incumbent_score = None, None
fits_used, proposals = 0, 0

proposer = GPProposer(space, rng.randint(2**31-1))   # (§5) fresh per (start, zone)
proposer.observe(start_point, None)   # register the seed point as the FIRST
                                       # proposal (analogous to Optuna's
                                       # enqueue_trial) - the first call to
                                       # proposer.suggest() must return start_point
                                       # itself before any GP fitting happens

while fits_used < n_iter and proposals < max_asks:
    idx = proposer.suggest()          # §5: cold-start pick, or GP+EI argmax
    proposals += 1
    frac = zones[zone_i]
    if (idx, frac) in cache:
        proposer.observe(idx, cache[(idx, frac)])
        n_cache_hits += 1
        continue
    score = evaluate_batch(frac, [idx])[0]              (§2 items 1-2)
    fits_used += 1
    cache[(idx, frac)] = score
    proposer.observe(idx, score)

    if incumbent is None or score > incumbent_score:
        move = 0.0 if incumbent is None else space.distance(incumbent, idx)
        incumbent, incumbent_score = idx, score
        new_zone = controller.observe_improvement(move)   (§6)
        if new_zone > zone_i:
            zone_i = new_zone                              # RATCHET: never down
            top = proposer.top_k_observed(promote_k)        # by score, unique idx
            proposer = GPProposer(space, rng.randint(2**31-1))   # fresh per zone
            for cfg_idx in top:
                score_k = cache-or-evaluate(cfg_idx, zones[zone_i])  # counts fits
                cache it; proposer.observe(cfg_idx, score_k)
            incumbent = best of the re-scored top; incumbent_score = its NEW score
            # (never compare across fractions: incumbent_score is always at the
            #  current fraction)

# ---- forced final polish for THIS start (always) ----
if zones[zone_i] < 1.0:
    top = proposer.top_k_observed(promote_k)  (incumbent guaranteed included)
    for cfg_idx in top: cache-or-evaluate at frac 1.0   # "final-polish" events

return {"start_point": start_point, "incumbent": incumbent,
        "score": <incumbent's score AT FRACTION 1.0>, "history": [...]}
```

Notes:
- `proposer.observe(idx, score=None)` with `score=None` is how the seed point is
  registered as "must be proposed next" without yet having an observed score —
  `GPProposer.suggest()` must special-case this: if there is a pending
  unscored-seed observation, return it verbatim on the very next `suggest()` call
  (§5.2). This replaces Optuna's `enqueue_trial`.
- "fresh `GPProposer` per zone" replaces "fresh `Study` per zone" — same
  never-mix-fractions rule as before, just a different object doing the resetting.
- `top_k_observed`: this start's observed (idx, score) pairs at the zone it's
  leaving, sorted by score desc, dedup by idx, take k.

---

## 5. `GPProposer` — the from-scratch Bayesian search core

New file `src/pattern_search_cv/_gp.py`. This is the component §8 validates
against Optuna's `GPSampler`. Design it as a small, independently testable class
with **no dependency on `BaseSearchCV`, `evaluate_candidates`, or anything
sklearn-search-specific** — it only needs a `Space` and observed (index, score)
pairs. This independence is exactly what makes the synthetic-function validation
in §8.1 possible.

```python
class GPProposer:
    def __init__(self, space, random_state=None, xi=0.01):
        ...
    def observe(self, idx, score):
        """Record an observation. score=None marks a pending seed point that
        MUST be returned by the very next suggest() call."""
    def suggest(self):
        """Return the next index tuple to evaluate."""
    def top_k_observed(self, k):
        """This proposer's observed (idx, score) pairs, sorted desc by score,
        deduped by idx, top k."""
```

### 5.1 Feature representation (reuses `Space`, does not reinvent it)

For each observed/candidate index tuple, build a numeric feature vector:
numeric dimensions → their existing normalized coordinate (`Dimension.coord`-style,
same normalization `Space.distance` already uses); categorical dimensions →
one-hot encoded (`n` binary features for an `n`-value categorical dimension).
This is the same geometry `Space.distance`'s Hamming term is implicitly built on
— write a small `_space.py` or `_gp.py` helper, `_featurize(space, idx) -> np.ndarray`,
and use it consistently for every GP fit/predict call.

### 5.2 `suggest()` algorithm

1. **Pending seed**: if `observe()` was last called with `score=None` and no
   observation has superseded it yet, return that index immediately. (Handles
   the multi-start seed point, §4.3.)
2. **Cold start**: if fewer than 2 *scored* observations exist yet, propose an
   unobserved point via a small QMC/random candidate draw (reuse the same
   QMC-pool pattern `select_starts` uses, seeded from this proposer's own
   `random_state`) — a GP fit on 0–1 points has no meaningful variance estimate,
   so don't trust it yet. This cold-start length (2) is a fixed internal
   constant, **not the same as, and not coupled to, `warmup`** (which governs
   `BullseyeController`'s ring calibration, §6, a different readiness concept for
   a different mechanism) — keep them decoupled and say so in a code comment, to
   avoid a future maintainer assuming they're meant to be the same knob.
3. **GP + Expected Improvement** (≥2 scored observations):
   - Featurize all observed points (§5.1) → `X_train`; observed scores → `y_train`.
   - Fit `sklearn.gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=2.5),
     normalize_y=True, alpha=1e-6, n_restarts_optimizer=2,
     random_state=<this proposer's seed>)` on `(X_train, y_train)`. Matern 5/2 is
     the standard default kernel for Bayesian optimization in the literature —
     not an arbitrary choice.
   - Build the candidate set: every remaining (unobserved-by-this-proposer) grid
     point if the space is small enough to enumerate cheaply (a few thousand
     points — the official benchmark grid is 1,014 points, trivial), else a
     QMC-sampled candidate pool (same reused pattern, capped at e.g. 2,000).
   - `mu, sigma = gp.predict(featurize(candidates), return_std=True)`.
   - Expected Improvement (maximizing; sklearn scores are greater-is-better):
     `f_best = max(y_train)`; for each candidate,
     `z = (mu - f_best - xi) / sigma` (guard `sigma <= 0` → `EI = 0`),
     `EI = (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)` (`scipy.stats.norm`).
   - Return the candidate with max EI. Tie-break deterministically (e.g. lowest
     index tuple, lexicographic) — required for the determinism guarantee (§7).

### 5.3 Unit-testability (drives §8.1's design)

Because `GPProposer` takes no dependency on the estimator machinery, §8's
synthetic-function validation can drive it directly: call `.observe(idx, score)`
in a loop against a hand-computed objective and inspect `.suggest()`'s trajectory,
with zero model fitting involved.

---

## 6. `BullseyeController` — extract the fidelity methodology into a shared class

New file `src/pattern_search_cv/_fidelity.py`. Encapsulates EXACTLY the rules
currently embedded in `Climber._commit_move/_calibrate/_zone_for`
(`_climber.py` lines ~227–270 — read them first; they are the normative source):

- `warmup` counts **positions, starting point included** (so `warmup=3` = start +
  2 improvements = 2 readings). During warm-up: no data purchases.
- Readings = displacement (normalized `Space.distance`) between successive
  incumbent updates. **Zero displacement is not a reading** (for BO: the first
  observation initializes the incumbent with move=0 → position count starts at 1,
  no reading recorded).
- Calibration at warm-up end: `D = mean(readings)`, floored to a whole number of
  grid steps: `D = max(min_step, floor(D/min_step)*min_step)`.
- Boundaries: `n_b = len(zones)-1`; `b_k = max(min_step, D*(n_b-k)/n_b)` for
  k=1..n_b, descending.
- Zone for a move: innermost k whose `b_k >= move`; ratchet (zone index never
  decreases); before calibration completes, zone stays 0.
- API sketch: `observe_improvement(move) -> int` (returns the ratcheted target
  zone index), plus readonly `D`, `boundaries`, `n_positions` for logging/tests.
- **One `BullseyeController` instance per start** (§4.3) — independent
  calibration per start, exactly mirroring how each `Climber` calibrates on its
  own trajectory in `PatternSearchCV`. Explicitly decoupled from `GPProposer`'s
  own cold-start logic (§5.2) — different mechanisms, coincidentally similar
  "don't trust the model yet" idea, deliberately not the same knob.

**Scope control: do NOT refactor `Climber` to use this class in this task.** Add a
one-line comment in `_climber.py` noting the future unification. Keeping the
proven PatternSearchCV path untouched is deliberate — its behavior is pinned by
trace tests and benchmark history.

Unit-test the controller standalone against hand-computed sequences (see §9).

---

## 7. Determinism

Given `random_state`: one `check_random_state(...)` rng created in `_prepare_run`,
used for (in order, so the sequence is reproducible): (a) `select_starts`'s QMC
seed, (b) a per-(start, zone-proposer) `GPProposer` seed drawn from that same rng
for every fresh `GPProposer(...)` construction across every start, in
start-then-zone order (this seed drives both the cold-start QMC draws and the
`GaussianProcessRegressor`'s `random_state` for its internal optimizer restarts).
Two fits with the same `random_state` (including with `n_starts>1`) must produce
identical `cv_results_` and identical `local_optima_`. No torch/Optuna involved
anywhere in this — test it directly, unconditionally, in the plain package venv.

---

## 8. Validation: compare `GPProposer` against Optuna's `GPSampler`

**Purpose**: we are replacing a mature, widely-used library's GP implementation
with a from-scratch one. Before trusting it for the real benchmark (§10), verify
it behaves like a correct Bayesian optimizer should, using Optuna's `GPSampler` as
an external reference — not because the two must match exactly (they won't:
different kernels, different acquisition details, different candidate-optimization
strategy, different RNG internals all make bit-identical trajectories unrealistic
and not a meaningful bar), but to catch a *qualitatively* broken implementation
(sign errors, an acquisition function that never explores, a kernel that never
fits, systematically worse results) before it contaminates the main benchmark.

**This validation is dev-time-only.** It needs Optuna and torch as an external
reference (§1.1) and is **not** part of the pytest suite in §9 — it does not run
in CI, it does not gate merges, and neither Optuna nor torch may leak into the
package's dependencies because of it. Build it as a notebook,
`GP_Validation_vs_Optuna.ipynb`, in the repo root (matching this project's
existing convention — see `Optuna_Baseline.ipynb`, `PE_Round_*.ipynb`), run from
the `psc-opt` venv (§1.1).

### 8.1 Part A — synthetic function, isolated optimizer logic (fast, no model fits)

Exercises `GPProposer` directly (§5.3) — no `BayesHalvingSearchCV`, no CV, no
real model fitting.

1. Define at least two small discretized objective functions on a grid via a
   plain `Space` (reuse `Space` for this too — it's exactly what it's for):
   - A unimodal one (e.g. a discretized 2D quadratic bowl) with a known, unique
     optimum — the sanity-check case both optimizers should nail.
   - A multimodal one (e.g. a discretized Rastrigin-style or multi-bump function)
     — where reasonable divergence between the two is expected and acceptable;
     the point is to see *both* explore sensibly, not to demand they agree.
2. For each function, with a fixed seed and a fixed evaluation budget (15–20,
   matching this project's usual scale): run `GPProposer` via its own
   `observe`/`suggest` loop, and run Optuna's `GPSampler` via its own ask/tell
   loop (`optuna.create_study(direction="maximize",
   sampler=optuna.samplers.GPSampler(seed=..., deterministic_objective=True))`),
   against the identical objective function.
3. Report, per function: the full proposed-point sequence from each ("paths"),
   side by side; final best point and value from each; whether each recovers the
   known optimum (unimodal case) or lands in a reasonable basin (multimodal
   case). **Do not require identical paths** — different implementations of the
   same algorithm family will diverge in specifics; look for *qualitatively*
   sound behavior (spread-out early proposals, convergence toward the best region
   later, no wasted repeated proposals of already-observed points).

### 8.2 Part B — real grid, fixed cheap fraction (0.25%), path comparison

**Goal here is different from 8.1: not "does it work on a toy function" but "does
it choose the same path as Optuna on the real objective" — made affordable by
using a single fixed cheap data fraction instead of paying full-data price per
trial, the same way this project's own P/E experiments made everything else
affordable.** This deliberately does **not** exercise `BayesHalvingSearchCV`'s
multi-fidelity climbing (that is already covered by `test_bayes.py` items 16 and
by §10's full benchmark) — both optimizers see one fixed data size throughout, so
any difference in their proposal sequences reflects the optimizers, not a
changing objective.

1. **Build one fixed, shared objective, once**: run the real pipeline (same as
   every other benchmark notebook — copy `Prototype_Replication.ipynb` cells[1]),
   compute the priority order via `stratified_order` (the same call
   `subsample="auto"` would make for `TimeSeriesSplit`), and take its **top 0.25%
   prefix** (~1,046 rows on the 418K-row training set) as a fixed row subset.
   Wrap it in one `ZoneSplitter(TimeSeriesSplit(n_splits=5), that_subset)`. Define
   `objective(idx) -> float`: build params via `Space.params(idx)`, fit the
   official-grid `ExtraTreesRegressor` config, score via that one fixed splitter
   (MAE, negated to match the greater-is-better convention). **This exact
   `objective` function, unchanged, is what both optimizers below actually
   optimize** — the point of fixing it once is that any difference in behavior is
   attributable to the optimizer, not to a shifting target.
2. Run `GPProposer`'s raw `observe`/`suggest` loop directly against `objective`
   (no `BayesHalvingSearchCV`, no zones, no `BaseSearchCV` machinery — same
   isolation as §8.1) for a fixed budget, `n_iter=15` (matching this project's
   established scale), fixed seed.
3. Run Optuna's `GPSampler` via ask/tell against the **same** `objective`
   function, same seed, same `n_iter=15`, in the same session/script immediately
   before or after step 2 (same-session wall-clock pairing, per this project's
   established machine-noise rule — see `EXPERIMENTS.md`).
4. Report: the full proposed-point sequence from each side by side (the "path" —
   this is the primary output of this test); final best point and best MAE from
   each; wall-clock for each. **Do not require identical paths** (§8's opening
   caveat still applies — different kernel/acquisition/RNG internals make exact
   agreement unrealistic), but do check they land in the **same or an adjacent
   basin** — and note explicitly if both converge to the (4,130,17)/(4,150,17)-class
   optimum this project keeps finding at low data fractions (Experiments 7–11),
   since that would be a strong, specific correctness signal beyond "some
   reasonable-looking answer."

   **Quantify "how many points are different," precisely, two ways** (a single
   vague "different" count is not meaningful without saying which of these it
   is — report both):
   - **Set overlap (order-independent, the headline number)**: of the (up to) 15
     distinct index tuples each side visited, how many appear in *both* sets
     (`len(gp_proposer_points & optuna_points)`), and how many are unique to each
     side. This answers "did they explore the same region of the grid."
   - **Position-by-position match (order-dependent, a stricter secondary stat)**:
     for step `i` in `1..15`, does `gp_proposer_path[i] == optuna_path[i]`? Report
     the count of matching positions out of 15.
   - **Expect the position-by-position count to be low, and say so in the
     report — this is not a red flag.** Both optimizers spend their first ~2
     proposals in a cold-start/random-initialization phase before their
     surrogate model has enough data to be meaningful (`GPProposer`'s own
     cold-start rule, §5.2 item 2; Optuna's `GPSampler` has an analogous
     random-startup phase internally). Two independent random-initialization
     draws, even from "the same seed," will not produce the same points unless
     the two implementations happen to use numerically identical RNG algorithms
     and call sequences — they don't. This means **early positions are the
     *least* informative** for judging correctness; if the set-overlap count is
     healthy and the *later* proposals in both paths visibly cluster around the
     same region, that is the meaningful signal, not early-position agreement.
5. This 0.25%-fraction run is **not** directly comparable in MAE terms to
   Experiment 4's 100%-data-Optuna numbers (805.730 at 100% data) — different
   data size, different landscape, as this project has established repeatedly.
   Do not present this test's MAE values as beating or matching Experiment 4;
   present them only as the same-conditions GPProposer-vs-Optuna comparison they
   actually are.

Log this validation's outcome as a short section in `EXPERIMENTS.md` (a new
numbered experiment, append-only, same format as everything else) — it is a real
experiment with a real result, not just scratch work.

---

## 9. Tests (add `tests/test_bayes.py` + `tests/test_fidelity.py` +
`tests/test_starts.py` + `tests/test_gp.py`; keep every existing test green —
currently 105 pass, 2 skip)

**Step 0, before writing any estimator code**: perform the §2.1 refactor, then
run `pytest tests -q` and confirm 105 passed / 2 skipped, unchanged. Hard gate.

**All tests below run unconditionally in the plain package `.venv` — no torch, no
Optuna, no skip markers.** This is a direct, real benefit of dropping Optuna/torch
as runtime dependencies: full test coverage in one environment.

`tests/test_starts.py` (pure unit, no sklearn):
1. `select_starts` produces identical output to `PatternSearchCV`'s pre-refactor
   `_select_starts` for a range of `(n_starts, start_points, seed)` combinations.
2. Calling `select_starts` directly with identical arguments from both
   estimators' calling conventions produces identical results.

`tests/test_fidelity.py` (pure unit, no sklearn):
3. warmup counting: with `warmup=3`, first two improvements buy nothing; the
   calibration uses exactly the 2 readings; `D` floored to `min_step` multiples.
4. boundary formula, descending, innermost floored at `min_step`.
5. ratchet: a big move after a climb never lowers the zone.
6. zero-move is not a reading.

`tests/test_gp.py` (pure unit, `GPProposer` directly, no sklearn estimator, no
model fitting — small `Space` instances only):
7. cold start: `suggest()` returns unobserved points for the first 2 scored
   observations, doesn't crash on 0/1 observations.
8. seed handling: `observe(idx, None)` then `suggest()` returns exactly `idx`.
9. on a simple discretized unimodal quadratic `Space` (small, enumerable), after
   a reasonable budget (~15–20 `observe`/`suggest` cycles), the best observed
   point is at or adjacent to the true optimum.
10. `top_k_observed(k)`: sorted desc, deduped, correct length when fewer than k
    observations exist.
11. determinism: same `random_state` → identical `suggest()` sequence given the
    same sequence of `observe()` calls.
12. never proposes an already-observed point while unobserved points remain.

`tests/test_bayes.py` (use `DecisionTreeRegressor`, `make_regression(400)`, small grids):
13. basic fit, `n_starts=1`: `best_params_` from grid; `n_resources` in
    `cv_results_`; `best_index_` row has max `n_resources`; ledger rows unique per
    (params, n_resources); `local_optima_` has exactly one entry.
14. determinism: two identical-seed fits (`n_starts=1` and `n_starts=3`) → each
    pair produces identical `cv_results_` scores and identical `local_optima_`.
15. budget: number of `cv_results_` rows per start ≤ `n_iter` + (promote/polish
    overhead bound: `promote_k+1` per climb + final polish); cache prevents
    duplicate (params, fraction) fits **across starts, not just within one**.
16. zones ratchet + final polish: fractions in `search_history_` are
    non-decreasing **per start**; at least one evaluation at fraction 1.0 exists
    per start; `best_*` overall comes from the best such row across all starts.
17. `data_zones=1` (ladder off) works and every row is full-data.
18. multi-start: `n_starts=4` produces `local_optima_` with between 1 and 4
    entries (dedup by final point); `search_history_` entries carry a start
    index; `start_points` explicit list takes priority seats (mirror
    `PatternSearchCV`'s `test_start_points_take_seats`); the shared cache
    measurably reduces total genuine fits vs. an equivalent case with no overlap
    possible (assert `n_cache_hits_ > 0` for a case constructed to guarantee
    overlap, e.g. two identical `start_points`).
19. verbose header names the metric, zones, and `n_starts` (caplog, mirroring
    `test_verbose_header_*` in `test_estimator.py`); verbose=0 emits nothing and
    runs no CV-summary fits.
20. invalid params raise: bad `n_iter<1`, bad `promote_k<1`, bad zones, bad
    `n_starts` (mirror `PatternSearchCV`'s validation-error tests).
21. `parametrize_with_checks` on a `BayesHalvingSearchCV(DecisionTreeClassifier...,
    n_iter=8, cv=3, random_state=0)` instance — full sklearn gate, same as
    `test_sklearn_compat.py` does for `PatternSearchCV` (append there or new
    file; expect the same tag/pickling requirements to bite if §2 items were
    skipped).
22. pickling: fit with `n_jobs=2` and `n_starts=2` succeeds and the fitted
    estimator pickles (regression for §2 item 5, exercised with multi-start).

Run: `.venv\Scripts\python.exe -m pytest tests -q` from the repo root.

---

## 10. Benchmark deliverable (after code is green AND §8 validation is logged)

Notebook `BHS_vs_PSC_26grid.ipynb` in the repo root, patterned on
`PE_Round_0.5_1_10_100.ipynb` (same pipeline cell — copy it from
`Prototype_Replication.ipynb` cells[1] like every other benchmark notebook does;
same official grid; `TimeSeriesSplit(5)`; MAE). **Runs in the plain package
`.venv`** (no torch needed — this is now true for the full benchmark, not just
the estimator's tests, since `BayesHalvingSearchCV` has no torch dependency at
all).

Primary arms (all `random_state=0`, `subsample="stratified"`, zones
`(0.005,0.01,0.1,1.0)`, **`n_starts=1`** — directly comparable to the reference
rows in §0):
1. `BayesHalvingSearchCV(n_iter=25, n_starts=1)`
2. `PatternSearchCV` current defaults (patient, `n_starts=1`) — fresh run, same
   session, for a same-machine wall-clock pairing.

Optional follow-up arm, once the primary comparison is logged: `n_starts=4` on
both estimators, same total-fit-budget framing already used for `PatternSearchCV`'s
own multi-start ablations.

Report per arm, in the user's standard comparison-table format (columns = runs,
rows = exactly): zones ladder, evaluations, full-fit equiv, wall-clock, best
point, CV MAE of best. Plus the trial-by-trial (start index, params, fraction,
MAE) history for the BHS arm(s). Machine-noise rule from `EXPERIMENTS.md`
applies: wall-clock differences under ~15–25% are noise; full-fit equivalents is
the primary metric; NEVER use median-of-per-eval-ratios (documented bad metric —
see the Experiment 6 correction).

Log results as a new numbered experiment in `EXPERIMENTS.md` following the
existing format. Do not renumber existing experiments. Commit messages: explain
what and why; end with the Co-Authored-By trailer used throughout this repo's
history (`git log` shows the pattern).

---

## 11. Known pitfalls from this project's history (each cost real time — read)

1. **Pickling during parallel fit** (§2 item 5). The #1 landmine.
2. `evaluate_candidates` results are **cumulative** — always slice `[-len(batch):]`.
3. `more_results` key must be passed on **every** call or columns desync.
4. The zone splitter must yield the **same `n_splits`** as the original cv —
   `BaseSearchCV` asserts it. `ZoneSplitter` already guarantees this; don't wrap it.
5. Windows + torch (§1.1): this only matters for the dev-time §8 validation now,
   not for anything shipped. Never `pip install torch` (unpinned) in the deep
   `.venv`; the validation notebook uses `psc-opt` exclusively.
6. Building notebooks programmatically: escape-mangling through shell heredocs
   corrupted an f-string once — write builder scripts to a file (Write tool /
   `.py` file), `ast.parse` every cell before saving, and syntax-check before
   executing.
7. `from time import time` in one namespace shadowed the `time` module for library
   code sharing that namespace — keep `import time` module-style in notebooks.
8. This machine's wall-clock noise between back-to-back identical runs is
   ±15–25%. Never claim a wall-clock effect from a single pair; equivalents and
   summed fit-work are the honest metrics.
9. `select_dtypes(["object"])` emits a Pandas4Warning on this stack — harmless,
   ignore; do not "fix" the shared pipeline cell.
10. sklearn's estimator checks: the `y=None` and empty-`X` guards (§2 item 7) and
    tag delegation (§2 item 6) are what make `check_requires_y_none`,
    `check_estimators_empty_data_messages`, `check_estimators_nan_inf`,
    `check_supervised_y_2d`, and the pickle checks pass. `PatternSearchCV`'s
    solutions are the template.
11. **The `_select_starts` extraction (§2.1) is the one place this task touches
    `PatternSearchCV`'s file.** Any other change to `_search.py`, `_climber.py`,
    or `_engine.py` is a scope violation.
12. `GaussianProcessRegressor` can throw/warn on numerically degenerate inputs
    (e.g. all-identical `y_train`, or near-duplicate feature rows if two distinct
    index tuples happen to featurize very close together for a mostly-categorical
    space). `alpha=1e-6` is a working nugget for this; if a real degenerate case
    surfaces in testing, increase the nugget rather than catching-and-ignoring
    the warning.

## 12. Out of scope (do not do)

- Do not modify `Climber` or `Engine` at all.
- The **only** permitted change to `PatternSearchCV`/`_search.py` is the
  mechanical, behavior-preserving extraction described in §2.1 (§11 item 11).
- Do not touch `EXPERIMENTS.md` history or renumber experiments (append only,
  including for the §8 validation result).
- Do not add ASHA/successive-halving elimination between starts — the fidelity
  schedule is the bullseye, and multi-start runs every start to completion.
- Do not invent a cross-start "merging" heuristic (§3.2) — the shared cache is
  the intended cost-saving mechanism.
- **Do not add `optuna` or `torch` to `pyproject.toml` in any form** — no
  `dependencies` entry, no `optional-dependencies` group. They are dev-time-only
  tools for §8, installed ad hoc in `psc-opt`, never declared by the package.
- Do not expose GP kernel/acquisition internals (`xi`, kernel choice, nugget,
  restarts) as public constructor parameters in this revision — they are
  internal constants (§5.2) until evidence justifies exposing one.
- `OpenQuestions.md` is a local untracked scratch file — leave it out of commits.
