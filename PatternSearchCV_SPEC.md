# PatternSearchCV — Technical Specification

**Status:** agreed design, ready for implementation
**Target:** standalone PyPI package (`pattern-search-cv`), built to scikit-learn-contrib
standards; contrib application after benchmarks; core scikit-learn not a near-term goal.
**Provenance:** distilled from the working prototype in
`DatasetSize_and_ParamOpt_WORKING_(3Large_Aug_30_2025).ipynb` (cells 106–110) and the
design discussion of 2026-07-12. Where this spec contradicts the prototype, the spec wins.

---

## 0. Summary

`PatternSearchCV` is a scikit-learn-compatible hyperparameter search estimator implementing
Hooke-Jeeves pattern search over discretized parameter grids, extended with:

1. **Multi-fidelity data growth** ("the bullseye"): each search starts on a small,
   representative subsample and buys more data only as its own movement indicates it is
   approaching an optimum. Active even for a single start.
2. **Multi-start via scatter search**: independent
   "climbers" from maximally-spread start points, all run to completion, no score-based
   elimination; best full-data optimum wins.
3. **Batch parallelism**: complete-poll exploration and cross-climber batching feed one
   joblib pool (`n_jobs`), exploiting the fact that pattern search probes are independent
   — unlike sequential Bayesian optimization.

Design pillars: deterministic given `random_state`; scores are only ever compared at the
same data fraction; the dataset is never copied — only index arrays move; every duplicate
evaluation is a cache hit.

---

## 1. Public API

```python
PatternSearchCV(
    estimator,
    param_grid,                  # dict: name -> list of values | (low, high, num) tuple
    *,
    scoring=None,
    n_jobs=None,
    refit=True,
    cv=None,                     # sklearn default (5-fold / splitter object)
    verbose=0,
    random_state=None,
    error_score=np.nan,
    return_train_score=False,
    # --- pattern search ---
    poll="auto",                 # "auto" | "complete" | "opportunistic"
    mesh_expansion=1.0,          # 1.0 = off (default); 2.0 = MATLAB GPS parity
    # --- multi-fidelity ---
    data_zones=(0.10, 0.20, 0.50, 1.0),  # int n -> n even levels (4 -> [.25,.5,.75,1]);
                                 #  or explicit ascending values ending in 1.0 for
                                 #  uneven ladders; 1 disables the ladder
    warmup=3,                    # best-updates before rings calibrate; higher = data
                                 #  added closer to the optimum (the "patience dial")
    subsample="auto",            # "auto" | "expanding" | "stratified" | "random"
    subsample_columns=None,      # optional column subset for "stratified"
    # --- multi-start ---
    n_starts=1,
    start_points=None,           # optional list of param dicts (CustomStartPointSet analog)
)
```

**Removed relative to earlier drafts** (superseded during design):
`factor` and `min_resources` (survivor-elimination rungs replaced by the bullseye ladder;
the floor survives as an internal check), the promote-fewer toggle (no elimination exists),
and all velocity/deceleration parameters (replaced by ring geometry).

### Fitted attributes (sklearn contract: trailing underscore, exist only after `fit`)

- `best_params_`, `best_score_`, `best_index_`, `best_estimator_` (if `refit`),
  `cv_results_`, `n_splits_`, `refit_time_`, `scorer_`, `multimetric_`, `n_features_in_`
  — standard SearchCV surface. `best_*` are drawn **only from evaluations at
  fraction 1.0** (`_select_best_index` override restricted to full-resource rows).
- `cv_results_` gains a `n_resources` key (rows used per evaluation). The key name
  deliberately mirrors the halving classes even though the parameter is `data_zones`,
  so halving-aware tooling reads our results unchanged. Every row is a genuine fit —
  cache hits never reach the ledger.
- `local_optima_`: list of dicts, one per distinct converged optimum across all starts:
  `{"params", "score", "n_starts_converged", "start_points"}` — the `MultiStart`-style
  solution map. Merged climbers' pre-merge bests are recorded in `search_history_` but
  only *converged* optima appear here.
- `search_history_`: per-evaluation record (climber id, params, fraction, score,
  move kind) — feeds trajectory plots and the benchmark tooling.

### Estimator-contract rules (hard requirements)

- `__init__` stores constructor args verbatim; **no** validation, computation, or printing.
  All work happens in `fit`. `fit` returns `self`.
- No `print`; output gated on `verbose`, warnings via `warnings.warn`.
- `random_state` consumed via `check_random_state` inside `fit` only.
- Passes `sklearn.utils.estimator_checks.parametrize_with_checks`.
- numpydoc docstrings; doctest-able example; PEP 8 (ruff); semantic versioning; changelog.

---

## 2. scikit-learn integration

- Subclass `sklearn.model_selection._search.BaseSearchCV`; implement `_run_search`.
  Do **not** vendor the base class (prototype's `BaseSearchCV2` is deleted) and do **not**
  inherit from `HalvingGridSearchCV` (experimental import; assumes a static candidate set).
- Fidelity uses the same extension point as sklearn's own halving:
  `evaluate_candidates(params_list, cv=subsampled_splitter, more_results=...)`, where the
  splitter is our reimplementation of `_SubsampleMetaSplitter` yielding index-subsampled
  train/test splits over the untouched X.
- Dependency pins: `scikit-learn` tested range (>=1.6, cap at last verified minor),
  `numpy>=2`, `scipy` (for `stats.qmc`). **No pandas, pyarrow, or numba in the core.**
  CI: GitHub Actions matrix over supported sklearn versions **plus a nightly-sklearn job**
  so private-API drift is caught early.
- Data handling: X/y validated with sklearn utilities (`indexable`, `check_cv`,
  `check_scoring`); rows selected only via index arrays (`_safe_indexing`), which keeps
  numpy / pandas / Arrow-backed / polars inputs working without conversion.
- Scoring follows sklearn conventions exactly (neg-metrics stay negated in `cv_results_`;
  the prototype's `abs()` display hack is dropped).

---

## 3. Search space: `Dimension`

Input per parameter: an explicit list of values, or a `(low, high, num)` tuple expanded
to a linspace (ints preserved when endpoints are integral). Each dimension is a sorted
grid addressed by integer index.

- **Numeric dimensions**: coordinate = `index / (len - 1)`, normalizing every dimension
  to [0, 1] so a 26-point axis and a 3-point axis weigh equally in any distance.
- **Categorical dimensions** (values not all numeric, e.g. `max_features='sqrt'|'log2'`):
  index order is arbitrary, so index *arithmetic* is meaningless. Distance contribution is
  **Hamming**: 0 if unchanged, 1 if changed. Polling still enumerates neighbors in index
  order (cheap at small cardinality).
- Per-dimension state: `delta` (step size in index units), initialized to half the grid
  width (`(len-1) - midpoint_index`), floor 1; dimensions of length 1 are fixed
  (delta 0, skipped in polls).

Distances (velocity readings, scatter-search maximin, ring membership) are Euclidean over
normalized numeric coordinates + Hamming terms, always in this index space.

---

## 4. Single-start core: the `Climber` state machine

A `Climber` never fits a model. It is a pure state machine that **proposes** batches of
candidate points and consumes scores fed back by the engine — unit-testable against
hand-computed Hooke-Jeeves traces with zero fitting.

### 4.1 State

| Field | Meaning |
|---|---|
| `start` | birth point (scatter-search or grid midpoint) |
| `position` | current exploration center — may be a *tentative* pattern point |
| `best` | confirmed optimum of this run: params + score + **fraction scored at** |
| `delta` | per-dimension step vector |
| `fraction` | current data rung (ratcheted, never decreases) |
| `last_move_size` | displacement of the latest `best` update, normalized space |
| `pattern_ref` | previous base point (pattern-move reference; None after reset) |
| `path` | every (params, fraction, score) visited — merge checks, plots, history |
| `status` | `running` / `converged` / `merged(into=k)` |

`position` vs `best` is semantic, not redundant: after a pattern move the climber explores
around unconfirmed ground; a disappointing sweep retreats `position` to `best`.
**Displacement is measured between successive `best` updates only** — failed pattern
excursions are not movement and produce no ring reading.

### 4.2 The loop (per climber)

1. **Poll (exploratory sweep)** around `position` with per-dimension steps `±delta`:
   - `poll="opportunistic"`: classic HJ — probe dimensions sequentially, accept
     improvements immediately, later dimensions probe from the drifted point.
   - `poll="complete"`: propose all `2N` probes as **one batch** around the fixed
     `position` (MATLAB `UseCompletePoll`); on return, additionally test the
     **composite move** (all improving dimensions applied at once) as a bonus candidate.
   - `poll="auto"`: complete when the parallel budget has headroom
     (roughly `n_jobs / n_splits >= 2`), else opportunistic.
2. **Pattern move** when the sweep improved: extrapolate
   `2 * new_base - pattern_ref`, snapped to the nearest grid point; **compounding** —
   consecutive successful pattern moves extend the vector (the prototype's reset of the
   reference point after success is a bug and is fixed). A pattern point whose score is
   already cached is *compared using the cached score*, not skipped.
3. **Contraction** (`delta -> max(1, round(delta / 2))`) fires **only when an exploratory
   sweep around a confirmed base fails**. A failed pattern move does *not* contract —
   the climber returns to exploration at the same delta (fixes the prototype's
   premature-contraction bug). With `mesh_expansion > 1`, a successful sweep multiplies
   delta by the factor, capped at grid width (off by default; documented for
   fine/continuous-like grids).
4. **Fidelity check** after every `best` update (see §5).
5. **Termination**: converged when data is at 1.0, all deltas are at 1, and a full sweep
   plus pattern move yield no improvement (three consecutive such passes, matching the
   prototype's `Continue` intent, now stated explicitly). If the climber converges while
   `fraction < 1.0`, it jumps to 1.0, re-scores, and runs a final polish before the
   convergence test applies.

### 4.3 Dedup cache

One dict shared by the whole search (all climbers): key =
`(dimension index tuple, fraction)` — **integers only**, no float comparisons, no
`np.isclose` (the prototype's O(rows × dims) DataFrame scan with type coercion is
deleted). Value = mean CV score. The engine strips cache hits from every proposed batch
before calling `evaluate_candidates`, so sklearn's ledger (`cv_results_`) records only
genuine fits; cached scores are fed back to climbers as if fresh.

---

## 5. Multi-fidelity: the bullseye

Active by default, **including `n_starts=1`**. `data_zones=1` disables it.

`data_zones` accepts an int — n evenly divided levels, e.g. `4 -> [0.25, 0.5, 0.75, 1.0]`,
`3 -> [1/3, 2/3, 1.0]` — or explicit ascending values ending in 1.0 for uneven ladders.
**Default: `(0.10, 0.20, 0.50, 1.0)`** — front-loaded cheap zones for the probe-heavy
early phases, and only four levels because measured searches make ~3–5 best-moves total
(§ warm-up rationale): more levels than moves would never be traversed. Validation:
int >= 1, or values in (0, 1], strictly ascending, last element 1.0. (Default is a tuple
only because sklearn convention forbids mutable default arguments; lists are accepted.)

### 5.1 Mechanism

The rings are bands of **move size**, centered at zero movement (we cannot center on the
optimum — its location is the unknown). Ring boundaries pair 1:1 with the `data_zones`
ladder; there is no trend/deceleration machinery — one speedometer reading, one lookup:

| Latest `best`-to-`best` displacement (normalized) | Fraction (default zones) |
|---|---|
| above ring boundary 1 (striding / traveling) | 0.10 |
| below boundary 1 | 0.20 |
| below boundary 2 | 0.50 |
| below boundary 3 (endgame-scale moves) | 1.00 |

Rules:

- **Ratchet**: fraction is a high-water mark — moves re-enlarging after a bump never
  reduce data; the climber just earns no more until it crosses the next unclaimed ring.
- **Warm-up + self-calibration** (`warmup`, exposed, default 3, minimum 3): `warmup`
  counts **best moves, starting point included** — so `warmup=3` = start + two confirmed
  best-moves = two nonzero displacement readings, during which no data is bought (the
  starting sample is representative; early readings are not worth reacting to). At
  warm-up end the climber calibrates its own ring geometry: `D = mean` of all warm-up
  readings — averaging is the anomaly handling (a fluky edge-clamped small move or an
  early compounding leap is damped, not special-cased); mean rather than max because an
  inflated `D` pushes boundaries outward so ordinary moves read as inner-ring and buy
  data early — the expensive failure; a small `D` merely buys late, the cheap failure.
  `D` is then **floored to a whole number of minimal grid steps**
  (`floor(D / min_step) * min_step`) — truncation, not rounding, so any fractional
  remainder tightens the rings and keeps the search on small data longer. Boundaries are
  `D` divided evenly across the ring count
  (e.g. `3D/4, D/2, D/4, ...`), with the **innermost boundary floored at one minimal
  grid step** (a literal-zero boundary would be unreachable, since zero displacement is
  not a reading). `warmup` is thereby the *patience dial*: larger values calibrate later,
  when the climber already moves slower, so `D` is smaller, rings are tighter, and data
  arrives closer to the optimum. Per-climber calibration (multi-start): each climber
  measures its own `D`; fractions still snap to the shared `data_zones` ladder, so cache
  keys stay aligned. A climber converging before warm-up ends is covered by the
  jump-to-1.0 rule. Default rationale (measured on the prototype's
  logged runs, ExtraTrees 3-dim grid): searches make only ~3–5 confirmed best-updates
  total, so `warmup=3` (two moves consumed) leaves 1–3 calibrated moves to drive the
  ladder; larger values
  are for larger grids/dimensionality (docstring guidance) or they starve data growth
  until the forced jump.
- **Zero displacement is not a reading** (no improvement -> no ring check; the "5, 0, 5"
  rule).
- **Convergence below 1.0 forces the jump to 1.0** for the final polish.
- **On every fraction change**: the climber's `best` is re-scored at the new fraction
  before any comparison — scores are never compared across fractions. One extra
  (often cached) evaluation per climb.
- Ring count comes from the `data_zones` levels (n boundaries = n fractions − 1); ring
  *radii* come from the warm-up self-calibration above — no fixed geometric constants
  remain. The benchmark ablation sweeps `warmup` (e.g. {3, 5, 7, 10}) instead of raw
  radii: a "patience vs. cost/quality" curve.
- Floor: fraction 0 must give every CV split enough rows; if not, the ladder is truncated
  from below (fewer, larger fractions) with a warning.

Honest framing (for docs/article): realized move size shrinks both when pattern moves
stop compounding (proximity) and when delta contracts after failures (local structure
resolved at that scale). The bullseye deliberately reads their combination — both are
signals that finer distinctions now deserve more data.

### 5.2 Subsampling — index arrays only, computed once per `fit`

A **priority ordering** of row indices is computed once; every fraction is a prefix slice,
so rung samples are strictly **nested** (more data only ever *adds* information) and each
climb costs an array slice. X is sliced only inside CV fits via `_safe_indexing`.

Modes (`subsample=`):

- **`"expanding"`** — chronological prefix: oldest rows first, so 12.5% = the oldest
  eighth, matching expanding-window forecast practice (train on old, test on newer within
  `TimeSeriesSplit`). Default for time-series splitters.
- **`"stratified"`** — transition sampling (novel; the package's signature sampler).
  Walk rows in order; consecutive rows with identical watched-feature values form a
  *run*. Priority: (0) first-ever occurrence of each unique feature combination —
  guaranteed seats; (1) run boundaries and run midpoints, **alternating per run** so the
  edge/typical mix stays ~50/50 under any budget; (2) recursive bisection of runs
  (quarters, eighths, ...) to fill large budgets. Over-budget levels are thinned
  **evenly across time** (every k-th, never truncated) so all seasons stay represented.
  Watches **all columns by default** (post-feature-selection columns are all meaningful);
  `subsample_columns` narrows optionally. Degenerate case (a continuous column changes
  every row): collapses gracefully to systematic every-k-th sampling. One O(n) vectorized
  pass: `(X[1:] != X[:-1]).any(axis=1)` and follow-on numpy ops — no compiled extension
  needed (~ms at 500K x 30 vs minutes per fit).
- **`"random"`** — uniform row sampling for i.i.d. data (sklearn-halving-standard).
  Docs carry an explicit leakage warning against using it on temporal data.
  (Possible v2 refinement: stratify by class for classification.)
- **`"auto"`** — `"expanding"` when `cv` is time-ordered (`TimeSeriesSplit`),
  else `"random"`. `"stratified"` is opt-in.

---

## 6. Multi-start layer

A thin layer over K independent climbers. **Single-start is a swarm of one** — same code.

### 6.1 Initialization (scatter search)

- Candidate pool: ~`10 * n_starts` points via stratified QMC (`scipy.stats.qmc`,
  LHS/Sobol, seeded by `random_state`) in normalized index space, snapped to grid.
- Greedy **maximin** selection: seed with the grid midpoint (continuity with
  single-start), then repeatedly add the pool point maximizing distance to its nearest
  selected point, until `n_starts`. Distances per §3 (Hamming for categoricals).
- `start_points` (list of param dicts) overrides/extends generated starts.

### 6.2 Execution semantics — MATLAB `MultiStart`, not `GlobalSearch`/ASHA

- **No score-based elimination, no synchronization barriers.** Every climber runs its own
  bullseye ladder to convergence at fraction 1.0. Rationale: eliminating on small-data
  scores can kill the basin containing the true optimum — the entire purpose of
  multi-start is thorough exploration. (An ASHA-style culling mode may be *benchmarked*
  as an ablation; it is not shipped behavior.)
- Cost containment is by **convergence, not elimination**: climbers entering the same
  basin ride the shared cache (identical `(params, fraction)` keys), so full-data cost
  scales with the number of **distinct optima**, not `n_starts`.
- **State-match merging**: if a climber reaches a full state — `position`, `delta`
  vector, `fraction`, `pattern_ref` — that any climber has already occupied, its future
  is provably identical (deterministic algorithm + shared cache), so it merges: status
  `merged`, its pre-merge `best` preserved in `search_history_`. Position-only matches do
  **not** merge (different delta/fraction ⇒ different futures). In practice merges fire
  near optima where deltas are 1, pattern refs reset, and fractions align.
- Final selection: among converged climbers' full-data bests, the maximum is
  `best_params_`; the deduplicated set becomes `local_optima_`.

### 6.3 Engine: climbers propose, the engine disposes

Per engine tick: collect proposal batches from all runnable climbers -> strip cache hits
-> one `evaluate_candidates` call (candidates × CV folds in one joblib batch — this is
where `n_jobs=-1` saturates: 8 climbers × 6 probes × 5 folds = 240 parallel fits) ->
write scores to cache -> feed each climber its results -> climbers advance state.
Batching order must not affect any climber's decision (climbers are independent), which
preserves **full determinism given `random_state`** — no race-dependent behavior exists
anywhere in the shipped design.

**Build order** (per agreement): (1) single-start climber + fidelity ladder, trace-based
unit tests; (2) profiling/runtime optimization pass; (3) engine + multi-start layer;
(4) estimator shell + `parametrize_with_checks` green; (5) benchmark suite.

---

## 7. Engineering rules

- **Vectorize data-sized work** (boundary detection, priority ordering, maximin
  distances); keep **plain readable loops for trial-sized decisions** (the search loop
  iterates over dozens of dependent decisions — clarity there is a review asset).
- No pandas/pyarrow in core (index-array design makes them unnecessary; benchmarks may
  use pandas freely).
- Tests: `parametrize_with_checks`; `Climber` trace tests against hand-computed HJ paths;
  cache/dedup tests; subsample-mode tests incl. nesting and time-thinning; determinism
  tests (same `random_state` ⇒ identical `cv_results_`); edge cases (1 dimension,
  length-1 dimensions, all-NaN scores, multimetric, `refit=False`, tiny datasets hitting
  the resource floor); an end-to-end test asserting the derived ladder and full-data-only
  `best_index_`.

---

## 8. Benchmark suite (separate `benchmarks/` layer)

- **Adapters** over a common interface: `PatternSearchCV`; skopt `BayesSearchCV` /
  `gp_minimize` (legacy continuity — project unmaintained); Optuna `TPESampler` **and**
  `GPSampler`; `RandomizedSearchCV` (mandatory baseline); `HalvingRandomSearchCV`
  (multi-fidelity baseline).
- **Protocol**: identical space/estimator/CV/scoring per scenario; budget parity in
  model fits (cache hits count free — reported explicitly); ≥10 seeds; every trial logged
  to tidy CSV (optimizer, seed, trial, params, score, fit time, overhead time, fraction);
  figures regenerate from CSVs only.
- **Figures**: anytime curves (best-so-far vs fits, and vs wall-clock; median + IQR);
  fits-to-target bars; overhead decomposition; held-out validation of chosen configs;
  the bullseye trajectory diagram.
- **Ablations**: `n_starts ∈ {1,2,4,8}` (same-fit-budget and same-wall-clock framings);
  bullseye vs fixed-schedule growth (+ ring-geometry sweep); `mesh_expansion` on/off;
  `"expanding"` vs `"stratified"` subsampling on the retail data; run-all vs ASHA-culling.
- **Datasets**: the 500K-row retail time series (ExtraTrees / XGBoost / LightGBM) plus
  2–3 public OpenML regression datasets.

---

## 9. Known honest limitations (state these in docs and the article)

1. Pattern search converges to **stationary/local** optima (Torczon 1997; Audet & Dennis
   2002); no global guarantee exists — MATLAB's own six-solver comparison shows
   `patternsearch` trapped on Rastrigin. Multi-start is the globalization strategy; we
   never claim global optimality.
2. `BaseSearchCV` is a private-module import — mitigated by version pinning + nightly CI,
   same posture as skopt/tune-sklearn.
3. Early rungs on `"expanding"` subsamples see only the oldest regime (drift caveat);
   `"stratified"` is the mitigation and the comparison is benchmarked.
4. A fluky small move can buy one rung early (bullseye has no trend gate); bounded by
   warm-up + ratchet + cheap low rungs; measured in the ablation.
