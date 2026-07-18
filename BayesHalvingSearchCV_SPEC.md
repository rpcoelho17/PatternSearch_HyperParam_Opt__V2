# BayesHalvingSearchCV — Spec / Architectural Document

**Audience:** Claude Sonnet 5, implementing and testing this estimator inside the
existing `pattern-search-cv` repository.
**Status:** agreed design, ready for implementation. Revision 2 (2026-07-15):
added same-package/same-environment packaging, shared search-space standard, and
shared multi-start methodology, per explicit user amendments to revision 1.
**Author of record:** design session 2026-07-15 (user + Claude Fable 5).

---

## 0. Mission

`Optuna GPSampler` (all trials at 100% data) found this project's benchmark optimum
in 15 trials — fewer *evaluations* than `PatternSearchCV` — but paid full price for
every one of them: **15.00 full-fit equivalents vs PatternSearchCV's ~5.04–5.85**
(see `EXPERIMENTS.md`, Experiment 4 vs Experiments 7–11). Its weakness is not the
search logic; it is that it has no multi-fidelity machinery.

**Build `BayesHalvingSearchCV`: a scikit-learn-compatible estimator, shipped in the
SAME pip package as `PatternSearchCV`, that keeps this package's entire
multi-fidelity infrastructure — stratified priority ordering, data zones, the
bullseye ring methodology, the shared dedup cache, the full-data-only best
selection, the scatter-search multi-start layer — but replaces the Hooke-Jeeves
search logic with Optuna's GP-based Bayesian optimization.** It must use the
identical `param_grid` search-space standard as `PatternSearchCV` (same `Space`
class, no parallel space abstraction) and the identical multi-start mechanism
(`n_starts`, `start_points`, scatter-search selection). Afterwards it will be
benchmarked head-to-head against `PatternSearchCV` on the official test space.

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

## 1. Packaging and environment — ONE pip library, minimal new dependencies

**Hard requirement: both estimators ship in the same package, `pattern_search_cv`,
from the same `pyproject.toml`, installed by the same `pip install pattern-search-cv`.**
There is no separate package, no separate repo, no separate `src/` tree.

- New module: `src/pattern_search_cv/_bayes.py` (the estimator), alongside the
  existing `_search.py`, `_climber.py`, `_engine.py`, `_sampling.py`, `_space.py`.
- `src/pattern_search_cv/__init__.py`: add `BayesHalvingSearchCV` to the imports
  and `__all__`, next to `PatternSearchCV`, `Space`, `Dimension`.
- **Core dependencies do not change.** `pyproject.toml`'s `dependencies` stays
  `numpy`, `scipy`, `scikit-learn` only — installing/using `PatternSearchCV` must
  never require `optuna` or `torch`. Import `optuna` **lazily, inside
  `BayesHalvingSearchCV.fit`** (exactly like `_search.py`'s `_log_cv_summary`
  imports `cross_validate` lazily), with an informative `ImportError` if missing,
  pointing at the extras group below.
- Add one new extras group to `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  test = ["pytest"]
  bayes = ["optuna>=3.6,<5"]
  ```
  `pip install pattern-search-cv[bayes]` is sufficient for `sampler="tpe"`
  (no torch needed — verify this: TPESampler has no torch dependency). Do **not**
  add a `bayes-gp` extras group promising GP support via pip — we have two
  *verified* platform failures installing torch in a deep venv path on this
  Windows box (§9 item 5); do not paper over that with an extras flag that would
  silently fail for users on the same platform. Document the manual GP install
  path (torch pinned to 2.5.1+cpu, short filesystem path) in the class docstring
  instead, exactly as this project's own benchmark venv (`psc-opt`, see below)
  had to be built.

**Consequence for testing**: this means `BayesHalvingSearchCV` with
`sampler="tpe"` runs in the **exact same environment** as `PatternSearchCV` —
same `.venv`, same test suite, same CI story, same `pip install -e .[test,bayes]`.
Only `sampler="gp"` needs anything different, and that difference is isolated to
one parameter value, not a different package or environment.

- Repo root: `C:\FILES\Code\Benchmarking\Working_on_Train_Set\V2025\pattern-search-cv`
  (git: `https://github.com/rpcoelho17/PatternSearch_HyperParam_Opt__V2`, branch `main`).
- Package venv: `.venv` inside the repo — Python 3.11, sklearn 1.9.0, numpy 2.4.6.
  Install optuna here too (`pip install -e .[test,bayes]`) — this venv is now the
  canonical dev/test environment for **both** estimators' non-GP code paths.
- **torch is NOT installable in the package venv**: two verified Windows
  failures — (a) torch's nested license paths exceed the 260-char limit under this
  deep venv path; (b) torch 2.13 wheels' `c10.dll` fails to initialize on this
  Windows 10 build regardless of path.
- Torch-capable venv (verified working, for GP-only runs): `C:\FILES\Code\Benchmarking\psc-opt`
  — torch 2.5.1+cpu, optuna 4.9.0, sklearn 1.9.0, numpy 2.4.6. Install this package
  there too (`pip install -e .[test,bayes]` from the repo, using that venv's pip)
  so GP-path tests/benchmarks can import `pattern_search_cv` normally. The known-good
  torch recipe is `pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu`
  from a SHORT filesystem path. Never `pip install torch` unpinned in the deep venv.
- Jupyter kernel `psc-venv` is registered for the package venv; benchmark notebooks
  are executed headlessly with
  `.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=psc-venv <nb>`.

---

## 2. What to reuse (exact inventory — do not reimplement these)

All in `src/pattern_search_cv/`:

| Component | Location | What it gives you |
|---|---|---|
| `Space` | `_space.py` | grid dims from `param_grid` (lists or `(low,high,num)` tuples); integer index tuples; `params(idx)`/`indices(params)`; `distance(a,b)` = Euclidean over normalized numeric coords + Hamming for categoricals; `min_step`; `midpoint()` |
| `stratified_order(X, columns=None)` | `_sampling.py` | the priority ordering (novel-first + boundary/midpoint + bit-reversed thinning); degenerates safely to a bit-reversed full-timeline permutation when every row is unique |
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
   instance during `fit`: no optuna `Study`, no GP/torch objects, no logging
   handlers, no generators. Keep them in local frames of `_run_search`; stash only
   plain-data results into `self._ctx["results"]`; `self._ctx = None` in `fit`'s
   `finally`.
6. **Tag delegation** (`__sklearn_tags__`): copy PatternSearchCV's — `allow_nan`
   and `target_tags` from the sub-estimator (BaseSearchCV misses these).
7. **`y is None` guard and `n_samples < 1` guard** with the exact error-message
   styles used there (estimator checks grep for them).
8. **Verbose conventions**: logger `pattern_search_cv` with NullHandler default;
   `verbose>=1` attaches a StreamHandler and MUST print the header (optimizing
   metric via `_scoring_label()`-style resolution, cv class name, every dimension's
   values, the estimator's own knobs — for this estimator: sampler, n_iter,
   promote_k, warmup, zones, n_starts) — and log every fidelity decision and every
   start's activity as it happens (mirror `Climber`'s per-decision logging, one
   line per event, prefixed with the start index), plus the end-of-run
   `_log_cv_summary` (per-fold EV/MAE/MSE/RMSE/R2 + means, fit/score times — copy
   or share PatternSearchCV's implementation). `verbose=0` = silent, no extra fits.
   **Log every explicitly-passed knob too** — a prior bug logged `poll` only when
   auto-resolved; don't repeat that mistake here for `sampler`/`n_starts`/etc.

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
                 sampler="gp",            # "gp" (GPSampler, needs torch) | "tpe"
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

### 3.1 Search space — mandatory: identical standard to `PatternSearchCV`

- `param_grid`: **exactly** the same accepted forms as `PatternSearchCV.param_grid`
  — dict mapping name → explicit list of values, or → `(low, high, num)` tuple
  expanded to a linspace. Both estimators build their space via the same call:
  `space = Space(self.param_grid)`. **No parallel search-space abstraction is
  permitted** — do not expose ConfigSpace, do not expose raw Optuna
  `suggest_*`/distribution objects in the public API, do not invent a second
  `Dimension`-like class. `Space`/`Dimension` (`_space.py`, unmodified) is the
  single search-space standard for the whole package.
- Internally, when asking Optuna for a trial, translate `Space`'s index geometry
  into Optuna's suggestion API (an *internal* detail, invisible to the user):
  - numeric dimension `d` → `trial.suggest_int(name, 0, d.n - 1)` (Optuna sees the
    ordinal index structure; convert index → real value via `Space.params(idx)`
    when building the estimator's parameters).
  - categorical dimension → `trial.suggest_categorical(name, list(range(d.n)))`.
- `space.distance(a, b)` (Euclidean over normalized numeric coords + Hamming for
  categoricals) is reused unmodified for the multi-start selection (§3.2) and for
  the `BullseyeController`'s displacement readings (§5) — same geometry, same
  code, both estimators.

### 3.2 Multi-start — mandatory: identical mechanism and philosophy to `PatternSearchCV`

- `n_starts`, `start_points`: same parameters, same defaults, same semantics as
  `PatternSearchCV`. Starts are selected via the shared `select_starts(space,
  n_starts, start_points, rng)` function from §2.1 — **the exact same call
  PatternSearchCV makes**, not a reimplementation.
- **MultiStart philosophy carries over unchanged: no elimination.** Every one of
  the `n_starts` independent Bayesian searches runs to its own completion (its own
  `n_iter` budget — see §4.2), exactly mirroring "every climber runs to
  completion; best full-data optimum wins" (spec section 6.2 of
  `PatternSearchCV_SPEC.md`). Do not add ASHA-style culling between starts (this
  was explicitly rejected for `PatternSearchCV` and the reasoning applies
  identically here — see that spec's §6.2 for the "MultiStart not GlobalSearch"
  argument if you want the full justification).
- **What does NOT carry over, and why**: `PatternSearchCV`'s state-match merging
  (two `Climber`s whose full deterministic state coincides get merged) has no
  clean analog for a stochastic Bayesian search — there is no notion of two GP
  optimizers being in an "identical state" the way two deterministic Hooke-Jeeves
  climbers can be bit-for-bit identical. **The shared dedup cache is the
  cost-saving mechanism instead** (§4.2): redundant `(params, fraction)` proposals
  across different starts' studies are served from cache for free, which is where
  multi-start's efficiency gain actually comes from in this estimator. Do not
  attempt to invent a merging heuristic — it is explicitly out of scope.
- Fitted attribute `local_optima_`: list of dicts, one per **distinct** converged
  start result (dedup by final incumbent index tuple, best score first) — same
  shape/spirit as `PatternSearchCV.local_optima_`: `{"params", "score",
  "n_starts_converged", "start_points"}`. `n_starts=1` still populates this with
  one entry (consistent with `PatternSearchCV`'s "single-start is a swarm of
  one" framing).

### 3.3 Everything else

- `sampler`: `"gp"` = `optuna.samplers.GPSampler(seed=<int>, deterministic_objective=True)`;
  `"tpe"` = `optuna.samplers.TPESampler(seed=<int>)`. optuna imported **inside
  fit** (§1) with an informative `ImportError` if missing; when `sampler="gp"`,
  catch the GPSampler-needs-torch failure and re-raise with the §1 install recipe
  in the message. Suppress `optuna.exceptions.ExperimentalWarning` for GPSampler
  and set `optuna.logging.set_verbosity(WARNING)` during fit.
- `n_iter`: budget of *genuine fits* **per start**, across all zones combined for
  that start, excluding the final-polish re-scores (which are ≤ `promote_k`+1
  extra evaluations per start, reported in `cv_results_` like everything else).
  Cache-served proposals (whether from this start's own earlier history or from
  another start's studies) don't consume budget; guard with `max_asks = 10 *
  n_iter` total ask() calls **per start** to make infinite dedup loops
  impossible (if hit, log a warning and proceed to that start's final polish).
- Fitted attributes: standard SearchCV surface + `n_resources` key in
  `cv_results_` + `local_optima_` (§3.2) + `search_history_` (list of plain
  dicts: **start index**, trial number, params, fraction, score, event tag e.g.
  "trial"/"climb-rescore"/"final-polish") + `n_cache_hits_`.
- Add the class to `__init__.py`'s `__all__` and exports (§1).
- Docstring: numpydoc, same caveat-forward style as `PatternSearchCV`'s (evidence
  provenance, name-vs-mechanism note from §0, and a note that `param_grid` and
  multi-start are governed by the same standard as `PatternSearchCV` — point the
  reader there for the detailed rationale rather than duplicating prose).

---

## 4. Search algorithm (normative)

### 4.1 Optuna integration mode

Use optuna's **ask/tell** API — no `study.optimize`, no objective callback, no
nameservers. One `Study` per **(start, data zone)** pair — a fresh GP/TPE model
per fidelity level *within* each independent start, so scores from different
fractions are never mixed in one surrogate (the same never-compare-across-fractions
rule `PatternSearchCV` enforces), and each start's search is fully independent of
every other start's (per §3.2).

Suggestion space = **index space**, via `Space` (§3.1) — identical for every
start; only the *seeding* differs per start.

### 4.2 Outer loop: multi-start (new in revision 2)

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
                            cache, ...)     # §4.3, appends its cache hits to
                                            # n_cache_hits, uses/extends `cache`
    per_start_results.append(result)

local_optima_ = dedup(per_start_results by final incumbent idx, best score first)
best_start    = argmax(per_start_results, key=score at fraction 1.0)
# best_* attributes populated by the standard _select_best_index path (§2 item 3)
# operating over the UNION of all starts' cv_results_ rows, restricted to
# fraction==1.0 rows, exactly as it already does for a single search.
```

All of this lives in local frames of `_run_search` (§2 item 5) — the per-start
loop, the studies, everything. Only `cache`'s *scores* (plain floats/tuples) and
the final plain-dict results may end up in anything durable.

### 4.3 Inner loop: one start (per-start Bayesian search with bullseye fidelity)

Same core loop as revision 1, run once per start, parameterized by `start_point`:

```
controller = BullseyeController(space.min_step, n_boundaries=len(zones)-1,
                                warmup=warmup)      (§5; ONE controller per start,
                                                     independent state, mirrors
                                                     each Climber calibrating on
                                                     its own trajectory)
zone_i     = 0
incumbent, incumbent_score = None, None
fits_used, asks = 0, 0

study = new_study()   # optuna.create_study(direction="maximize", sampler=fresh
                       # seeded sampler)  -- maximize: sklearn scores are
                       # greater-is-better
study.enqueue_trial(space.params(start_point))   # first ask() returns exactly
                                                  # the scatter-search start point

while fits_used < n_iter and asks < max_asks:
    trial = study.ask(); asks += 1
    idx   = tuple(suggested indices)
    frac  = zones[zone_i]
    if (idx, frac) in cache:
        study.tell(trial, cache[(idx, frac)])          # free, no budget
        n_cache_hits += 1
        continue
    score = evaluate_batch(frac, [idx])[0]              (§2 items 1-2)
    fits_used += 1
    cache[(idx, frac)] = score
    study.tell(trial, score)

    if incumbent is None or score > incumbent_score:
        move = 0.0 if incumbent is None else space.distance(incumbent, idx)
        incumbent, incumbent_score = idx, score
        new_zone = controller.observe_improvement(move)   (§5)
        if new_zone > zone_i:
            zone_i = new_zone                              # RATCHET: never down
            top = top_k_unique_configs(study, promote_k)
            study = new_study()
            for cfg_idx in top:
                score_k = cache-or-evaluate(cfg_idx, zones[zone_i])  # counts fits
                cache it; study.add_trial(completed trial with cfg_idx, score_k)
            incumbent = best of the re-scored top; incumbent_score = its NEW score
            # (never compare across fractions: incumbent_score is always at the
            #  current fraction)

# ---- forced final polish for THIS start (always) ----
if zones[zone_i] < 1.0:
    top = top_k_unique_configs(study, promote_k)  (incumbent guaranteed included)
    for cfg_idx in top: cache-or-evaluate at frac 1.0   # "final-polish" events

return {"start_point": start_point, "incumbent": incumbent,
        "score": <incumbent's score AT FRACTION 1.0>, "history": [...]}
```

Notes:
- `study.enqueue_trial(...)` (NOT a manual `create_trial`) is the correct optuna
  API for seeding the very next `ask()` with a specific point — use it for the
  start-point seed only, at the beginning of each start's first study.
- `study.add_trial` with a manually-constructed completed
  `optuna.trial.create_trial(params=..., distributions=..., value=...)` is the
  supported way to seed a *fresh per-zone* study with already-known scores when
  promoting — use it there (NOT `enqueue_trial`, which would re-fit instead of
  injecting the known score).
- `top_k_unique_configs`: completed trials sorted by value desc, dedup by idx
  tuple, take k. Fewer than k exist → take what exists.

---

## 5. `BullseyeController` — extract the fidelity methodology into a shared class

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
  own trajectory in `PatternSearchCV`.

**Scope control: do NOT refactor `Climber` to use this class in this task.** Add a
one-line comment in `_climber.py` noting the future unification. Keeping the
proven PatternSearchCV path untouched is deliberate — its behavior is pinned by
trace tests and benchmark history. (This is the same "behavior-preserving only"
discipline as the `_select_starts` extraction in §2.1 — extract what's genuinely
shared, touch nothing else.)

Unit-test the controller standalone against hand-computed sequences (see §7).

---

## 6. Determinism

Given `random_state`: one `check_random_state(...)` rng created in `_prepare_run`,
used for (in order, so the sequence is reproducible): (a) `select_starts`'s QMC
seed, (b) a per-(start, zone-study) optuna sampler seed drawn from that same rng
for every `new_study()` call across every start, in start-then-zone order.
`GPSampler(seed=..., deterministic_objective=True)`. Two fits with the same
`random_state` (including with `n_starts>1`) must produce identical
`cv_results_` and identical `local_optima_` — test this with `sampler="tpe"` so
it runs without torch.

---

## 7. Tests (add `tests/test_bayes.py` + `tests/test_fidelity.py` +
`tests/test_starts.py`; keep every existing test green — currently 105 pass, 2 skip)

**Step 0, before writing any new estimator code**: perform the §2.1 refactor,
then run `pytest tests -q` and confirm 105 passed / 2 skipped, unchanged. This is
a hard gate, not a suggestion.

`tests/test_starts.py` (pure unit, no sklearn):
1. `select_starts` produces identical output to `PatternSearchCV`'s pre-refactor
   `_select_starts` for a range of `(n_starts, start_points, seed)` combinations
   (regression proof the extraction was behavior-preserving).
2. Calling `select_starts` directly with identical arguments from both
   "as PatternSearchCV would call it" and "as BayesHalvingSearchCV would call it"
   produces identical results — proving true shared code, not parallel logic.

`tests/test_fidelity.py` (pure unit, no sklearn):
3. warmup counting: with `warmup=3`, first two improvements buy nothing; the
   calibration uses exactly the 2 readings; `D` floored to `min_step` multiples.
4. boundary formula, descending, innermost floored at `min_step`.
5. ratchet: a big move after a climb never lowers the zone.
6. zero-move is not a reading.

`tests/test_bayes.py` (use `DecisionTreeRegressor`, `make_regression(400)`, small
grids, `sampler="tpe"` everywhere except tests explicitly marked
`@pytest.mark.skipif(torch missing)`):
7. basic fit, `n_starts=1`: `best_params_` from grid; `n_resources` in
   `cv_results_`; `best_index_` row has max `n_resources`; ledger rows unique per
   (params, n_resources); `local_optima_` has exactly one entry.
8. determinism: two identical-seed fits (`n_starts=1` and `n_starts=3`) → each
   pair produces identical `cv_results_` scores and identical `local_optima_`.
9. budget: number of `cv_results_` rows per start ≤ `n_iter` + (promote/polish
   overhead bound: `promote_k+1` per climb + final polish); cache prevents
   duplicate (params, fraction) fits **across starts, not just within one**.
10. zones ratchet + final polish: fractions in `search_history_` are
    non-decreasing **per start**; at least one evaluation at fraction 1.0 exists
    per start; `best_*` overall comes from the best such row across all starts.
11. `data_zones=1` (ladder off) works and every row is full-data.
12. multi-start: `n_starts=4` produces `local_optima_` with between 1 and 4
    entries (dedup by final point); `search_history_` entries carry a start
    index; `start_points` explicit list takes priority seats (mirror
    `PatternSearchCV`'s `test_start_points_take_seats`); the shared cache
    measurably reduces total genuine fits vs. running 4 independent
    single-start fits with `n_jobs`/cache reset between them (assert
    `n_cache_hits_ > 0` for a case constructed to guarantee overlap, e.g. two
    identical `start_points`).
13. verbose header names the metric, sampler, zones, and `n_starts` (caplog,
    mirroring `test_verbose_header_*` in `test_estimator.py`); verbose=0 emits
    nothing and runs no CV-summary fits.
14. invalid params raise: bad `sampler`, `n_iter<1`, `promote_k<1`, bad zones,
    bad `n_starts` (mirror `PatternSearchCV`'s validation-error tests).
15. optuna missing → informative ImportError (monkeypatch the import).
16. `parametrize_with_checks` on a `BayesHalvingSearchCV(DecisionTreeClassifier...,
    sampler="tpe", n_iter=8, cv=3, random_state=0)` instance — full sklearn gate,
    same as `test_sklearn_compat.py` does for `PatternSearchCV` (append there or
    new file; expect the same tag/pickling requirements to bite if §2 items were
    skipped).
17. pickling: fit with `n_jobs=2` and `n_starts=2` succeeds and the fitted
    estimator pickles (regression for §2 item 5, exercised with multi-start).

Run: `.venv\Scripts\python.exe -m pytest tests -q` from the repo root. GP-marked
tests additionally runnable from `psc-opt` after `pip install -e .[test,bayes]` there.

---

## 8. Benchmark deliverable (after code is green)

Notebook `BHS_vs_PSC_26grid.ipynb` in the repo root, patterned on
`PE_Round_0.5_1_10_100.ipynb` (same pipeline cell — copy it from
`Prototype_Replication.ipynb` cells[1] like every other benchmark notebook does;
same official grid; `TimeSeriesSplit(5)`; MAE). **Must run from the `psc-opt` venv
(GP needs torch)** — register a kernel for it or run as a script writing JSON, the
way `run_gp.py` did for Experiment 4.

Primary arms (all `random_state=0`, `subsample="stratified"`, zones
`(0.005,0.01,0.1,1.0)`, **`n_starts=1`** — directly comparable to the reference
rows in §0):
1. `BayesHalvingSearchCV(sampler="gp", n_iter=25, n_starts=1)`
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

## 9. Known pitfalls from this project's history (each cost real time — read)

1. **Pickling during parallel fit** (§2 item 5). The #1 landmine.
2. `evaluate_candidates` results are **cumulative** — always slice `[-len(batch):]`.
3. `more_results` key must be passed on **every** call or columns desync.
4. The zone splitter must yield the **same `n_splits`** as the original cv —
   `BaseSearchCV` asserts it. `ZoneSplitter` already guarantees this; don't wrap it.
5. Windows + torch: §1. Never `pip install torch` (unpinned) in the deep venv.
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
    or `_engine.py` is a scope violation — if something seems to require a
    second change there, stop and reconsider the `BayesHalvingSearchCV`-side
    design instead of reaching for a shared-file edit.

## 10. Out of scope (do not do)

- Do not modify `Climber` or `Engine` at all.
- The **only** permitted change to `PatternSearchCV`/`_search.py` is the
  mechanical, behavior-preserving extraction described in §2.1 (§9 item 11).
- Do not touch `EXPERIMENTS.md` history or renumber experiments (append only).
- Do not add ASHA/successive-halving elimination between starts — the fidelity
  schedule is the bullseye, and multi-start runs every start to completion, per
  the user's explicit design (§3.2).
- Do not invent a cross-start "merging" heuristic (§3.2) — the shared cache is
  the intended cost-saving mechanism.
- Do not vendor or reimplement GP internals — optuna is the engine.
- Do not add a `bayes-gp` (torch-including) extras group to `pyproject.toml` (§1)
  — we have verified this fails on this platform; don't advertise it as installable.
- `OpenQuestions.md` is a local untracked scratch file — leave it out of commits.
