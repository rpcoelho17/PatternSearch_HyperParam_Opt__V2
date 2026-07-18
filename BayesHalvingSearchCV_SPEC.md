# BayesHalvingSearchCV — Spec / Architectural Document

**Audience:** Claude Sonnet 5, implementing and testing this estimator inside the
existing `pattern-search-cv` repository.
**Status:** agreed design, ready for implementation.
**Author of record:** design session 2026-07-15 (user + Claude Fable 5).

---

## 0. Mission

`Optuna GPSampler` (all trials at 100% data) found this project's benchmark optimum
in 15 trials — fewer *evaluations* than `PatternSearchCV` — but paid full price for
every one of them: **15.00 full-fit equivalents vs PatternSearchCV's ~5.04–5.85**
(see `EXPERIMENTS.md`, Experiment 4 vs Experiments 7–11). Its weakness is not the
search logic; it is that it has no multi-fidelity machinery.

**Build `BayesHalvingSearchCV`: a scikit-learn-compatible estimator that keeps this
package's entire multi-fidelity infrastructure — stratified priority ordering, data
zones, the bullseye ring methodology, the shared dedup cache, the full-data-only
best selection — but replaces the Hooke-Jeeves search logic with Optuna's GP-based
Bayesian optimization.** Afterwards it will be benchmarked head-to-head against
`PatternSearchCV` on the official test space.

Reference numbers the new estimator is trying to beat (official space:
`max_features` {2,3,4} × `n_estimators` {10..260 step 10} × `max_depth` {5..17},
523K-row retail dataset, `TimeSeriesSplit(5)`, MAE):

| | evaluations | full-fit equiv | best CV MAE | wall-clock |
|---|---|---|---|---|
| Optuna GP, all-100%-data (Exp. 4) | 15 | 15.00 | 805.730 at (4,150,17) | 964.6 s |
| PatternSearchCV, current defaults (Exp. 10) | 22 | 5.09 | 805.038 at (4,130,17) | 443.8 s |

Success looks like: GP-quality answers (≈805) at well under 15 equivalents.
Anything ≤ ~7 equivalents with the right basin is a strong result.

**Naming note:** the user chose the name `BayesHalvingSearchCV`. The fidelity
mechanism is the bullseye-zones design, *not* classic successive halving — keep the
user's name; add one line in the docstring clarifying the mechanism.

---

## 1. Where it lives, environments, hard-won platform facts

- Repo root: `C:\FILES\Code\Benchmarking\Working_on_Train_Set\V2025\pattern-search-cv`
  (git: `https://github.com/rpcoelho17/PatternSearch_HyperParam_Opt__V2`, branch `main`).
- Package venv: `.venv` inside the repo — Python 3.11, sklearn 1.9.0, numpy 2.4.6,
  optuna 4.9.0 installed. **torch is NOT installable in this venv**: two verified
  Windows failures — (a) torch's nested license paths exceed the 260-char limit
  under this deep venv path; (b) torch 2.13 wheels' `c10.dll` fails to initialize
  on this Windows 10 build regardless of path.
- Torch-capable venv (verified working today): `C:\FILES\Code\Benchmarking\psc-opt`
  — torch 2.5.1+cpu, optuna 4.9.0, sklearn 1.9.0, numpy 2.4.6. The known-good torch
  recipe is `pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu`
  from a SHORT filesystem path. `pattern-search-cv` is not yet installed there;
  `pip install -e` it when running GP-dependent tests/benchmarks.
- Jupyter kernel `psc-venv` is registered for the package venv; benchmark notebooks
  are executed headlessly with
  `.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=psc-venv <nb>`.
- `optuna.samplers.GPSampler` requires torch ⇒ **GP-path tests cannot run in the
  package venv.** This drives the `sampler` parameter design in §3 (TPE fallback
  keeps the estimator testable everywhere; GP tests are `skipif torch missing`).

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
   call — preferred).
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
   promote_k, warmup, zones), every fidelity decision as it happens, and the
   end-of-run `_log_cv_summary` (per-fold EV/MAE/MSE/RMSE/R2 + means, fit/score
   times — copy or share PatternSearchCV's implementation). `verbose=0` = silent,
   no extra fits. **Log every explicitly-passed knob too** — a prior bug logged
   `poll` only when auto-resolved; don't repeat it.

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
                 n_iter=25,               # total NEW model evaluations (cache hits free)
                 promote_k=3,             # top-k configs re-scored when a zone climbs
                 # --- multi-fidelity (identical semantics to PatternSearchCV) ---
                 data_zones=(0.005, 0.01, 0.1, 1.0),
                 warmup=3,
                 subsample="auto",        # "auto"|"expanding"|"stratified"|"random"
                 subsample_columns=None):
```

- `param_grid`: same forms as PatternSearchCV (lists or `(low, high, num)` tuples)
  → `Space`.
- `sampler`: `"gp"` = `optuna.samplers.GPSampler(seed=<int from random_state>,
  deterministic_objective=True)`; `"tpe"` = `optuna.samplers.TPESampler(seed=...)`.
  optuna imported **inside fit** with an informative `ImportError` message if
  missing; same for the torch requirement when `sampler="gp"` (catch the
  GPSampler-needs-torch failure and re-raise with the §1 install recipe in the
  message). Suppress `optuna.exceptions.ExperimentalWarning` for GPSampler and set
  `optuna.logging.set_verbosity(WARNING)` during fit.
- `n_iter`: budget of *genuine fits* across all zones combined, **excluding** the
  final-polish re-scores (which are ≤ `promote_k`+1 extra evaluations, reported in
  `cv_results_` like everything else). Cache-served proposals don't consume budget;
  guard with `max_asks = 10 * n_iter` total ask() calls to make infinite dedup
  loops impossible (if hit, log a warning and proceed to final polish).
- Fitted attributes: standard SearchCV surface + `n_resources` key in
  `cv_results_` + `search_history_` (list of plain dicts: trial number, params,
  fraction, score, event tag e.g. "trial"/"climb-rescore"/"final-polish") +
  `n_cache_hits_`. No `local_optima_` (single-model search; omit).
- Add the class to `__init__.py`'s `__all__` and exports.
- Docstring: numpydoc, same caveat-forward style as PatternSearchCV's (evidence
  provenance, name-vs-mechanism note from §0).

---

## 4. Search algorithm (normative)

### 4.1 Optuna integration mode

Use optuna's **ask/tell** API — no `study.optimize`, no objective callback, no
nameservers. One `Study` per data zone (fresh GP per fidelity — scores from
different fractions are never mixed in one surrogate; this is the same
never-compare-across-fractions rule PatternSearchCV enforces).

Suggestion space = **index space**, mirroring `Space`'s geometry:
- numeric dimension `d` → `trial.suggest_int(name, 0, d.n - 1)` (the GP sees
  ordinal structure; convert index→value via `Space` when building params).
- categorical dimension → `trial.suggest_categorical(name, list(range(d.n)))`.

### 4.2 Main loop (pseudocode)

```
zones, sizes  = validated ladder + resource floor      (reuse §2 item 4)
order         = priority ordering per `subsample`      (reuse; "auto": stratified
                for TimeSeries* cv, else random — same rule as PatternSearchCV)
splitters     = {frac: ZoneSplitter(cv, order[:size]) or None for full data}
cache         = {}   # (idx_tuple, frac) -> score      (integer keys, exact)
controller    = BullseyeController(space.min_step, n_boundaries=len(zones)-1,
                                   warmup=warmup)      (§5)
zone_i        = 0
incumbent     = None (idx_tuple), incumbent_score = None
fits_used     = 0; asks = 0

new_study():  study = optuna.create_study(direction=maximize, sampler=fresh
              seeded sampler)   # maximize: sklearn scores are greater-is-better
study = new_study()

while fits_used < n_iter and asks < max_asks:
    trial = study.ask(); asks += 1
    idx   = tuple(suggested indices)
    frac  = zones[zone_i]
    if (idx, frac) in cache:
        study.tell(trial, cache[(idx, frac)])          # free, no budget
        n_cache_hits += 1
        continue
    score = evaluate_batch(frac, [idx])[0]             # §2 items 1–2
    fits_used += 1
    cache[(idx, frac)] = score
    study.tell(trial, score)

    if incumbent is None or score > incumbent_score:
        move = 0.0 if incumbent is None else space.distance(incumbent, idx)
        incumbent, incumbent_score = idx, score
        new_zone = controller.observe_improvement(move)   # §5
        if new_zone > zone_i:
            zone_i = new_zone                              # RATCHET: never down
            # promote: top-k configs of the finished study by value
            top = top_k_unique_configs(study, promote_k)
            study = new_study()
            for cfg_idx in top:
                score_k = cache-or-evaluate(cfg_idx, zones[zone_i])  # counts fits
                cache it; study.add_trial(completed trial with cfg_idx, score_k)
            incumbent = best of the re-scored top; incumbent_score = its NEW score
            # (never compare across fractions: incumbent_score is always at the
            #  current fraction)

# ---- forced final polish (always, mirrors PatternSearchCV) ----
if zones[zone_i] < 1.0:
    top = top_k_unique_configs(study, promote_k)  (incumbent guaranteed included)
    for cfg_idx in top: cache-or-evaluate at frac 1.0   # "final-polish" events
# best_* selection then happens automatically via _select_best_index (full-data
# rows only). If nothing was ever evaluated at 1.0 (cannot happen given the
# polish), that's a bug.
```

Notes:
- optuna `study.add_trial` with a manually-constructed completed
  `optuna.trial.create_trial(params=..., distributions=..., value=...)` is the
  supported way to seed the fresh per-zone study; use it (NOT `enqueue_trial`,
  which would re-fit instead of injecting the known score).
- `top_k_unique_configs`: completed trials sorted by value desc, dedup by idx
  tuple, take k. Fewer than k exist → take what exists.
- All of this lives in local frames of `_run_search` (§2 item 5).

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

**Scope control: do NOT refactor `Climber` to use this class in this task.** Add a
one-line comment in `_climber.py` noting the future unification. Keeping the
proven PatternSearchCV path untouched is deliberate — its behavior is pinned by
trace tests and benchmark history.

Unit-test the controller standalone against hand-computed sequences (see §7).

---

## 6. Determinism

Given `random_state`: seed the optuna sampler with `check_random_state(...).randint(2**31-1)`
once per study creation, drawing from ONE rng created in `_prepare_run` (so the
per-zone studies get a deterministic seed sequence). `GPSampler(seed=...,
deterministic_objective=True)`. Two fits with the same `random_state` must produce
identical `cv_results_` (test this with `sampler="tpe"` so it runs without torch).

---

## 7. Tests (add `tests/test_bayes.py` + `tests/test_fidelity.py`; keep every
existing test green — currently 105 pass, 2 skip)

`test_fidelity.py` (pure unit, no sklearn):
1. warmup counting: with `warmup=3`, first two improvements buy nothing; the
   calibration uses exactly the 2 readings; `D` floored to `min_step` multiples.
2. boundary formula, descending, innermost floored at `min_step`.
3. ratchet: a big move after a climb never lowers the zone.
4. zero-move is not a reading.

`test_bayes.py` (use `DecisionTreeRegressor`, `make_regression(400)`, small grids,
`sampler="tpe"` everywhere except tests explicitly marked
`@pytest.mark.skipif(torch missing)`):
5. basic fit: `best_params_` from grid; `n_resources` in `cv_results_`;
   `best_index_` row has max `n_resources`; ledger rows unique per
   (params, n_resources).
6. determinism: two identical-seed fits → identical `cv_results_` scores.
7. budget: number of `cv_results_` rows ≤ `n_iter` + (promote/polish overhead
   bound: `promote_k+1` per climb + final polish); cache prevents duplicate
   (params, fraction) fits.
8. zones ratchet + final polish: fractions in `search_history_` are
   non-decreasing; at least one evaluation at fraction 1.0 exists; `best_*` comes
   from it.
9. `data_zones=1` (ladder off) works and every row is full-data.
10. verbose header names the metric, sampler, and zones (caplog, mirroring
    `test_verbose_header_*` in `test_estimator.py`); verbose=0 emits nothing and
    runs no CV-summary fits.
11. invalid params raise: bad `sampler`, `n_iter<1`, `promote_k<1`, bad zones.
12. optuna missing → informative ImportError (monkeypatch the import).
13. `parametrize_with_checks` on a `BayesHalvingSearchCV(DecisionTreeClassifier...,
    sampler="tpe", n_iter=8, cv=3, random_state=0)` instance — full sklearn gate,
    same as `test_sklearn_compat.py` does for PatternSearchCV (append there or new
    file; expect the same tag/pickling requirements to bite if §2 items were
    skipped).
14. pickling: fit with `n_jobs=2` succeeds and the fitted estimator pickles
    (regression for §2 item 5).

Run: `.venv\Scripts\python.exe -m pytest tests -q` from the repo root. GP-marked
tests additionally runnable from `psc-opt` after `pip install -e .[test]` there.

---

## 8. Benchmark deliverable (after code is green)

Notebook `BHS_vs_PSC_26grid.ipynb` in the repo root, patterned on
`PE_Round_0.5_1_10_100.ipynb` (same pipeline cell — copy it from
`Prototype_Replication.ipynb` cells[1] like every other benchmark notebook does;
same official grid; `TimeSeriesSplit(5)`; MAE). **Must run from the `psc-opt` venv
(GP needs torch)** — register a kernel for it or run as a script writing JSON, the
way `run_gp.py` did for Experiment 4.

Arms (all `random_state=0`, `subsample="stratified"`, zones `(0.005,0.01,0.1,1.0)`):
1. `BayesHalvingSearchCV(sampler="gp", n_iter=25)`
2. `PatternSearchCV` current defaults (patient) — fresh run, same session, for a
   same-machine wall-clock pairing.

Report per arm, in the user's standard comparison-table format (columns = runs,
rows = exactly): zones ladder, evaluations, full-fit equiv, wall-clock, best
point, CV MAE of best. Plus the trial-by-trial (params, fraction, MAE) history
for the BHS arm. Machine-noise rule from EXPERIMENTS.md applies: wall-clock
differences under ~15–25% are noise; full-fit equivalents is the primary metric;
NEVER use median-of-per-eval-ratios (documented bad metric — see the Experiment 6
correction).

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
    `check_supervised_y_2d`, and the pickle checks pass. PatternSearchCV's
    solutions are the template.

## 10. Out of scope (do not do)

- Do not modify `Climber`, `Engine`, or any `PatternSearchCV` behavior/defaults.
- Do not touch `EXPERIMENTS.md` history or renumber experiments (append only).
- Do not add ASHA/successive-halving elimination — the fidelity schedule is the
  bullseye, per the user's explicit design.
- Do not vendor or reimplement GP internals — optuna is the engine.
- `OpenQuestions.md` is a local untracked scratch file — leave it out of commits.
