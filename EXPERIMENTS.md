# Experiment Log — PatternSearchCV vs. Prototype vs. Bayesian Optimization

Plain-language record of every benchmark run so far. Newest entries at the bottom.

---

## The task

Tune an `ExtraTreesRegressor` on the Italian retail sales dataset
(`C:\FILES\Code\Benchmarking\train.csv`, 523,021 rows; training = first 80% =
418,416 rows; target = `NumberOfSales`).

- **Search space — there are TWO, because the prototype notebook holds two
  recorded runs** (clarified 2026-07-13): common axes `max_features` ∈ {2, 3, 4}
  and `max_depth` ∈ {5, …, 17}; **Run A** (earlier prototype version) used
  `n_estimators` ∈ {10, 20, …, 260} (26 values → 1,014 grid points); **Run B**
  (later version, cell 110) used `n_estimators` ∈ {10, 70, 130, 190, 250}
  (5 values → 195 points). Experiment 1 matches Run A's grid; Experiments 2 and
  3 match Run B's.
- **CV**: `TimeSeriesSplit(n_splits=5)` — **Scoring**: mean absolute error (lower = better)
- **Data pipeline**: exact replication of the prototype notebook — int64→int32,
  object→category codes (`Date` included), five visibility/gust/cloud columns dropped,
  `NumberOfCustomers` kept as a feature (as the prototype had it), 80/20 chronological split.
- **This machine**: 8 cores, Windows 10, Python 3.11, scikit-learn 1.9.0, numpy 2.4.6.

**Cost metric.** Different methods evaluate on different amounts of data, so raw
evaluation counts mislead. *Full-fit equivalents* = each evaluation weighted by the
fraction of rows it used (an evaluation on 10% of data costs 0.10). For methods that
always use all rows, equivalents = evaluations.

---

## Reference records (from the original prototype notebook, not re-run)

Source: `DatasetSize_and_ParamOpt_WORKING_(3Large_Aug_30_2025).ipynb` (kept untouched).

| method | grid | evaluations | full-fit equiv | best (their scoring) | wall-clock | machine |
|---|---|---|---|---|---|---|
| Prototype **Run A** (V1, scored R²) | 26-value (1,014 pts) | 18 | 18.00 | R² 0.80998 at `{150, 4, 17}`¹ | 867.34 s | original |
| Prototype **Run B** (V2, scored MAE) | 5-value (195 pts) | 11 | 11.00 | MAE **805.038** at `{130, 4, 17}` | 1212.28 s | original |
| skopt `gp_minimize`, `n_calls=15` | 26-value | 15 | 15.00 | n/a — optimized R² | n/a (2019-era run) | original |

¹ Params shown as `{n_estimators, max_features, max_depth}`.

Notes: skopt enforces a minimum of 10 calls; it is archived/unmaintained and cannot run
on this numpy 2.x stack, so its numbers enter as recorded facts only.

---

## Experiment 1 — New `PatternSearchCV`, default configuration (2026-07-13)

Notebook: `Prototype_Replication.ipynb` (executed; full logs inside).

> **Grid note (resolved)**: this run uses the 26-value grid — which matches
> prototype **Run A** exactly. Head-to-head vs Run A: **same optimum found,
> `{150, 4, 17}`** (Run A scored R², this run MAE — different objectives, same
> argmax); 23 evals / **6.80** equivalents vs Run A's 18 evals / **18.00**
> equivalents — 62% less compute. The +5 evaluation difference on the same grid
> is almost exactly the fidelity protocol's overhead (re-score on each data
> climb + the full-data polish sweep). Experiment 3 covers the Run B (5-value)
> grid.

**Configuration**: defaults — 1 start (grid midpoint `{130, 3, 11}`), data zones
10/20/50/100%, `warmup=3`, `subsample='auto'` → **expanding** (oldest rows first,
because CV is TimeSeriesSplit), `poll='auto'` → **opportunistic** (8 cores ÷ 5 folds
< 2, so one probe's folds already fill the machine), `n_jobs=-1`, `random_state=0`.

**Results**

| metric | value |
|---|---|
| evaluations | 23 (18 at 10% data, 5 at 100%) |
| full-fit equivalents | **6.80** |
| cache hits (fits avoided) | 19 |
| wall-clock (this machine) | **746.8 s** |
| best params | `{n_estimators: 150, max_features: 4, max_depth: 17}` |
| best CV MAE (full data) | 805.730 |
| held-out MAE (last 20% of rows) | 784.714 |

**Findings**

1. **Same answer, 38% less compute than the prototype.** 6.80 vs 11.00 full-fit
   equivalents; CV MAE 805.73 vs 805.04 (0.09% apart — inside the ±22 fold-std noise);
   same basin (`max_features=4, max_depth=17`, 150 vs 130 trees).
2. **vs Bayesian optimization: under half the compute.** 6.80 vs 15.00 equivalents.
   Raw evaluation *count* is higher (23 vs 15) — the honest claim is "matches the
   optimum at less than half the compute," not "fewer evaluations."
3. **The intermediate zones (20%, 50%) were never used.** The search converged at 10%
   data in ~3 moves (warm-up consumed most of them), then took the forced jump straight
   to the 100% polish. All savings came from the two-phase effect: cheap exploration
   (18 × 0.10 = 1.8 equiv) + full-data confirmation (5 × 1.0 = 5.0 equiv). On small
   grids the graduated ladder barely engages — multi-start / larger grids are where the
   middle rungs should matter (to be tested).
4. Calibration readings were volatile (0.855 then 0.040) — the mean+floor rule handled
   it, but it confirms the choice of mean over max.

**Bug found & fixed during this experiment**: with `n_jobs=-1`, sklearn pickles the
estimator for every parallel task; the fitted instance was carrying unpicklable state
(logging handler, live generators) → `PicklingError`. Fixed (machinery now lives only
in local frames); regression test added (`test_parallel_n_jobs_pickles_estimator`).
Suite: 91 passed, 2 skipped, including sklearn's `parametrize_with_checks`.

---

## Experiment 2 — Prototype re-run on THIS machine (2026-07-13, done)

Notebook: `Prototype_Original_Timed.ipynb`. Purpose: put prototype and new version on
the *same hardware and same sklearn 1.9*, removing the cross-machine caveat from the
wall-clock comparison.

Method: original cells reproduced **verbatim** with two disclosed changes only —
data path localized, and `sklearn.utils._joblib` import shimmed to `joblib` (pure
re-export removed from sklearn ≥1.3; smoke-tested identical on toy data). The original
notebook remains untouched as the archive of the original timings.

**Results**

| metric | value |
|---|---|
| evaluations | 11 (10 + initial), all at 100% data |
| best params | `{n_estimators: 130, max_features: 4, max_depth: 17}` |
| best CV MAE | **805.038061 — exact reproduction** of the original record |
| wall-clock (this machine) | **1126.28 s** (original machine: 1212.28 s) |

**Findings**

1. **Determinism verified end-to-end**: same fits, same path, same MAE to 6 decimals
   as the Aug 2025 run — on different hardware and a newer sklearn.
2. **Same-machine wall-clock: new version is 1.51× faster** (746.8 s vs 1126.3 s) —
   *while searching a 5× larger grid* (see discovery below). The exact-grid rematch
   (Experiment 3) will state the clean number.
3. **Discovery — the real prototype grid is smaller than assumed.** The execution
   cell's `param_dist` uses `n_estimators = linspace(10, 250, 5)`; the 26-value
   version is commented out ("Reduced num for faster example"). Experiment 1's grid
   was therefore wrong; Experiments 3 and 4 use the exact 195-point grid.

---

## Experiment 3 — CANCELLED (user decision)

The 5-value-grid comparison (`Replication_ExactGrid.ipynb`) was cancelled
mid-run: the grid of record for all comparisons is Run A's 26-value grid.
Run B remains in the reference table only as the determinism proof
(Experiment 2 reproduced its MAE to 6 decimals on this machine).

---

## Experiment 4 — Trajectory match vs Run A (SUPERSEDED, never executed)

Stopped before running at the user's request; replaced by Experiments 6 and 7,
which compare full runs rather than trajectories. Notebook remains available.

## Experiment 4 (original plan below, kept for the record)

Notebook: `Trajectory_Match_RunA.ipynb`. The new package configured exactly
like Run A: 26-value grid in Run A's dimension order, `scoring='r2'`,
`data_zones=1` (all evaluations on 100% of rows), single midpoint start.
Compares every visited grid point 1:1 against Run A's 18 recorded rows, checks
score agreement on shared points (cross-machine determinism), and isolates the
effect of the two bug fixes from the effect of the bullseye (which is off here).

Expected divergences, predicted in advance: initial `n_estimators` step 12 vs
Run A's 13 (probe #2 becomes 250 vs 260 — different half-width convention on
even-length dims), and different behavior after failed pattern moves (Run A
contracted — the premature-contraction bug; the new version re-explores).

*(results pending)*

---

## Experiment 5 — Optuna Bayesian baselines (2026-07-14, TPE done; GP running)

Notebook: `Optuna_Baseline.ipynb` (TPE) + `C:\FILES\Code\Benchmarking\psc-opt\run_gp.py`
(GPSampler — needs torch, which required a short-path venv: torch's nested file
paths exceed Windows' 260-char limit under the package venv, and torch 2.13's
DLLs fail on this Windows 10 build; torch 2.5.1 works). 15 trials per sampler
(the recorded `gp_minimize(n_calls=15)` budget), 26-value grid, MAE objective,
every trial on 100% of rows, seeded.

**TPE results**

| metric | value |
|---|---|
| trials | 15 (15.00 full-fit equivalents) |
| wall-clock | 828.7 s |
| best MAE | 810.553 at (4, 100, 17) |
| best after 11 trials | 811.500 |

TPE never found the (4, 150, 17) optimum: its best is 4.8 MAE (0.6%) worse than
both pattern searches, at the same wall-clock as the new algorithm's entire
run and 2.2× its compute. Trial curve: 10 scattered startup trials (worst
1415 MAE), then convergence toward max_depth=17 without time to refine
n_estimators.

**GP results**

| metric | value |
|---|---|
| trials | 15 (15.00 full-fit equivalents) |
| wall-clock | 964.6 s |
| best MAE | **805.730 at (4, 150, 17) — found the optimum** (first reached at trial 13/15) |
| best after 11 trials | 811.500 |

---

## FINAL FOUR-WAY TABLE — 26-value grid, this machine, same sklearn 1.9

| | V1 prototype (pasted, R²) | **NEW PatternSearchCV** (MAE) | Optuna TPE (MAE) | Optuna GP (MAE) |
|---|---|---|---|---|
| evaluations | 33 | 23 | 15 | 15 |
| full-fit equivalents | 33.00 | **6.80** | 15.00 | 15.00 |
| wall-clock | 1710.9 s | **824.1 s** | 828.7 s | 964.6 s |
| best point | (4, 150, 17) | (4, 150, 17) | (4, 100, 17) | (4, 150, 17) |
| CV MAE of best | 805.730 | 805.730 | 810.553 | 805.730 |
| CV R² of best | 0.809981 | 0.809981 | — | 0.809981 |

**Conclusions**

1. Three of four methods found the identical optimum (4, 150, 17); TPE missed it
   (0.6% worse) within the 15-trial budget.
2. The new PatternSearchCV matched the best Bayesian sampler's answer at
   **2.21× less compute** (6.80 vs 15.00 equivalents) and 1.17× faster
   wall-clock — and GP only reached the optimum at trial 13, i.e. it needed
   ~13.0 equivalents to first hit what the new algorithm reached within 6.80
   total.
3. At the prototype's historical 11-evaluation budget, both Bayesian samplers
   were still at 811.5 MAE — worse than every pattern-search variant.
4. vs its own prototype ancestor (as pasted): identical answer, 2.08× faster,
   4.85× less compute.
5. The original project claim — "fewer iterations and faster than Bayesian
   optimization" — is now MEASURED on the project's own dataset in its honest
   form: *same answer as the best modern Bayesian optimizer at less than half
   the compute, with determinism the Bayesian methods lack.*

---

## Experiment 6 — Head-to-head, 26-value grid, MAE objective (2026-07-14, done)

Notebook: `HeadToHead_26grid.ipynb`. New algorithm (opportunistic forced, default
4-zone bullseye) vs the V2 prototype cell (bugs intact) with its own commented
26-value grid line activated. Same kernel, sequential, same MAE objective.

| | OLD (V2 prototype, bugs) | NEW (PatternSearchCV) |
|---|---|---|
| evaluations | 17 | 23 |
| full-fit equivalents | 17.00 | **6.80** |
| wall-clock | 1546.2 s | **824.1 s** |
| best point | (4, 130, 17) | (4, 150, 17) |
| CV MAE of best | **805.038** | 805.730 |
| CV R² of best | 0.809692 | **0.809981** |

Speedup 1.88× wall-clock, 2.50× compute; quality a statistical tie split both
ways (Δ far inside the ±22 fold std). Zones used: 10% and 100% only, exactly as
predicted. New run's winner = Run A's recorded optimum.

---

## Experiment 7 — V1 prototype exactly as pasted (2026-07-14, done)

Notebook: `V1_Prototype_26grid.ipynb`. The user-specified prototype side: the V1
class + its "#Execute Pattern Search:" cell **verbatim** (default scoring = R²,
`clf n_jobs=-1`), 26-value grid. Six disclosed plumbing shims (joblib path,
sklearn-1.x base class, `iid=`, `error_score` value, `df.append`→`pd.concat`,
`time` shadowing in timing lines); the search loop byte-identical.

| metric | value |
|---|---|
| evaluations | 33, all at 100% data |
| full-fit equivalents | 33.00 |
| wall-clock | 1710.9 s |
| best point | (4, 150, 17) |
| CV R² of best | 0.809981 |
| CV MAE of best | 805.730 |

**THE COMPARISON THE USER ASKED FOR** (both on this machine, 26-value grid):

| | V1 prototype (as pasted, R²) | NEW PatternSearchCV (MAE) |
|---|---|---|
| evaluations | 33 | 23 |
| full-fit equivalents | 33.00 | **6.80** (4.85× less) |
| wall-clock | 1710.9 s | **824.1 s** (2.08× faster) |
| best point | (4, 150, 17) | (4, 150, 17) — **identical** |
| CV R² of best | 0.809981 | 0.809981 — identical |
| CV MAE of best | 805.730 | 805.730 — identical |

**Identical optimum, identical quality, 2.08× wall-clock, 4.85× compute.**

Caveats recorded honestly: (a) the two searches optimized different objectives
(R² vs MAE — "exactly as pasted" implies R²) and still chose the same point;
(b) this V1 run took 33 evals vs Run A's recorded 18 — its trajectory shows
both-directions polling around a fixed center with quarter-width initial steps,
i.e. the V1 class cell as saved differs from whatever state produced Run A's
867 s record (the notebook's class cell was evidently edited after Run A ran).
The saved-code pairing is what was benchmarked; (c) V1's wall-clock includes
nested parallelism (`clf n_jobs=-1` inside search `n_jobs=-1`), as in the
original configuration.

---

## Open questions queued for future experiments

- `subsample='stratified'` (transition sampling) vs `'expanding'` on this dataset —
  does a full-timeline 10% sample find the basin faster/more reliably than the
  oldest-tenth sample? (Signature feature of the package; article figure.)
- `n_starts ∈ {1, 2, 4, 8}` at same budget and same wall-clock (multi-start ablation).
- `poll='complete'` on this 8-core machine (expected ~neutral single-start; relevant
  with multi-start batching).
- Whether intermediate data zones engage on larger grids / multi-start (finding 3).
- Optuna (TPE + GPSampler) as the modern Bayesian baseline — skopt is unmaintained.
