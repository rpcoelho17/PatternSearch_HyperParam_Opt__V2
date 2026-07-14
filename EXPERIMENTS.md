# Experiment Log — PatternSearchCV vs. Prototype vs. Bayesian Optimization

Plain-language record of every benchmark run. Newest entries at the bottom.

---

## The task

Tune an `ExtraTreesRegressor` on the Italian retail sales dataset
(`C:\FILES\Code\Benchmarking\train.csv`, 523,021 rows; training = first 80% =
418,416 rows; target = `NumberOfSales`).

**The official test space** (1,014 grid points) — every experiment in this log
ran on it:

- `max_features` ∈ {2, 3, 4}
- `n_estimators` ∈ {10, 20, 30, …, 260} (26 values)
- `max_depth` ∈ {5, 6, …, 17} (13 values)

- **CV**: `TimeSeriesSplit(n_splits=5)` — **Scoring**: mean absolute error
  (lower = better) unless noted (the V1 prototype defaults to R²)
- **Data pipeline**: exact replication of the prototype notebook — int64→int32,
  object→category codes (`Date` included), five visibility/gust/cloud columns
  dropped, `NumberOfCustomers` kept as a feature (as the prototype had it),
  80/20 chronological split.
- **This machine**: 8 cores, Windows 10, Python 3.11, scikit-learn 1.9.0,
  numpy 2.4.6. All wall-clocks below are same-machine unless marked otherwise.

**Cost metric.** Different methods evaluate on different amounts of data, so raw
evaluation counts mislead. *Full-fit equivalents* = each evaluation weighted by
the fraction of rows it used (an evaluation on 10% of data costs 0.10). For
methods that always use all rows, equivalents = evaluations.

---

## Reference records (from the original prototype notebook, not re-run)

Source: `DatasetSize_and_ParamOpt_WORKING_(3Large_Aug_30_2025).ipynb` (kept
untouched as the archive).

| method | evaluations | full-fit equiv | best (its scoring) | wall-clock | machine |
|---|---|---|---|---|---|
| Prototype **Run A** (V1, scored R²) | 18 | 18.00 | R² 0.80998 at `{150, 4, 17}`¹ | 867.34 s | original |
| skopt `gp_minimize`, `n_calls=15` | 15 | 15.00 | n/a — optimized R² | n/a (2019-era run) | original |

¹ Params shown as `{n_estimators, max_features, max_depth}`.

Notes: skopt enforces a minimum of 10 calls; it is archived/unmaintained and
cannot run on this numpy 2.x stack, so its numbers enter as recorded facts only.
Caveat on Run A: the V1 class cell as saved in the notebook does **not**
reproduce Run A's 18-evaluation trajectory (see Experiment 4) — the class was
evidently edited after Run A was recorded; Run A stays as the historical
reference, Experiment 4 as the measured one.

---

## Experiment 1 — New `PatternSearchCV`, default configuration (2026-07-13)

Notebook: `Prototype_Replication.ipynb` (executed; full logs inside).

**Configuration**: defaults — 1 start (grid midpoint), data zones 10/20/50/100%,
`warmup=3`, `subsample='auto'` → **expanding** (oldest rows first, because CV is
TimeSeriesSplit), `poll='auto'` → **opportunistic** (8 cores ÷ 5 folds < 2),
`n_jobs=-1`, `random_state=0`, MAE objective.

**Results**

| metric | value |
|---|---|
| evaluations | 23 (18 at 10% data, 5 at 100%) |
| full-fit equivalents | **6.80** |
| cache hits (fits avoided) | 19 |
| wall-clock | **746.8 s** |
| best params | `{n_estimators: 150, max_features: 4, max_depth: 17}` |
| best CV MAE (full data) | 805.730 |
| held-out MAE (last 20% of rows) | 784.714 |

**Findings**

1. **Found Run A's recorded optimum** — (4, 150, 17), R² 0.809981 matching the
   original notebook to six decimals — while optimizing MAE instead of R².
2. **The intermediate zones (20%, 50%) were never used.** The search converged
   at 10% data in ~3 moves (warm-up consumed most of them), then took the
   forced jump straight to the 100% polish. All savings came from the two-phase
   effect: cheap exploration (18 × 0.10 = 1.8 equiv) + full-data confirmation
   (5 × 1.0 = 5.0 equiv). On small grids the graduated ladder barely engages.
3. Calibration readings were volatile (0.855 then 0.040) — the mean+floor rule
   handled it, confirming the choice of mean over max.

**Bug found & fixed during this experiment**: with `n_jobs=-1`, sklearn pickles
the estimator for every parallel task; the fitted instance was carrying
unpicklable state (logging handler, live generators) → `PicklingError`. Fixed
(machinery now lives only in local frames); regression test added. Suite green
including sklearn's `parametrize_with_checks`.

---

## Experiment 2 — Head-to-head vs prototype (bugs intact), MAE objective (2026-07-14)

Notebook: `HeadToHead_26grid.ipynb`. New algorithm (opportunistic forced,
default 4-zone bullseye) vs the prototype's V2 search cell **verbatim, bugs
intact** (premature contraction on failed/duplicate pattern moves,
non-compounding pattern references, O(n) dedup scans), both optimizing MAE,
same kernel, sequential.

| | OLD (prototype, bugs) | NEW (PatternSearchCV) |
|---|---|---|
| evaluations | 17 | 23 |
| full-fit equivalents | 17.00 | **6.80** |
| wall-clock | 1546.2 s | **824.1 s** |
| best point | (4, 130, 17) | (4, 150, 17) |
| CV MAE of best | **805.038** | 805.730 |
| CV R² of best | 0.809692 | **0.809981** |

Speedup 1.88× wall-clock, 2.50× compute; quality a statistical tie split both
ways (Δ far inside the ±22 fold std). Zones used: 10% and 100% only. Note the
patient run's wall-clock here (824.1 s) vs Experiment 1's (746.8 s) — same
policy, different dimension order and machine mood: a 77 s spread that matters
for interpreting single-run timing differences (see Experiment 5).

---

## Experiment 3 — V1 prototype exactly as the user's cell (2026-07-14)

Notebook: `V1_Prototype_26grid.ipynb`. The V1 class + its "#Execute Pattern
Search:" cell **verbatim** (default scoring = R², `clf n_jobs=-1`). Six
disclosed plumbing shims (joblib path, sklearn-1.x base class, `iid=`,
`error_score` value, `df.append`→`pd.concat`, `time` shadowing in timing
lines); the search loop byte-identical.

| metric | value |
|---|---|
| evaluations | 33, all at 100% data |
| full-fit equivalents | 33.00 |
| wall-clock | 1710.9 s |
| best point | (4, 150, 17) |
| CV R² of best | 0.809981 |
| CV MAE of best | 805.730 |

**Two-way comparison (the user's requested matchup):**

| | V1 prototype (as pasted, R²) | NEW PatternSearchCV (MAE) |
|---|---|---|
| evaluations | 33 | 23 |
| full-fit equivalents | 33.00 | **6.80** (4.85× less) |
| wall-clock | 1710.9 s | **824.1 s** (2.08× faster) |
| best point | (4, 150, 17) | (4, 150, 17) — identical |
| CV R² / MAE of best | 0.809981 / 805.730 | identical |

**Identical optimum, identical quality, 2.08× wall-clock, 4.85× compute.**

Caveats recorded honestly: (a) the two searches optimized different objectives
(R² vs MAE) and still chose the same point; (b) this V1 run took 33 evals vs
Run A's recorded 18 — its trajectory shows both-directions polling around a
fixed center with quarter-width initial steps, i.e. the V1 class cell as saved
differs from whatever code state produced Run A's 867 s record; (c) V1's
wall-clock includes nested parallelism (`clf n_jobs=-1` inside search
`n_jobs=-1`), as in the original configuration.

---

## Experiment 4 — Optuna Bayesian baselines, TPE + GPSampler (2026-07-14)

Notebook: `Optuna_Baseline.ipynb` (TPE) + `C:\FILES\Code\Benchmarking\psc-opt\run_gp.py`
(GPSampler — needs torch, which required a short-path venv: torch's nested file
paths exceed Windows' 260-char limit under the package venv, and torch 2.13's
DLLs fail on this Windows 10 build; torch 2.5.1 works). 15 trials per sampler
(the recorded `gp_minimize(n_calls=15)` budget), MAE objective, every trial on
100% of rows, seeded.

| | TPE | GP |
|---|---|---|
| trials | 15 (15.00 equiv) | 15 (15.00 equiv) |
| wall-clock | 828.7 s | 964.6 s |
| best MAE | 810.553 at (4, 100, 17) | **805.730 at (4, 150, 17)** — found the optimum (first at trial 13/15) |
| best after 11 trials | 811.500 | 811.500 |

TPE never found the optimum (0.6% worse). GP found it, but needed ~13 full-fit
equivalents to first reach it — the new algorithm's entire search cost 6.80.

---

## Experiment 5 — Eager contraction, controlled single-variable test (2026-07-14)

Notebook: `Eager_26grid.ipynb`. `contraction="eager"` (prototype-faithful:
failed pattern moves also contract), everything else identical to
Experiment 2's new-algorithm run (same seed, MAE objective, opportunistic poll,
default bullseye).

| | patient (default) | eager |
|---|---|---|
| evaluations | 23 | 24 |
| full-fit equivalents | 6.80 | 6.90 |
| wall-clock | 824.1 s | **688.8 s — 135 s (16%) faster; under investigation** |
| best point | (4, 150, 17) | (4, 150, 17) — identical |
| CV MAE / R² | 805.730 / 0.809981 | identical |

**Finding 1: evaluation cost is policy-neutral here.** The earlier belief that
eager-style contraction saves evaluations was a confounded inference from
Experiment 2 (the prototype's 17 evals mixed contraction policy with
sweep-drift semantics, dedup handling, compounding refs, and fidelity-protocol
evals). Isolated, the policy changed the path but not the evaluation count
(24 vs 23). Docstring and spec corrected accordingly.

**Open point — the 135-second wall-clock difference.** Eager finished 135 s
(16%) sooner despite one more evaluation. If real, this matters for the Optuna
comparison: patient's 824 s roughly matches TPE's 828.7 s, while eager's 689 s
clears every Bayesian time. Against it: patient itself has been measured at
746.8 s and 824.1 s (a 77 s spread between two runs of the same policy), so a
single 135 s gap on one sample each could be machine fluctuation; and eager
doing *more* row-weighted work in *less* time suggests either a cheaper fit mix
on its path (fewer trees per probe — possible, since equivalents weight rows
but not n_estimators) or machine drift.

**RESOLVED in Experiment 6: machine fluctuation.** Two runs with *byte-identical
evaluation sequences* differed by 217 s (14%) in the opposite direction, and
identical fits measured ~23% slower in the second run. This machine's
run-to-run noise is ±15–25%; the 135 s gap is inside it. Wall-clock claims on
this box need repeated runs or fit-work instrumentation.

**Five-way table (the official space, this machine):**

| | V1 proto (R²) | NEW patient | NEW eager | Optuna TPE | Optuna GP |
|---|---|---|---|---|---|
| evaluations | 33 | 23 | 24 | 15 | 15 |
| full-fit equiv | 33.00 | **6.80** | 6.90 | 15.00 | 15.00 |
| wall-clock | 1710.9 s | 824.1 s | 688.8 s | 828.7 s | 964.6 s |
| best point | (4,150,17) | (4,150,17) | (4,150,17) | (4,100,17) | (4,150,17) |
| CV MAE | 805.730 | 805.730 | 805.730 | 810.553 | 805.730 |

---

## Experiment 6 — Patient/Eager instrumented round, zones [0.05, 0.10, 0.20, 1.0] (running)

Notebook: `PE_Round_5_10_20_100.ipynb`. One P,E round, back-to-back in a single
kernel, per-evaluation fit times recorded (`mean_fit_time` × 5 = "total fit
work", a machine-noise-resistant measure of computation performed). Purpose:
resolve Experiment 5's open 135 s question — policy effect, cheaper fit mix on
eager's path, or machine fluctuation. The shared-evaluation fit-time ratio
(identical params AND rows in both runs) is the drift control: ~1.0 = steady
machine, real differences; <1.0 = machine was simply faster during eager.
Also first exercise of a 4-zone ladder starting at 5% (~20.9K rows).

**Results**

| | PATIENT | EAGER |
|---|---|---|
| evaluations | 24 | 24 — **identical sequences** (24 shared, 0 unique) |
| full-fit equivalents | 9.75 | 9.75 |
| wall-clock | 1552.3 s | 1769.4 s |
| summed fit work | 4422.9 s | 5091.3 s |
| best point | (4, 230, 17) | (4, 230, 17) |
| best CV MAE | 815.373 | 815.373 |
| zones used | 5% and 100% | 5% and 100% |

**Finding 1 — the 135 s question is answered: machine fluctuation.** On this
ladder the two policies happened to produce *byte-identical* evaluation
sequences (no divergence event on this path), i.e. two runs of the exact same
workload — yet wall-clocks differed by 217 s (14%), this time with eager
*slower*, and the shared-evaluation fit-time ratio shows identical fits taking
median 1.23× longer in the second run. The machine drifts ±15–25% between
back-to-back runs. Conclusion: Experiment 5's 135 s eager advantage was the
machine, not the policy; the docstring's "cost-neutral" stands, now with
controlled evidence. Any future wall-clock claim on this box requires repeats
or fit-work instrumentation.

**Finding 2 — the 5% rung is below this dataset's reliability floor.** The
[5, 10, 20, 100] ladder found a *worse* optimum — (4, 230, 17), MAE 815.373,
1.2% worse than the (4, 150, 17) / 805.730 found by every run with a 10% rung —
at a *higher* cost (9.75 vs 6.80 equivalents: the noisy 5% landscape converged
the search quickly toward the wrong n_estimators region, and the forced
full-data polish then spent ~9 full-price evaluations confirming a suboptimal
neighborhood). Cheap exploration only pays if the small sample's landscape is
faithful: rung-0 size is a real quality knob, and the default 10% start beat
the 5% start on both cost and answer. (Also an argument for testing
`subsample='stratified'`, whose whole purpose is making small rungs faithful.)

---

## Open questions queued for future experiments

- `subsample='stratified'` (transition sampling) vs `'expanding'` on this dataset —
  does a full-timeline 10% sample find the basin faster/more reliably than the
  oldest-tenth sample? (Signature feature of the package; article figure.)
- `n_starts ∈ {1, 2, 4, 8}` at same budget and same wall-clock (multi-start ablation).
- `poll='complete'` on this 8-core machine (expected ~neutral single-start; relevant
  with multi-start batching).
- Whether intermediate data zones engage on larger grids / multi-start.
- Multi-seed variance bands for the Optuna samplers (both runs above are seed=0).
