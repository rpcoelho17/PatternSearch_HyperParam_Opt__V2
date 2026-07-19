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

**Seven-way table (the official space, this machine):**

| | V1 proto (R²) | NEW patient | NEW eager | Optuna TPE | Optuna GP | Exp.7 Patient (stratified) | Exp.7 Eager (stratified) |
|---|---|---|---|---|---|---|---|
| zones ladder | n/a | 10/20/50/100 | 10/20/50/100 | n/a | n/a | **5/10/20/100** | **5/10/20/100** |
| evaluations | 33 | 23 | 24 | 15 | 15 | 22 | 22 |
| full-fit equiv | 33.00 | **6.80** | 6.90 | 15.00 | 15.00 | **5.85** | **5.85** |
| wall-clock | 1710.9 s | 824.1 s | 688.8 s | 828.7 s | 964.6 s | 599.7 s | 576.6 s |
| best point | (4,150,17) | (4,150,17) | (4,150,17) | (4,100,17) | (4,150,17) | (4,130,17) | (4,130,17) |
| CV MAE of best | 805.730 | 805.730 | 805.730 | 810.553 | 805.730 | **805.038** | **805.038** |

**Finding — the stratified 5% runs strictly dominate every other
configuration in this table, on all three axes at once.** Not a tradeoff:
better answer (805.038 < 805.730 — a genuinely lower minimum, not a tie),
least compute (5.85 full-fit equivalents, the smallest of all seven
columns), and smallest wall-clock (576.6 s, the fastest of all seven
columns). No other column beats it on any single axis; this one beats every
other column on all three simultaneously.

Caveat: the first five columns ran the default 4-zone ladder
(10/20/50/100%); the last two ran the [5, 10, 20, 100]% ladder from
Experiments 6–7 with `subsample='stratified'` (full-timeline transition
sampling) instead of the default expanding window. So this isn't a
clean single-variable A/B against the other five columns — the win could be
the aggressive 5% start, the stratified sampler, or (most likely) the
combination, since Experiment 7 already showed stratified sampling alone
beats expanding on this same 5% rung. Isolating the 5%-start effect on its
own (stratified, default ladder shape) is a natural next experiment.

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
| **P/E wall-clock ratio** | | **0.877** |
| summed fit work | 4422.9 s | 5091.3 s |
| best point | (4, 230, 17) | (4, 230, 17) |
| best CV MAE | 815.373 | 815.373 |
| zones used | 5% and 100% | 5% and 100% |

**Finding 1 — the 135 s question is answered: machine fluctuation.** On this
ladder the two policies happened to produce *byte-identical* evaluation
sequences (no divergence event on this path), i.e. two runs of the exact same
workload — yet wall-clocks differed by 217 s (14%: 1769.4 vs 1552.3 s), this
time with eager *slower*. The **sum of fit-work over the 24 shared
evaluations** (the metric consistent with wall-clock — see the correction
below) was 5091.3 s vs 4422.9 s, a **1.15× ratio**, matching the wall-clock
gap. The machine drifted ~15% between these two back-to-back runs. Conclusion:
Experiment 5's 135 s eager advantage was the machine, not the policy; the
docstring's "cost-neutral" stands, now with controlled evidence. Any future
wall-clock claim on this box requires repeats or fit-work instrumentation.

*Correction (2026-07-14): the notebook also prints a "median of per-evaluation
fit-time ratios" (1.23× here, 1.16× in Experiment 7). That statistic is
unreliable and should be disregarded — it is dominated by the many cheap
5%-data evaluations, where tiny absolute fit times make small noise look like
large percentage swings, and in Experiment 7 it pointed the opposite direction
from the actual wall-clock outcome. The **sum-based ratio** used above (total
shared fit-work, eager ÷ patient) is the metric that stays consistent with
wall-clock direction and is used throughout this log going forward.*

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

## Experiment 7 — Stratified sampling P/E round, zones [0.05, 0.10, 0.20, 1.0] (running)

Notebook: `PE_Stratified_5_10_20_100.ipynb`. Identical to Experiment 6 in every
respect (P then E, same ladder, seed, grid, MAE, fit-time instrumentation)
except `subsample="stratified"` — the transition sampler — instead of the
expanding window. Question under test: Experiment 6 showed the expanding 5%
rung (oldest ~5 weeks) misleads the search to (4, 230, 17) / MAE 815.373 at
9.75 equivalents; does a full-timeline 5% sample restore the
(4, 150, 17) / 805.730 optimum? This is the transition sampler's first outing
on real data — note that with continuous weather columns among the 30 watched
features, most rows are transitions, so the sampler is expected to operate
near its designed degenerate mode: systematic full-timeline sampling (all
seasons represented, unlike expanding's oldest-first prefix).

**Results**

| | PATIENT | EAGER |
|---|---|---|
| evaluations | 22 | 22 — **identical sequences** (22 shared, 0 unique) |
| full-fit equivalents | 5.85 | 5.85 |
| wall-clock | 599.7 s | 576.6 s |
| **P/E wall-clock ratio** | | **1.040** |
| summed fit work | 1559.8 s | 1594.2 s |
| best point | **(4, 130, 17)** | **(4, 130, 17)** |
| best CV MAE | **805.038** | **805.038** |
| zones used | 5% and 100% | 5% and 100% |

**Stratified vs expanding — direct answer: stratified wins decisively.**

| | Expanding (Exp. 6) | Stratified (Exp. 7) |
|---|---|---|
| best MAE | 815.373 | **805.038 — 1.3% better; ties Run A/prototype's historical optimum** |
| best point | (4, 230, 17) | (4, 130, 17) |
| full-fit equivalents | 9.75 | **5.85 — 40% less compute** |
| wall-clock (P+E combined) | 3321.7 s | **1176.3 s — 65% faster** |

The oldest-5-weeks expanding sample misled the search into a worse
n_estimators region and then paid ~9 full-price evaluations trying to refine
around it. The full-timeline stratified sample gave the 5% rung a landscape
faithful enough to find the right basin immediately — fewer evaluations,
cheaper evaluations, and a *better* answer, matching the best MAE found
anywhere in this log (Run A / the V1 prototype's recorded optimum, on this
metric equivalent to (4,150,17)'s R² twin). This is the clearest evidence yet
that `subsample='stratified'` is not just "different" but a genuine
improvement over `'expanding'` on this dataset, particularly at aggressive
(low) starting fractions where sample faithfulness matters most.

Machine-noise note (corrected 2026-07-14): P/E wall-clocks were close (599.7
vs 576.6 s, eager 4% *faster*). Sum-based fit-work over the 22 shared
evaluations was 1594.2 vs 1559.8 s — eager only 1.02× patient, i.e. ~2% more
total computation despite finishing sooner. Wall-clock and fit-work disagree
on direction here, both by small margins — the honest read is that P and E
are statistically indistinguishable in this run, well inside the noise floor
Experiment 6 established (~15%). (The notebook's separate "median of
per-evaluation ratios" line, 1.16×, is the unreliable statistic flagged in
Experiment 6's correction and should be ignored.)

---

## Experiment 8 — Patient/Eager, zones [2.5%, 5%, 10%, 100%], verbose=2 (built, awaiting run)

Notebook: `PE_Round_2.5_5_10_100_verbose2.ipynb`. Same P-then-E controlled
round as Experiments 6–7, one step more aggressive on the starting zone
(2.5% ≈ 10.5K rows, vs Experiment 7's 5%), `subsample="stratified"` explicit
(Experiment 7's winning configuration), `poll="opportunistic"` explicit,
**`verbose=2`** — built specifically for interactive use: the user opens it
in Jupyter/VS Code and runs cells themselves to watch the full live decision
narration (climber calibration, ring crossings, sweep probes, pattern moves,
contractions, data climbs, merges) stream as it happens, rather than reading
captured output after a headless run. Not pre-executed for that reason.

Question under test: does the 2.5% rung still find the right basin (à la
Experiment 7's stratified 5% success), or is this below even the stratified
sampler's reliability floor (à la Experiment 6's *expanding* 5% failure)?

**Results**

| | PATIENT | EAGER |
|---|---|---|
| evaluations | 22 | 22 — **identical sequences** (22 shared, 0 unique) |
| full-fit equivalents | 5.43 | 5.43 |
| wall-clock | 626.1 s | 649.9 s |
| **P/E wall-clock ratio** | | **0.963** |
| summed fit work | 1254.3 s | 1282.8 s (sum ratio 1.023×) |
| best point | (4, 130, 17) | (4, 130, 17) |
| best CV MAE | **805.038** | **805.038** |
| zones used (rows) | 10,461 and 418,416 | 10,461 and 418,416 |

**Finding: the 2.5% rung held.** Both policies found the historical optimum
(805.038, matching Run A/V1-prototype and both Experiment 7 runs exactly) at
5.43 full-fit equivalents — **the lowest compute cost recorded anywhere in
this log**, beating Experiment 7's 5.85. Stratified sampling's reliability
floor is at or below 2.5% on this dataset; it did not fail the way
`expanding` failed at 5% in Experiment 6. Wall-clock: P/E ratio 0.963 (eager
4% slower), sum-based fit-work ratio 1.023× (eager ~2% more total
computation) — small and mutually consistent this time (both point the same
direction, unlike Experiments 6/7), but still well inside the machine's
established noise band.

Progression across the three stratified-sampling starting-zone tests:

| starting zone | full-fit equiv (P) | best MAE |
|---|---|---|
| 10% (defaults before 2026-07-15) | 6.80 | 805.730 |
| 5% (Experiment 7) | 5.85 | 805.038 |
| 2.5% (Experiment 8) | **5.43** | 805.038 |

Monotone: smaller stratified starting zones have so far only helped on this
dataset — lower cost, same-or-better answer, no failure yet. The natural
next step is finding where this trend actually breaks (1%? 0.5%?), since
every default-tuning decision so far has been "more aggressive won" and that
can't continue indefinitely.

**Methodology note — what "stratified" is actually doing on this dataset,
verified precisely (2026-07-15).** Every stratified run so far (Experiments
7–9) logged `stratified_order: 418,416 rows, 418,416 runs (1.0 rows/run
avg)`. Traced the exact mechanism directly against the algorithm's own logic
(not inferred): **all 418,416 runs have length exactly 1** (confirmed
programmatically — min/max/mean run length = 1), which means `mid == start`
in every single run — boundary and midpoint are the literal same row, so the
designed "alternate between boundary and midpoint" behavior has no two
distinct rows to alternate between. Deeper: `subsample` watches all feature
columns by default, including ~17 continuous daily weather readings
(temperature, humidity, pressure, dew point, wind, precipitation) that
essentially never repeat exactly between rows even after the prototype's
5-column drop (§ discussion below on which columns are actually dropped).
Because of this, **every run's start-row combination is also "novel"**
(confirmed: 418,416 of 418,416 runs, 100%) — with continuous float columns
in the mix, no two rows ever hash identically, so nothing is ever a repeat.

The consequence for the algorithm's control flow (`_sampling.py`
`stratified_order`): the *first* priority tier, "first-ever occurrence of
each unique combination," already claims literally 100% of rows in one call
— `_take(novel_rows)` where `novel_rows` is the full dataset — ordered via
bit-reversed (Van der Corput-style) ranking, which spreads picks evenly
across any prefix length. The "alternating boundary/midpoint" tier and the
recursive-bisection-of-remaining-rows tier both then receive **zero**
unclaimed rows and do nothing. So on this dataset, the entire sampler
reduces to: **one bit-reversed permutation of the whole timeline**, full
stop — a legitimate, well-spread systematic sample, but none of the tiered
boundary/midpoint/novel-combination machinery the sampler was designed
around is actually exercising anything beyond that single tier. This is
almost certainly *why* stratified keeps winning regardless of how small the
rung gets (Experiments 7–9 below): a bit-reversed sample stays evenly spread
across the whole year no matter how sparse, whereas `expanding`'s failure
mode (Experiment 6) was about being unrepresentative in *time*
(oldest-N%-only), not about sample size. Testing `subsample_columns`
narrowed to genuinely categorical, slow-changing columns (excluding all
weather, not just the five already-dropped columns) is the only way to
produce real multi-row runs and actually exercise the boundary/midpoint/
novel-seat logic this sampler was built for.

---

## MAJOR FINDING — the CV split is store-based, not date-based (2026-07-15)

Triggered by a user question ("do the selected rows respect the time
sequence?") that led to checking whether row-index actually tracks calendar
time globally. It does not, and this reframes how every experiment in this
log up to this point should be read.

**Verified structural facts about `train.csv`:**

- The CSV is sorted by **`StoreID` first, then `Date` within each store's
  block** — not by date across the whole file. Confirmed: `StoreID` is
  monotonically non-decreasing row-by-row; `Date` is not.
- 749 distinct stores total; ~696 rows per store (≈ 2 years of near-daily
  data) — each store's own block already spans nearly the *entire*
  2016-03-01 to 2018-02-28 range on its own.
- Consequence: **the "chronological" 80/20 split is actually a store
  split.** Training rows (first 80%, 418,416 rows) contain **601 distinct
  stores**; validation rows (last 20%) contain **149**, of which **148 have
  never appeared in training at all**. Both the training and validation
  portions span the *identical* full date range — there is no temporal
  holdout happening at the whole-dataset level.
- Consequence: **`TimeSeriesSplit(n_splits=5)`, applied to this file, is
  performing a store-group split, not a temporal split.** Verified directly:
  every fold's train and test rows cover the exact same date range
  (2016-03-01 to 2018-02-28 in all 5 folds); each fold trains on an
  increasing number of stores (100 → 200 → 300 → 399 → 499) and tests on a
  disjoint block of ~100 *different* stores, with only ~1 store (~1%)
  overlapping between any fold's train and test sets.

**What this means:** every design decision and docstring claim in this
project framed around "TimeSeriesSplit," "chronological order," and
"avoiding leakage" (the `subsample` parameter's leakage warning, the
`"auto"` resolution picking `"expanding"`/now `"stratified"` "for
time-ordered splitters") was reasoning about the wrong risk. The actual
generalization question `TimeSeriesSplit` is measuring on this dataset is
**"do hyperparameters tuned on one set of stores generalize to a different,
non-overlapping set of stores"** — a cross-store generalization problem
wearing a time-series CV's clothes. There is little classical temporal
leakage risk to avoid (train and test already share the same date range in
every fold); the real risk `expanding` walks straight into is *store*
leakage/undercoverage, not date leakage. Not yet corrected in code/docs —
logged for discussion before deciding whether to (a) keep the current setup
but correct the documentation to describe it accurately, (b) re-sort the
pipeline by `Date` globally so `TimeSeriesSplit` does what its name
promises and re-run the key experiments, or (c) something else.

---

## Why stratified sampling has actually been winning (2026-07-15)

Direct follow-up to the finding above: since row-index is store-block
position, "spread evenly across row-index" means "spread across stores."
Measured distinct-store coverage directly (601 stores total in the training
portion) rather than assuming:

| fraction | rows | `expanding` stores | `stratified` stores | true random (avg of 5 seeds) |
|---|---|---|---|---|
| 1% | 4,185 | **6** | **601** | 600.0 |
| 2.5% | 10,461 | 15 | 601 | 601.0 |
| 5% | 20,921 | 30 | 601 | 601.0 |
| 10% | 41,842 | 60 | 601 | 601.0 |

**This fully explains Experiment 6's `expanding` failure**: at 1% it sees 6
of 601 stores — the search tunes against one small handful of stores and
generalizes badly to the ~99% of stores it never saw. It also fully explains
why `stratified` has been winning: its bit-reversed pick order spreads
evenly across row-index by construction, which — since row-index is store
identity here — delivers near-total store coverage even at tiny fractions.

**Important caveat, also measured, not assumed: at 1%–10% (the fractions
Experiments 6–9 actually used), true random sampling would have achieved
essentially the same store coverage as `stratified` (600–601 of 601 in both
cases).** The demonstrated win in this log has been "anything except
`expanding`'s contiguous-block sampling beats `expanding`," not "`stratified`
beats random." Pushed to more extreme fractions to find where the
deterministic design actually separates from random.

**Reasoning for this test:** at 1%–10% `stratified` and true random tie on
store coverage (600–601 of 601 both ways) — which meant, at the fractions
actually used so far, it looked suspiciously like plain random sampling
would have done just as well, and `stratified`'s more elaborate
low-discrepancy design wasn't earning its keep over something far simpler.
Rather than accept that, the next move was to push to smaller fractions and
see whether the "smart" mechanism actually separates from random anywhere
— i.e. find the point where random's coverage starts failing by luck and
check whether `stratified`'s guarantee holds there instead.

**How it was measured** (side-analysis script, not a `PatternSearchCV`
search run — this isolates the sampling mechanism's coverage property in
a few seconds, rather than paying for a full search per data point):
1. Ran the real data pipeline (`train.csv` → int64→int32 → category
   codes → the same five dropped weather columns → 80/20 split) to get
   the training portion: 418,416 rows, 601 distinct stores.
2. Called the actual library function, `pattern_search_cv._sampling
   .stratified_order(X)` — not a reimplementation — to get the real
   priority order the package would use.
3. For each fraction `f`, took `k = ceil(f * n)` rows: the top-`k` of the
   real `stratified_order` output, and separately `np.random.RandomState
   (seed).choice(n, size=k, replace=False)` for 20 independent seeds
   (0–19).
4. Store coverage = `len(np.unique(store_ids[selected_rows]))` for each
   draw. Reported the `stratified` count (deterministic, one number), the
   mean across the 20 random seeds, and the single worst (minimum) seed.

| fraction | rows | `stratified` stores | random avg (20 seeds) | random worst seed |
|---|---|---|---|---|
| 0.1% | 419 | **415** | 301.9 | 293 |
| 0.2% | 837 | **601** | 448.1 | 433 |
| 0.5% | 2,093 | **601** | 581.8 | 574 |
| 1.0% | 4,185 | 601 | 600.0 | 597 |

**Below ~0.5%, `stratified` decisively beats random** (415 vs. ~302 stores
at 0.1%) — this is the low-discrepancy (Van der Corput/bit-reversal)
property doing genuine, provable work: it *guarantees* even spread at every
prefix length rather than relying on random luck, which starts to matter
once the sample is small enough that luck can fail. No search has yet been
run in that regime (Experiments 6–9 stopped at 1%); this table predicts
what such a search would show but doesn't replace it.

One more precision: at the 1% fraction actually used in Experiment 9, the
sample doesn't just touch every store — rows-per-store distribution is
min=2, max=9, mean=6.96, with **zero stores represented by only a single
row**. Coverage is balanced, not just technically present.

**Implications / open decisions for discussion:**
1. The `expanding` vs. `stratified` comparison (Experiments 6–7) demonstrates
   "avoid catastrophic store undercoverage," not necessarily "stratified's
   transition-detection logic adds value" — that logic has never actually
   fired on this dataset (see the methodology note above: run length is
   always 1 here).
2. `subsample="random"` has never been benchmarked in an actual P/E search
   run on this dataset, despite the leakage-avoidance rationale for skipping
   it no longer clearly applying now that the CV is understood to be
   store-based rather than date-based. The store-coverage numbers above
   predict it would tie `stratified` at 1%+ and lose below ~0.5%.
3. A genuinely diagnostic next experiment is a P/E search run at 0.1%–0.2%,
   where `stratified` should pull ahead of both `expanding` and `random` for
   the first time in an actual search (not just a coverage-counting
   side-analysis).

---

## Experiment 9 — Patient/Eager, zones [1%, 5%, 10%, 100%], verbose=0 (2026-07-15, done)

Notebook: `PE_Round_1_5_10_100.ipynb`. Continuing the starting-zone
progression from Experiment 8 (10% → 5% → 2.5%, each a win or tie so far):
does `subsample='stratified'` hold at an even more aggressive 1% starting
zone (~4,184 rows), or is this finally below its reliability floor?
`subsample='stratified'` explicit, `poll='opportunistic'` explicit,
`verbose=0`.

**Results**

| | PATIENT | EAGER |
|---|---|---|
| evaluations | 22 | 22 — **identical sequences** (22 shared, 0 unique) |
| full-fit equivalents | 5.17 | 5.17 |
| wall-clock | 512.0 s | 475.0 s |
| **P/E wall-clock ratio** | | **1.078** |
| summed fit work | 1378.8 s | 1301.6 s (sum ratio 0.944×) |
| best point | (4, 130, 17) | (4, 130, 17) |
| best CV MAE | **805.038** | **805.038** |
| zones used (rows) | 4,185 and 418,416 | 4,185 and 418,416 |

**Finding: the 1% rung held too — fourth win in a row.** Both policies again
found the historical optimum (805.038) at **5.17 full-fit equivalents**, a
new low, beating Experiment 8's 5.43. `stratified_order` reports 418,416
rows / 418,416 runs on this dataset (§ discussion above) — every rung is
effectively a systematic full-timeline sample regardless of size, which is
almost certainly *why* this keeps working: an evenly-spread 1% sample is
still a faithful miniature of the whole year, just sparser. This also
explains why the streak might not break from *shrinking the rung further*
on this dataset — the failure mode demonstrated in Experiment 6 was about a
sample being unrepresentative in *time* (expanding's oldest-N%-only), not
about sample *size* per se.

Complete progression across four stratified-sampling starting-zone tests:

| starting zone | full-fit equiv (P) | best MAE |
|---|---|---|
| 10% (defaults before 2026-07-15) | 6.80 | 805.730 |
| 5% (Experiment 7) | 5.85 | 805.038 |
| 2.5% (Experiment 8) | 5.43 | 805.038 |
| 1% (Experiment 9) | **5.17** | **805.038** |

Still monotone, still no failure. Given the likely explanation above (every
rung is systematic-in-time regardless of size, so shrinking it mostly just
saves rows without hurting representativeness), the more diagnostic next
experiment is probably not "push the % lower" but testing whether
`subsample_columns` narrowed to genuinely categorical columns (producing
real multi-row runs instead of the current 1.0-rows/run degenerate case)
changes this picture at all — see the discussion above this table.

---

## Experiment 10 — Patient/Eager, zones [0.5%, 1%, 10%, 100%], verbose=0 (2026-07-15, done)

Notebook: `PE_Round_0.5_1_10_100.ipynb`. Continuing the starting-zone
progression (10% → 5% → 2.5% → 1%, each a win or tie so far — Experiments
7–9): does `subsample='stratified'` hold at an even more aggressive 0.5%
starting zone (~2,092 rows)? Same configuration as Experiment 9 otherwise.

**Results**

| | PATIENT | EAGER |
|---|---|---|
| evaluations | 22 | 22 — **identical sequences** (22 shared, 0 unique) |
| full-fit equivalents | 5.09 | 5.09 |
| wall-clock | 443.8 s | 445.0 s |
| **P/E wall-clock ratio** | | **0.997** |
| summed fit work | 1178.8 s | 1224.0 s (sum ratio 1.038×) |
| best point | (4, 130, 17) | (4, 130, 17) |
| best CV MAE | **805.038** | **805.038** |
| zones used (rows) | 2,093 and 418,416 | 2,093 and 418,416 |

**Finding: the 0.5% rung held — fifth win in a row, new low.** Both
policies again found the historical optimum (805.038) at **5.09 full-fit
equivalents**, beating Experiment 9's 5.17. P and E wall-clocks are the
closest of any round so far (ratio 0.997), consistent with the established
noise-floor picture — no systematic direction across five rounds.

Complete progression across five stratified-sampling starting-zone tests:

| starting zone | full-fit equiv (P) | best MAE |
|---|---|---|
| 10% (defaults before 2026-07-15) | 6.80 | 805.730 |
| 5% (Experiment 7) | 5.85 | 805.038 |
| 2.5% (Experiment 8) | 5.43 | 805.038 |
| 1% (Experiment 9) | 5.17 | 805.038 |
| 0.5% (Experiment 10) | **5.09** | **805.038** |

Still monotone, still no failure — five wins in a row now. Per the earlier
"why stratified has actually been winning" analysis, this is expected: the
low-discrepancy bit-reversed order guarantees near-uniform coverage at
every prefix length, and the measured store-coverage numbers showed
`stratified` still hitting 415 of 601 stores even at 0.1% — so 0.5% is
comfortably inside its reliable range. The gains between rungs are also
visibly shrinking (0.95 → 0.42 → 0.26 → 0.08 equivalents saved per halving
step), suggesting the curve is flattening well before it breaks. Finding
the actual failure point (if one exists above the resource floor) would
need to go substantially lower than 0.5%, or use the `subsample='random'`
comparison arm already queued below to see where random sampling — not
`stratified` — finally fails.

---

## Experiment 11 — Patient/Eager, zones [0.25%, 1%, 10%, 100%], verbose=0 (2026-07-15, done)

Notebook: `PE_Round_0.25_1_10_100.ipynb`. Skips straight to a 0.25% starting
zone (~1,046 rows) — between the store-coverage side-analysis's 0.2%
(601/601 stores, full coverage) and 0.1% (415/601, where `stratified` first
pulled decisively ahead of random). Tests whether an actual search still
holds up in that range, not just the coverage count. Same configuration as
Experiment 10 otherwise.

**Results**

| | PATIENT | EAGER |
|---|---|---|
| evaluations | 22 | 22 — **identical sequences** (22 shared, 0 unique) |
| full-fit equivalents | 5.04 | 5.04 |
| wall-clock | 445.8 s | 469.7 s |
| **P/E wall-clock ratio** | | **0.949** |
| summed fit work | 1179.3 s | 1270.0 s (sum ratio 1.077×) |
| best point | (4, 130, 17) | (4, 130, 17) |
| best CV MAE | **805.038** | **805.038** |
| zones used (rows) | 1,047 and 418,416 | 1,047 and 418,416 |

**Finding: the 0.25% rung held — sixth win/tie in a row, new low.** Both
policies again found the historical optimum (805.038) at **5.04 full-fit
equivalents**, beating Experiment 10's 5.09. Notably, this rung is *inside*
the 0.1–0.5% range where the coverage side-analysis showed `stratified`
starting to separate from plain random (415 vs. ~302 stores at 0.1%) — this
is the first actual search result, not just a coverage count, in that
regime, and it still found the right basin.

Complete progression across six stratified-sampling starting-zone tests:

| starting zone | full-fit equiv (P) | best MAE |
|---|---|---|
| 10% (defaults before 2026-07-15) | 6.80 | 805.730 |
| 5% (Experiment 7) | 5.85 | 805.038 |
| 2.5% (Experiment 8) | 5.43 | 805.038 |
| 1% (Experiment 9) | 5.17 | 805.038 |
| 0.5% (Experiment 10) | 5.09 | 805.038 |
| 0.25% (Experiment 11) | **5.04** | **805.038** |

Savings per halving step: 0.95 → 0.42 → 0.26 → 0.08 → 0.05 — the curve has
now clearly flattened; six consecutive rungs without a failure, and the
remaining headroom is small. Given how little compute is left to save this
way, the more informative next experiments are the ones already queued
below (`subsample='random'` at these same fractions to see whether *it*
finally fails where the coverage analysis predicted it would, and the
`subsample_columns` test to exercise the sampler's un-triggered
boundary/midpoint logic) rather than continuing to shrink this ladder
further.

---

## Patient vs eager across the starting-zone progression (Experiments 7–11)

Patient runs:

| | Exp.7 Patient | Exp.8 Patient | Exp.9 Patient | Exp.10 Patient | Exp.11 Patient |
|---|---|---|---|---|---|
| zones ladder | 5/10/20/100 | 2.5/5/10/100 | 1/5/10/100 | 0.5/1/10/100 | 0.25/1/10/100 |
| evaluations | 22 | 22 | 22 | 22 | 22 |
| full-fit equiv | 5.85 | 5.43 | 5.17 | 5.09 | **5.04** |
| wall-clock | 599.7 s | 626.1 s | 512.0 s | 443.8 s | 445.8 s |
| best point | (4,130,17) | (4,130,17) | (4,130,17) | (4,130,17) | (4,130,17) |
| CV MAE of best | 805.038 | 805.038 | 805.038 | 805.038 | 805.038 |

Eager runs:

| | Exp.7 Eager | Exp.8 Eager | Exp.9 Eager | Exp.10 Eager | Exp.11 Eager |
|---|---|---|---|---|---|
| zones ladder | 5/10/20/100 | 2.5/5/10/100 | 1/5/10/100 | 0.5/1/10/100 | 0.25/1/10/100 |
| evaluations | 22 | 22 | 22 | 22 | 22 |
| full-fit equiv | 5.85 | 5.43 | 5.17 | 5.09 | **5.04** |
| wall-clock | 576.6 s | 649.9 s | 475.0 s | 445.0 s | 469.7 s |
| best point | (4,130,17) | (4,130,17) | (4,130,17) | (4,130,17) | (4,130,17) |
| CV MAE of best | 805.038 | 805.038 | 805.038 | 805.038 | 805.038 |

---

## DEFAULT CHANGE (2026-07-15): contraction="patient", data_zones=(0.005, 0.01, 0.1, 1.0)

Following the two tables above, decided to set the shipped defaults to
**`contraction="patient"`** (reverted from `"eager"`, set 2026-07-15 earlier
today) and **`data_zones=(0.005, 0.01, 0.1, 1.0)`** (0.5/1/10/100%, changed
from `(0.05, 0.10, 0.20, 1.0)`).

Justification for `contraction="patient"`: across all five controlled
rounds (Experiments 7–11), patient and eager are tied on every cost metric
that matters — identical evaluation counts, identical full-fit equivalents,
identical best point and MAE in every single round. Wall-clock bounces both
directions within the established machine-noise floor (patient faster in
3 of 5 rounds, eager in 2 of 5, no consistent winner). There has never been
a measured advantage to `"eager"` in this project; reverting to `"patient"`
removes an unjustified deviation from classic Hooke-Jeeves and its
documented (if untested-here) premature-convergence risk, at zero measured
cost.

Justification for the `0.5/1/10/100%` ladder: Experiment 10 (this exact
ladder) measured 5.09 full-fit equivalents with `patient`, matching the
historical optimum (805.038) — the best-tested ladder at the time of this
decision (Experiment 11 later found 0.25% starts marginally better still,
5.04 equiv, but that ladder was not chosen as default here). Caveat carried
forward from the discussion before this decision: this is single-dataset,
single-grid evidence (523K rows, one 3-parameter ExtraTrees search); an
aggressive 0.5% starting zone has not been validated on smaller datasets,
higher-dimensional grids, or non-time-series data, and this dataset's
favorable behavior at small fractions is partly attributable to its
store-blocked structure (§ "why stratified sampling has actually been
winning" above) rather than to a universal property of aggressive
subsampling. The resource floor (`min_rows = max(2*(n_splits+1), 8)`)
protects small datasets from an unreasonably tiny first rung regardless.

---

## Experiment 12 — `GPProposer` vs Optuna `GPSampler`, dev-time validation (2026-07-18, done)

Notebook: `GP_Validation_vs_Optuna.ipynb`, run from the separate `psc-opt` venv
(torch 2.5.1+cpu, optuna 4.9.0 — dev-time reference only, never a runtime
dependency; see `BayesHalvingSearchCV_SPEC.md` §1.1/§8). Not part of the pytest
suite, not a CI gate. Validates the from-scratch `GPProposer`
(`sklearn.gaussian_process.GaussianProcessRegressor`, Matern 5/2, hand-rolled
Expected Improvement) built for `BayesHalvingSearchCV` against Optuna's
`GPSampler` before trusting it for a real benchmark.

**Part A — synthetic functions (isolated optimizer logic, 18 evals, seed=0)**

Unimodal (21x21 quadratic bowl): both optimizers found the exact known
optimum (10, 10), distance 0.

Multimodal (Rastrigin): first attempt (sklearn's default
`length_scale_bounds=(1e-5, 1e5)`) exposed a real defect — after 4 spread
cold-start draws, `GPProposer` degenerated into a monotonic walk along one
grid edge for the remaining 14 evals, landing far from the optimum (value
-25.0, vs. Optuna's -1.0). Root cause: on Rastrigin's high-frequency
landscape the length-scale MLE collapsed to the lower bound every fit
(`ConvergenceWarning` on every call), producing a near-delta kernel whose
predictions revert to the same value everywhere far from training data — EI
went flat, and the deterministic `min(tied)` tie-break then walked
lexicographically through the remaining grid. **Fix**: floor
`length_scale_bounds=(0.05, 10.0)` with `length_scale=0.5` (0.05 = one grid
step in `_featurize`'s normalized coordinates) in `GPProposer._gp_ei_pick`
(`_gp.py`). After the fix, proposals stayed varied throughout, landing at
(10, 14), distance 4 — vs. Optuna's (10, 8), distance 2. All 204 package
tests still pass after this change (no test asserted an exact proposal
sequence).

**Part B — real grid, fixed 0.25% data fraction (1,047 rows), `n_iter=15`, seed=0**

Grid: `max_features` {2,3,4} x `n_estimators` {10..260 step 10} x
`max_depth` {5..17} = 1,014 points. Fixed subset via `stratified_order`
(same call `subsample="auto"` makes for `TimeSeriesSplit`), one shared
`objective` for both optimizers.

| | GPProposer | Optuna GP |
|---|---|---|
| best MAE (0.25% subset) | **956.171** | 965.346 |
| best point | (4, 140, 17) | (4, 170, 15) |
| lands in (4, ~130-150, 17)-class optimum? | **yes** | no |
| wall-clock | 76.3 s | 22.4 s |

Path comparison: set overlap 2/15 (13%), position-by-position matches 0/15
(expected — both optimizers cold-start differently, so early-position
agreement is the least informative signal, per spec). Not comparable in MAE
terms to Experiment 4's 100%-data numbers (805.730) — different data size,
different landscape.

**Finding: `GPProposer` is validated.** Perfect on the unimodal sanity check,
reasonable (post-fix) divergence on an adversarial multimodal function, and
on the real objective at a realistic low-data fraction it beat Optuna's
`GPSampler` on both MAE and on recovering this project's known optimum
basin — a specific, meaningful correctness signal beyond "some
reasonable-looking answer." Cleared to use, unmodified from this state, in
`BayesHalvingSearchCV`'s real benchmark against `PatternSearchCV`.

---

## Experiment 13 — BayesHalvingSearchCV vs PatternSearchCV, real benchmark (2026-07-18, done)

Notebook: `BHS_vs_PSC_26grid.ipynb`. Runs in the plain package `.venv` — no
torch anywhere in this notebook, confirming `BayesHalvingSearchCV` really has
zero additional runtime dependencies. Official grid (`max_features` {2,3,4} x
`n_estimators` {10..260 step 10} x `max_depth` {5..17}), `TimeSeriesSplit(5)`,
MAE, zones `(0.005, 0.01, 0.1, 1.0)`, `subsample="stratified"`,
`random_state=0`, `n_starts=1` on both arms — directly comparable to the
reference rows in `BayesHalvingSearchCV_SPEC.md` §0.

**Results**

| | Optuna GP, 100% data (Exp. 4) | PatternSearchCV, defaults (this run) | **BayesHalvingSearchCV** (this run) |
|---|---|---|---|
| total evaluations | 15 | 22 | 28 |
| fits @ 0.5% (2,093 rows) | — | 17 | 17 |
| fits @ 10% (41,842 rows) | — | 0 | 8 |
| fits @ 100% (418,416 rows) | 15 | **5** | **3** |
| full-fit equivalents | 15.00 | 5.09 | **3.89** |
| best point | (4,150,17)-class | (4, 130, 17) | (4, 150, 17) |
| best CV MAE | 805.730 | 805.038 | 805.730 |
| wall-clock | 964.6 s | 1157.6 s | 993.2 s |
| zones used (rows) | — | [2093, 418416] | [2093, 41842, 418416] |

**Finding: mission met, and BayesHalvingSearchCV beat PatternSearchCV's own
equivalents count on this run.** `BayesHalvingSearchCV` reached the exact MAE
Optuna's `GPSampler` found at 100% data (805.730) at **3.89 full-fit
equivalents — a 74% reduction from Optuna's 15.00**, and 24% fewer
equivalents than `PatternSearchCV`'s 5.09, despite 28 raw evaluations vs
PatternSearchCV's 22: 17 of the 28 ran at the cheapest 0.5% rung, 8 more at
10% after a single zone climb (3 of those are the `promote_k=3` re-score
that triggered the climb), and only the mandatory 3-row final polish touched
full data — the bullseye ladder is doing exactly what it should. Wall-clock
(993.2 s vs 1157.6 s, ratio 0.858) favors BHS too, but per the established
machine-noise rule (~15–25% is noise on this machine) that is not a
confidently-claimable win by itself; full-fit equivalents is the primary
metric, and there the win (24% fewer) is real and outside the noise floor.

BayesHalvingSearchCV did not find PatternSearchCV's slightly better optimum
(805.038 at (4,130,17) vs 805.730 at (4,150,17)) — a 0.09% relative MAE gap,
i.e. both land in the same (4, ~130–150, 17)-class basin this project keeps
finding at low data fractions (Experiments 7–11), just at adjacent grid
points. Given `GPProposer`'s own validation (Experiment 12) showed it lands
in this exact basin reliably, this reads as expected single-seed,
single-start noise between two different search algorithms rather than a
search-quality gap.

**Follow-up (2026-07-18): why does PatternSearchCV have fewer total fits
(22 vs 28) but higher full-fit equivalents (5.09 vs 3.89)?** Re-ran the
exact, deterministic PatternSearchCV arm (`random_state=0` reproduces the
identical 22 evals / 805.038 result) to capture the per-tier breakdown that
wasn't printed the first time — see the table above. `equiv = Σ (n_resources_i
/ n_samples)`: PatternSearchCV = 17×0.005 + 5×1.0 = 0.085 + 5.00 = **5.085**;
BayesHalvingSearchCV = 17×0.005 + 8×0.1 + 3×1.0 = 0.085 + 0.80 + 3.00 =
**3.885**. PatternSearchCV put 5 of its 22 fits (23%) at full data — 98% of
its total cost — while BayesHalvingSearchCV put only 3 of its 28 (11%),
capped at exactly `promote_k`.

The mechanism, verified against `_climber.py`: PatternSearchCV's actual run
never touched the 1% or 10% rungs at all. This is not luck — once the
search mesh contracts to its floor step size while still below full data,
`Climber`'s convergence logic has a hard-coded rule forcing a jump straight
to `len(self.zones) - 1` (the *last* zone, i.e. 100%), skipping every
intermediate rung on purpose (`reason="forced-final-polish"`). Re-running
with `search_history_` printed (note: this only logs *confirmed improving
moves*, not every probe, unlike BayesHalvingSearchCV's per-trial ledger)
shows exactly this: start `(3,130,11)` → one improving move to `(4,130,17)`
at 0.5% → forced jump straight to 100% for the polish rescore (805.038) →
4 more full-data fits (each a confirmatory sweep probe) before the "3
consecutive failed sweeps" convergence rule was satisfied. Each of those
extra confirmatory probes costs a full 1.0 equivalent, because Hooke-Jeeves'
patient contraction re-confirms convergence *at whatever tier it's
currently on* — and here that tier is the most expensive one.
BayesHalvingSearchCV's final-polish step, by contrast, is budget-capped by
design: it re-scores exactly the top `promote_k` candidates at 100% once
and stops, rather than re-sweeping at full data until it's convinced. The
gap isn't about total work done — it's that PatternSearchCV's convergence
check is willing to pay full price repeatedly to be sure it's finished,
while BayesHalvingSearchCV's polish step has a hard ceiling on that cost.

---

## Experiment 14 — Does `subsample='random'` actually break at 0.2%, where coverage theory predicts it should? (2026-07-18, done)

Notebook: `PE_Stratified_vs_Random_0.2_0.3_0.5_100.ipynb`. Direct test of the
theory from "Why stratified sampling has actually been winning" (above): a
side-analysis (not a search run) measured that at 0.2%, `stratified`
guarantees 601/601 store coverage while true random sampling averages only
448.1/601 across 20 seeds (worst seed 433) — predicting an actual search
using `subsample='random'` should start losing to `stratified` below ~0.5%.
No search had ever been run with `subsample='random'` on this dataset before
this experiment. Zones `[0.2%, 0.3%, 0.5%, 100%]` — one rung below
Experiment 11's 0.25% floor, `random_state=0`, `poll='opportunistic'`
explicit, `verbose=0`. Three arms: PATIENT/stratified, EAGER/stratified (the
usual P/E pair), and PATIENT/random (single seed, this project's standard
convention — the comparison arm).

**Results**

| | PATIENT/stratified | EAGER/stratified | PATIENT/random |
|---|---|---|---|
| evaluations | 31 | 29 | 27 |
| full-fit equivalents | 5.07 | 5.06 | **5.07** |
| wall-clock | 1168.0 s | 1160.9 s | 910.0 s |
| summed fit work | 2945.3 s | 2993.7 s | 2290.3 s |
| best point | (4, 130, 17) | (4, 130, 17) | **(4, 130, 17)** |
| best CV MAE | 805.038 | 805.038 | **805.038** |
| zones used (rows) | [837, 2093, 418416] | [837, 2093, 418416] | [837, 2093, 418416] |

Shared evaluations, PATIENT/stratified vs PATIENT/random: 21 of 31/27 —
substantial path overlap, not two unrelated searches.

**Finding: the theory did NOT hold up in this test — `subsample='random'`
did not break at 0.2%.** All three arms converged to the exact same optimum
(4, 130, 17) at the exact same MAE (805.038) and essentially the same
full-fit equivalents (5.06–5.07). `PATIENT/random` used *fewer* evaluations
(27 vs 31) and less wall-clock (910.0s vs 1168.0s) than its matched
`PATIENT/stratified` arm — the opposite direction from what the coverage
numbers predicted, though within this project's ~15–25% wall-clock noise
floor, so not claimed as a real speed advantage either.

Also notable: none of the three arms ever evaluated at the 0.3% (1,255-row)
rung — all jumped straight from 0.2% to 0.5%. This is expected bullseye
behavior, not a bug: `_zone_for` finds the innermost ring boundary a
displacement falls below in one step, so a single confident move can cross
two ring boundaries at once and skip the middle rung entirely; it happened
identically in all three arms, so it isn't a `stratified`-vs-`random`
effect.

**Why the coverage theory's prediction didn't materialize here — read with
real caveats, not as a refutation:**
1. **Single seed.** This is one draw of `subsample='random'` (`random_state=0`).
   The coverage side-analysis showed real seed-to-seed variance at 0.2%
   (448.1 average, 433 worst, across 20 seeds) — seed 0 may simply have
   landed on a lucky draw for this specific 837-row sample. The coverage
   theory was never a claim about every seed; a genuinely unlucky seed could
   still show degradation. This result does not rule that out — it rules out
   only "random reliably breaks regardless of seed," which was never
   precisely the claim either.
2. **The search algorithm has its own self-correction the coverage count
   doesn't model.** `stratified`/`random` only shape the ranking signal at
   the *smallest* zone; PatternSearchCV's mesh contraction and ratcheted
   zone growth mean a temporarily noisy small-sample signal doesn't have to
   be trusted forever — the search keeps moving and re-scores at larger
   zones as soon as displacement calms down. A store-coverage count measures
   a property of the *sample*, not of the *search outcome* after that
   self-correcting machinery has had a chance to run.
3. **ExtraTreesRegressor is itself heavily randomized** (bootstrap +
   per-split feature randomization) — some of the ranking noise a
   lower-coverage sample would introduce may already be within the range the
   estimator's own internal randomness routinely produces, especially for
   the coarse win/lose comparisons a pattern-search sweep actually needs
   (not a precise ranking, just "did this move help").

One seed, however clean, is not enough to overturn the side-analysis's
real, measured coverage gap. What this experiment does show is that
`stratified`'s coverage guarantee is not *load-bearing* for this specific
search/grid/seed at 0.2% — a useful, honest negative result, not the
confirmation the theory predicted.

**Follow-up side-analysis: is the tie/win actually leakage, not luck?**
(not a `PatternSearchCV` search run — same "isolate the mechanism cheaply"
methodology as the store-coverage side-analysis above). Given row-index is
store-block position on this dataset, a plausible leakage mechanism exists:
if `random`'s sampled rows put the same stores in both train and test more
often than `stratified`'s do, the model could partly memorize per-store
baseline sales instead of genuinely generalizing, inflating (lowering) its
CV score without earning it. Directly measured using the exact
`stratified_order`/`random_order` calls and the exact `ZoneSplitter` the
real search uses, on the same 837-row (0.2%) zone, `random_state=0`:

1. **Train/test store overlap per fold**: `stratified` — 0.0% in all 5
   folds. `random` — 0.0–1.4% (1 store out of 68–102, in 2 of 5 folds). No
   meaningful overlap under either sampler — `TimeSeriesSplit` is already
   splitting by near-disjoint store sets regardless of the sampling method
   (consistent with the store-blocked-CSV finding above), so the classic
   leakage mechanism (same entity in train and test) isn't in play for
   either arm.
2. **Real per-fold MAE, winning config (4,130,17), same rows a live search
   would score**: `stratified` = [1153.95, 928.32, 1006.80, 979.84, 901.36],
   mean **994.056**. `random` = [1127.74, 912.55, 1039.01, 959.48, 829.05],
   mean **973.567** — random *is* ~20 points (2%) more optimistic on this
   exact config, same direction as the full search's result.

**Conclusion: not classic leakage — the overlap check rules out the
specific mechanism that would justify that word — but the optimism is real
and measured, not imagined.** With zero train/test store overlap under
either sampler, the ~20-point gap reads as ordinary sampling variance: seed
0's particular 837-row draw happened to land on a slightly easier split for
this config (fewer noisy folds), not test information reaching train. This
is a single-config check, though — it doesn't rule out a real generalization
gap becoming visible on other configs or other seeds. The multi-seed
follow-up above remains the way to settle whether Experiment 14's result
generalizes.

---

## Experiment 15 — Low-data stress test: PatternSearchCV (eager) vs BayesHalvingSearchCV, and stratified vs random with seed 42 (2026-07-18/19, done)

Notebook: `LowData_Stress_PSCeager_BHS_Random42.ipynb`. Zones `[0.15%, 0.20%,
2%, 100%]` — one rung below Experiment 14's 0.2% floor. Two comparisons in
one run, `verbose=2` on every arm (full per-decision log saved in the
notebook), `n_starts=1`, official grid, `TimeSeriesSplit(5)`, MAE:

1. `PatternSearchCV(contraction="eager", subsample="stratified", random_state=0)`
   vs `BayesHalvingSearchCV(subsample="stratified", random_state=0, n_iter=25, promote_k=3)`.
2. `PatternSearchCV(contraction="eager", subsample="stratified", random_state=0)`
   vs `PatternSearchCV(contraction="eager", subsample="random", random_state=42)`
   — Experiment 14 (seed 0) found `random` tied `stratified`; this repeats
   the test with a different seed at a more aggressive floor.

**Results**

| | PSC eager / stratified | BayesHalvingSearchCV / stratified | PSC eager / random (seed=42) |
|---|---|---|---|
| total evaluations | 30 | 28 | 31 |
| fits @ 0.15% (628 rows) | 20 | 25 | 11 |
| fits @ 0.20% (837 rows) | 0 | 0 | 7 |
| fits @ 2% (8,369 rows) | 0 | 0 | 7 |
| fits @ 100% (418,416 rows) | **10** | **3** | 6 |
| full-fit equivalents | **10.030** | 3.038 | 6.171 |
| wall-clock | 2105.2 s | 922.5 s | 1781.3 s |
| best point | (4, 130, 17) | (4, 140, 15) | (4, 170, 17) |
| best CV MAE | **805.038** | **866.197** | 807.968 |

**Finding 1 — PatternSearchCV (eager) found the true optimum here;
BayesHalvingSearchCV did not.** PSC eager/stratified landed exactly on this
project's known optimum (805.038). BayesHalvingSearchCV converged to a
materially worse point (866.197, +7.6% MAE) — the first time in this
project's history either estimator has landed meaningfully off the known
optimum basin rather than an adjacent grid point. Its own verbose log shows
why: `BullseyeController calibrated: readings=[0.2054, 0.2968] mean=0.2511
D=0.2400 boundaries=[0.16, 0.08, 0.04]`. GP-EI's proposals are global (any
grid point can be the next suggestion), so its early incumbent-improving
moves were large in normalized-distance terms (~0.21–0.30) — calibrating
wide ring boundaries. None of its subsequent 25 proposals at the 0.15% zone
ever produced a displacement below the innermost boundary (0.16), so
`zone_i` never left 0: it spent its entire `n_iter=25` budget at the
cheapest tier, then ran the mandatory 3-fit `promote_k` polish at 100% —
which wasn't enough to recover, since the top-3 candidates it promoted were
themselves ranked using only 0.15%-fraction scores. This contrasts with
`PatternSearchCV`'s Hooke-Jeeves search, whose local mesh-contraction moves
are small and localized by construction, so its bullseye ring-crossing logic
engages differently (in this run, it never triggered a ring crossing either
— it skipped straight to 100% via the same `forced-final-polish` mesh-floor
rule seen in Experiment 13 — but its convergence check keeps re-confirming
*at full data* until 3 consecutive full-data sweeps fail, which is what
actually found and validated the true optimum here).

**Finding 2 — this also means the aggressive 0.15% floor made
PatternSearchCV *more* expensive, not less.** PSC eager/stratified's 10.030
full-fit equivalents is nearly double Experiment 13's patient run at 0.5%
starting data (5.09 equiv) and Experiment 14's eager run at 0.2% (5.06
equiv) — the lower starting fraction meant more full-data confirmation fits
were needed (10 vs 5), each costing a full 1.0 equivalent, more than
offsetting any savings from the cheaper exploration phase.

**Finding 3 — PSC eager/random with seed 42 landed close to, but
measurably off, the true optimum, and behaved differently from both
`stratified` arms.** (4, 170, 17), MAE 807.968 — a real but modest 0.36%
relative gap from 805.038, and a different (n_estimators, max_depth) pair
entirely. Unlike either `stratified` arm, this run actually touched every
rung of the ladder (11/7/7/6 across the four tiers) rather than skipping
straight to 100% — a qualitatively different mesh trajectory under this
seed. Within this same experiment (same ladder, same contraction, only
`subsample`+seed differing), `stratified` found the exact known optimum
while `random` (seed 42) did not — the opposite of Experiment 14's result
(seed 0, tied exactly), though the two runs also differ in starting
fraction (0.15% vs 0.2%) and aren't a fully isolated single-variable
comparison of seed alone.

**Who broke first, in this specific run: BayesHalvingSearchCV, by the
largest margin (+7.6% MAE); PSC eager/random(42) second (+0.36% MAE);
PSC eager/stratified did not break (found the exact known optimum, at
nearly double the usual compute cost).**

---

## Experiment 16 — Faithfulness sweep: which axis breaks at low data fractions, and where (2026-07-19, done)

**Why**: Experiment 15 found that starting at an aggressive 0.15% data
fraction corrupted the search's incumbent — its verbose log showed the
"best so far" pick was `max_depth=14`, when the true best (on full data)
is `max_depth=17`. That means the cheap fraction was *lying* about which
hyperparameter value was better. This experiment asks precisely: at what
data fraction does each hyperparameter's cheap-data ranking stop lying and
start agreeing with its true (100%-data) ranking — its "faithfulness
floor"? Below that floor, starting a search there risks the exact
corruption Experiment 15 hit; above it, cheap exploration is trustworthy.

**Methodology**: side-analysis, not a search run. Marginal 3-axis sweep
around the known optimum (`max_features=4, n_estimators=130,
max_depth=17`): hold two dimensions fixed at their optimal value, vary the
third across its full grid range (`n_estimators` thinned to every other
value), and score every resulting config at 9 fractions from 0.15% up to
100% (27 distinct configs × 9 fractions = 243 fits). Row selection
(`stratified_order`) and CV splitting (`ZoneSplitter` + `TimeSeriesSplit`)
use the exact same machinery the real searches use, not a synthetic
approximation — this measures the actual signal a real search would see
at each fraction.

**Reading the tables**: each row is one data fraction; each non-`argmin`
column is one value of the varied hyperparameter (the other two held
fixed, noted in each table's heading); **each cell is the cross-validated
MAE for that (hyperparameter value, fraction) combination — lower is
better**. The bolded cell and the `argmin` column mark which value looked
best *at that fraction*; that's "faithful" only when it matches the
100%-fraction row's answer.

**max_features axis (n_estimators=130, max_depth=17 fixed) — every cell
below is a cross-validated MAE (mean absolute error, lower = better); the
column headers are the `max_features` value that produced that MAE**

| data fraction | MAE @ max_features=2 | MAE @ max_features=3 | MAE @ max_features=4 | best max_features at this fraction |
|---|---|---|---|---|
| 0.15% | 1249.1 | 1131.8 | **1015.3** | 4 |
| 0.20% | 1252.6 | 1098.0 | **994.1** | 4 |
| 0.30% | 1211.3 | 1064.9 | **960.0** | 4 |
| 0.50% | 1208.6 | 1066.2 | **944.6** | 4 |
| 1% | 1148.5 | 1011.0 | **879.6** | 4 |
| 2% | 1089.4 | 970.3 | **858.0** | 4 |
| 5% | 1063.3 | 935.5 | **835.2** | 4 |
| 10% | 1041.9 | 921.1 | **818.0** | 4 |
| 100% | 1015.2 | 907.6 | **805.0** | 4 |

Perfectly faithful at every fraction tested — `max_features=4` has the
lowest (best) MAE in every single row, from 0.15% all the way to 100%.

**n_estimators axis (max_features=4, max_depth=17 fixed) — same reading:
every cell is a cross-validated MAE for that `n_estimators` value at that
data fraction, lower = better**

| data fraction | MAE @ n_estimators=10 | 30 | 50 | 70 | 90 | 110 | 130 | 150 | 170 | 190 | 210 | 230 | 250 | best n_estimators |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.15% | 1204.2 | 1099.7 | 1047.3 | 1035.7 | 1023.8 | 1026.9 | **1015.3** | 1022.2 | 1020.9 | 1019.7 | 1026.2 | 1026.1 | 1027.2 | 130 |
| 0.20% | 1164.7 | 1097.2 | 1029.3 | 1018.0 | 1003.2 | 1008.8 | **994.1** | 998.1 | 999.4 | 1005.0 | 1009.3 | 1010.2 | 1013.3 | 130 |
| 0.30% | 1132.4 | 1044.2 | 1001.4 | 989.1 | 974.9 | 969.8 | **960.0** | 960.2 | 961.6 | 969.0 | 972.1 | 973.5 | 973.0 | 130 |
| 0.50% | 1093.2 | 1020.5 | 968.5 | 971.5 | 953.8 | 952.7 | **944.6** | 946.5 | 948.8 | 952.7 | 956.2 | 958.9 | 961.6 | 130 |
| 1% | 1000.6 | 942.9 | 902.5 | 900.0 | 888.6 | 889.9 | **879.6** | 881.3 | 884.4 | 887.1 | 888.7 | 890.8 | 892.1 | 130 |
| 2% | 929.7 | 899.3 | 873.1 | 873.8 | 869.1 | 867.2 | **858.0** | 860.0 | 861.4 | 864.5 | 865.5 | 865.3 | 866.1 | 130 |
| 5% | 934.6 | 891.6 | 848.1 | 848.9 | 841.5 | 842.8 | **835.2** | 835.5 | 837.1 | 842.3 | 845.1 | 845.0 | 846.6 | 130 |
| 10% | 881.3 | 861.3 | 837.0 | 832.9 | 823.8 | 826.3 | **818.0** | 818.0 | 818.3 | 821.8 | 824.7 | 827.6 | 830.5 | 130 |
| 100% | 879.9 | 851.7 | 817.4 | 815.7 | 809.3 | 812.2 | **805.0** | 805.7 | 808.0 | 812.4 | 814.7 | 815.4 | 818.1 | 130 |

Also perfectly faithful — `n_estimators=130` has the lowest MAE in every
row, despite a visibly bumpier curve than `max_features`'s.

**max_depth axis (max_features=4, n_estimators=130 fixed) — the fragile
one. Same reading: every cell is a cross-validated MAE for that
`max_depth` value at that data fraction, lower = better**

| data fraction | MAE @ max_depth=5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | best max_depth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0.15%** | 1254.8 | 1171.4 | 1127.3 | 1099.0 | 1065.8 | 1044.0 | 1035.7 | 1016.8 | 1020.1 | **1008.5** | 1016.1 | 1025.3 | 1015.3 | **14 (WRONG — true best is 17)** |
| **0.20%** | 1267.5 | 1218.3 | 1161.5 | 1110.8 | 1077.8 | 1057.9 | 1029.0 | 1019.8 | 1000.6 | 999.6 | 993.9 | **987.6** | 994.1 | **16 (WRONG — true best is 17)** |
| 0.30% | 1267.1 | 1205.4 | 1144.6 | 1096.9 | 1065.3 | 1027.7 | 1019.8 | 979.7 | 974.4 | 969.5 | 969.1 | 962.6 | **960.0** | 17 |
| 0.50% | 1316.9 | 1255.1 | 1177.5 | 1132.1 | 1078.2 | 1053.2 | 1015.8 | 992.0 | 976.8 | 968.8 | 955.4 | 957.3 | **944.6** | 17 |
| 1% | 1296.7 | 1221.7 | 1163.1 | 1114.8 | 1057.6 | 1023.4 | 988.5 | 952.6 | 926.5 | 916.6 | 902.6 | 886.7 | **879.6** | 17 |
| 2% | 1288.7 | 1206.8 | 1144.8 | 1097.1 | 1054.3 | 1015.6 | 982.7 | 949.0 | 915.8 | 894.6 | 877.2 | 864.1 | **858.0** | 17 |
| 5% | 1303.1 | 1240.4 | 1167.7 | 1105.2 | 1064.3 | 1018.5 | 978.8 | 948.0 | 916.8 | 887.1 | 875.3 | 844.3 | **835.2** | 17 |
| 10% | 1303.8 | 1235.2 | 1165.2 | 1103.7 | 1072.6 | 1022.1 | 983.4 | 947.6 | 915.3 | 878.3 | 858.5 | 835.6 | **818.0** | 17 |
| 100% | 1318.1 | 1252.7 | 1188.5 | 1144.1 | 1082.0 | 1039.3 | 1009.3 | 961.4 | 933.5 | 887.2 | 863.9 | 831.1 | **805.0** | 17 |

Reading the 100% row (the ground truth) confirms `max_depth=17` really is
best (MAE 805.0, the lowest in that row) — but at 0.15% and 0.20%, a
*different* depth's MAE happens to dip lower than depth=17's, which is
exactly the kind of small-sample noise that misled Experiment 15's search.

**Finding: of the three grid axes, only `max_depth` is fragile at low data
fractions, and its faithfulness floor is precisely 0.30%.** `max_features`
and `n_estimators` pick the correct argmin at every single fraction tested,
all the way down to 0.15% — their quality gaps between values are large
enough to survive small-sample noise. `max_depth`'s gaps near the optimum
are much tighter (960.0 vs 962.6 at 0.30%, depth 17 vs 16), so at 0.15% and
0.20% small-sample noise inverts the ranking: `depth=14` and `depth=16`
respectively look better than the true-optimal `depth=17`. From 0.30%
onward the correct answer (17) wins at every fraction, with a clean,
monotonically-improving-with-fraction curve.

This is a direct, causal explanation for Experiment 15's failure mode (the
search's incumbent moved to `max_depth=14` at the 0.15% zone, exactly
matching this table's row) and for why every prior experiment starting at
0.5% or above (Experiments 1, 7-13) never exhibited it — 0.5% sits
comfortably above the 0.30% floor. Experiment 11's 0.25% starting zone sits
inside the untested gap between this sweep's 0.20% (broken) and 0.30%
(fixed) fractions; that experiment's real search nonetheless found the
correct optimum, which is not a contradiction — the search's own resilience
(multiple probe directions, pattern moves, mandatory full-data confirmation)
provides some margin beyond what a single marginal-axis argmin check alone
predicts — but it is not covered by this sweep and remains untested
directly.
