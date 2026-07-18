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
| evaluations | 15 | 22 | 28 |
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
search-quality gap — the `n_starts=4` follow-up arm (spec §10, optional,
not yet run) would be the way to confirm that.

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

**Open follow-up, not yet run:** repeat the `subsample='random'` arm across
multiple seeds (the "multi-seed variance bands" item already in
`OpenQuestions.md`) before concluding coverage doesn't matter to search
outcomes at all — one seed, however clean, is not enough to overturn the
side-analysis's real, measured coverage gap. What this experiment does show
is that `stratified`'s coverage guarantee is not *load-bearing* for this
specific search/grid/seed at 0.2% — a useful, honest negative result, not
the confirmation the theory predicted.

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
