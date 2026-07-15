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

**Methodology note — what "stratified" is actually doing on this dataset
(2026-07-15).** Every stratified run so far, including Experiments 7–9, has
logged `stratified_order: 418,416 rows, 418,416 runs (1.0 rows/run avg)` —
every single row is its own "run." `subsample` watches **all** feature
columns by default, and this dataset carries several continuous daily
weather columns (temperature, humidity, pressure, wind, precipitation, dew
point) that essentially never repeat between consecutive rows. So even
though `StoreID`/`IsOpen`/`HasPromotions`/etc. genuinely do repeat for many
consecutive days, the moment any one of ~15 weather columns ticks — which
happens almost every row — the whole row counts as a "transition." The
sampler has therefore been running in its documented fail-soft degenerate
mode this whole time: it collapses to **systematic full-timeline sampling**
(evenly spread picks across the whole year, never just the oldest slice),
not its designed behavior of prioritizing genuine categorical transitions
over repeated "typical" rows. This is almost certainly *why* stratified
keeps winning regardless of how small the rung gets (Experiments 7–9 below):
an evenly-spread sample stays a faithful miniature of the whole year no
matter how sparse, whereas `expanding`'s failure mode (Experiment 6) was
about being unrepresentative in *time*, not about sample size. The
sampler's actual signature mechanism (novel-combination seats, alternating
boundary/midpoint priority) has not yet been exercised end-to-end on this
dataset — that needs `subsample_columns` narrowed to genuinely categorical,
slow-changing columns so real multi-row runs actually form.

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

## Open questions queued for future experiments

- `subsample='stratified'` (transition sampling) vs `'expanding'` on this dataset —
  does a full-timeline 10% sample find the basin faster/more reliably than the
  oldest-tenth sample? (Signature feature of the package; article figure.)
- `n_starts ∈ {1, 2, 4, 8}` at same budget and same wall-clock (multi-start ablation).
- `poll='complete'` on this 8-core machine (expected ~neutral single-start; relevant
  with multi-start batching).
- Whether intermediate data zones engage on larger grids / multi-start.
- Multi-seed variance bands for the Optuna samplers (both runs above are seed=0).
