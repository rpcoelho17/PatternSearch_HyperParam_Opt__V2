"""PatternSearchCV: sklearn-compatible Hooke-Jeeves hyperparameter search
with bullseye multi-fidelity data growth and scatter-search multi-start.

Spec: PatternSearchCV_SPEC.md.
"""

from __future__ import annotations

import logging
import math
import os
import time
from copy import deepcopy
from numbers import Integral

import numpy as np
from scipy.stats import qmc
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state, get_tags
from sklearn.utils.validation import _num_samples

from ._climber import Climber
from ._engine import Engine
from ._sampling import ZoneSplitter, expanding_order, random_order, stratified_order
from ._space import Space

logger = logging.getLogger("pattern_search_cv")

_DEFAULT_ZONES = (0.05, 0.10, 0.20, 1.0)


class PatternSearchCV(BaseSearchCV):
    """Hooke-Jeeves pattern search over a discrete hyperparameter grid.

    Parameters
    ----------
    estimator : estimator object
        The estimator to tune.
    param_grid : dict
        Maps parameter names to either an explicit list of values or a
        ``(low, high, num)`` tuple expanded to a linspace grid.
    poll : {"auto", "complete", "opportunistic"}, default="opportunistic"
        Exploratory sweep mode.  "complete" evaluates all +/-delta probes
        around the fixed center in one parallel batch (MATLAB UseCompletePoll)
        plus the composite of improving dimensions; "opportunistic" is the
        classic 1961 sequential sweep with immediate acceptance.  "auto" picks
        "complete" when ``n_jobs / n_splits >= 2``, else "opportunistic".
        CAVEAT: the "opportunistic" default was chosen because it is what
        "auto" resolved to on every machine this package has been benchmarked
        on (5-fold CV, <=8 cores); "complete" poll has never actually been
        measured.  If you have many more cores than CV folds, "auto" (or
        explicit "complete") may parallelize better - this has not been
        verified either way.
    mesh_expansion : float, default=1.0
        Step-size multiplier applied after a successful sweep.  1.0 (default)
        is classic Hooke-Jeeves (contraction only); 2.0 matches MATLAB GPS.
        Raise it on fine, continuous-like grids.
    contraction : {"patient", "eager"}, default="eager"
        When the mesh contracts.  "patient" (classic Hooke-Jeeves): only after
        a failed exploratory sweep.  "eager" (prototype-faithful, default): a
        failed pattern move also contracts, spending step resolution faster.
        CAVEAT: in every controlled single-variable test run on the retail
        benchmark (three separate experiments), "eager" measured cost-neutral
        to slightly worse than "patient" on full-fit equivalents (e.g. 6.90 vs
        6.80; byte-identical evaluation sequences in two other runs) - there
        is no measured compute advantage to this default, and "eager" carries
        a real (untested-on-rugged-landscapes) premature-convergence risk that
        "patient" does not.  It was made default by explicit user decision,
        not by benchmark evidence.  Consider "patient" if you hit poor optima
        on a landscape with many local structures; pair "eager" with
        n_starts > 1 to hedge the risk.
    data_zones : int or sequence of float, default=(0.05, 0.10, 0.20, 1.0)
        The data ladder.  An int n gives n evenly divided levels
        (``4 -> [0.25, 0.5, 0.75, 1.0]``); a sequence gives explicit ascending
        fractions ending at 1.0; ``1`` disables multi-fidelity.  This
        front-loaded, aggressive-start ladder was set as the default after it
        strictly dominated the previous default (0.10, 0.20, 0.50, 1.0) on the
        retail benchmark - better optimum, less compute (5.85 vs 6.80 full-fit
        equivalents), and faster wall-clock - when paired with
        ``subsample="stratified"``.  Evidence is from one dataset/grid; a 5%
        starting zone risks an unrepresentative sample on data where
        ``subsample`` cannot make small rungs faithful (see ``subsample``).
    warmup : int, default=3
        Number of positions (starting point included) before the bullseye
        rings self-calibrate.  The patience dial: higher = data is added
        closer to the optimum.  Minimum 3 (two displacement readings).
    subsample : {"auto", "expanding", "stratified", "random"}, default="auto"
        How the data-zone priority ordering is built.  "auto" picks
        "stratified" for time-ordered splitters (TimeSeriesSplit), else
        "random".  "stratified" (the transition sampler) measurably beat
        "expanding" on the retail benchmark's aggressive 5% starting zone
        (lower MAE, less compute, faster wall-clock) and is fail-soft by
        design (degrades to systematic sampling in the worst case), which is
        why it is now the time-series default instead of "expanding".
        "expanding" remains available explicitly.  "random" on temporal data
        leaks future rows - see docs.
    subsample_columns : sequence of int, optional
        Column subset watched by the "stratified" transition sampler.
    n_starts : int, default=1
        Independent climbers.  Starts are chosen by scatter search (QMC pool +
        greedy maximin); every climber runs to completion (no elimination) and
        the best full-data optimum wins.
    start_points : list of dict, optional
        Explicit start points (parameter dicts); they take seats before
        scatter-search generation fills the rest.

    Notes
    -----
    ``verbose >= 1`` narrates every search decision as it happens (moves,
    contractions, ring crossings, data climbs, merges) and, at the end of
    ``fit``, logs a full ``cross_validate`` pass on the winning parameters
    over the complete dataset with the user's own ``cv`` splitter -
    mirroring a typical post-search sanity check.  This adds ``n_splits``
    extra fits and is skipped entirely at ``verbose=0`` (the default), so it
    never costs anything unless requested.  ``verbose >= 2`` additionally
    logs per-probe debug detail.
    """

    _required_parameters = ["estimator", "param_grid"]

    def __init__(self, estimator, param_grid, *, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0, random_state=None,
                 pre_dispatch="2*n_jobs", error_score=np.nan,
                 return_train_score=False,
                 poll="opportunistic", mesh_expansion=1.0, contraction="eager",
                 data_zones=_DEFAULT_ZONES, warmup=3,
                 subsample="auto", subsample_columns=None,
                 n_starts=1, start_points=None):
        super().__init__(estimator=estimator, scoring=scoring, n_jobs=n_jobs,
                         refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)
        self.param_grid = param_grid
        self.random_state = random_state
        self.poll = poll
        self.mesh_expansion = mesh_expansion
        self.contraction = contraction
        self.data_zones = data_zones
        self.warmup = warmup
        self.subsample = subsample
        self.subsample_columns = subsample_columns
        self.n_starts = n_starts
        self.start_points = start_points

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None, **params):
        # NOTE: with n_jobs > 1, sklearn pickles `self` for every parallel
        # task, so nothing unpicklable (handlers, engines, generators) may
        # ever be stored on the instance during fit.
        handler = self._prepare_run(X, y)
        try:
            super().fit(X, y, **params)
            results = (self._ctx or {}).get("results")
            if results is not None:
                self.local_optima_ = results["local_optima"]
                self.search_history_ = results["history"]
                self.n_cache_hits_ = results["cache_hits"]
            if self.verbose and hasattr(self, "best_params_"):
                self._log_cv_summary(X, y)
        finally:
            if handler is not None:
                logger.removeHandler(handler)
            self._ctx = None
        return self

    def __sklearn_tags__(self):
        # BaseSearchCV delegates most sub-estimator tags but not these two;
        # we support NaN inputs / multi-output y exactly iff the wrapped
        # estimator does.
        tags = super().__sklearn_tags__()
        sub = get_tags(self.estimator)
        tags.input_tags.allow_nan = sub.input_tags.allow_nan
        tags.target_tags = deepcopy(sub.target_tags)
        return tags

    def _scoring_label(self):
        """Human-readable name of the metric being optimized (for the
        verbose header and the end-of-run summary)."""
        if self.scoring is None:
            return "estimator default (R^2 for regressors, accuracy for classifiers)"
        if isinstance(self.scoring, str):
            return self.scoring
        if isinstance(self.scoring, dict):
            names = ", ".join(self.scoring.keys())
            which = self.refit if isinstance(self.refit, str) else (
                "callable" if callable(self.refit) else "first")
            return f"{names} (best selected by: {which})"
        if callable(self.scoring):
            return getattr(self.scoring, "__name__", repr(self.scoring))
        return str(self.scoring)

    def _prepare_run(self, X, y):
        # verbose -> package logger (spec forbids print); the handler is
        # returned (not stored on self) so the instance stays picklable
        handler = None
        if self.verbose and not any(
            not isinstance(h, logging.NullHandler) for h in logger.handlers
        ):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if self.verbose >= 2 else logging.INFO)

        space = Space(self.param_grid)
        rng = check_random_state(self.random_state)
        n_samples = _num_samples(X)
        if n_samples < 1:
            raise ValueError(
                f"Found array with {n_samples} sample(s) "
                f"(shape={getattr(X, 'shape', None)}) while a minimum of 1 "
                "is required by PatternSearchCV.")
        if y is None and self.__sklearn_tags__().target_tags.required:
            raise ValueError(
                f"{type(self).__name__} requires y to be passed, but the "
                "target y is None.")

        # ---- validate the ladder (spec 5) -------------------------------
        zones = self.data_zones
        if isinstance(zones, Integral) and not isinstance(zones, bool):
            if zones < 1:
                raise ValueError(f"data_zones must be >= 1, got {zones}")
            zones = [k / zones for k in range(1, zones + 1)]
        else:
            zones = [float(z) for z in zones]
            if (not zones or any(z <= 0 or z > 1 for z in zones)
                    or sorted(zones) != zones or len(set(zones)) != len(zones)
                    or zones[-1] != 1.0):
                raise ValueError(
                    "data_zones must be strictly ascending fractions in (0, 1] "
                    f"ending at 1.0, got {self.data_zones}")
        if not isinstance(self.warmup, Integral) or self.warmup < 3:
            raise ValueError(f"warmup must be an int >= 3, got {self.warmup}")
        if self.poll not in ("auto", "complete", "opportunistic"):
            raise ValueError(f"poll must be 'auto', 'complete' or "
                             f"'opportunistic', got {self.poll!r}")
        if self.mesh_expansion < 1.0:
            raise ValueError(
                f"mesh_expansion must be >= 1.0, got {self.mesh_expansion}")
        if self.contraction not in ("patient", "eager"):
            raise ValueError(f"contraction must be 'patient' or 'eager', "
                             f"got {self.contraction!r}")
        if not isinstance(self.n_starts, Integral) or self.n_starts < 1:
            raise ValueError(f"n_starts must be an int >= 1, got {self.n_starts}")

        # ---- header: what is being optimized, and over what grid? -------
        logger.info("PatternSearchCV: optimizing metric = %s",
                    self._scoring_label())
        logger.info("cv = %s", type(self.cv).__name__ if self.cv is not None
                    else "5-fold KFold (sklearn default)")
        for d in space.dims:
            logger.info("  %s : %s", d.name, d.values)

        # ---- resource floor: every zone must feed the CV enough rows ----
        n_splits_guess = getattr(self.cv, "n_splits", None) or (
            self.cv if isinstance(self.cv, Integral) else 5)
        min_rows = max(2 * (int(n_splits_guess) + 1), 8)
        sizes = []
        for z in zones:
            k = min(n_samples, max(min_rows, int(math.ceil(z * n_samples))))
            if not sizes or k > sizes[-1]:
                sizes.append(k)
        sizes[-1] = n_samples
        sizes = sorted(set(sizes))
        eff_zones = [s / n_samples for s in sizes]
        eff_zones[-1] = 1.0
        if len(eff_zones) < len(zones):
            logger.info("data ladder truncated by resource floor: %s -> %s",
                        zones, [round(z, 4) for z in eff_zones])

        # ---- subsample priority ordering (once per fit, spec 5.2) -------
        mode = self.subsample
        if mode == "auto":
            cv_name = type(self.cv).__name__ if self.cv is not None else ""
            mode = "stratified" if "TimeSeries" in cv_name else "random"
        if mode == "expanding":
            order = expanding_order(n_samples)
        elif mode == "random":
            order = random_order(n_samples, rng)
        elif mode == "stratified":
            order = stratified_order(X, self.subsample_columns)
        else:
            raise ValueError(f"subsample must be 'auto', 'expanding', "
                             f"'stratified' or 'random', got {self.subsample!r}")
        logger.info("subsample mode=%s, zones=%s (rows %s)", mode,
                    [round(z, 4) for z in eff_zones], sizes)

        # ---- starts (scatter search, spec 6.1) ---------------------------
        starts = self._select_starts(space, rng)
        logger.info("starts (%d): %s", len(starts),
                    [space.params(s) for s in starts])

        self._ctx = {
            "space": space, "zones": eff_zones, "sizes": sizes,
            "order": order, "n_samples": n_samples, "starts": starts,
        }
        return handler

    def _select_starts(self, space, rng):
        starts = []
        if self.start_points:
            for p in self.start_points:
                idx = space.indices(p)
                if idx not in starts:
                    starts.append(idx)
            starts = starts[: self.n_starts]
        if len(starts) < self.n_starts:
            mid = space.midpoint()
            if mid not in starts:
                starts.append(mid)
        if len(starts) < self.n_starts:
            # QMC candidate pool, then greedy maximin selection
            active = space.active
            seed = rng.randint(np.iinfo(np.int32).max)
            sampler = qmc.LatinHypercube(d=max(1, len(active)), seed=seed)
            pool_u = sampler.random(10 * self.n_starts)
            pool = set()
            for row in pool_u:
                idx = list(space.midpoint())
                for j, dim_i in enumerate(active):
                    n = space.dims[dim_i].n
                    idx[dim_i] = min(n - 1, int(row[j] * n))
                pool.add(tuple(idx))
            pool -= set(starts)
            pool = sorted(pool)  # determinism
            while len(starts) < self.n_starts and pool:
                best_p = max(
                    pool,
                    key=lambda p: min(space.distance(p, s) for s in starts),
                )
                starts.append(best_p)
                pool.remove(best_p)
        return starts[: self.n_starts]

    # ------------------------------------------------------------ search
    def _run_search(self, evaluate_candidates, **kwargs):
        ctx = self._ctx
        space, zones, sizes = ctx["space"], ctx["zones"], ctx["sizes"]

        # refit metric drives internal comparisons
        if isinstance(self.refit, str):
            metric = self.refit
        else:
            metric = "score"
        score_key = f"mean_test_{metric}"

        # poll mode (spec 4.2): complete pays fits for wall-clock only when
        # cores exceed the CV folds' appetite
        poll = self.poll
        if poll == "auto":
            n_jobs = self.n_jobs if self.n_jobs and self.n_jobs > 0 else (
                (os.cpu_count() or 1) if self.n_jobs == -1 else 1)
            poll = "complete" if n_jobs / max(1, self.n_splits_) >= 2 else (
                "opportunistic")
            logger.info("poll='auto' resolved to %r (n_jobs=%s, n_splits=%d)",
                        poll, self.n_jobs, self.n_splits_)

        splitters = {}
        for frac, size in zip(zones, sizes):
            if frac >= 1.0:
                splitters[frac] = None  # original CV
            else:
                splitters[frac] = ZoneSplitter(
                    self._checked_cv_orig, ctx["order"][:size])
        size_of = dict(zip(zones, sizes))

        def evaluate_batch(frac, points):
            params_list = [space.params(p) for p in points]
            results = evaluate_candidates(
                params_list,
                cv=splitters[frac],
                more_results={"n_resources": [size_of[frac]] * len(points)},
            )
            scores = results[score_key][-len(points):]
            return [float(s) for s in scores]

        climbers = [
            Climber(cid=i, space=space, start=s, zones=zones,
                    warmup=self.warmup, poll_mode=poll,
                    mesh_expansion=self.mesh_expansion,
                    contraction=self.contraction)
            for i, s in enumerate(ctx["starts"])
        ]
        # the engine lives ONLY in this frame: it holds live generators,
        # which must never sit on `self` while workers pickle the estimator
        engine = Engine(climbers, space, evaluate_batch)
        engine.run()
        ctx["results"] = {
            "local_optima": engine.local_optima(),
            "history": self._build_history(engine),
            "cache_hits": engine.n_cache_hits,
        }

    # ---------------------------------------------------- best selection
    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """best_* come ONLY from full-data evaluations (spec 1)."""
        if callable(refit):
            return refit(results)
        n_res = np.asarray(results["n_resources"])
        mask = n_res == n_res.max()
        scores = np.asarray(results[f"mean_test_{refit_metric}"], dtype=float)
        masked = np.where(mask, scores, -np.inf)
        masked = np.where(np.isnan(masked), -np.inf, masked)
        return int(np.argmax(masked))

    # ---------------------------------------------------- CV summary log
    def _log_cv_summary(self, X, y):
        """Extra ``cross_validate`` pass on the winning params, over the
        full data and the user's own ``cv`` splitter - replicates the
        prototype's ``CrossEval()`` printout.  Verbose-gated only: this adds
        ``n_splits`` extra fits and never runs unless the user asked to see
        it (``verbose >= 1``).
        """
        from sklearn.base import clone
        from sklearn.model_selection import cross_validate

        tags = get_tags(self.estimator)
        est = clone(self.estimator).set_params(**self.best_params_)

        if tags.estimator_type == "regressor":
            scoring = ("r2", "explained_variance",
                      "neg_mean_absolute_error", "neg_mean_squared_error")
        else:
            s = self.scoring
            if isinstance(s, dict):
                scoring = tuple(s.keys())
            elif isinstance(s, (list, tuple)):
                scoring = tuple(s)
            elif isinstance(s, str):
                scoring = (s,)
            else:
                scoring = ("accuracy",)

        logger.info("Cross Validation Performance (best params, full data):")
        t0 = time.time()
        scores = cross_validate(est, X, y, cv=self.cv, scoring=scoring)
        elapsed = time.time() - t0
        logger.info("Cross Validation Time: %.6f", elapsed)

        if tags.estimator_type == "regressor":
            ev = scores["test_explained_variance"]
            mae = -scores["test_neg_mean_absolute_error"]
            mse = -scores["test_neg_mean_squared_error"]
            rmse = np.sqrt(mse)
            r2 = scores["test_r2"]
            logger.info("EV per fold: %s", ev)
            logger.info("EV: %.6f", ev.mean())
            logger.info("MAE per fold: %s", mae)
            logger.info("MAE: %.6f", mae.mean())
            logger.info("MSE per fold: %s", mse)
            logger.info("MSE: %.6f", mse.mean())
            logger.info("RMSE per fold: %s", rmse)
            logger.info("RMSE: %.6f", rmse.mean())
            logger.info("R2 per fold: %s", r2)
            logger.info("Cross Validation R2: %.6f", r2.mean())
        else:
            for key in scoring:
                vals = scores[f"test_{key}"]
                logger.info("%s per fold: %s", key, vals)
                logger.info("%s: %.6f", key, vals.mean())

        logger.info("fit_time per fold: %s", scores["fit_time"])
        logger.info("fit_time: %.6f", scores["fit_time"].mean())
        logger.info("score_time per fold: %s", scores["score_time"])
        logger.info("score_time: %.6f", scores["score_time"].mean())

    # ------------------------------------------------------------- misc
    def _build_history(self, engine):
        history = []
        for c in engine.climbers:
            for point, frac, score in c.path:
                history.append({
                    "climber": c.id,
                    "params": engine.space.params(point),
                    "fraction": frac,
                    "score": score,
                })
            history.append({
                "climber": c.id, "params": engine.space.params(c.best),
                "fraction": c.fraction, "score": c.best_score,
                "status": c.status,
            })
        return history
