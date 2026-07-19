"""BayesHalvingSearchCV: sklearn-compatible Bayesian hyperparameter search
(Gaussian Process + Expected Improvement) with bullseye multi-fidelity data
growth and scatter-search multi-start.

Spec: BAYESHALVINGSearchCV_SPEC.md.
"""

from __future__ import annotations

import logging
import math
import time
from copy import deepcopy
from numbers import Integral

import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_random_state, get_tags
from sklearn.utils.validation import _num_samples

from ._climber import _better
from ._fidelity import BullseyeController
from ._gp import GPProposer
from ._sampling import ZoneSplitter, expanding_order, random_order, stratified_order
from ._space import Space
from ._starts import select_starts

logger = logging.getLogger("pattern_search_cv")

_DEFAULT_ZONES = (0.005, 0.01, 0.1, 1.0)


class BayesHalvingSearchCV(BaseSearchCV):
    """Gaussian-Process Bayesian search over a discrete hyperparameter grid,
    with bullseye multi-fidelity data growth and scatter-search multi-start.

    Despite the name, the fidelity mechanism is this package's self-calibrating
    bullseye-ring data growth (see ``PatternSearchCV``), not classic successive
    halving; the name was chosen by the project's author and kept as-is.

    ``param_grid`` and multi-start (``n_starts``/``start_points``) follow the
    exact same standard as :class:`PatternSearchCV` — see its docstring and
    ``PatternSearchCV_SPEC.md`` for the search-space and scatter-search
    conventions; they are not duplicated here. This estimator has **zero
    additional dependencies** beyond ``PatternSearchCV``'s own (``numpy``,
    ``scipy``, ``scikit-learn`` — the GP surrogate is
    ``sklearn.gaussian_process.GaussianProcessRegressor`` plus a hand-rolled
    Expected Improvement acquisition; no Optuna, no torch).

    Parameters
    ----------
    estimator : estimator object
        The estimator to tune.
    param_grid : dict
        Maps parameter names to either an explicit list of values or a
        ``(low, high, num)`` tuple expanded to a linspace grid.
    n_iter : int, default=25
        Budget of genuine (non-cache-served) model evaluations **per start**,
        across all data zones combined, excluding the bounded final-polish
        re-scores (at most ``promote_k`` + 1 extra evaluations per start).
    promote_k : int, default=3
        Number of top-scored configurations re-scored (and used to seed a
        fresh Gaussian Process) whenever the bullseye rings climb to a new
        data zone, and again for the mandatory final polish at full data.
    data_zones : int or sequence of float, default=(0.005, 0.01, 0.1, 1.0)
        The data ladder. Identical semantics to ``PatternSearchCV.data_zones``.
    warmup : int, default=3
        Positions (starting point included) before the bullseye rings
        self-calibrate. Identical semantics to ``PatternSearchCV.warmup``.
    subsample : {"auto", "expanding", "stratified", "random"}, default="auto"
        Identical semantics to ``PatternSearchCV.subsample``.
    subsample_columns : sequence of int, optional
        Column subset watched by the "stratified" transition sampler.
    n_starts : int, default=1
        Independent Bayesian searches. Starts are chosen by the same
        scatter-search mechanism as ``PatternSearchCV`` (QMC pool + greedy
        maximin); every start runs to completion (no elimination between
        starts) and the best full-data optimum wins. Unlike
        ``PatternSearchCV``, there is no state-match merging between starts
        (no clean analog for a stochastic search) — the shared dedup cache is
        the cost-saving mechanism instead.
    start_points : list of dict, optional
        Explicit start points (parameter dicts); they take seats before
        scatter-search generation fills the rest.

    Notes
    -----
    ``verbose >= 1`` narrates every search decision as it happens (proposals,
    ring crossings, data climbs, final polish) and, at the end of ``fit``,
    logs a full ``cross_validate`` pass on the winning parameters over the
    complete dataset with the user's own ``cv`` splitter, exactly mirroring
    ``PatternSearchCV``. This adds ``n_splits`` extra fits and is skipped
    entirely at ``verbose=0`` (the default). ``verbose >= 2`` additionally
    logs per-proposal debug detail.
    """

    _required_parameters = ["estimator", "param_grid"]

    def __init__(self, estimator, param_grid, *, scoring=None, n_jobs=None,
                refit=True, cv=None, verbose=0, random_state=None,
                pre_dispatch="2*n_jobs", error_score=np.nan,
                return_train_score=False,
                n_iter=25, promote_k=3,
                data_zones=_DEFAULT_ZONES, warmup=3,
                subsample="auto", subsample_columns=None,
                n_starts=1, start_points=None):
        super().__init__(estimator=estimator, scoring=scoring, n_jobs=n_jobs,
                         refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)
        self.param_grid = param_grid
        self.random_state = random_state
        self.n_iter = n_iter
        self.promote_k = promote_k
        self.data_zones = data_zones
        self.warmup = warmup
        self.subsample = subsample
        self.subsample_columns = subsample_columns
        self.n_starts = n_starts
        self.start_points = start_points

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None, **params):
        # NOTE: with n_jobs > 1, sklearn pickles `self` for every parallel
        # task, so nothing unpicklable (handlers, live GP fit state,
        # generators) may ever be stored on the instance during fit.
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
                "is required by BayesHalvingSearchCV.")
        if y is None and self.__sklearn_tags__().target_tags.required:
            raise ValueError(
                f"{type(self).__name__} requires y to be passed, but the "
                "target y is None.")

        # ---- validate the ladder (same rule as PatternSearchCV) ---------
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
        if not isinstance(self.n_iter, Integral) or self.n_iter < 1:
            raise ValueError(f"n_iter must be an int >= 1, got {self.n_iter}")
        if not isinstance(self.promote_k, Integral) or self.promote_k < 1:
            raise ValueError(f"promote_k must be an int >= 1, got {self.promote_k}")
        if not isinstance(self.n_starts, Integral) or self.n_starts < 1:
            raise ValueError(f"n_starts must be an int >= 1, got {self.n_starts}")

        # ---- header: what is being optimized, and over what grid? -------
        logger.info("BayesHalvingSearchCV: optimizing metric = %s",
                    self._scoring_label())
        logger.info("cv = %s", type(self.cv).__name__ if self.cv is not None
                    else "5-fold KFold (sklearn default)")
        for d in space.dims:
            logger.info("  %s : %s", d.name, d.values)
        logger.info("n_iter=%d, promote_k=%d, n_starts=%d, warmup=%d",
                    self.n_iter, self.promote_k, self.n_starts, self.warmup)

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

        # ---- subsample priority ordering (once per fit) ------------------
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

        # ---- starts (scatter search, same mechanism as PatternSearchCV) --
        starts = select_starts(space, self.n_starts, self.start_points, rng)
        logger.info("starts (%d): %s", len(starts),
                    [space.params(s) for s in starts])

        self._ctx = {
            "space": space, "zones": eff_zones, "sizes": sizes,
            "order": order, "n_samples": n_samples, "starts": starts,
            "rng": rng,
        }
        return handler

    # ------------------------------------------------------------ search
    def _run_search(self, evaluate_candidates, **kwargs):
        ctx = self._ctx
        space, zones, sizes = ctx["space"], ctx["zones"], ctx["sizes"]
        rng = ctx["rng"]

        if isinstance(self.refit, str):
            metric = self.refit
        else:
            metric = "score"
        score_key = f"mean_test_{metric}"

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

        max_asks = 10 * self.n_iter
        cache = {}
        per_start_results = []
        for start_i, start_point in enumerate(ctx["starts"]):
            result = _run_one_start(
                start_i, start_point, space, zones, self.promote_k,
                self.n_iter, self.warmup, evaluate_batch, cache, rng, max_asks,
            )
            per_start_results.append(result)
            logger.info("start %d converged: %s score=%.6g (%d cache hits so far)",
                        start_i, space.params(result["incumbent"]),
                        result["score"] if result["score"] == result["score"]
                        else float("nan"), result["cache_hits"])

        history = []
        for r in per_start_results:
            history.extend(r["history"])

        ctx["results"] = {
            "local_optima": _build_local_optima(space, per_start_results),
            "history": history,
            "cache_hits": sum(r["cache_hits"] for r in per_start_results),
        }

    # ---------------------------------------------------- best selection
    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """best_* come ONLY from full-data evaluations, across every start."""
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
        full data and the user's own ``cv`` splitter. Verbose-gated only:
        this adds ``n_splits`` extra fits and never runs unless requested
        (``verbose >= 1``). Mirrors ``PatternSearchCV``'s implementation.
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


# ---------------------------------------------------------------------- #
# module-level helpers (no `self`: kept out of the class so nothing here  #
# is ever a candidate for accidentally landing on `self` during fit)      #
# ---------------------------------------------------------------------- #

def _run_one_start(start_i, start_point, space, zones, promote_k, n_iter,
                   warmup, evaluate_batch, cache, rng, max_asks):
    """One independent Bayesian search (BAYESHALVINGSearchCV_SPEC.md §4.3):
    GP-EI proposals with bullseye multi-fidelity climbing, sharing `cache`
    (dedup, across starts) and drawing all randomness from `rng` (determinism,
    §7)."""
    controller = BullseyeController(space.min_step, n_boundaries=len(zones) - 1,
                                    warmup=warmup)
    zone_i = 0
    incumbent, incumbent_score = None, None
    fits_used = 0
    cache_hits = 0
    history = []

    def score_at(idx, frac, event, trial):
        nonlocal fits_used, cache_hits
        key = (idx, frac)
        if key in cache:
            cache_hits += 1
            s = cache[key]
        else:
            s = evaluate_batch(frac, [idx])[0]
            fits_used += 1
            cache[key] = s
        history.append({"start": start_i, "trial": trial, "params": space.params(idx),
                        "fraction": frac, "score": s, "event": event})
        return s

    proposer = GPProposer(space, rng.randint(np.iinfo(np.int32).max))
    proposer.observe(start_point, None)

    proposals = 0
    while fits_used < n_iter and proposals < max_asks:
        idx = proposer.suggest()
        proposals += 1
        frac = zones[zone_i]
        key = (idx, frac)
        if key in cache:
            cache_hits += 1
            proposer.observe(idx, cache[key])
            continue
        score = evaluate_batch(frac, [idx])[0]
        fits_used += 1
        cache[key] = score
        proposer.observe(idx, score)
        history.append({"start": start_i, "trial": proposals, "params": space.params(idx),
                        "fraction": frac, "score": score, "event": "trial"})

        if incumbent is None or _better(score, incumbent_score):
            move = 0.0 if incumbent is None else space.distance(incumbent, idx)
            incumbent, incumbent_score = idx, score
            new_zone = controller.observe_improvement(move)
            if new_zone > zone_i:
                zone_i = new_zone
                top = proposer.top_k_observed(promote_k)
                proposer = GPProposer(space, rng.randint(np.iinfo(np.int32).max))
                rescored = []
                for cfg_idx in top:
                    s = score_at(cfg_idx, zones[zone_i], "climb-rescore", proposals)
                    proposer.observe(cfg_idx, s)
                    rescored.append((cfg_idx, s))
                incumbent, incumbent_score = max(
                    rescored, key=lambda t: float("-inf") if t[1] != t[1] else t[1])

    if proposals >= max_asks and fits_used < n_iter:
        logger.warning("start %d hit max_asks=%d proposal cap before exhausting "
                       "n_iter=%d fits (%d used); proceeding to final polish",
                       start_i, max_asks, n_iter, fits_used)

    # ---- forced final polish for this start (always) --------------------
    if zones[zone_i] < 1.0:
        top = proposer.top_k_observed(promote_k)
        polished = []
        for cfg_idx in top:
            s = score_at(cfg_idx, 1.0, "final-polish", proposals)
            polished.append((cfg_idx, s))
        incumbent, incumbent_score = max(
            polished, key=lambda t: float("-inf") if t[1] != t[1] else t[1])

    return {"start_point": start_point, "incumbent": incumbent,
            "score": incumbent_score, "history": history, "cache_hits": cache_hits}


def _build_local_optima(space, per_start_results):
    """Distinct converged optima across starts, best first — dedup by final
    incumbent index tuple (BAYESHALVINGSearchCV_SPEC.md §3.2), same shape as
    PatternSearchCV's Engine.local_optima()."""
    groups = {}
    for r in per_start_results:
        groups.setdefault(r["incumbent"], []).append(r)
    out = []
    for point, rs in groups.items():
        score = max((r["score"] for r in rs),
                    key=lambda s: float("-inf") if s != s else s)
        out.append({
            "params": space.params(point),
            "score": score,
            "n_starts_converged": len(rs),
            "start_points": [space.params(r["start_point"]) for r in rs],
        })
    out.sort(key=lambda d: float("-inf") if d["score"] != d["score"]
             else d["score"], reverse=True)
    return out
