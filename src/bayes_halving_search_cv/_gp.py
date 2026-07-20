"""GPProposer: the from-scratch Bayesian search core (BAYESHALVINGSearchCV_SPEC.md
section 5). sklearn's GaussianProcessRegressor + a hand-rolled Expected
Improvement acquisition - no Optuna, no torch. Entirely independent of
BaseSearchCV / the estimator machinery, so it can be driven and validated in
isolation (spec section 5.3, section 8).
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import check_random_state

logger = logging.getLogger("bayes_halving_search_cv")

# Scored observations needed before trusting the GP surrogate. Deliberately
# NOT the same knob as `warmup` (BullseyeController's ring-calibration
# readiness is a different concept for a different mechanism) - kept as an
# internal constant, not user-configurable (spec 3.3).
_COLD_START = 2
_CANDIDATE_ENUM_CAP = 2000
_XI = 0.01  # Expected Improvement explore/exploit trade-off


def _featurize(space, idx_list):
    """Index tuples -> a 2D numeric feature matrix: normalized coordinate for
    numeric dims, one-hot for categorical dims. Reuses Space's own geometry
    (the same normalization Space.distance's numeric term uses) rather than
    inventing a parallel one.
    """
    rows = []
    for idx in idx_list:
        feats = []
        for d, i in zip(space.dims, idx):
            if d.n == 1:
                continue
            if d.is_categorical:
                onehot = [0.0] * d.n
                onehot[i] = 1.0
                feats.extend(onehot)
            else:
                feats.append(d.coord(i))
        rows.append(feats)
    return np.asarray(rows, dtype=float)


class GPProposer:
    """One independent Bayesian-optimization proposer over a Space.

    No dependency on BaseSearchCV/evaluate_candidates - drive it directly via
    ``observe()``/``suggest()`` for unit tests and the validation in spec
    section 8.
    """

    def __init__(self, space, random_state=None, xi=_XI):
        self.space = space
        self.rng = check_random_state(random_state)
        self.xi = xi
        self._idx = []        # observed index tuples, in order
        self._scores = []     # parallel observed scores (None = pending seed)
        self._observed_set = set()

    def observe(self, idx, score):
        """Record an observation. ``score=None`` marks a pending seed point
        that MUST be returned by the very next ``suggest()`` call."""
        idx = tuple(idx)
        self._idx.append(idx)
        self._scores.append(score)
        self._observed_set.add(idx)

    def suggest(self):
        """Return the next index tuple to evaluate."""
        seed = self._pending_seed()
        if seed is not None:
            return seed

        scored = self._scored_pairs()
        candidates = self._candidate_pool()
        if not candidates:
            # search space exhausted: fall back to the best known point
            # (harmless - a guaranteed cache hit at the estimator level).
            if scored:
                return max(scored, key=lambda t: t[1])[0]
            return self.space.midpoint()

        if len(scored) < _COLD_START:
            return self._random_pick(candidates, size=1)[0]

        return self._gp_ei_pick(scored, candidates)

    def top_k_observed(self, k):
        """This proposer's observed (idx, score) pairs, sorted desc by score,
        deduped by idx, top k."""
        scored = self._scored_pairs()
        best_by_idx = {}
        for i, s in scored:
            if i not in best_by_idx or s > best_by_idx[i]:
                best_by_idx[i] = s
        ranked = sorted(best_by_idx.items(), key=lambda t: t[1], reverse=True)
        return [i for i, _ in ranked[:k]]

    # ---- internals ---------------------------------------------------
    def _pending_seed(self):
        if self._scores and self._scores[-1] is None:
            return self._idx[-1]
        return None

    def _scored_pairs(self):
        return [(i, s) for i, s in zip(self._idx, self._scores) if s is not None]

    def _candidate_pool(self):
        """All unobserved grid points if the grid is small enough to
        enumerate; otherwise a QMC-sampled pool (same low-discrepancy
        machinery `select_starts` uses for its scatter-search pool)."""
        total = 1
        for d in self.space.dims:
            total *= d.n
            if total > _CANDIDATE_ENUM_CAP:
                break
        if total <= _CANDIDATE_ENUM_CAP:
            ranges = [range(d.n) for d in self.space.dims]
            return [p for p in itertools.product(*ranges)
                   if p not in self._observed_set]

        pool = set()
        active = self.space.active
        seed = int(self.rng.randint(np.iinfo(np.int32).max))
        sampler = qmc.LatinHypercube(d=max(1, len(active)), seed=seed)
        pool_u = sampler.random(_CANDIDATE_ENUM_CAP)
        for row in pool_u:
            idx = list(self.space.midpoint())
            for j, dim_i in enumerate(active):
                n = self.space.dims[dim_i].n
                idx[dim_i] = min(n - 1, int(row[j] * n))
            p = tuple(idx)
            if p not in self._observed_set:
                pool.add(p)
        return sorted(pool)

    def _random_pick(self, candidates, size):
        """Deterministic-given-seed pick(s) from `candidates`, this
        proposer's own rng (used for the cold-start phase)."""
        sel = self.rng.choice(len(candidates), size=min(size, len(candidates)),
                              replace=False)
        return [candidates[i] for i in sorted(np.atleast_1d(sel))]

    def _gp_ei_pick(self, scored, candidates):
        idx_train = [i for i, _ in scored]
        y_train = np.asarray([s for _, s in scored], dtype=float)
        X_train = _featurize(self.space, idx_train)

        gp_seed = int(self.rng.randint(np.iinfo(np.int32).max))
        # length_scale_bounds floored at 0.05 (~one grid step in _featurize's
        # normalized coordinates, see Dimension.coord): sklearn's default
        # lower bound of 1e-5 lets the optimizer fit a length scale far
        # shorter than the grid spacing on rugged objectives, collapsing the
        # kernel to near-delta (every point its own island) and EI to flat -
        # verified via GP_Validation_vs_Optuna.ipynb's multimodal case, which
        # degenerated into a deterministic lexicographic tie-break walk
        # before this floor was added.
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale=0.5,
                          length_scale_bounds=(0.05, 10.0)),
            normalize_y=True, alpha=1e-6,
            n_restarts_optimizer=2, random_state=gp_seed,
        )
        gp.fit(X_train, y_train)

        X_cand = _featurize(self.space, candidates)
        mu, sigma = gp.predict(X_cand, return_std=True)

        f_best = float(np.max(y_train))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mu - f_best - self.xi) / sigma
            ei = (mu - f_best - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.where(sigma > 0, ei, 0.0)

        best_ei = float(np.max(ei))
        tied = [candidates[i] for i in range(len(candidates)) if ei[i] == best_ei]
        return min(tied)  # deterministic tie-break: lexicographically smallest
