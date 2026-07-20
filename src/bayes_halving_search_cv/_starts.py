"""Scatter-search start selection, shared by PatternSearchCV and
BayesHalvingSearchCV (spec: PatternSearchCV_SPEC.md section 6.1,
BAYESHALVINGSearchCV_SPEC.md section 2.1).

Extracted verbatim from PatternSearchCV._select_starts so both estimators use
one implementation, not two that could silently drift.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def select_starts(space, n_starts, start_points, rng):
    """MATLAB MultiStart-style scatter search: explicit ``start_points`` take
    seats first, then the grid midpoint, then a QMC candidate pool filled by
    greedy maximin selection. Returns a list of index tuples, length
    ``<= n_starts``.
    """
    starts = []
    if start_points:
        for p in start_points:
            idx = space.indices(p)
            if idx not in starts:
                starts.append(idx)
        starts = starts[:n_starts]
    if len(starts) < n_starts:
        mid = space.midpoint()
        if mid not in starts:
            starts.append(mid)
    if len(starts) < n_starts:
        # QMC candidate pool, then greedy maximin selection
        active = space.active
        seed = rng.randint(np.iinfo(np.int32).max)
        sampler = qmc.LatinHypercube(d=max(1, len(active)), seed=seed)
        pool_u = sampler.random(10 * n_starts)
        pool = set()
        for row in pool_u:
            idx = list(space.midpoint())
            for j, dim_i in enumerate(active):
                n = space.dims[dim_i].n
                idx[dim_i] = min(n - 1, int(row[j] * n))
            pool.add(tuple(idx))
        pool -= set(starts)
        pool = sorted(pool)  # determinism
        while len(starts) < n_starts and pool:
            best_p = max(
                pool,
                key=lambda p: min(space.distance(p, s) for s in starts),
            )
            starts.append(best_p)
            pool.remove(best_p)
    return starts[:n_starts]
