"""Data-zone subsampling: priority orderings and the zone CV splitter.

The dataset is never copied.  A single *priority ordering* of row indices is
computed once per fit; every data zone is a prefix of it, so zone samples are
strictly nested and each climb costs an array slice (spec 5.2).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.utils import _safe_indexing

logger = logging.getLogger("bayes_halving_search_cv")


def _bit_reversed_order(n):
    """Permutation of arange(n) such that any prefix is near-evenly spread.

    Van-der-Corput-style: sort positions by their bit-reversed rank.  Used to
    thin priority levels evenly across time instead of truncating (spec 5.2).
    """
    if n <= 1:
        return np.arange(n)
    bits = int(np.ceil(np.log2(n)))
    idx = np.arange(n, dtype=np.uint64)
    rev = np.zeros_like(idx)
    for b in range(bits):
        rev |= ((idx >> np.uint64(b)) & np.uint64(1)) << np.uint64(bits - 1 - b)
    return np.argsort(rev, kind="stable")


def expanding_order(n_rows):
    """Chronological prefix: oldest rows first (rows assumed time-ordered)."""
    return np.arange(n_rows)


def random_order(n_rows, rng):
    """Uniform permutation for i.i.d. data (leakage warning lives in the docs)."""
    return rng.permutation(n_rows)


def stratified_order(X, columns=None):
    """Transition sampling (spec 5.2): prioritize rows where the watched feature
    combination changes, alternating run-boundary / run-midpoint, novel
    combinations first, each level thinned evenly across time.

    Returns a permutation of arange(n_rows).
    """
    Xa = np.asarray(X)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if columns is not None:
        Xa = Xa[:, list(columns)]
    n = Xa.shape[0]
    if n <= 2:
        return np.arange(n)

    # --- run detection: one vectorized pass -------------------------------
    if Xa.dtype == object:
        changed = np.array([not np.array_equal(Xa[i], Xa[i - 1])
                            for i in range(1, n)])
    else:
        changed = (Xa[1:] != Xa[:-1]).any(axis=1)
    run_starts = np.concatenate(([0], np.flatnonzero(changed) + 1))
    run_ends = np.concatenate((run_starts[1:], [n]))  # exclusive
    n_runs = len(run_starts)
    logger.info("stratified sampler: %d rows, %d runs (%.1f rows/run avg)",
                n, n_runs, n / n_runs)

    # --- level 0a: first-ever occurrence of each unique combination -------
    # hash rows of the run-start values to find novel states
    starts_view = Xa[run_starts]
    if starts_view.dtype == object:
        keys = [tuple(r) for r in starts_view]
    else:
        c = np.ascontiguousarray(starts_view)
        keys = [r.tobytes() for r in c]
    seen = set()
    novel_mask = np.zeros(n_runs, dtype=bool)
    for j, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            novel_mask[j] = True
    novel_rows = run_starts[novel_mask]

    # --- level 1: boundary/midpoint alternating per run -------------------
    mids = (run_starts + run_ends - 1) // 2
    alternating = np.empty(n_runs, dtype=np.int64)
    alternating[0::2] = run_starts[0::2]   # even runs contribute their boundary
    alternating[1::2] = mids[1::2]         # odd runs contribute their midpoint
    # the complementary picks form the next level
    complement = np.empty(n_runs, dtype=np.int64)
    complement[0::2] = mids[0::2]
    complement[1::2] = run_starts[1::2]

    # --- deeper levels: recursive bisection of runs ------------------------
    # order every row inside its run by bit-reversed rank; concatenate by depth
    order = np.full(n, -1, dtype=np.int64)
    rank = 0

    def _take(rows):
        nonlocal rank
        rows = rows[order[rows] == -1]
        if len(rows) == 0:
            return
        # thin evenly across time: bit-reversed prefix ordering
        rows = rows[_bit_reversed_order(len(rows))]
        order[rows] = np.arange(rank, rank + len(rows))
        rank += len(rows)

    _take(novel_rows)
    _take(alternating)
    _take(complement)
    # remaining rows, per-run bisection depth: assign depth = bit-reversed rank
    remaining = np.flatnonzero(order == -1)
    if len(remaining) > 0:
        # depth key: position of the row inside its run under bit reversal
        run_of = np.searchsorted(run_starts, remaining, side="right") - 1
        depth_key = np.empty(len(remaining), dtype=np.int64)
        for j in np.unique(run_of):
            sel = run_of == j
            rows_j = remaining[sel]
            depth_key[sel] = _bit_reversed_order(len(rows_j)).argsort(kind="stable")
        # interleave shallow-first across runs, evenly in time inside a depth
        for depth in np.unique(depth_key):
            _take(remaining[depth_key == depth])

    result = np.argsort(order, kind="stable")
    return result


class ZoneSplitter:
    """CV splitter over a nested prefix subset of the rows.

    Wraps the user's splitter: splits are computed on the subset (preserving
    original row order, which keeps TimeSeriesSplit semantics valid) and mapped
    back to original indices.  Same n_splits as the base splitter, which is
    what BaseSearchCV.evaluate_candidates asserts.
    """

    def __init__(self, base_cv, subset):
        self.base_cv = base_cv
        self.subset = np.sort(np.asarray(subset))  # preserve time order

    def split(self, X, y=None, groups=None):
        Xs = _safe_indexing(X, self.subset)
        ys = _safe_indexing(y, self.subset) if y is not None else None
        gs = _safe_indexing(groups, self.subset) if groups is not None else None
        for train, test in self.base_cv.split(Xs, ys, gs):
            yield self.subset[train], self.subset[test]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.base_cv.get_n_splits(X, y, groups)
