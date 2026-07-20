"""Discrete search space: dimensions, index coordinates, and distances.

Every point in the space is addressed by a tuple of integer indices (one per
dimension).  All geometry (velocity readings, ring membership, scatter-search
maximin) happens in *normalized index space*: numeric dimensions are scaled to
[0, 1] so axes of different lengths weigh equally, and categorical dimensions
contribute Hamming distance (0 if unchanged, 1 if changed) because their index
order is arbitrary.
"""

from __future__ import annotations

import logging
import math
from numbers import Integral, Real

import numpy as np

logger = logging.getLogger("bayes_halving_search_cv")


def _expand_grid_spec(name, spec):
    """Turn a param_grid entry (list of values or (low, high, num)) into a list."""
    if isinstance(spec, tuple) and len(spec) == 3 and all(
        isinstance(v, (Integral, Real)) and not isinstance(v, bool) for v in spec[:2]
    ) and isinstance(spec[2], Integral):
        low, high, num = spec
        if num < 1:
            raise ValueError(f"Dimension {name!r}: tuple form needs num >= 1, got {num}")
        values = np.linspace(low, high, int(num))
        if float(low).is_integer() and float(high).is_integer():
            # integral endpoints -> keep integer dtype (deduplicate after rounding)
            values = np.unique(np.round(values).astype(int))
        return [v.item() if isinstance(v, np.generic) else v for v in values]
    values = list(spec)
    if len(values) == 0:
        raise ValueError(f"Dimension {name!r} has no values")
    return values


class Dimension:
    """One axis of the grid: sorted values, integer index addressing."""

    def __init__(self, name, spec):
        self.name = name
        values = _expand_grid_spec(name, spec)
        self.is_categorical = not all(
            isinstance(v, (Integral, Real)) and not isinstance(v, bool) for v in values
        )
        if not self.is_categorical:
            values = sorted(set(values))
        self.values = values
        self.n = len(values)
        # step size (index units); half the grid width, floor 1; 0 = fixed dim
        self.initial_delta = 0 if self.n == 1 else max(1, (self.n - 1) // 2)
        # normalized length of one index step (numeric dims only)
        self.step_norm = 0.0 if self.n == 1 else 1.0 / (self.n - 1)

    @property
    def midpoint_index(self):
        return (self.n - 1) // 2

    def clip(self, index):
        return max(0, min(self.n - 1, index))

    def coord(self, index):
        """Normalized coordinate of an index (numeric dims)."""
        return 0.0 if self.n == 1 else index / (self.n - 1)

    def index_of(self, value):
        """Exact match first; numeric dims fall back to nearest grid value."""
        for i, v in enumerate(self.values):
            if v == value or (
                isinstance(v, Real) and isinstance(value, Real)
                and not self.is_categorical and math.isclose(float(v), float(value))
            ):
                return i
        if self.is_categorical:
            raise ValueError(
                f"Value {value!r} not in categorical dimension {self.name!r}"
            )
        arr = np.asarray(self.values, dtype=float)
        return int(np.argmin(np.abs(arr - float(value))))

    def snap(self, target):
        """Nearest index to a (possibly off-grid) numeric target value."""
        if self.is_categorical:
            raise ValueError(f"Cannot snap on categorical dimension {self.name!r}")
        arr = np.asarray(self.values, dtype=float)
        return int(np.argmin(np.abs(arr - float(target))))


class Space:
    """Ordered collection of Dimensions; all point math lives here."""

    def __init__(self, param_grid):
        if not param_grid:
            raise ValueError("param_grid must contain at least one dimension")
        self.dims = [Dimension(k, v) for k, v in param_grid.items()]
        self.names = [d.name for d in self.dims]
        self.n_dims = len(self.dims)
        self.active = [i for i, d in enumerate(self.dims) if d.n > 1]
        if not self.active:
            logger.warning("All dimensions have a single value; nothing to search.")
        # smallest nonzero move in normalized space (min_step for ring flooring)
        numeric_steps = [
            d.step_norm for i, d in enumerate(self.dims)
            if i in self.active and not d.is_categorical
        ]
        cat_steps = [1.0 for i in self.active if self.dims[i].is_categorical]
        all_steps = numeric_steps + cat_steps
        self.min_step = min(all_steps) if all_steps else 1.0

    # ---- point conversions -------------------------------------------------
    def midpoint(self):
        return tuple(d.midpoint_index for d in self.dims)

    def params(self, idx):
        """Index tuple -> {name: value} dict."""
        return {d.name: d.values[i] for d, i in zip(self.dims, idx)}

    def indices(self, params):
        """{name: value} dict -> index tuple (nearest grid point)."""
        return tuple(d.index_of(params[d.name]) for d in self.dims)

    # ---- geometry ----------------------------------------------------------
    def distance(self, a, b):
        """Normalized-space distance: Euclidean numeric + Hamming categorical."""
        acc = 0.0
        for d, ia, ib in zip(self.dims, a, b):
            if d.n == 1:
                continue
            if d.is_categorical:
                acc += 0.0 if ia == ib else 1.0
            else:
                acc += (d.coord(ia) - d.coord(ib)) ** 2
        return math.sqrt(acc)

    def extrapolate(self, base, ref):
        """Pattern move target: 2*base - ref, snapped/clipped to the grid.

        Categorical dims have no direction: the pattern move keeps base's value.
        """
        out = []
        for d, ib, ir in zip(self.dims, base, ref):
            if d.n == 1 or d.is_categorical:
                out.append(ib)
            else:
                out.append(d.clip(2 * ib - ir))
        return tuple(out)

    def diagonal(self):
        """Length of the space diagonal in normalized space."""
        acc = 0.0
        for i in self.active:
            acc += 1.0  # each active dim spans [0,1] (categoricals: Hamming max 1)
        return math.sqrt(acc)
