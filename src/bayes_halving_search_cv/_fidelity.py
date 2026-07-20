"""BullseyeController: the multi-fidelity ring methodology (bullseye), extracted
from Climber so BayesHalvingSearchCV can reuse the identical rules
(BAYESHALVINGSearchCV_SPEC.md section 6).

Normative source: Climber._commit_move/_calibrate/_zone_for in _climber.py.
Climber itself is NOT refactored to use this class (deliberate scope control,
see the spec) - its behavior stays pinned by its own trace tests and benchmark
history. This class is a faithful re-derivation of the same rules for a second,
independent caller.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger("bayes_halving_search_cv")


class BullseyeController:
    """Self-calibrating ring geometry for multi-fidelity data growth.

    ``warmup`` counts positions, the starting point included: with the
    default ``warmup=3`` the controller collects two displacement readings
    (position 2 and position 3) before calibrating its own ring boundaries
    from the mean of those readings. Zero-displacement observations are not
    readings. Zone assignment ratchets: once data has grown, it never shrinks.
    """

    def __init__(self, min_step, n_boundaries, warmup):
        self.min_step = min_step
        self.n_boundaries = n_boundaries
        self.warmup = warmup
        self.readings = []
        self.n_positions = 1  # the starting point counts as position 1
        self.D = None
        self.boundaries = None
        self.zone_i = 0

    def observe_improvement(self, move):
        """Record an incumbent-improving move (normalized-space displacement
        from the previous incumbent; 0.0 for the very first incumbent, which
        is not a reading). Returns the ratcheted zone index the caller should
        be at now (unchanged if no boundary was crossed, or if still
        warming up / uncalibrated).
        """
        self.n_positions += 1
        if self.D is None:
            # warm-up: collect readings; no data purchases yet
            if move > 0:
                self.readings.append(move)
            if self.n_positions >= self.warmup and len(self.readings) >= 2:
                self._calibrate()
        else:
            zone = self._zone_for(move)
            if zone > self.zone_i:
                self.zone_i = zone
        return self.zone_i

    def _calibrate(self):
        """Ring geometry from this run's own observed travel speed."""
        d_mean = sum(self.readings) / len(self.readings)
        step = self.min_step
        # floor to whole grid steps: truncation tightens rings (cheap direction)
        self.D = max(step, math.floor(d_mean / step) * step)
        n_b = self.n_boundaries
        self.boundaries = [
            max(step, self.D * (n_b - k) / n_b) for k in range(1, n_b + 1)
        ]
        logger.debug("BullseyeController calibrated: readings=%s mean=%.4f "
                    "D=%.4f boundaries=%s",
                    [round(r, 4) for r in self.readings], d_mean, self.D,
                    [round(b, 4) for b in self.boundaries])

    def _zone_for(self, move):
        """Innermost zone whose boundary the displacement falls below."""
        zone = 0
        for k, b in enumerate(self.boundaries, start=1):
            if move <= b:
                zone = k
        return zone
