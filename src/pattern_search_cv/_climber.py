"""The Climber: a pure Hooke-Jeeves state machine with bullseye data growth.

A Climber never fits a model.  Its ``_run`` generator yields *requests* — lists
of ``(index_tuple, fraction)`` pairs — and receives the corresponding list of
mean CV scores via ``send``.  The engine owns evaluation, caching and
parallelism; the climber owns every algorithmic decision.  This makes the whole
search logic unit-testable against hand-computed traces in microseconds.

Spec: PatternSearchCV_SPEC.md sections 4 and 5.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger("pattern_search_cv")

_CONVERGENCE_STREAK = 3  # failed sweep+pattern passes at full data before stopping


def _better(score, than):
    """NaN-safe comparison; higher is better (sklearn convention)."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return False
    if than is None or (isinstance(than, float) and math.isnan(than)):
        return True
    return score > than


class Climber:
    """One independent pattern search run (single start = a swarm of one)."""

    def __init__(self, cid, space, start, zones, warmup, poll_mode,
                 mesh_expansion=1.0):
        self.id = cid
        self.space = space
        self.start = tuple(start)
        self.position = tuple(start)      # current exploration center
        self.best = tuple(start)          # confirmed best point of this run
        self.best_score = None            # score of `best` at `fraction`
        self.delta = [d.initial_delta for d in space.dims]
        self.zones = list(zones)          # ascending fractions ending at 1.0
        self.zone_i = 0
        self.warmup = warmup              # counts POSITIONS, start included
        self.poll_mode = poll_mode        # "complete" | "opportunistic"
        self.mesh_expansion = mesh_expansion
        self.n_positions = 1              # start counts as position 1
        self.readings = []                # warm-up displacement readings
        self.D = None                     # calibrated travel scale
        self.boundaries = None            # ring boundaries, descending
        self.last_move_size = 0.0
        self.pattern_ref = None           # previous base (compounding reference)
        self.path = []                    # (idx, fraction, score) confirmed bests
        self.status = "running"
        self.merged_into = None
        self._gen = self._run()
        self.pending = None               # current request awaiting scores

    # ---- engine protocol -----------------------------------------------
    @property
    def fraction(self):
        return self.zones[self.zone_i]

    def ask(self):
        """Prime or return the pending request."""
        if self.pending is None and self.status == "running":
            try:
                self.pending = next(self._gen)
            except StopIteration:
                self.status = "converged"
        return self.pending

    def tell(self, scores):
        """Feed scores for the pending request; advance to the next one."""
        self.pending = None
        try:
            self.pending = self._gen.send(list(scores))
        except StopIteration:
            self.status = "converged"
            logger.info("climber %s converged at %s score=%s",
                        self.id, self.space.params(self.best), self.best_score)

    def merge(self, other_id):
        self.status = "merged"
        self.merged_into = other_id
        self._gen.close()
        self.pending = None
        logger.info("climber %s merged into climber %s (pre-merge best=%s score=%s)",
                    self.id, other_id, self.space.params(self.best), self.best_score)

    def state_key(self):
        """Full deterministic state: identical key => identical future (spec 6.2)."""
        return (self.position, tuple(self.delta), self.zone_i, self.pattern_ref)

    # ---- the algorithm ---------------------------------------------------
    def _run(self):
        (s,) = yield [(self.position, self.fraction)]
        self.best_score = s
        self.path.append((self.best, self.fraction, s))
        logger.debug("climber %s start %s score=%.6g frac=%.3g",
                     self.id, self.position, s if s == s else float("nan"),
                     self.fraction)
        streak = 0
        while True:
            improved, new_pos, new_score = yield from self._sweep()
            if improved:
                streak = 0
                self.pattern_ref = self.position
                yield from self._commit_move(new_pos, new_score, kind="sweep")
                if self.mesh_expansion > 1.0:
                    self._expand()
                # ---- compounding pattern moves (spec 4.2 step 2) ----
                while self.pattern_ref is not None:
                    target = self.space.extrapolate(self.position, self.pattern_ref)
                    if target == self.position:
                        self.pattern_ref = None
                        break
                    (ps,) = yield [(target, self.fraction)]
                    if _better(ps, self.best_score):
                        logger.debug("climber %s pattern move %s -> %s score=%.6g",
                                     self.id, self.position, target, ps)
                        self.pattern_ref = self.position  # compound: ref = prev base
                        yield from self._commit_move(target, ps, kind="pattern")
                    else:
                        # failed pattern move: back to exploring, NO contraction
                        logger.debug("climber %s pattern move to %s failed (%.6g <= %.6g)",
                                     self.id, target, ps if ps == ps else float("nan"),
                                     self.best_score)
                        self.pattern_ref = None
            else:
                # exploration around a confirmed base failed -> contract mesh
                self.pattern_ref = None
                self._contract()
                if self._mesh_at_floor():
                    if self.zone_i < len(self.zones) - 1:
                        # converged on partial data: forced jump to 100% polish
                        yield from self._climb_to(len(self.zones) - 1,
                                                  reason="forced-final-polish")
                        streak = 0
                    else:
                        streak += 1
                        logger.debug("climber %s no-improvement streak %d/%d",
                                     self.id, streak, _CONVERGENCE_STREAK)
                        if streak >= _CONVERGENCE_STREAK:
                            return

    # ---- sweeps ----------------------------------------------------------
    def _sweep(self):
        if self.poll_mode == "complete":
            return (yield from self._sweep_complete())
        return (yield from self._sweep_opportunistic())

    def _sweep_opportunistic(self):
        """Classic HJ exploratory move: sequential probes, immediate acceptance,
        later dimensions probe from the drifted point."""
        center = list(self.position)
        current = self.best_score
        moved = False
        for i in self.space.active:
            if self.delta[i] == 0:
                continue
            for direction in (1, -1):
                ni = self.space.dims[i].clip(center[i] + direction * self.delta[i])
                if ni == center[i]:
                    continue
                cand = tuple(center[:i] + [ni] + center[i + 1:])
                (sc,) = yield [(cand, self.fraction)]
                if _better(sc, current):
                    center[i] = ni
                    current = sc
                    moved = True
                    break
        return moved, tuple(center), current

    def _sweep_complete(self):
        """Complete poll (MATLAB UseCompletePoll): all 2N probes around the fixed
        center in ONE batch, then the composite of all improving dimensions as a
        bonus candidate."""
        center = list(self.position)
        probes = []
        probe_dims = []
        for i in self.space.active:
            if self.delta[i] == 0:
                continue
            for direction in (1, -1):
                ni = self.space.dims[i].clip(center[i] + direction * self.delta[i])
                if ni == center[i]:
                    continue
                cand = tuple(center[:i] + [ni] + center[i + 1:])
                if cand not in (p for p, _ in probes):  # edge-clip duplicates
                    probes.append((cand, i))
        if not probes:
            return False, self.position, self.best_score
        scores = yield [(p, self.fraction) for p, _ in probes]
        improving = [(p, i, s) for (p, i), s in zip(probes, scores)
                     if _better(s, self.best_score)]
        if not improving:
            return False, self.position, self.best_score
        # best single-dimension move
        best_p, _, best_s = max(improving, key=lambda t: t[2])
        # composite: apply every improving dimension's best direction at once
        per_dim = {}
        for p, i, s in improving:
            if i not in per_dim or s > per_dim[i][1]:
                per_dim[i] = (p[i], s)
        if len(per_dim) > 1:
            comp = list(center)
            for i, (idx_val, _) in per_dim.items():
                comp[i] = idx_val
            comp = tuple(comp)
            if comp != best_p:
                (cs,) = yield [(comp, self.fraction)]
                logger.debug("climber %s composite move %s score=%.6g vs best probe %.6g",
                             self.id, comp, cs if cs == cs else float("nan"), best_s)
                if _better(cs, best_s):
                    return True, comp, cs
        return True, best_p, best_s

    # ---- move bookkeeping & the bullseye (spec 5.1) -----------------------
    def _commit_move(self, new_pos, new_score, kind):
        move = self.space.distance(self.best, new_pos)
        self.position = new_pos
        self.best = new_pos
        self.best_score = new_score
        self.last_move_size = move
        self.n_positions += 1
        self.path.append((self.best, self.fraction, new_score))
        logger.debug("climber %s move #%d (%s) -> %s score=%.6g d=%.4f frac=%.3g",
                     self.id, self.n_positions - 1, kind, new_pos, new_score,
                     move, self.fraction)
        if self.D is None:
            # warm-up: collect readings; no data purchases (spec 5.1)
            if move > 0:
                self.readings.append(move)
            if self.n_positions >= self.warmup and len(self.readings) >= 2:
                self._calibrate()
        else:
            zone = self._zone_for(move)
            if zone > self.zone_i:
                yield from self._climb_to(zone, reason=f"ring-crossing d={move:.4f}")

    def _calibrate(self):
        """Ring geometry from the climber's own observed travel speed."""
        d_mean = sum(self.readings) / len(self.readings)
        step = self.space.min_step
        # floor to whole grid steps: truncation tightens rings (cheap direction)
        self.D = max(step, math.floor(d_mean / step) * step)
        n_b = len(self.zones) - 1
        self.boundaries = [
            max(step, self.D * (n_b - k) / n_b) for k in range(1, n_b + 1)
        ]
        logger.info("climber %s calibrated: readings=%s mean=%.4f D=%.4f "
                    "boundaries=%s", self.id,
                    [round(r, 4) for r in self.readings], d_mean, self.D,
                    [round(b, 4) for b in self.boundaries])

    def _zone_for(self, move):
        """Innermost zone whose boundary the displacement falls below."""
        zone = 0
        for k, b in enumerate(self.boundaries, start=1):
            if move <= b:
                zone = k
        return zone

    def _climb_to(self, zone, reason):
        """Ratcheted data growth + mandatory re-score of best at new fraction."""
        old = self.fraction
        self.zone_i = zone
        logger.info("climber %s data %.3g -> %.3g (%s)",
                    self.id, old, self.fraction, reason)
        (rs,) = yield [(self.best, self.fraction)]
        self.best_score = rs  # scores are never compared across fractions
        self.path.append((self.best, self.fraction, rs))

    # ---- mesh ------------------------------------------------------------
    def _contract(self):
        before = list(self.delta)
        for i in self.space.active:
            if self.delta[i] > 1:
                self.delta[i] = max(1, int(0.5 + self.delta[i] / 2))
        if self.delta != before:
            logger.debug("climber %s contract delta %s -> %s",
                         self.id, before, self.delta)

    def _expand(self):
        before = list(self.delta)
        for i in self.space.active:
            cap = self.space.dims[i].n - 1
            self.delta[i] = min(cap, max(1, int(self.delta[i] * self.mesh_expansion)))
        if self.delta != before:
            logger.debug("climber %s expand delta %s -> %s",
                         self.id, before, self.delta)

    def _mesh_at_floor(self):
        return all(self.delta[i] <= 1 for i in self.space.active)
