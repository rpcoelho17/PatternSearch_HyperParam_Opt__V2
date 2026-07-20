"""The engine: climbers propose, the engine disposes (spec 6.3).

Per tick: gather every running climber's pending request, strip cache hits,
group the misses by data fraction, run ONE evaluate_candidates call per
fraction (candidates x folds parallelized by sklearn/joblib), write scores to
the shared cache, feed each climber its answers.  Batching order never affects
any climber's decision, so the search stays deterministic given random_state.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("bayes_halving_search_cv")


class Engine:
    def __init__(self, climbers, space, evaluate_batch):
        """evaluate_batch(fraction, [idx_tuple, ...]) -> list of mean scores."""
        self.climbers = climbers
        self.space = space
        self.evaluate_batch = evaluate_batch
        self.cache = {}          # (idx_tuple, zone_fraction) -> mean score
        self.n_cache_hits = 0
        self.n_evaluations = 0

    def run(self):
        for c in self.climbers:
            c.ask()
        tick = 0
        while True:
            running = [c for c in self.climbers if c.status == "running"]
            if not running:
                break
            tick += 1
            self._merge_check(running)
            running = [c for c in self.climbers if c.status == "running"]

            # ---- gather requests, strip cache hits ----------------------
            by_fraction = {}
            for c in running:
                for point, frac in (c.pending or []):
                    key = (point, frac)
                    if key not in self.cache:
                        by_fraction.setdefault(frac, []).append(point)

            # ---- one batch per fraction ---------------------------------
            for frac, points in sorted(by_fraction.items()):
                unique = list(dict.fromkeys(points))  # order-preserving dedup
                scores = self.evaluate_batch(frac, unique)
                self.n_evaluations += len(unique)
                for p, s in zip(unique, scores):
                    self.cache[(p, frac)] = s
                logger.debug("tick %d: evaluated %d points at frac=%.3g "
                             "(%d requested, %d cache-served this tick)",
                             tick, len(unique), frac, len(points),
                             len(points) - len(unique))

            # ---- answer every climber -----------------------------------
            for c in running:
                request = c.pending or []
                answers = []
                for point, frac in request:
                    key = (point, frac)
                    if key in self.cache:
                        answers.append(self.cache[key])
                    else:  # pragma: no cover - defensive
                        raise RuntimeError(f"engine failed to score {key}")
                served_from_cache = sum(
                    1 for point, frac in request
                    if (point, frac) not in
                    {(p, f) for f, ps in by_fraction.items() for p in ps}
                )
                self.n_cache_hits += served_from_cache
                c.tell(answers)

        logger.info("engine done: %d evaluations, %d cache hits, climbers: %s",
                    self.n_evaluations, self.n_cache_hits,
                    {c.id: c.status for c in self.climbers})

    def _merge_check(self, running):
        """State-match merging (spec 6.2): identical full state => identical
        future => the trailing climber merges, keeping its pre-merge best."""
        seen = {}
        for c in running:
            key = c.state_key()
            if key in seen:
                c.merge(seen[key])
            else:
                seen[key] = c.id

    # ---- results ---------------------------------------------------------
    def finished(self):
        return [c for c in self.climbers if c.status == "converged"]

    def local_optima(self):
        """Distinct converged optima, best first (spec: local_optima_)."""
        groups = {}
        for c in self.finished():
            groups.setdefault(c.best, []).append(c)
        out = []
        for point, cs in groups.items():
            score = max((c.best_score for c in cs),
                        key=lambda s: float("-inf") if s != s else s)
            out.append({
                "params": self.space.params(point),
                "score": score,
                "n_starts_converged": len(cs),
                "start_points": [self.space.params(c.start) for c in cs],
            })
        out.sort(key=lambda d: float("-inf") if d["score"] != d["score"]
                 else d["score"], reverse=True)
        return out
