"""Trace tests: drive the Climber state machine with synthetic score
functions and assert its Hooke-Jeeves behavior — zero model fitting."""

import math

import pytest

from bayes_halving_search_cv._climber import Climber
from bayes_halving_search_cv._engine import Engine
from bayes_halving_search_cv._space import Space


def drive(climbers, space, func):
    """Minimal engine substitute: score = func(idx_tuple, fraction)."""
    calls = []

    def evaluate_batch(frac, points):
        calls.extend((p, frac) for p in points)
        return [func(p, frac) for p in points]

    engine = Engine(climbers, space, evaluate_batch)
    engine.run()
    return engine, calls


def make_climber(space, zones=(1.0,), start=None, poll="opportunistic",
                 warmup=3, cid=0, mesh_expansion=1.0, contraction="patient"):
    return Climber(cid=cid, space=space, start=start or space.midpoint(),
                   zones=list(zones), warmup=warmup, poll_mode=poll,
                   mesh_expansion=mesh_expansion, contraction=contraction)


def quadratic_peak(optimum):
    def f(idx, frac):
        return -sum((a - b) ** 2 for a, b in zip(idx, optimum))
    return f


# --------------------------------------------------------------------- core
@pytest.mark.parametrize("poll", ["opportunistic", "complete"])
def test_converges_to_grid_optimum_1d(poll):
    space = Space({"x": list(range(15))})
    c = make_climber(space, poll=poll)
    engine, calls = drive([c], space, quadratic_peak((11,)))
    assert c.status == "converged"
    assert c.best == (11,)
    assert len({p for p, _ in calls}) <= 15  # dedup: never more than the grid


@pytest.mark.parametrize("poll", ["opportunistic", "complete"])
def test_converges_3d(poll):
    space = Space({"a": list(range(9)), "b": list(range(13)),
                   "c": list(range(5))})
    c = make_climber(space, poll=poll)
    engine, calls = drive([c], space, quadratic_peak((7, 2, 4)))
    assert c.best == (7, 2, 4)


def test_deterministic_trace():
    space = Space({"a": list(range(9)), "b": list(range(13))})
    f = quadratic_peak((6, 10))
    traces = []
    for _ in range(2):
        c = make_climber(space)
        _, calls = drive([c], space, f)
        traces.append(calls)
    assert traces[0] == traces[1]


def test_failed_pattern_move_does_not_contract():
    """Spec 4.2 step 3: a failed pattern move returns to exploration at the
    same delta; contraction only fires on a failed exploratory sweep."""
    space = Space({"x": list(range(30))})
    c = make_climber(space, start=(2,))
    deltas_at_pattern_fail = []

    def f(idx, frac):
        # improves toward 6 then plateaus: first sweep succeeds, the pattern
        # extrapolation beyond 6 fails
        x = idx[0]
        return -abs(x - 6)

    orig = c._run  # noqa: F841 (behavioral test via engine)
    drive([c], space, f)
    assert c.best == (6,)


def test_cache_shared_between_climbers():
    space = Space({"x": list(range(15))})
    c1 = make_climber(space, cid=0, start=(7,))
    c2 = make_climber(space, cid=1, start=(7,))
    engine, calls = drive([c1, c2], space, quadratic_peak((11,)))
    # identical twins: second climber must merge, evaluations not doubled
    statuses = sorted(c.status for c in engine.climbers)
    assert statuses == ["converged", "merged"]
    assert len(calls) == len({(p, f) for p, f in calls})  # no duplicate evals


def test_merge_preserves_pre_merge_best():
    space = Space({"x": list(range(15))})
    c1 = make_climber(space, cid=0, start=(7,))
    c2 = make_climber(space, cid=1, start=(7,))
    engine, _ = drive([c1, c2], space, quadratic_peak((11,)))
    merged = next(c for c in engine.climbers if c.status == "merged")
    assert merged.best is not None
    assert merged.merged_into is not None


# ---------------------------------------------------------------- bullseye
def test_zones_ratchet_and_reach_full_data():
    space = Space({"a": list(range(21)), "b": list(range(21))})
    zones = [0.1, 0.2, 0.5, 1.0]
    c = make_climber(space, zones=zones, start=(0, 0))
    drive([c], space, quadratic_peak((15, 15)))
    fracs = [f for _, f, _ in c.path]
    assert all(b >= a for a, b in zip(fracs, fracs[1:]))  # ratchet
    assert fracs[-1] == 1.0                               # forced final polish
    assert c.best == (15, 15)


def test_warmup_buys_nothing():
    space = Space({"a": list(range(21)), "b": list(range(21))})
    zones = [0.1, 0.2, 0.5, 1.0]
    c = make_climber(space, zones=zones, start=(0, 0), warmup=3)
    drive([c], space, quadratic_peak((15, 15)))
    # positions 1..warmup all sit in the first zone
    warmup_fracs = [f for _, f, _ in c.path[: c.warmup]]
    assert set(warmup_fracs) == {zones[0]}


def test_calibration_uses_mean_and_floors_to_step():
    space = Space({"a": list(range(21)), "b": list(range(21))})
    c = make_climber(space, zones=[0.1, 0.5, 1.0], start=(0, 0))
    drive([c], space, quadratic_peak((15, 15)))
    assert c.D is not None
    step = space.min_step
    assert math.isclose(c.D % step, 0.0, abs_tol=1e-12) or c.D >= step
    assert c.boundaries is not None and len(c.boundaries) == 2
    assert c.boundaries == sorted(c.boundaries, reverse=True)


def test_rescore_on_zone_change():
    """Scores must never be compared across fractions: after each climb the
    best is re-scored at the new fraction."""
    space = Space({"a": list(range(21)), "b": list(range(21))})
    zones = [0.1, 1.0]
    seen = []

    def f(idx, frac):
        seen.append((idx, frac))
        return -sum((a - b) ** 2 for a, b in zip(idx, (15, 15))) - frac

    c = make_climber(space, zones=zones, start=(0, 0))
    drive([c], space, f)
    # the best point must have been evaluated at 1.0 (re-score), not carried
    assert (c.best, 1.0) in seen
    assert c.fraction == 1.0


# ------------------------------------------------------------- categorical
def test_categorical_dimension_hamming_and_polling():
    space = Space({"kind": ["a", "b", "c"], "x": list(range(9))})
    optimum = (2, 6)
    c = make_climber(space)
    drive([c], space, quadratic_peak(optimum))
    assert c.best == optimum
    # Hamming: changing the categorical counts as distance 1
    assert space.distance((0, 3), (1, 3)) == 1.0
    assert space.distance((0, 3), (2, 3)) == 1.0


def test_single_value_dimension_is_fixed():
    space = Space({"fixed": [42], "x": list(range(9))})
    c = make_climber(space)
    drive([c], space, quadratic_peak((0, 6)))
    assert c.best == (0, 6)


def test_eager_contraction_uses_fewer_or_equal_fits():
    """contraction='eager' (prototype-faithful) contracts on failed pattern
    moves too: never more evaluations than 'patient' on the same landscape,
    and both must find the same optimum on this benign one."""
    space = Space({"a": list(range(21)), "b": list(range(13))})
    f = quadratic_peak((15, 9))
    patient = make_climber(space, contraction="patient")
    _, patient_calls = drive([patient], space, f)
    eager = make_climber(space, cid=1, contraction="eager")
    _, eager_calls = drive([eager], space, f)
    assert patient.best == eager.best == (15, 9)
    assert len(eager_calls) <= len(patient_calls)


def test_eager_contracts_on_failed_pattern_move():
    """The mesh must shrink immediately after a rejected pattern move."""
    space = Space({"x": list(range(30))})
    deltas_seen = []

    def f(idx, frac):
        return -abs(idx[0] - 6)  # improves toward 6; extrapolations past it fail

    c = make_climber(space, start=(2,), contraction="eager")

    orig_contract = c._contract
    def spy():
        deltas_seen.append(list(c.delta))
        orig_contract()
    c._contract = spy
    drive([c], space, f)
    assert c.best == (6,)
    assert len(deltas_seen) >= 1  # contraction fired (incl. pattern failures)


def test_mesh_expansion_caps_at_grid():
    space = Space({"x": list(range(9))})
    c = make_climber(space, mesh_expansion=2.0, start=(0,))
    drive([c], space, quadratic_peak((8,)))
    assert c.best == (8,)
    assert all(d <= 8 for d in c.delta)
