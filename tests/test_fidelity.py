"""BullseyeController - pure unit tests, no sklearn, no Space, hand-computed
sequences (BAYESHALVINGSearchCV_SPEC.md section 9, items 3-6)."""

from bayes_halving_search_cv._fidelity import BullseyeController


def test_warmup_buys_nothing_and_uses_exactly_two_readings():
    # warmup=3: start (position 1) + 2 improvements (positions 2, 3) -> 2
    # readings collected, calibration fires exactly at the 2nd improvement.
    c = BullseyeController(min_step=0.1, n_boundaries=3, warmup=3)
    assert c.D is None
    z1 = c.observe_improvement(0.5)   # position 2, reading #1
    assert c.D is None and z1 == 0    # no data purchase during warm-up
    assert c.readings == [0.5]
    z2 = c.observe_improvement(0.3)   # position 3, reading #2 -> calibrates
    assert c.D is not None
    assert c.readings == [0.5, 0.3]
    # d_mean = 0.4, floored to 0.1 steps -> D = 0.4
    assert abs(c.D - 0.4) < 1e-9


def test_boundary_formula_descending_and_floored_at_min_step():
    c = BullseyeController(min_step=0.1, n_boundaries=3, warmup=3)
    c.observe_improvement(0.6)
    c.observe_improvement(0.6)  # d_mean=0.6 -> D=0.6
    # n_b=3: boundaries = D*(3-k)/3 for k=1,2,3 = [0.4, 0.2, 0.0->floored 0.1]
    assert c.boundaries is not None
    assert len(c.boundaries) == 3
    assert c.boundaries == sorted(c.boundaries, reverse=True)
    assert c.boundaries[-1] >= c.min_step  # innermost floored, never below min_step
    for b in c.boundaries:
        assert b >= c.min_step


def test_ratchet_never_decreases_zone():
    c = BullseyeController(min_step=0.1, n_boundaries=2, warmup=3)
    c.observe_improvement(1.0)
    c.observe_improvement(1.0)  # calibrates, D=1.0, boundaries=[0.5, 0.1]
    z_far = c.observe_improvement(2.0)   # large move -> stays at zone 0 (above b1)
    assert z_far == 0
    z_near = c.observe_improvement(0.05)  # tiny move -> deep zone
    assert z_near > 0
    # a subsequent LARGE move must not pull the zone back down
    z_back_out = c.observe_improvement(5.0)
    assert z_back_out == z_near


def test_zero_move_is_not_a_reading():
    c = BullseyeController(min_step=0.1, n_boundaries=3, warmup=3)
    c.observe_improvement(0.0)   # e.g. BO's first incumbent: no prior to move from
    assert c.readings == []
    c.observe_improvement(0.4)
    assert c.readings == [0.4]
    c.observe_improvement(0.0)   # another zero: still not recorded
    assert c.readings == [0.4]
    c.observe_improvement(0.2)
    assert c.readings == [0.4, 0.2]
    assert c.D is not None  # calibrated once 2 nonzero readings existed


def test_n_boundaries_zero_never_crosses():
    # data_zones effectively disabled (single fraction): no boundaries to cross.
    c = BullseyeController(min_step=0.1, n_boundaries=0, warmup=3)
    c.observe_improvement(1.0)
    c.observe_improvement(1.0)
    assert c.boundaries == []
    assert c.observe_improvement(0.01) == 0


def test_n_positions_counts_start_included():
    c = BullseyeController(min_step=0.1, n_boundaries=3, warmup=3)
    assert c.n_positions == 1
    c.observe_improvement(0.5)
    assert c.n_positions == 2
    c.observe_improvement(0.3)
    assert c.n_positions == 3
