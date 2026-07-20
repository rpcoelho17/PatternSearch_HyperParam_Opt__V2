"""GPProposer - pure unit tests, no sklearn estimator, no model fitting harness
around it, small Space instances only (BAYESHALVINGSearchCV_SPEC.md section 9,
items 7-12)."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from bayes_halving_search_cv._gp import GPProposer
from bayes_halving_search_cv._space import Space


SPACE = Space({"a": list(range(9)), "b": list(range(13)), "c": ["x", "y", "z"]})


def test_cold_start_returns_unobserved_and_does_not_crash():
    gp = GPProposer(SPACE, random_state=0)
    seen = set()
    for _ in range(2):
        idx = gp.suggest()
        assert idx not in seen
        seen.add(idx)
        gp.observe(idx, float(np.random.RandomState(0).rand()))
    # 0 observations already handled above (first suggest()); also check a
    # brand-new proposer with literally zero prior calls doesn't crash.
    gp2 = GPProposer(SPACE, random_state=1)
    idx0 = gp2.suggest()
    assert idx0 in [tuple(p) for p in
                    __import__("itertools").product(range(9), range(13), range(3))]


def test_seed_handling_returns_exact_pending_seed():
    gp = GPProposer(SPACE, random_state=0)
    seed_idx = (4, 6, 1)
    gp.observe(seed_idx, None)
    assert gp.suggest() == seed_idx


def test_finds_optimum_of_simple_unimodal_quadratic():
    # small enumerable quadratic space, true optimum at (5, 5)
    space = Space({"x": list(range(11)), "y": list(range(11))})
    true_opt = (5, 5)

    def score(idx):
        return -((idx[0] - 5) ** 2 + (idx[1] - 5) ** 2)

    gp = GPProposer(space, random_state=0)
    for _ in range(18):
        idx = gp.suggest()
        gp.observe(idx, score(idx))

    best_idx = gp.top_k_observed(1)[0]
    dist = abs(best_idx[0] - true_opt[0]) + abs(best_idx[1] - true_opt[1])
    assert dist <= 1


def test_top_k_observed_sorted_deduped_and_short_list():
    gp = GPProposer(SPACE, random_state=0)
    gp.observe((0, 0, 0), 1.0)
    gp.observe((1, 1, 0), 3.0)
    gp.observe((1, 1, 0), 5.0)  # same idx, better score -> replaces, not duplicates
    gp.observe((2, 2, 0), 2.0)

    top = gp.top_k_observed(2)
    assert top == [(1, 1, 0), (2, 2, 0)]

    # fewer than k observations
    gp2 = GPProposer(SPACE, random_state=0)
    gp2.observe((0, 0, 0), 1.0)
    assert gp2.top_k_observed(5) == [(0, 0, 0)]


def test_determinism_same_seed_same_suggest_sequence():
    def run(seed):
        gp = GPProposer(SPACE, random_state=seed)
        seq = []
        rng = check_random_state(0)  # deterministic score generator, shared
        for _ in range(10):
            idx = gp.suggest()
            seq.append(idx)
            gp.observe(idx, float(rng.rand()))
        return seq

    seq_a = run(7)
    seq_b = run(7)
    assert seq_a == seq_b


def test_never_reproposes_observed_point():
    gp = GPProposer(SPACE, random_state=0)
    seen = set()
    for i in range(25):
        idx = gp.suggest()
        assert idx not in seen, f"re-proposed {idx} at step {i}"
        seen.add(idx)
        gp.observe(idx, float(i % 5))
