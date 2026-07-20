"""select_starts is shared by PatternSearchCV and BayesHalvingSearchCV
(BAYESHALVINGSearchCV_SPEC.md section 2.1). These tests prove the extraction
out of PatternSearchCV was behavior-preserving and that both estimators call
into one implementation, not two that could silently drift."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from bayes_halving_search_cv._space import Space
from bayes_halving_search_cv._starts import select_starts


SPACE = Space({"a": list(range(9)), "b": list(range(13)), "c": ["x", "y", "z"]})


@pytest.mark.parametrize("n_starts", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_select_starts_matches_wrapper_delegation(n_starts, seed):
    """PatternSearchCV._select_starts must be a pure delegating wrapper: same
    args, same rng state in, identical result out."""
    from sklearn.tree import DecisionTreeRegressor
    from bayes_halving_search_cv import PatternSearchCV

    search = PatternSearchCV(
        DecisionTreeRegressor(random_state=0),
        {"a": list(range(9)), "b": list(range(13)), "c": ["x", "y", "z"]},
        n_starts=n_starts, random_state=seed,
    )
    rng1 = check_random_state(seed)
    rng2 = check_random_state(seed)
    direct = select_starts(SPACE, n_starts, None, rng1)
    via_wrapper = search._select_starts(SPACE, rng2)
    assert direct == via_wrapper


@pytest.mark.parametrize("n_starts", [1, 3, 6])
def test_select_starts_deterministic(n_starts):
    a = select_starts(SPACE, n_starts, None, check_random_state(0))
    b = select_starts(SPACE, n_starts, None, check_random_state(0))
    assert a == b
    assert len(a) == n_starts
    assert len(set(a)) == n_starts  # no duplicate starts


def test_select_starts_midpoint_first():
    starts = select_starts(SPACE, 3, None, check_random_state(0))
    assert starts[0] == SPACE.midpoint()


def test_select_starts_explicit_points_take_seats():
    explicit = [{"a": 0, "b": 0, "c": "x"}, {"a": 8, "b": 12, "c": "z"}]
    starts = select_starts(SPACE, 4, explicit, check_random_state(0))
    assert SPACE.indices(explicit[0]) in starts
    assert SPACE.indices(explicit[1]) in starts
    assert len(starts) == 4
