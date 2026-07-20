"""Priority-ordering and zone-splitter tests."""

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

from bayes_halving_search_cv._sampling import (
    ZoneSplitter,
    expanding_order,
    random_order,
    stratified_order,
)


def test_expanding_is_chronological():
    assert np.array_equal(expanding_order(10), np.arange(10))


def test_random_is_seeded_permutation():
    rng = np.random.RandomState(0)
    a = random_order(100, rng)
    b = random_order(100, np.random.RandomState(0))
    assert np.array_equal(a, b)
    assert np.array_equal(np.sort(a), np.arange(100))


def test_stratified_is_full_permutation():
    rng = np.random.RandomState(0)
    X = np.column_stack([
        np.repeat(np.arange(20), 50),           # "store id": 20 runs of 50
        rng.randint(0, 2, 1000).cumsum() % 2,   # churny binary column
    ])
    order = stratified_order(X)
    assert np.array_equal(np.sort(order), np.arange(1000))


def test_stratified_prefix_prioritizes_transitions():
    # 5 runs of 200 identical rows: boundaries must appear early
    X = np.repeat(np.arange(5), 200).reshape(-1, 1)
    order = stratified_order(X)
    boundaries = {0, 200, 400, 600, 800}
    early = set(order[:20].tolist())
    assert boundaries & early  # run starts (or their midpoints) come first
    # prefixes nest by construction (single ordering)
    assert set(order[:100]) <= set(order[:500])


def test_stratified_even_sampling_continuous_column():
    # every row differs -> falls back to Even Sampling (near-systematic),
    # still a full permutation, no crash
    X = np.arange(500, dtype=float).reshape(-1, 1)
    order = stratified_order(X)
    assert np.array_equal(np.sort(order), np.arange(500))
    # any prefix spans the timeline reasonably evenly
    prefix = np.sort(order[:50])
    gaps = np.diff(prefix)
    assert gaps.max() <= 500 / 50 * 4


def test_zone_splitter_maps_to_original_indices():
    X = np.arange(200).reshape(-1, 1).astype(float)
    y = np.arange(200).astype(float)
    subset = np.arange(0, 100)
    zs = ZoneSplitter(KFold(n_splits=4), subset)
    for train, test in zs.split(X, y):
        assert set(train) <= set(subset.tolist())
        assert set(test) <= set(subset.tolist())
        assert len(set(train) & set(test)) == 0
    assert zs.get_n_splits() == 4


def test_zone_splitter_preserves_time_order():
    X = np.arange(300).reshape(-1, 1).astype(float)
    y = np.arange(300).astype(float)
    subset = np.random.RandomState(0).permutation(300)[:120]
    zs = ZoneSplitter(TimeSeriesSplit(n_splits=3), subset)
    for train, test in zs.split(X, y):
        # every training row precedes every test row in ORIGINAL time
        assert train.max() < test.min()
