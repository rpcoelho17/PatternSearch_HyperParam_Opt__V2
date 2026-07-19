"""End-to-end BayesHalvingSearchCV tests against real (small, fast) models
(BAYESHALVINGSearchCV_SPEC.md section 9, items 13-20 and 22)."""

import pickle

import pytest
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from pattern_search_cv import BayesHalvingSearchCV


GRID = {"max_depth": [2, 3, 4, 5, 6, 8, 10, 12],
       "min_samples_leaf": [1, 2, 4, 8, 16]}


def make_search(**kw):
    defaults = dict(cv=3, random_state=0, n_iter=10, data_zones=[0.25, 0.5, 1.0])
    defaults.update(kw)
    return BayesHalvingSearchCV(DecisionTreeRegressor(random_state=0), GRID,
                                **defaults)


@pytest.fixture(scope="module")
def data():
    return make_regression(n_samples=400, n_features=8, noise=10, random_state=0)


# ---- 13: basic fit, n_starts=1 ---------------------------------------------
def test_basic_fit_and_attributes(data):
    X, y = data
    search = make_search(n_starts=1).fit(X, y)
    assert search.best_params_["max_depth"] in GRID["max_depth"]
    assert search.best_params_["min_samples_leaf"] in GRID["min_samples_leaf"]
    assert hasattr(search, "best_estimator_")
    res = search.cv_results_
    assert "n_resources" in res
    assert res["n_resources"][search.best_index_] == max(res["n_resources"])
    seen = list(zip(map(tuple, (sorted(p.items()) for p in res["params"])),
                    res["n_resources"]))
    assert len(seen) == len(set(seen))
    assert len(search.local_optima_) == 1
    assert search.predict(X[:5]).shape == (5,)


# ---- 14: determinism --------------------------------------------------------
@pytest.mark.parametrize("n_starts", [1, 3])
def test_determinism(data, n_starts):
    X, y = data
    r1 = make_search(n_starts=n_starts).fit(X, y)
    r2 = make_search(n_starts=n_starts).fit(X, y)
    assert list(r1.cv_results_["mean_test_score"]) == list(
        r2.cv_results_["mean_test_score"])
    assert r1.local_optima_ == r2.local_optima_


# ---- 15: budget + cross-start dedup ----------------------------------------
def test_budget_bound_and_cross_start_dedup(data):
    X, y = data
    n_iter, promote_k = 8, 3
    search = make_search(n_starts=3, n_iter=n_iter, promote_k=promote_k).fit(X, y)

    per_start_rows = {}
    for h in search.search_history_:
        per_start_rows.setdefault(h["start"], []).append(h)
    n_zones = len(set(z for h in search.search_history_ for z in [h["fraction"]]))
    for start_i, rows in per_start_rows.items():
        # generous bound: n_iter genuine trials + up to promote_k rows for
        # every possible climb/polish event (at most n_zones of them)
        assert len(rows) <= n_iter + promote_k * max(1, n_zones)

    # cache dedups across starts too: no (params, n_resources) pair repeats
    res = search.cv_results_
    seen = list(zip(map(tuple, (sorted(p.items()) for p in res["params"])),
                    res["n_resources"]))
    assert len(seen) == len(set(seen))


# ---- 16: zones ratchet + final polish --------------------------------------
def test_zones_ratchet_and_final_polish_per_start(data):
    X, y = data
    search = make_search(n_starts=2, n_iter=10).fit(X, y)

    per_start_fracs = {}
    for h in search.search_history_:
        per_start_fracs.setdefault(h["start"], []).append(h["fraction"])
    for start_i, fracs in per_start_fracs.items():
        assert fracs == sorted(fracs)  # non-decreasing per start
        assert 1.0 in fracs  # mandatory final polish reached full data

    # best_* comes from the best full-data row across all starts
    res = search.cv_results_
    full = [i for i, n in enumerate(res["n_resources"]) if n == max(res["n_resources"])]
    assert search.best_index_ in full


# ---- 17: data_zones=1 (ladder off) ------------------------------------------
def test_zones_disabled(data):
    X, y = data
    search = make_search(data_zones=1).fit(X, y)
    assert set(search.cv_results_["n_resources"]) == {400}


# ---- 18: multi-start --------------------------------------------------------
def test_multi_start_local_optima_and_history(data):
    X, y = data
    search = make_search(n_starts=4).fit(X, y)
    assert 1 <= len(search.local_optima_) <= 4
    total = sum(o["n_starts_converged"] for o in search.local_optima_)
    assert total == 4
    starts_seen = {h["start"] for h in search.search_history_}
    assert starts_seen == set(range(4))


def test_start_points_take_seats(data):
    X, y = data
    search = make_search(
        n_starts=2, start_points=[{"max_depth": 2, "min_samples_leaf": 16}],
    ).fit(X, y)
    first = search.search_history_[0]
    assert first["params"] == {"max_depth": 2, "min_samples_leaf": 16}


def test_shared_cache_reduces_fits_across_identical_starts(data):
    X, y = data
    dup_point = {"max_depth": 4, "min_samples_leaf": 4}
    search = make_search(
        n_starts=2, start_points=[dup_point, dup_point],
    ).fit(X, y)
    assert search.n_cache_hits_ > 0


# ---- 19: verbose header -----------------------------------------------------
def test_verbose_header_names_metric_zones_and_starts(data, caplog):
    X, y = data
    with caplog.at_level("INFO", logger="pattern_search_cv"):
        make_search(scoring="neg_mean_absolute_error", verbose=1,
                   n_starts=2).fit(X, y)
    messages = [r.message for r in caplog.records]
    header = next(m for m in messages if "optimizing metric" in m)
    assert "neg_mean_absolute_error" in header
    assert any("n_iter=" in m and "n_starts=2" in m for m in messages)
    assert any("zones=" in m for m in messages)


def test_verbose_zero_skips_cv_summary(data, caplog):
    """The extra cross_validate pass must never run at verbose=0 - it costs
    real fits and must be strictly opt-in (mirrors PatternSearchCV's
    test_verbose_zero_skips_cv_summary)."""
    X, y = data
    with caplog.at_level("INFO", logger="pattern_search_cv"):
        make_search(verbose=0).fit(X, y)
    assert not any("Cross Validation Performance" in r.message
                  for r in caplog.records)


# ---- 20: invalid params ------------------------------------------------------
def test_invalid_params_raise(data):
    X, y = data
    with pytest.raises(ValueError, match="n_iter"):
        make_search(n_iter=0).fit(X, y)
    with pytest.raises(ValueError, match="promote_k"):
        make_search(promote_k=0).fit(X, y)
    with pytest.raises(ValueError, match="data_zones"):
        make_search(data_zones=[0.5, 0.2, 1.0]).fit(X, y)
    with pytest.raises(ValueError, match="n_starts"):
        make_search(n_starts=0).fit(X, y)


# ---- 22: pickling with multi-start + n_jobs>1 --------------------------------
def test_pickling_with_n_jobs_and_multi_start(data):
    X, y = data
    search = make_search(n_jobs=2, n_starts=2).fit(X, y)
    assert search.best_params_
    pickle.dumps(search)
