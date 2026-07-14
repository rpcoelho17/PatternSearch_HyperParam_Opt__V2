"""End-to-end estimator tests against real (small, fast) models."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from pattern_search_cv import PatternSearchCV


GRID = {"max_depth": [2, 3, 4, 5, 6, 8, 10, 12],
        "min_samples_leaf": [1, 2, 4, 8, 16]}


def make_search(**kw):
    defaults = dict(cv=3, random_state=0, data_zones=[0.25, 0.5, 1.0])
    defaults.update(kw)
    return PatternSearchCV(DecisionTreeRegressor(random_state=0), GRID,
                           **defaults)


@pytest.fixture(scope="module")
def data():
    return make_regression(n_samples=400, n_features=8, noise=10,
                           random_state=0)


def test_basic_fit_and_attributes(data):
    X, y = data
    search = make_search().fit(X, y)
    assert search.best_params_["max_depth"] in GRID["max_depth"]
    assert search.best_params_["min_samples_leaf"] in GRID["min_samples_leaf"]
    assert hasattr(search, "best_estimator_")
    assert search.local_optima_
    assert search.search_history_
    # best_* only from full-data rows
    res = search.cv_results_
    assert "n_resources" in res
    assert res["n_resources"][search.best_index_] == max(res["n_resources"])
    # ledger only contains genuine fits: params+resources pairs unique
    seen = list(zip(map(tuple, (sorted(p.items()) for p in res["params"])),
                    res["n_resources"]))
    assert len(seen) == len(set(seen))
    # predictions work (refit happened)
    assert search.predict(X[:5]).shape == (5,)


def test_parallel_n_jobs_pickles_estimator(data):
    """n_jobs>1 makes sklearn pickle `self` per task: no handlers, engines
    or generators may live on the instance during fit (regression test)."""
    X, y = data
    search = make_search(n_jobs=2, verbose=1).fit(X, y)
    assert search.best_params_
    assert search.local_optima_


def test_determinism(data):
    X, y = data
    r1 = make_search(n_starts=3).fit(X, y)
    r2 = make_search(n_starts=3).fit(X, y)
    assert r1.best_params_ == r2.best_params_
    assert list(r1.cv_results_["mean_test_score"]) == list(
        r2.cv_results_["mean_test_score"])


def test_multi_start_runs_all_to_completion(data):
    X, y = data
    search = make_search(n_starts=4).fit(X, y)
    statuses = {h.get("status") for h in search.search_history_
                if "status" in h}
    assert statuses <= {"converged", "merged"}
    assert any(s == "converged" for s in statuses)
    assert len(search.local_optima_) >= 1
    total = sum(o["n_starts_converged"] for o in search.local_optima_)
    n_merged = sum(1 for h in search.search_history_
                   if h.get("status") == "merged")
    assert total + n_merged == 4


def test_zones_disabled(data):
    X, y = data
    search = make_search(data_zones=1).fit(X, y)
    assert set(search.cv_results_["n_resources"]) == {400}


def test_int_zones(data):
    X, y = data
    search = make_search(data_zones=2).fit(X, y)  # -> [0.5, 1.0]
    assert set(search.cv_results_["n_resources"]) <= {200, 400}


def test_start_points_take_seats(data):
    X, y = data
    search = make_search(
        n_starts=2,
        start_points=[{"max_depth": 2, "min_samples_leaf": 16}],
    ).fit(X, y)
    first = search.search_history_[0]
    assert first["params"] == {"max_depth": 2, "min_samples_leaf": 16}


def test_time_series_auto_uses_expanding(data):
    X, y = data
    search = make_search(cv=TimeSeriesSplit(n_splits=3)).fit(X, y)
    assert search.best_params_  # smoke: expanding path exercised


def test_refit_false(data):
    X, y = data
    search = make_search(refit=False).fit(X, y)
    assert search.best_params_
    assert not hasattr(search, "best_estimator_")


def test_clone_and_params_roundtrip():
    search = make_search(n_starts=3, warmup=4, poll="complete")
    cloned = clone(search)
    assert cloned.get_params()["warmup"] == 4
    assert cloned.get_params()["n_starts"] == 3


def test_invalid_params_raise(data):
    X, y = data
    with pytest.raises(ValueError, match="warmup"):
        make_search(warmup=2).fit(X, y)
    with pytest.raises(ValueError, match="data_zones"):
        make_search(data_zones=[0.5, 0.2, 1.0]).fit(X, y)
    with pytest.raises(ValueError, match="poll"):
        make_search(poll="bogus").fit(X, y)
    with pytest.raises(ValueError, match="mesh_expansion"):
        make_search(mesh_expansion=0.5).fit(X, y)


def test_tuple_grid_spec(data):
    X, y = data
    search = PatternSearchCV(
        DecisionTreeRegressor(random_state=0),
        {"max_depth": (2, 12, 6), "min_samples_leaf": [1, 4, 16]},
        cv=3, random_state=0, data_zones=[0.5, 1.0],
    ).fit(X, y)
    assert 2 <= search.best_params_["max_depth"] <= 12
