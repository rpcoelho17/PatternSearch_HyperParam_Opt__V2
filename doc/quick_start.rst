.. _quick_start:

###############
Getting started
###############

Installation
============

.. code-block:: bash

   pip install BayesHalvingSearchCV

``bayes_halving_search_cv`` has exactly three runtime dependencies: ``numpy``,
``scipy``, and ``scikit-learn`` — that holds for *both* estimators, including
:class:`~bayes_halving_search_cv.BayesHalvingSearchCV`'s Gaussian Process search
(built on ``sklearn.gaussian_process.GaussianProcessRegressor`` plus a
hand-rolled Expected Improvement acquisition — no Optuna, no torch).

For development (running the test suite from a source checkout), from the
repository root:

.. code-block:: bash

   python -m venv .venv
   .venv/Scripts/pip install -e .[test]
   .venv/Scripts/python -m pytest

Specifying ``param_grid``
==========================

``param_grid`` is a plain dict. Each value can be **either** an explicit
list of values, **or** a ``(low, high, num)`` tuple that gets expanded into
an evenly-spaced grid (like ``numpy.linspace``) — and you can freely mix
both forms in the same grid. Both estimators build their search space from
``param_grid`` in exactly the same way — this is not something you
configure per-estimator.

.. code-block:: python

   # Form 1: explicit lists - use when you know exactly which values matter
   param_grid = {
       "max_features": [2, 3, 4],
       "criterion": ["squared_error", "absolute_error"],
   }

   # Form 2: (low, high, num) tuples - use for a regular sweep across a range
   param_grid = {
       "n_estimators": (10, 260, 26),   # -> 10, 20, 30, ..., 260 (26 values)
       "max_depth": (5, 17, 13),        # -> 5, 6, 7, ..., 17 (13 values)
   }

   # Both forms together, in one grid:
   param_grid = {
       "max_features": [2, 3, 4],        # explicit list
       "n_estimators": (10, 260, 26),    # tuple spec
       "max_depth": (5, 17, 13),         # tuple spec
   }

Integer-endpoint tuples (like the two above) produce an integer grid
automatically; float endpoints produce a float grid.

Pattern search (``PatternSearchCV``)
=====================================

.. code-block:: python

   from bayes_halving_search_cv import PatternSearchCV
   from sklearn.model_selection import TimeSeriesSplit

   search = PatternSearchCV(
       estimator,
       {"max_depth": [3, 5, 7, 9, 12, 16], "min_samples_leaf": [1, 2, 4, 8]},
       cv=TimeSeriesSplit(n_splits=5),
       scoring="neg_mean_absolute_error",
       n_starts=4,               # scatter-search multi-start
       subsample="stratified",   # transition sampling for time-series data
       random_state=0,
   )
   search.fit(X, y)
   search.best_params_       # chosen ONLY from full-data evaluations
   search.local_optima_      # the map: every distinct optimum found
   search.cv_results_        # every point evaluated, and its score
   search.search_history_    # every confirmed-improving move across every start

Bayesian search (``BayesHalvingSearchCV``)
============================================

.. code-block:: python

   from bayes_halving_search_cv import BayesHalvingSearchCV
   from sklearn.model_selection import TimeSeriesSplit

   search = BayesHalvingSearchCV(
       estimator,
       {"max_depth": [3, 5, 7, 9, 12, 16], "min_samples_leaf": [1, 2, 4, 8]},
       cv=TimeSeriesSplit(n_splits=5),
       scoring="neg_mean_absolute_error",
       n_iter=25,                 # per-start budget of genuine evaluations
       subsample="stratified",
       random_state=0,
   )
   search.fit(X, y)
   search.best_params_
   search.local_optima_      # the map: every distinct optimum found
   search.cv_results_        # every point evaluated, and its score
   search.search_history_    # every trial: start index, params, fraction, score

``param_grid``, ``subsample``/``subsample_columns``, and multi-start
(``n_starts``/``start_points``) follow the *same* standard on both
estimators — see :ref:`api` for the full parameter list, or the design specs
``PatternSearchCV_SPEC.md`` / ``BAYESHALVINGSearchCV_SPEC.md`` in the repository
root for the reasoning behind each default.

Logging
=======

Both estimators log every algorithmic decision (moves, contractions, ring
calibrations and crossings, data climbs, cache statistics) to the
``SearchCV`` logger. ``verbose=1`` attaches a stream handler at
``INFO``, ``verbose=2`` at ``DEBUG`` (also cascades into scikit-learn's own
native per-fold ``[CV] END ...`` printing, since ``verbose`` is passed
through to ``BaseSearchCV``).
