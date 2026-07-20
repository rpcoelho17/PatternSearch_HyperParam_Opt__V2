.. _quick_start:

###############
Getting started
###############

Installation
============

From the repository root:

.. code-block:: bash

   python -m venv .venv
   .venv/Scripts/pip install -e .[test]
   .venv/Scripts/python -m pytest

``bayes_halving_search_cv`` has exactly three runtime dependencies: ``numpy``,
``scipy``, and ``scikit-learn`` — that holds for *both* estimators, including
:class:`~bayes_halving_search_cv.BayesHalvingSearchCV`'s Gaussian Process search
(built on ``sklearn.gaussian_process.GaussianProcessRegressor`` plus a
hand-rolled Expected Improvement acquisition — no Optuna, no torch).

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
``bayes_halving_search_cv`` logger. ``verbose=1`` attaches a stream handler at
``INFO``, ``verbose=2`` at ``DEBUG`` (also cascades into scikit-learn's own
native per-fold ``[CV] END ...`` printing, since ``verbose`` is passed
through to ``BaseSearchCV``).
