.. pattern-search-cv documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:notoc:

##################
pattern-search-cv
##################

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/rpcoelho17/PatternSearch_HyperParam_Opt__V2>`__ |
`Issues & Ideas <https://github.com/rpcoelho17/PatternSearch_HyperParam_Opt__V2/issues>`__

Two scikit-learn-compatible hyperparameter search estimators, sharing one
multi-fidelity "bullseye" data-growth mechanism and one scatter-search
multi-start layer:

* :class:`~pattern_search_cv.PatternSearchCV` — Hooke-Jeeves pattern search.
* :class:`~pattern_search_cv.BayesHalvingSearchCV` — a from-scratch Gaussian
  Process + Expected Improvement Bayesian search (no Optuna, no torch), on
  the exact same multi-fidelity infrastructure.

Both start on a small, representative subsample of the training data and buy
more data as their own search trajectory shows them converging on an
optimum, confirming every reported result only on full data.

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:

   quick_start
   api

Getting started
================

Information on installing the package and a first working example — see
:ref:`quick_start`.

API reference
=============

Every parameter each estimator accepts, generated from the classes' own
NumPyDoc docstrings — see :ref:`api`.
