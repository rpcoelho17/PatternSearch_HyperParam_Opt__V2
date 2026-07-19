.. _api:

#############
API Reference
#############

This page documents every public class in ``pattern_search_cv`` and every
parameter each one accepts. The text below is generated directly from each
class's own docstring (NumPyDoc format) — the source of truth lives in the
docstrings themselves (``src/pattern_search_cv/_search.py`` and
``src/pattern_search_cv/_bayes.py``), not here.

.. currentmodule:: pattern_search_cv

Estimators
==========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   PatternSearchCV
   BayesHalvingSearchCV

Search space
============

Both estimators above share the exact same search-space standard — see
``PatternSearchCV_SPEC.md`` and ``BAYESHALVINGSearchCV_SPEC.md`` §3.1 — built
on the two classes below.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Space
   Dimension
