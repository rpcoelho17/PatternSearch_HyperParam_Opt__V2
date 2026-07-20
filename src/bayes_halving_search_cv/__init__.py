"""bayes-halving-search-cv: scikit-learn hyperparameter search with bullseye
multi-fidelity data growth and scatter-search multi-start. Provides
BayesHalvingSearchCV (a from-scratch GP + Expected Improvement Bayesian
search) and PatternSearchCV (Hooke-Jeeves pattern search)."""

import logging

from ._bayes import BayesHalvingSearchCV
from ._search import PatternSearchCV
from ._space import Dimension, Space

__all__ = ["PatternSearchCV", "BayesHalvingSearchCV", "Space", "Dimension"]
__version__ = "0.1.0"

logging.getLogger("SearchCV").addHandler(logging.NullHandler())
