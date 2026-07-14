"""pattern-search-cv: Hooke-Jeeves hyperparameter search for scikit-learn,
with bullseye multi-fidelity data growth and scatter-search multi-start."""

import logging

from ._search import PatternSearchCV
from ._space import Dimension, Space

__all__ = ["PatternSearchCV", "Space", "Dimension"]
__version__ = "0.1.0.dev0"

logging.getLogger("pattern_search_cv").addHandler(logging.NullHandler())
