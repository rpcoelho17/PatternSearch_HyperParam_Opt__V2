"""scikit-learn estimator-contract checks (the scikit-learn-contrib gate)."""

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from pattern_search_cv import BayesHalvingSearchCV, PatternSearchCV


@parametrize_with_checks([
    PatternSearchCV(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 2, 4]},
        cv=3,
        random_state=0,
    ),
    BayesHalvingSearchCV(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 2, 4]},
        n_iter=8,
        cv=3,
        random_state=0,
    ),
])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)
