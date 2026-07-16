import numpy as np
from c4fairness.scoring import make_categorical_error_scorer, make_composite_scorer


def test_categorical_error_scorer_rewards_separation():
    # error category perfectly aligned with clusters -> strong separation -> ~1
    error = np.array(["a"] * 10 + ["b"] * 10)
    labels = np.array([0] * 10 + [1] * 10)
    scorer = make_categorical_error_scorer(error)
    assert scorer(None, labels) > 0.9


def test_categorical_error_scorer_penalizes_no_separation():
    # error category independent of clusters -> weak separation -> low score
    error = np.array(["a", "b"] * 10)
    labels = np.array([0] * 10 + [1] * 10)
    scorer = make_categorical_error_scorer(error)
    assert scorer(None, labels) < 0.5


def test_categorical_error_scorer_single_cluster_is_zero():
    error = np.array(["a", "b", "a"])
    labels = np.array([0, 0, 0])
    scorer = make_categorical_error_scorer(error)
    assert scorer(None, labels) == 0.0


def test_composite_multiclass_uses_categorical_error_scorer():
    # A string (categorical) error column must not crash the composite scorer.
    error = np.array(["correct"] * 10 + ["0→1"] * 10)
    sens = np.array([0, 1] * 10)
    labels = np.array([0] * 10 + [1] * 10)
    scorer = make_composite_scorer(error_data=error, sensitive_data=sens,
                                   error_type="multiclass")
    score = scorer(np.random.RandomState(0).rand(20, 2), labels)
    assert 0.0 <= score <= 1.0
