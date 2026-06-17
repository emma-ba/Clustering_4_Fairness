import numpy as np
import pandas as pd
import pytest
from src.fairness_metrics import multiclass_error_types

# Reference confusion pattern used across tests:
#   row0: 0->0 correct   row1: 0->1 error(true 0)
#   row2: 1->1 correct   row3: 1->2 error(true 1)
#   row4: 2->2 correct
Y_TRUE = [0, 0, 1, 1, 2]
Y_PRED = [0, 1, 1, 2, 2]


def test_accuracy_is_binary_incorrect_indicator():
    out = multiclass_error_types(Y_TRUE, Y_PRED, "accuracy")
    assert list(out.columns) == ["error"]
    assert out["error"].tolist() == [0, 1, 0, 1, 0]


def test_per_class_labels_true_class_for_errors_correct_otherwise():
    out = multiclass_error_types(Y_TRUE, Y_PRED, "per_class")
    assert list(out.columns) == ["error"]
    assert out["error"].tolist() == ["correct", "0", "correct", "1", "correct"]


def test_per_cell_labels_confusion_cell_for_errors():
    out = multiclass_error_types(Y_TRUE, Y_PRED, "per_cell")
    assert list(out.columns) == ["error"]
    assert out["error"].tolist() == ["correct", "0→1", "correct", "1→2", "correct"]


def test_onehot_one_binary_column_per_class_one_vs_all_error():
    out = multiclass_error_types(Y_TRUE, Y_PRED, "onehot")
    assert list(out.columns) == ["error=0", "error=1", "error=2"]
    assert out["error=0"].tolist() == [0, 1, 0, 0, 0]
    assert out["error=1"].tolist() == [0, 0, 0, 1, 0]
    assert out["error=2"].tolist() == [0, 0, 0, 0, 0]


def test_unknown_option_raises():
    with pytest.raises(ValueError):
        multiclass_error_types(Y_TRUE, Y_PRED, "nonsense")


def test_accepts_string_class_labels():
    out = multiclass_error_types(["cat", "cat", "dog"], ["cat", "dog", "dog"], "per_cell")
    assert out["error"].tolist() == ["correct", "cat→dog", "correct"]
