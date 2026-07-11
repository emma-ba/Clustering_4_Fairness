import numpy as np
import pandas as pd
import pytest
from c4f.fairness_metrics import binary_error_rate_column, cluster_value
from c4f.experiments import make_recap

# Hand-built confusion matrix: TP=3, FP=2, FN=1, TN=4 (positive label = 1).
#   TP(x3): true=1 pred=1   FP(x2): true=0 pred=1
#   FN(x1): true=1 pred=0   TN(x4): true=0 pred=0
Y_TRUE = pd.Series([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
Y_PRED = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


def _rate(metric):
    return np.nanmean(binary_error_rate_column(Y_TRUE, Y_PRED, metric).values)


def test_fpr_is_fp_over_actual_negatives():
    # FP/(FP+TN) = 2/6
    assert _rate("fpr") == pytest.approx(2 / 6)


def test_fnr_is_fn_over_actual_positives():
    # FN/(FN+TP) = 1/4
    assert _rate("fnr") == pytest.approx(1 / 4)


def test_precision_error_is_fp_over_predicted_positives():
    # 1 - Precision = FP/(FP+TP) = 2/5
    assert _rate("precision") == pytest.approx(2 / 5)


def test_prec_neg_error_is_fn_over_predicted_negatives():
    # 1 - Precision-for-Negative = FN/(FN+TN) = 1/5
    assert _rate("prec_neg") == pytest.approx(1 / 5)


def test_denominator_rows_are_masked_not_zeroed():
    # FPR's denominator is actual-negatives only: the 4 actual-positive rows must be
    # NaN (excluded), not 0 (which would wrongly inflate the denominator to 10).
    col = binary_error_rate_column(Y_TRUE, Y_PRED, "fpr")
    assert col.notna().sum() == 6  # FP(2) + TN(4)
    assert col.isna().sum() == 4  # TP(3) + FN(1)


def test_masked_column_feeds_binary_cluster_value():
    # The whole pipeline relies on cluster_value(binary) dropping NaN and averaging
    # the {0,1} survivors -> the conditional rate. neg=0 => proportion of 1s = FPR.
    col = binary_error_rate_column(Y_TRUE, Y_PRED, "fpr")
    assert cluster_value(col, "binary", neg=0) == pytest.approx(2 / 6)


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        binary_error_rate_column(Y_TRUE, Y_PRED, "f1")


def test_make_recap_error_gap_uses_conditional_denominator():
    # A masked rate column (NaN = outside the denominator) must produce a one-vs-all
    # error_gap over the CONDITIONAL denominator, not the full row count. Here:
    #   cluster 0: FP=2, TN=2, 2 excluded -> FPR = 0.5
    #   cluster 1: FP=1, TN=3, 2 excluded -> FPR = 0.25
    # correct gap_0 = 0.5 - 0.25 = 0.25. The old n_error/rest_count shortcut would give
    # 0.5 - 1/6 = 0.333 by dividing by all 6 rest rows including the NaN ones.
    df = pd.DataFrame({
        "clusters": [0] * 6 + [1] * 6,
        "x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
        "err_rate": [1, 1, 0, 0, np.nan, np.nan] + [1, 0, 0, 0, np.nan, np.nan],
    })
    recap = make_recap(df, ["x"], sensitive_cols=[], error_col="err_rate",
                       error_type="binary")
    r0 = recap.loc[recap["c"] == 0].iloc[0]
    r1 = recap.loc[recap["c"] == 1].iloc[0]
    assert r0["error_value"] == pytest.approx(0.5)
    assert r1["error_value"] == pytest.approx(0.25)
    assert r0["error_gap"] == pytest.approx(0.25)
    assert r1["error_gap"] == pytest.approx(-0.25)
