import numpy as np
import pandas as pd
import pytest
from src.fairness_metrics import (
    feature_kind, cluster_value, cluster_value_cat,
    overview_gap, overview_gap_cat, onevsall_gap, onevsall_gap_cat,
    omnibus_separability_p, onevsall_categorical_p,
    size_metrics, extreme_pair_gap_p, onevsall_gap_p,
)

def test_feature_kind_binary():
    assert feature_kind(pd.Series([0, 1, 1, 0]), is_continuous=False) == "binary"

def test_feature_kind_multicat():
    assert feature_kind(pd.Series([0, 1, 2, 3, 2]), is_continuous=False) == "multicat"

def test_feature_kind_numeric_when_declared():
    assert feature_kind(pd.Series([0.0, 1.0, 2.0, 3.0]), is_continuous=True) == "numeric"
    assert feature_kind(pd.Series([0.0, 1.0, 2.0, 3.0]), is_continuous=False) == "multicat"

def test_cluster_value_binary_is_proportion_of_positive():
    assert cluster_value([0, 1, 1, 1], "binary") == pytest.approx(0.75)

def test_cluster_value_binary_all_negative_is_zero():
    # an all-negative cluster must read 0.0, not 1.0 (no positives present)
    assert cluster_value([0, 0, 0, 0], "binary") == pytest.approx(0.0)

def test_cluster_value_binary_all_positive_uses_global_neg():
    # all-positive cluster: the global negative label is supplied so it reads 1.0
    assert cluster_value([1, 1, 1], "binary", neg=0) == pytest.approx(1.0)

def test_cluster_value_binary_nonzero_coding():
    # binary coded {1,2}: 2 is positive (global negative = 1)
    assert cluster_value([1, 2, 2, 2], "binary", neg=1) == pytest.approx(0.75)

def test_cluster_value_numeric_is_median():
    assert cluster_value([1.0, 2.0, 9.0], "numeric") == pytest.approx(2.0)

def test_cluster_value_multicat_is_proportion_of_mode():
    # mode is 3, present in 2 of 4 rows -> 0.5
    assert cluster_value([3, 3, 1, 2], "multicat") == pytest.approx(0.5)

def test_cluster_value_cat_multicat_is_modal_label():
    assert cluster_value_cat([3, 3, 1, 2], "multicat") == 3

def test_cluster_value_cat_non_multicat_is_nan():
    assert np.isnan(cluster_value_cat([0, 1, 1], "binary"))
    assert np.isnan(cluster_value_cat([1.0, 2.0, 3.0], "numeric"))

def test_overview_gap_binary_spread():
    vals   = [1, 1, 0, 0, 0, 0]
    labels = [0, 0, 0, 1, 1, 1]
    assert overview_gap(vals, labels, "binary") == pytest.approx(0.6667, abs=1e-3)

def test_overview_gap_numeric_spread():
    vals   = [10, 10, 2, 2]
    labels = [0, 0, 1, 1]
    assert overview_gap(vals, labels, "numeric") == pytest.approx(8.0)

def test_overview_gap_multicat_worst_category():
    vals   = [2, 0, 2, 1, 1, 0]
    labels = [0, 0, 0, 1, 1, 1]
    assert overview_gap(vals, labels, "multicat") == pytest.approx(0.6667, abs=1e-3)

def test_overview_gap_single_cluster_is_nan():
    assert np.isnan(overview_gap([1, 0, 1], [0, 0, 0], "binary"))

def test_onevsall_gap_binary_signed():
    assert onevsall_gap([1, 1, 1], [0, 0, 1], "binary") == pytest.approx(1.0 - 1/3)

def test_onevsall_gap_numeric_signed_can_be_negative():
    assert onevsall_gap([1.0, 1.0], [5.0, 5.0], "numeric") == pytest.approx(-4.0)

def test_onevsall_gap_multicat_most_divergent_signed():
    assert onevsall_gap([0, 0, 0], [1, 1, 1], "multicat") == pytest.approx(1.0)

def test_omnibus_categorical_perfect_separation_low_p():
    vals   = [0, 0, 0, 1, 1, 1] * 5
    labels = [0, 0, 0, 1, 1, 1] * 5
    assert omnibus_separability_p(vals, labels, "binary") < 0.05

def test_omnibus_numeric_uses_anova():
    vals   = list(np.r_[np.zeros(20), np.ones(20) * 10])
    labels = [0] * 20 + [1] * 20
    assert omnibus_separability_p(vals, labels, "numeric") < 0.05

def test_omnibus_single_cluster_is_nan():
    assert np.isnan(omnibus_separability_p([0, 1, 0], [0, 0, 0], "binary"))

def test_onevsall_categorical_p_separation():
    p = onevsall_categorical_p([0, 0, 0] * 10, [1, 2, 1] * 10)
    assert p < 0.05

# --- winning-category columns (Phase C) ---

def test_overview_gap_cat_picks_largest_spread_category():
    # cat 1: 0/3 vs 2/3 spread 2/3; cat 2: 2/3 vs 0/3 spread 2/3 (tie, first wins).
    vals   = [2, 0, 2, 1, 1, 0]
    labels = [0, 0, 0, 1, 1, 1]
    assert overview_gap_cat(vals, labels) == 1

def test_overview_gap_cat_single_cluster_is_nan():
    assert np.isnan(overview_gap_cat([0, 1, 2], [0, 0, 0]))

def test_onevsall_gap_cat_most_divergent_category():
    # cat 2 differs most: 0 in cluster vs 1.0 in rest.
    assert onevsall_gap_cat([0, 0, 1], [2, 2, 2]) == 2

def test_onevsall_gap_cat_empty_side_is_nan():
    assert np.isnan(onevsall_gap_cat([], [1, 2, 3]))

# --- size metrics (Phase B) ---

def test_size_metrics_basic():
    m = size_metrics([0, 0, 0, 1, 1])
    assert m["min_size"] == 2
    assert m["min_prop"] == pytest.approx(0.4)
    assert m["max_prop"] == pytest.approx(0.6)

def test_size_metrics_ignores_noise():
    m = size_metrics([0, 0, 0, 1, 1, -1, -1])
    assert m["min_size"] == 2
    assert m["min_prop"] == pytest.approx(0.4)
    assert m["max_prop"] == pytest.approx(0.6)

def test_size_metrics_all_noise_is_nan():
    m = size_metrics([-1, -1, -1])
    assert np.isnan(m["min_size"]) and np.isnan(m["min_prop"]) and np.isnan(m["max_prop"])

# --- extreme-pair gap significance (Phase B) ---

def test_extreme_pair_binary_perfect_separation_low_p():
    vals   = [1] * 10 + [0] * 10
    labels = [0] * 10 + [1] * 10
    assert extreme_pair_gap_p(vals, labels, "binary") < 0.05

def test_extreme_pair_numeric_separation_low_p():
    vals   = list(np.r_[np.zeros(10), np.ones(10) * 10])
    labels = [0] * 10 + [1] * 10
    assert extreme_pair_gap_p(vals, labels, "numeric") < 0.05

def test_extreme_pair_single_cluster_is_nan():
    assert np.isnan(extreme_pair_gap_p([1, 0, 1], [0, 0, 0], "binary"))

# --- Detailed one-vs-all gap significance (Phase E: Fisher 2x2 / Mann-Whitney) ---

def test_onevsall_gap_p_binary_fisher_low_p():
    assert onevsall_gap_p([1] * 10, [0] * 10, "binary") < 0.05

def test_onevsall_gap_p_numeric_mannwhitney_low_p():
    assert onevsall_gap_p([0.0] * 10, [10.0] * 10, "numeric") < 0.05

def test_onevsall_gap_p_multicat_fisher_low_p():
    # category 'a' is exclusive to the cluster -> strong divergence
    assert onevsall_gap_p(["a"] * 10, ["b"] * 10, "multicat") < 0.05

def test_onevsall_gap_p_empty_side_is_nan():
    assert np.isnan(onevsall_gap_p([], [1, 0, 1], "binary"))
    assert np.isnan(onevsall_gap_p([1.0], [], "numeric"))
