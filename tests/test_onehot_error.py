import numpy as np
import pandas as pd
import pytest
from c4fairness.experiments import classify_column, make_recap, recap_quali_metrics


# --- classify_column: onehot per-class error columns ---

def test_classify_onehot_value():
    assert classify_column("error=High", "mispred") == ("error", "value", "mispred=High")

def test_classify_onehot_gap():
    assert classify_column("error=High_gap", "mispred") == ("error", "value", "mispred=High gap")

def test_classify_onehot_gap_sig():
    assert classify_column("error=High_gap_sig", "mispred") == ("error", "pvalue", "mispred=High gap sig.")

def test_classify_onehot_sep():
    assert classify_column("error=High_sep", "mispred") == ("error", "pvalue", "mispred=High sep.")

def test_classify_classwise_cat_and_gap_class():
    assert classify_column("error=A_cat", "err") == ("error", "category", "err=A cat.")
    assert classify_column("error=A_gap_cat", "err") == ("error", "category", "err=A gap cat.")
    assert classify_column("error=A_gap_class", "err") == ("error", "category", "err=A gap class")


# --- make_recap: onehot emits one binary error set per class ---

def _onehot_df():
    return pd.DataFrame({
        "clusters": [0] * 20 + [1] * 20,
        "gender":   [0, 1] * 20,
        "error=0":  [1] * 5 + [0] * 15 + [0] * 20,   # cluster0: 5/20, cluster1: 0
        "error=1":  [0] * 20 + [1] * 8 + [0] * 12,   # cluster0: 0, cluster1: 8/20
    })

def test_make_recap_onehot_emits_per_class_sets():
    df = _onehot_df()
    recap = make_recap(df, ["gender"], sensitive_cols=["gender"], error_col=None,
                       error_type="multiclass", multiclass_option="onehot",
                       error_cols=["error=0", "error=1"])
    cols = set(recap.columns)
    for ec in ("error=0", "error=1"):
        assert {ec, f"{ec}_gap", f"{ec}_gap_sig"} <= cols
    # value = per-cluster positive rate
    assert recap.loc[recap["c"] == 0, "error=0"].iloc[0] == pytest.approx(0.25)
    assert recap.loc[recap["c"] == 1, "error=1"].iloc[0] == pytest.approx(0.40)
    # a cluster with zero class errors must read 0.0, not 1.0
    assert recap.loc[recap["c"] == 1, "error=0"].iloc[0] == pytest.approx(0.0)
    assert recap.loc[recap["c"] == 0, "error=1"].iloc[0] == pytest.approx(0.0)
    # no single-error columns in onehot mode
    assert "error_gap" not in cols and "error_value" not in cols


# --- recap_quali_metrics: onehot Overview, one set per class ---

def test_overview_onehot_emits_per_class_sets():
    df = _onehot_df()
    recap = make_recap(df, ["gender"], sensitive_cols=["gender"], error_col=None,
                       error_type="multiclass", multiclass_option="onehot",
                       error_cols=["error=0", "error=1"])
    results = {"cond_name": ["c"], "cond_descr": ["c"],
               "cond_res": [df], "cond_recap": [recap]}
    chi_res = pd.DataFrame({"cond_descr": ["c"], "cond_name": ["c"],
                            "error=0_sep": [0.01], "error=1_sep": [0.2],
                            "gender_sep": [0.3]})
    ov = recap_quali_metrics(chi_res, results, sensitive_cols=["gender"],
                             error_col=None, error_type="multiclass",
                             multiclass_option="onehot", error_cols=["error=0", "error=1"])
    cols = set(ov.columns)
    for ec in ("error=0", "error=1"):
        assert {f"{ec}_sep", f"{ec}_gap", f"{ec}_gap_sig"} <= cols
    assert ov["error=0_sep"].iloc[0] == pytest.approx(0.01)
    # single-error columns absent in onehot
    assert "error_sep" not in cols and "error_gap" not in cols


# --- classwise: multi-categorical per-class error sets (TP/FN/FP/TN) ---

def _classwise_df():
    return pd.DataFrame({
        "clusters": [0] * 10 + [1] * 10,
        "gender":   [0, 1] * 10,
        "error=0":  (["TP"] * 6 + ["FN"] * 4) + (["TN"] * 10),
        "error=1":  (["TN"] * 10) + (["TP"] * 5 + ["FP"] * 5),
    })

def test_make_recap_classwise_emits_multicat_per_class_sets():
    df = _classwise_df()
    recap = make_recap(df, ["gender"], sensitive_cols=["gender"], error_col=None,
                       error_type="multiclass", multiclass_option="classwise",
                       error_cols=["error=0", "error=1"], error_cols_kind="multicat")
    cols = set(recap.columns)
    for ec in ("error=0", "error=1"):
        assert {ec, f"{ec}_cat", f"{ec}_gap", f"{ec}_gap_cat", f"{ec}_gap_sig"} <= cols
    # error=0 cluster0 modal cell 'TP' (6/10)
    assert recap.loc[recap["c"] == 0, "error=0"].iloc[0] == pytest.approx(0.6)
    assert recap.loc[recap["c"] == 0, "error=0_cat"].iloc[0] == "TP"

def test_overview_classwise_emits_gap_class():
    df = _classwise_df()
    recap = make_recap(df, ["gender"], sensitive_cols=["gender"], error_col=None,
                       error_type="multiclass", multiclass_option="classwise",
                       error_cols=["error=0", "error=1"], error_cols_kind="multicat")
    results = {"cond_name": ["c"], "cond_descr": ["c"],
               "cond_res": [df], "cond_recap": [recap]}
    chi_res = pd.DataFrame({"cond_descr": ["c"], "cond_name": ["c"],
                            "error=0_sep": [0.01], "error=1_sep": [0.2], "gender_sep": [0.3]})
    ov = recap_quali_metrics(chi_res, results, sensitive_cols=["gender"],
                             error_col=None, error_type="multiclass",
                             multiclass_option="classwise", error_cols=["error=0", "error=1"],
                             error_cols_kind="multicat")
    cols = set(ov.columns)
    for ec in ("error=0", "error=1"):
        assert {f"{ec}_sep", f"{ec}_gap", f"{ec}_gap_class", f"{ec}_gap_sig"} <= cols
