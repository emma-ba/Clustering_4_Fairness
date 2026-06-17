import numpy as np
import pandas as pd
import pytest
from src.experiments import classify_column, make_recap, recap_quali_metrics


# --- classify_column: onehot per-class error columns ---

def test_classify_onehot_value():
    assert classify_column("error=High", "mispred") == ("error", "value", "mispred=High")

def test_classify_onehot_gap():
    assert classify_column("error=High_gap", "mispred") == ("error", "value", "mispred=High gap")

def test_classify_onehot_gap_sig():
    assert classify_column("error=High_gap_sig", "mispred") == ("error", "pvalue", "mispred=High gap sig.")

def test_classify_onehot_sep():
    assert classify_column("error=High_sep", "mispred") == ("error", "pvalue", "mispred=High sep.")


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
    ov = recap_quali_metrics(chi_res, results, None, sensitive_cols=["gender"],
                             error_col=None, error_type="multiclass",
                             multiclass_option="onehot", error_cols=["error=0", "error=1"])
    cols = set(ov.columns)
    for ec in ("error=0", "error=1"):
        assert {f"{ec}_sep", f"{ec}_gap", f"{ec}_gap_sig"} <= cols
    assert ov["error=0_sep"].iloc[0] == pytest.approx(0.01)
    # single-error columns absent in onehot
    assert "error_sep" not in cols and "error_gap" not in cols
