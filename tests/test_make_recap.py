import numpy as np
import pandas as pd
import pytest
from src.experiments import make_recap, recap_quali_metrics

def _toy():
    df = pd.DataFrame({
        "clusters": [0] * 30 + [1] * 30,
        "gender":   [0, 1] * 15 + [1] * 30,
        "region":   ([0, 1, 2] * 10) + ([0] * 30),
        "error":    [0] * 15 + [1] * 15 + [0] * 30,
    })
    return df

def test_make_recap_columns_one_per_feature():
    df = _toy()
    recap = make_recap(df, ["gender", "region"], sensitive_cols=["gender", "region"],
                       error_col="error", error_type="binary")
    cols = set(recap.columns)
    for f in ("gender", "region"):
        assert f"{f}_value" in cols and f"{f}_gap_sig" in cols and f"{f}_gap" in cols
    assert "proportion" in cols and "silh" in cols
    assert not any(c.endswith("_entropy") for c in cols)
    assert not any("=" in c for c in cols)
    assert "region_prop" not in cols

def test_make_recap_proportion_sums_to_one():
    df = _toy()
    recap = make_recap(df, ["gender", "region"], sensitive_cols=["gender", "region"],
                       error_col="error", error_type="binary")
    assert np.isclose(recap["proportion"].sum(), 1.0)

def test_make_recap_multiclass_error_emits_category_columns():
    # 'error' is a multicat confusion-cell column (per_cell style).
    df = pd.DataFrame({
        "clusters": [0] * 20 + [1] * 20,
        "gender":   [0, 1] * 20,
        "error":    (["correct"] * 15 + ["0→1"] * 5)
                  + (["correct"] * 5 + ["1→2"] * 15),
    })
    recap = make_recap(df, ["gender"], sensitive_cols=["gender"],
                       error_col="error", error_type="multiclass",
                       multiclass_option="per_cell")
    cols = set(recap.columns)
    # Error treated as a multicat column: value (mode proportion) + winning-cat label,
    # plus one-vs-all gap / gap category / separability.
    assert {"error_value", "error_cat", "error_gap", "error_gap_cat", "error_gap_sig"} <= cols
    # cluster 0 mode is 'correct' (15/20), cluster 1 mode is '1→2' (15/20)
    assert recap.loc[recap["c"] == 0, "error_cat"].iloc[0] == "correct"
    assert recap.loc[recap["c"] == 1, "error_cat"].iloc[0] == "1→2"
    assert recap.loc[recap["c"] == 0, "error_value"].iloc[0] == pytest.approx(0.75)

def _multiclass_results():
    df = pd.DataFrame({
        "gender":   [0, 1] * 20,
        "errors":   (["correct"] * 15 + ["0→1"] * 5)
                  + (["correct"] * 5 + ["1→2"] * 15),
        "clusters": [0] * 20 + [1] * 20,
    })
    recap = make_recap(df, ["gender"], sensitive_cols=["gender"], error_col="errors",
                       error_type="multiclass", multiclass_option="per_cell")
    results = {"cond_name": ["c"], "cond_descr": ["c"],
               "cond_res": [df], "cond_recap": [recap]}
    chi_res = pd.DataFrame({"cond_descr": ["c"], "cond_name": ["c"],
                            "error_sep": [0.01], "gender_sep": [0.2]})
    return chi_res, results

def test_overview_multiclass_error_emits_gap_class():
    chi_res, results = _multiclass_results()
    ov = recap_quali_metrics(chi_res, results, None, sensitive_cols=["gender"],
                             error_col="errors", error_type="multiclass",
                             multiclass_option="per_cell")
    assert "error_gap_class" in ov.columns
    # '1→2' has the largest cross-cluster proportion spread (0.75)
    assert ov["error_gap_class"].iloc[0] == "1→2"
    assert ov["error_gap"].iloc[0] == pytest.approx(0.75, abs=1e-6)
