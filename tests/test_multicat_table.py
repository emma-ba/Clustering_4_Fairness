import numpy as np
import pandas as pd
import pytest
from c4f.cli import (
    _build_sensitive_analysis_list,
    _reconstruct_multicat,
    apply_salient_reconstruction,
)
from c4f.experiments import make_recap

MCD = {"region": ["region_mid", "region_north", "region_south"]}
ORIG = ["region", "gender"]


def test_analysis_list_onehot_keeps_dummies():
    # Euclidean layout: sensitive_cols already holds the dummies.
    sensitive = ["region_mid", "region_north", "region_south", "gender"]
    got = _build_sensitive_analysis_list(sensitive, MCD, ORIG, option="onehot")
    assert got == ["region_mid", "region_north", "region_south", "gender"]


def test_analysis_list_salient_uses_original_name():
    sensitive = ["region_mid", "region_north", "region_south", "gender"]
    got = _build_sensitive_analysis_list(sensitive, MCD, ORIG, option="salient")
    assert got == ["gender", "region"]  # dummies dropped, readable original added


def test_analysis_list_salient_gower_layout():
    # Gower layout: sensitive_cols holds the factorized original, not the dummies.
    sensitive = ["region", "gender"]
    got = _build_sensitive_analysis_list(sensitive, MCD, ORIG, option="salient")
    assert got == ["gender", "region"]


def test_reconstruct_multicat_recovers_labels():
    df = pd.DataFrame({
        "region_mid":   [0, 1, 0, 0],
        "region_north": [1, 0, 0, 1],
        "region_south": [0, 0, 1, 0],
    })
    rec = _reconstruct_multicat(df, "region", MCD["region"])
    assert list(rec) == ["north", "mid", "south", "north"]


def test_salient_make_recap_emits_multicat_columns():
    # After reconstruction the recap treats region as a single multi-categorical
    # column (winning-category), NOT one binary column per category.
    df = pd.DataFrame({
        "clusters": [0] * 4 + [1] * 4,
        "x": [0.0, 0.1, 0.2, 0.3, 5.0, 5.1, 5.2, 5.3],
        "gender": [0, 1, 0, 1, 0, 1, 0, 1],
        "err": [0, 1, 0, 1, 0, 1, 0, 1],
        "region_mid":   [1, 1, 0, 0, 0, 0, 0, 0],
        "region_north": [0, 0, 1, 1, 1, 1, 0, 0],
        "region_south": [0, 0, 0, 0, 0, 0, 1, 1],
    })
    analysis = _build_sensitive_analysis_list(
        ["region_mid", "region_north", "region_south", "gender"], MCD, ORIG, option="salient"
    )
    apply_salient_reconstruction(df, MCD, ORIG)
    recap = make_recap(df, ["x"], sensitive_cols=analysis, error_col="err",
                       error_type="binary")
    cols = set(recap.columns)
    assert "region_value" in cols and "region_cat" in cols  # multicat salient columns
    # the per-category dummy columns must NOT appear as their own feature groups
    assert "region_mid_value" not in cols and "region_north_value" not in cols
