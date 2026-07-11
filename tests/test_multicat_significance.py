import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency
from c4f.fairness_metrics import (
    _chi2_with_zerocell, _ova_fisher_fdr, _has_r, error_sep_p,
)


def test_chi2_dense_matches_scipy():
    # No zero cells -> pure delegation to scipy chi2_contingency.
    t = np.array([[30, 10, 5], [8, 25, 7], [6, 9, 28]])
    assert _chi2_with_zerocell(t) == pytest.approx(round(float(chi2_contingency(t).pvalue), 6))


def test_chi2_zero_cells_returns_finite():
    # Zero cells used to risk NaN; Haldane +0.5 keeps the p finite and in range.
    t = np.array([[12, 0], [0, 9]])
    p = _chi2_with_zerocell(t)
    assert np.isfinite(p) and 0.0 <= p <= 1.0


def test_ova_fisher_detects_association():
    # category 0 -> cluster 0, category 1 -> cluster 1: strong separability.
    assert _ova_fisher_fdr(np.array([[20, 1], [1, 20]])) < 0.05


def test_ova_fisher_independent_not_significant():
    assert _ova_fisher_fdr(np.array([[20, 20], [20, 20]])) > 0.05


def test_has_r_returns_bool_without_raising():
    assert isinstance(_has_r(), bool)


def test_error_sep_fisher_rxc_requires_r_when_absent():
    vals = pd.Series(["a"] * 10 + ["b"] * 10)
    labels = np.array([0] * 10 + [1] * 10)
    if not _has_r():
        with pytest.raises(ValueError):
            error_sep_p(vals, labels, "multicat", sig="fisher_rxc")
    else:
        assert np.isfinite(error_sep_p(vals, labels, "multicat", sig="fisher_rxc"))


def test_error_sep_auto_falls_back_to_scipy_without_r(monkeypatch):
    import c4f.fairness_metrics as fm
    monkeypatch.setattr(fm, "_has_r", lambda: False)
    vals = pd.Series(["a"] * 10 + ["b"] * 10)
    labels = np.array([0] * 10 + [1] * 10)
    p = error_sep_p(vals, labels, "multicat", sig="auto")
    assert np.isfinite(p) and p < 0.05  # perfectly separated -> significant


def test_error_sep_numeric_still_anova(monkeypatch):
    # sig choice must not touch the numeric (ANOVA) path.
    import c4f.fairness_metrics as fm
    vals = pd.Series([1.0, 1.1, 0.9] * 5 + [8.0, 8.1, 7.9] * 5)
    labels = np.array([0] * 15 + [1] * 15)
    assert error_sep_p(vals, labels, "numeric", sig="auto") < 0.05
