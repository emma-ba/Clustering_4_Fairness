"""End-to-end experiment-mode pipeline test, R-free.

run_batch_experiment was never covered because its omnibus error test needs
rpy2 + R. We stub only fisher_rxc_p (the one R call) with a scipy chi2, so the
REAL pipeline — argparse -> run_experiments_generic -> make_recap ->
make_chi_tests -> recap_quali_metrics -> CSV — runs with its real signatures.
This catches call/signature mismatches and column-wiring bugs (e.g. a stray
kwarg to make_chi_tests) that unit tests on individual functions miss.
"""
import glob
import os
import numpy as np
import pandas as pd
import pytest

import c4fairness.fairness_metrics as fm
import c4fairness.main as cli


@pytest.fixture
def tiny_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 90
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "g": rng.integers(0, 2, n),            # binary sensitive
        "r": rng.integers(0, 3, n),            # multi-class sensitive
        "err": rng.integers(0, 2, n),          # binary error
        "yt": rng.integers(0, 3, n),           # true class (3)
        "yp": rng.integers(0, 3, n),           # pred class (3)
    })
    p = tmp_path / "tiny.csv"
    df.to_csv(p, index=False)
    return str(p)


def _scipy_fisher_stub(table):
    from scipy.stats import chi2_contingency
    table = np.asarray(table, dtype=int)
    if table.ndim != 2 or min(table.shape) < 2:
        return float("nan")
    keep = table[:, table.sum(axis=0) > 0]
    keep = keep[keep.sum(axis=1) > 0, :]
    if min(keep.shape) < 2:
        return float("nan")
    try:
        return round(float(chi2_contingency(keep).pvalue), 6)
    except ValueError:
        return float("nan")


@pytest.mark.parametrize("extra", [
    ["--error_type", "binary", "--error_col", "err"],
    ["--error_type", "multiclass", "--y_true_col", "yt", "--y_pred_col", "yp",
     "--error_multiclass_option", "per_class"],
    ["--error_type", "multiclass", "--y_true_col", "yt", "--y_pred_col", "yp",
     "--error_multiclass_option", "classwise"],
])
def test_experiment_pipeline_runs(tiny_csv, tmp_path, monkeypatch, extra):
    monkeypatch.setattr(fm, "fisher_rxc_p", _scipy_fisher_stub)
    out = str(tmp_path / "out")
    argv = ["c4fairness", "--data_path", tiny_csv,
            "--regular_cols", "a", "--sensitive_cols", "g,r",
            "--algorithm", "kmeans", "--n_clusters", "3", "--seed", "1",
            "--experiment", "ERR", "--no_plots", "--output_dir", out] + extra
    monkeypatch.setattr("sys.argv", argv)
    cli.main()

    summ = glob.glob(os.path.join(out, "*experiment*", "results_summary.csv"))
    chi = glob.glob(os.path.join(out, "*experiment*", "chi_res.csv"))
    assert summ, "results_summary.csv not produced"
    assert chi, "chi_res.csv not produced"
    # Overview must carry the size + error-gap columns from the spec.
    s = pd.read_csv(summ[0])
    assert "silh" in s.columns and "min_size" in s.columns
    assert any(c == "error_gap" or c.endswith("_gap") for c in s.columns)


@pytest.mark.parametrize("mc_option", ["per_class", "per_cell"])
def test_experiment_err_group_multicat_error_does_not_crash(
    tiny_csv, tmp_path, monkeypatch, mc_option
):
    # ERR group INCLUDED (no exclusion): the categorical multi-class error column
    # (e.g. 'correct'/'0->1') must NOT be fed to the clustering StandardScaler as raw
    # strings. The ERR clustering feature must be a 0/1 misclassification indicator.
    monkeypatch.setattr(fm, "fisher_rxc_p", _scipy_fisher_stub)
    out = str(tmp_path / "out")
    argv = ["c4fairness", "--data_path", tiny_csv,
            "--regular_cols", "a", "--sensitive_cols", "g",
            "--algorithm", "kmeans", "--n_clusters", "3", "--seed", "1",
            "--experiment", "",  # include ALL groups, incl. ERR
            "--error_type", "multiclass", "--y_true_col", "yt", "--y_pred_col", "yp",
            "--error_multiclass_option", mc_option,
            "--no_plots", "--output_dir", out]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()  # must not raise ValueError: could not convert string to float

    summ = glob.glob(os.path.join(out, "*experiment*", "results_summary.csv"))
    assert summ, "results_summary.csv not produced"


def test_overview_error_gap_uses_the_rate_column(tiny_csv, tmp_path, monkeypatch):
    # With --binary_error_metric, the Overview's error_sep is copied from chi_res,
    # which is computed on the masked rate column. error_gap is computed here, and
    # used to read the raw 0/1 misclassification column instead, so the two sat on
    # different quantities in the same row. The Overview gap must equal the spread
    # of the per-cluster rates in that condition's Detailed table.
    monkeypatch.setattr(fm, "fisher_rxc_p", _scipy_fisher_stub)
    out = str(tmp_path / "out")
    argv = ["c4fairness", "--data_path", tiny_csv,
            "--regular_cols", "a", "--sensitive_cols", "g,r",
            "--algorithm", "kmeans", "--n_clusters", "3", "--seed", "1",
            "--experiment", "ERR", "--no_plots", "--output_dir", out,
            "--error_type", "binary", "--y_true_col", "yt", "--y_pred_col", "yp",
            "--binary_error_metric", "fpr"]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()

    run_dir = glob.glob(os.path.join(out, "*experiment*"))[0]
    overview = pd.read_csv(os.path.join(run_dir, "results_summary.csv"))
    checked = 0
    for _, row in overview.iterrows():
        detailed = os.path.join(run_dir, row["cond_name"].replace(" ", "") + ".csv")
        if not os.path.exists(detailed):
            continue
        clusters = pd.read_csv(detailed)
        clusters = clusters[clusters["c"].astype(str).str.isdigit()]
        rates = clusters["error_value"].astype(float).dropna()
        if len(rates) < 2 or pd.isna(row["error_gap"]):
            continue
        assert row["error_gap"] == pytest.approx(rates.max() - rates.min(), abs=1e-3)
        checked += 1
    assert checked, "no condition had a comparable error_gap"
