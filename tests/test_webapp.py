import sys
from c4f.webapp import _build_cmd


def test_build_cmd_binary_uses_error_col():
    c = _build_cmd("d.csv", "/out", "age", "race,sex", "binary", "err",
                   "", "", "per_class", "kmeans", "euclidean", 4, 42, "")
    assert "--experiment" in c and "--error_col" in c and "err" in c
    assert "--y_true_col" not in c


def test_build_cmd_multiclass_uses_y_cols_and_option():
    c = _build_cmd("d.csv", "/out", "", "race", "multiclass", "",
                   "yt", "yp", "classwise", "kmeans", "gower", 5, 1, "SPECIAL")
    assert "--y_true_col" in c and "--error_multiclass_option" in c and "classwise" in c
    assert "--error_col" not in c
    assert c[c.index("--experiment") + 1] == "SPECIAL"  # exclude groups passed


def test_build_cmd_kmeans_emits_max_iter_not_eps():
    c = _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "", "", "per_class",
                   "kmeans", "euclidean", 4, 42, "", eps=0.7, min_samples=8, max_iter=250)
    assert "--max_iter" in c and c[c.index("--max_iter") + 1] == "250"
    assert "--eps" not in c and "--min_samples" not in c


def test_build_cmd_dbscan_emits_eps_and_min_samples_not_max_iter():
    c = _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "", "", "per_class",
                   "dbscan", "euclidean", 4, 42, "", eps=0.7, min_samples=8, max_iter=250)
    assert "--eps" in c and c[c.index("--eps") + 1] == "0.7"
    assert "--min_samples" in c and c[c.index("--min_samples") + 1] == "8"
    assert "--max_iter" not in c


def test_build_cmd_hdbscan_emits_min_samples_only():
    c = _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "", "", "per_class",
                   "hdbscan", "euclidean", 4, 42, "", eps=0.7, min_samples=8, max_iter=250)
    assert "--min_samples" in c
    assert "--eps" not in c and "--max_iter" not in c


def test_build_cmd_always_emits_multicat_flags():
    c = _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "", "", "per_class",
                   "kmeans", "euclidean", 4, 42, "",
                   multicat_sig="fisher_ova", multicat_table_option="salient")
    assert c[c.index("--multicat_sig") + 1] == "fisher_ova"
    assert c[c.index("--multicat_table_option") + 1] == "salient"


def test_build_cmd_emits_column_roles_and_error_label():
    c = _build_cmd("d.csv", "/out", "age", "race,region", "binary", "err", "", "", "per_class",
                   "kmeans", "euclidean", 4, 42, "",
                   continuous_sensitive="age", proxy="zip", special="shap",
                   error_label="FN Rate")
    assert c[c.index("--continuous_sensitive_cols") + 1] == "age"
    assert c[c.index("--proxy_cols") + 1] == "zip"
    assert c[c.index("--special_cols") + 1] == "shap"
    assert c[c.index("--error_label") + 1] == "FN Rate"


def test_build_cmd_omits_empty_column_roles():
    c = _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "", "", "per_class",
                   "kmeans", "euclidean", 4, 42, "")
    assert "--continuous_sensitive_cols" not in c and "--proxy_cols" not in c
    assert "--special_cols" not in c and "--error_label" not in c
