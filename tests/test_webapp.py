import sys
from c4fairness.webapp import _build_cmd


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


def _base(**kw):
    return _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "", "", "per_class",
                      "kmeans", "euclidean", 4, 42, "", **kw)


def test_build_cmd_fixed_k_by_default():
    c = _base()
    assert "--n_clusters" in c and "--n_min" not in c and "--n_max" not in c


def test_build_cmd_k_search_replaces_fixed_k():
    c = _base(n_min=2, n_max=8)
    assert c[c.index("--n_min") + 1] == "2" and c[c.index("--n_max") + 1] == "8"
    assert "--n_clusters" not in c


def test_build_cmd_scoring_and_weights():
    c = _base(scoring="chi2_error", composite_weights="silhouette:0.2,error:0.6,fairness:0.2",
              feature_weights="sensitive:2.0", min_datapoints=25)
    assert c[c.index("--scoring") + 1] == "chi2_error"
    assert c[c.index("--composite_weights") + 1] == "silhouette:0.2,error:0.6,fairness:0.2"
    assert c[c.index("--feature_weights") + 1] == "sensitive:2.0"
    assert c[c.index("--min_datapoints") + 1] == "25"


def test_build_cmd_subset_seeds_projection_categorical():
    c = _base(subset="FP", seeds="1,2,3", projection="pca,tsne", categorical_cols="region")
    assert c[c.index("--subset") + 1] == "FP"
    assert c[c.index("--seeds") + 1] == "1,2,3"
    assert c[c.index("--projection") + 1] == "pca,tsne"
    assert c[c.index("--categorical_cols") + 1] == "region"


def test_build_cmd_boolean_flags():
    c = _base(separability_check=True, no_standardize=True)
    assert "--separability_check" in c and "--no_standardize" in c
    c2 = _base(separability_check=False, no_standardize=False)
    assert "--separability_check" not in c2 and "--no_standardize" not in c2


def test_build_cmd_subset_none_omitted():
    assert "--subset" not in _base(subset="none")


def test_build_cmd_binary_rate_uses_y_cols_and_metric():
    c = _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "yt", "yp", "per_class",
                   "kmeans", "euclidean", 4, 42, "", binary_error_metric="fpr")
    assert c[c.index("--binary_error_metric") + 1] == "fpr"
    assert "--y_true_col" in c and "--y_pred_col" in c
    assert "--error_col" not in c


def test_build_cmd_binary_raw_still_uses_error_col():
    c = _base()  # binary_error_metric defaults to 'raw'
    assert "--error_col" in c and "--binary_error_metric" not in c


def test_build_cmd_sensitive_gap_test():
    assert _base(sensitive_gap_test="fisher")[
        _base(sensitive_gap_test="fisher").index("--sensitive_gap_test") + 1] == "fisher"
    assert "--sensitive_gap_test" in _base()  # always emitted (default chi2)


def test_build_cmd_sensitive_labels():
    c = _base(sensitive_labels="race:Ethnicity,sex_Male:Male")
    assert c[c.index("--sensitive_labels") + 1] == "race:Ethnicity,sex_Male:Male"
    assert "--sensitive_labels" not in _base()  # omitted when empty
