"""Regression tests: multi-class *sensitive* features must be clustered under
Euclidean/Manhattan, the same as under Gower.

Bug: under `--distance euclidean`, multi-class sensitive columns (e.g. region)
were one-hot encoded for fairness *analysis* but their dummies were dropped from
`col_lists['sensitive']`, so they never entered the clustering feature matrix.
Under `--distance gower` the same columns stay in `sensitive_cols` as factorized
codes and ARE clustered. So "use all sensitive features for clustering" silently
held for Gower but not for Euclidean.

Intended behaviour:
  * Euclidean/Manhattan: multi-class sensitive dummies live IN sensitive_cols
    (the feature matrix) and are flagged as OHE so StandardScaler leaves them 0/1.
  * Proxy multi-class dummies stay excluded from the feature matrix (only
    sensitive features were requested for clustering).
  * The fairness-analysis sensitive list is deduplicated and never contains the
    factorized multi-class originals, under either distance.
"""
import numpy as np
import pandas as pd

from c4fairness.preprocessing import encode_categoricals
from c4fairness.cli import build_sensitive_analysis_list
from c4fairness.clustering import cluster

REGIONS = [
    "East Anglian Region", "Scotland", "London Region", "South Region",
    "North Western Region", "West Midlands Region", "South West Region",
]


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "studied_credits": rng.integers(30, 240, n).astype(float),
        "region": rng.choice(REGIONS, n),       # multi-class sensitive
        "gender": rng.integers(0, 2, n),         # binary sensitive
    })


def test_euclidean_includes_multiclass_sensitive_dummies_in_feature_matrix():
    col_lists = {"regular": ["studied_credits"], "sensitive": ["region", "gender"],
                 "proxy": [], "special": []}
    dfe, cl, catnames, mcd, ohe = encode_categoricals(
        _df(), col_lists, ["region"], "kmeans", distance="euclidean"
    )
    region_dummies = mcd["region"]
    assert region_dummies, "region should have produced multi-class dummies"
    for d in region_dummies:
        assert d in cl["sensitive"], f"{d} not in clustering feature matrix"
        assert d in ohe, f"{d} not flagged as OHE (would get StandardScaler'd)"
        assert set(dfe[d].unique()) <= {0, 1}, f"{d} should be 0/1"
    # Original factorized column is gone in the one-hot (Euclidean) path.
    assert "region" not in cl["sensitive"]
    assert "region" not in dfe.columns


def test_euclidean_includes_multiclass_proxy_dummies_in_feature_matrix():
    col_lists = {"regular": ["studied_credits"], "sensitive": ["gender"],
                 "proxy": ["region"], "special": []}
    dfe, cl, catnames, mcd, ohe = encode_categoricals(
        _df(), col_lists, ["region"], "kmeans", distance="euclidean"
    )
    for d in mcd["region"]:
        assert d in cl["proxy"], f"{d} should be clustered as a proxy feature"
        assert d in ohe, f"{d} not flagged as OHE (would get StandardScaler'd)"
    assert "region" not in cl["proxy"]
    assert "region" not in dfe.columns
    # A proxy multi-class column must NOT leak into the *sensitive* analysis list.
    analysis = build_sensitive_analysis_list(cl["sensitive"], mcd, ["gender"])
    assert analysis == ["gender"]


def test_analysis_list_dedup_under_euclidean():
    col_lists = {"regular": [], "sensitive": ["region", "gender"],
                 "proxy": [], "special": []}
    dfe, cl, catnames, mcd, ohe = encode_categoricals(
        _df(), col_lists, ["region"], "kmeans", distance="euclidean"
    )
    analysis = build_sensitive_analysis_list(cl["sensitive"], mcd, ["region", "gender"])
    assert len(analysis) == len(set(analysis)), "analysis list has duplicates"
    for d in mcd["region"]:
        assert analysis.count(d) == 1
    assert "gender" in analysis
    assert "region" not in analysis


def test_cluster_on_all_ohe_features_does_not_crash():
    # A condition clustering only on sensitive OHE dummies leaves an empty numeric
    # mask; standardization must be skipped, not crash StandardScaler.
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "race_A": rng.integers(0, 2, 200),
        "race_B": rng.integers(0, 2, 200),
        "sex_Male": rng.integers(0, 2, 200),
    })
    res = cluster(features=X, algorithm="kmeans", distance="euclidean",
                  n_clusters=3, standardize=True, ohe_features=[0, 1, 2])
    assert res.n_clusters >= 1
    # Inputs must remain unscaled 0/1 (nothing to standardize).
    assert set(np.unique(res.feature_matrix)) <= {0.0, 1.0}


def test_analysis_list_gower_drops_factorized_original():
    col_lists = {"regular": [], "sensitive": ["region", "gender"],
                 "proxy": [], "special": []}
    dfe, cl, catnames, mcd, ohe = encode_categoricals(
        _df(), col_lists, ["region"], "kmedoids", distance="gower"
    )
    # Gower keeps the factorized original in the feature matrix...
    assert "region" in cl["sensitive"]
    analysis = build_sensitive_analysis_list(cl["sensitive"], mcd, ["region", "gender"])
    # ...but analyses the readable dummies, never the float codes.
    assert "region" not in analysis
    for d in mcd["region"]:
        assert analysis.count(d) == 1
    assert "gender" in analysis
