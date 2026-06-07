"""
Fairness metrics for clustering analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, kruskal


def cluster_proportion(c_n, c_count):
    """Proportion of a binary group in a cluster."""
    return float(c_n) / c_count if c_count > 0 else 0.0


def one_vs_all_p_binary(c_n, c_count, rest_n, rest_count):
    """
    Poisson means test p-value for a binary attribute (one cluster vs rest).

    Handles zero counts by flipping to the complement count.
    """
    if (c_n < 1) or (c_count < 1) or (rest_n < 1) or (rest_count < 1):
        res = stats.poisson_means_test(
            c_count - c_n, c_count, rest_count - rest_n, rest_count
        )
    else:
        res = stats.poisson_means_test(c_n, c_count, rest_n, rest_count)
    return round(res.pvalue, 3)


def one_vs_all_p_continuous(c_vals, rest_vals):
    """
    Mann-Whitney U p-value for a continuous attribute (one cluster vs rest).

    Returns NaN when either group is empty or the test cannot be run.
    """
    if len(c_vals) == 0 or len(rest_vals) == 0:
        return np.nan
    try:
        _, p = mannwhitneyu(c_vals, rest_vals, alternative='two-sided')
        return round(float(p), 6)
    except ValueError:
        return np.nan


def mean_diff(c_vals, rest_vals):
    """Mean difference between a cluster and all other clusters (one-vs-all)."""
    if len(c_vals) == 0 or len(rest_vals) == 0:
        return np.nan
    return float(np.mean(c_vals)) - float(np.mean(rest_vals))


def feature_kind(series, is_continuous=False):
    """'binary' | 'multicat' | 'numeric'. Type is decided by declaration +
    cardinality, NOT dtype — gower factorizes categoricals to float codes."""
    if is_continuous:
        return "numeric"
    return "binary" if pd.Series(series).nunique(dropna=True) <= 2 else "multicat"


def cluster_value(values, kind):
    """Detail F_value for one cluster: binary->proportion of positive,
    numeric->median, multicat->mode (returns the category label)."""
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return np.nan
    if kind == "numeric":
        return float(s.median())
    if kind == "multicat":
        return s.mode().iloc[0]
    return float((s == s.max()).mean())  # binary: proportion of the '1' value


def _props_per_cluster(df, cat):
    return [(df.loc[df["c"] == cl, "v"] == cat).mean() for cl in sorted(df["c"].unique())]


def overview_gap(values, labels, kind):
    """Overview F_gap: spread across clusters, always >= 0. NaN if < 2 clusters."""
    df = pd.DataFrame({"v": pd.Series(values).values, "c": np.asarray(labels)})
    df = df[df["c"] != -1].dropna(subset=["v"])
    clusters = sorted(df["c"].unique())
    if len(clusters) < 2:
        return np.nan
    if kind == "numeric":
        meds = [df.loc[df["c"] == cl, "v"].median() for cl in clusters]
        return float(max(meds) - min(meds))
    if kind == "binary":
        pos = df["v"].max()
        props = [(df.loc[df["c"] == cl, "v"] == pos).mean() for cl in clusters]
        return float(max(props) - min(props))
    worst = 0.0  # multicat: worst-category spread
    for cat in sorted(df["v"].unique()):
        props = _props_per_cluster(df, cat)
        worst = max(worst, max(props) - min(props))
    return float(worst)


def onevsall_gap(cluster_vals, rest_vals, kind):
    """Detail F_gap: signed (cluster - rest)."""
    cs, rs = pd.Series(cluster_vals).dropna(), pd.Series(rest_vals).dropna()
    if len(cs) == 0 or len(rs) == 0:
        return np.nan
    if kind == "numeric":
        return float(cs.median() - rs.median())
    if kind == "binary":
        pos = pd.concat([cs, rs]).max()
        return float((cs == pos).mean() - (rs == pos).mean())
    best_signed, best_abs = np.nan, -1.0  # multicat: most divergent category, signed
    for cat in sorted(pd.concat([cs, rs]).unique()):
        d = (cs == cat).mean() - (rs == cat).mean()
        if abs(d) > best_abs:
            best_abs, best_signed = abs(d), d
    return float(best_signed)


def omnibus_separability_p(values, labels, kind):
    """Overview *_sep: one omnibus p across all clusters. Categorical -> chi2
    on (categories x clusters); numeric -> Kruskal-Wallis. NaN on degenerate
    input (preserves current 'don't crash' behavior; 0-cell policy deferred)."""
    df = pd.DataFrame({"v": pd.Series(values).values, "c": np.asarray(labels)})
    df = df[df["c"] != -1].dropna(subset=["v"])
    if df["c"].nunique() < 2:
        return np.nan
    if kind == "numeric":
        groups = [g["v"].values for _, g in df.groupby("c") if len(g) > 0]
        if len(groups) < 2:
            return np.nan
        try:
            return round(float(kruskal(*groups).pvalue), 6)
        except ValueError:
            return np.nan
    table = pd.crosstab(df["v"], df["c"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    try:
        return round(float(chi2_contingency(table).pvalue), 6)
    except ValueError:
        return np.nan


def onevsall_categorical_p(cluster_vals, rest_vals):
    """Detail multicat F_sep: chi2 on (categories x [cluster, rest])."""
    cs, rs = pd.Series(cluster_vals).dropna(), pd.Series(rest_vals).dropna()
    cats = sorted(set(cs.unique()) | set(rs.unique()))
    if len(cats) < 2:
        return np.nan
    table = np.array([[int((cs == k).sum()) for k in cats],
                      [int((rs == k).sum()) for k in cats]])
    table = table[:, table.sum(axis=0) > 0]  # drop all-absent categories
    if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
        return np.nan
    try:
        return round(float(chi2_contingency(table).pvalue), 6)
    except ValueError:
        return np.nan
