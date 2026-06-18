"""
Fairness metrics for clustering analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency


def cluster_proportion(c_n, c_count):
    """Proportion of a binary group in a cluster."""
    return float(c_n) / c_count if c_count > 0 else 0.0


def size_metrics(labels):
    """Overview size summary: smallest cluster size, and smallest / largest cluster
    proportion of the non-noise population. All NaN if there are no clusters."""
    lab = np.asarray(labels)
    lab = lab[lab != -1]
    if len(lab) == 0:
        return {"min_size": np.nan, "min_prop": np.nan, "max_prop": np.nan}
    counts = np.unique(lab, return_counts=True)[1]
    total = counts.sum()
    return {
        "min_size": int(counts.min()),
        "min_prop": float(counts.min() / total),
        "max_prop": float(counts.max() / total),
    }


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


def cluster_value(values, kind, neg=None):
    """Detail F_value for one cluster: binary->proportion of positive,
    numeric->median, multicat->proportion of the modal category. The modal
    category's label is reported separately by cluster_value_cat().

    For binary, `neg` is the GLOBAL negative label (e.g. 0). Proportion positive =
    fraction not equal to `neg`, so an all-negative cluster correctly reads 0 (using
    the per-cluster max would mislabel an all-zero cluster as 100% positive). Falls
    back to the local minimum when `neg` is unspecified (standalone use)."""
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return np.nan
    if kind == "numeric":
        return float(s.median())
    if kind == "multicat":
        return float((s == s.mode().iloc[0]).mean())
    if neg is None:
        neg = s.min()
    return float((s != neg).mean())  # binary: proportion of the non-negative value


def cluster_value_cat(values, kind):
    """Detail F_cat for one cluster: the modal category label of a multicat
    feature (the category whose proportion cluster_value() reports). NaN for
    binary / numeric features (no category column) or empty input."""
    if kind != "multicat":
        return np.nan
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return np.nan
    return s.mode().iloc[0]


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


def overview_gap_cat(values, labels):
    """Winning category behind a multicat Overview gap: the category whose
    per-cluster proportion has the largest spread (the value overview_gap()
    reports). NaN if < 2 clusters or no data. Only meaningful for multicat
    features."""
    df = pd.DataFrame({"v": pd.Series(values).values, "c": np.asarray(labels)})
    df = df[df["c"] != -1].dropna(subset=["v"])
    if df["c"].nunique() < 2 or len(df) == 0:
        return np.nan
    best_cat, best_spread = np.nan, -1.0
    for cat in sorted(df["v"].unique()):
        props = _props_per_cluster(df, cat)
        spread = max(props) - min(props)
        if spread > best_spread:
            best_spread, best_cat = spread, cat
    return best_cat


def _fisher_2x2(a_pos, a_neg, b_pos, b_neg):
    """Fisher exact p-value for a 2x2 table [[a_pos, a_neg], [b_pos, b_neg]]."""
    try:
        return round(float(stats.fisher_exact([[a_pos, a_neg], [b_pos, b_neg]]).pvalue), 6)
    except ValueError:
        return np.nan


def _fisher_2x2_p(df, hi, lo, pos):
    """Fisher exact p-value for the 2x2 [pos, neg] x [cluster hi, cluster lo] table."""
    def counts(cl):
        s = df.loc[df["c"] == cl, "v"]
        p = int((s == pos).sum())
        return p, len(s) - p
    ph, nh = counts(hi)
    pl, nl = counts(lo)
    return _fisher_2x2(ph, nh, pl, nl)


def extreme_pair_gap_p(values, labels, kind):
    """Overview *_gap_sig: significance of the max gap, comparing ONLY the two
    clusters that define it (the min- and max-statistic clusters). binary -> Fisher
    exact (2x2); multicat -> chi2 over all categories for the two extreme clusters;
    numeric -> one-way ANOVA. NaN on degenerate input."""
    df = pd.DataFrame({"v": pd.Series(values).values, "c": np.asarray(labels)})
    df = df[df["c"] != -1].dropna(subset=["v"])
    clusters = sorted(df["c"].unique())
    if len(clusters) < 2:
        return np.nan

    if kind == "numeric":
        meds = {cl: df.loc[df["c"] == cl, "v"].median() for cl in clusters}
        hi, lo = max(meds, key=meds.get), min(meds, key=meds.get)
        a = df.loc[df["c"] == hi, "v"].values
        b = df.loc[df["c"] == lo, "v"].values
        if len(a) < 2 or len(b) < 2:
            return np.nan
        try:
            return round(float(stats.f_oneway(a, b).pvalue), 6)
        except (ValueError, FloatingPointError):
            return np.nan

    if kind == "binary":
        pos = df["v"].max()
        props = {cl: (df.loc[df["c"] == cl, "v"] == pos).mean() for cl in clusters}
        hi, lo = max(props, key=props.get), min(props, key=props.get)
        return _fisher_2x2_p(df, hi, lo, pos)

    # multicat: winning category = largest cross-cluster spread; test its two
    # extreme clusters with a chi2 over all categories.
    best_spread, best_pair = -1.0, None
    for cat in sorted(df["v"].unique()):
        props = {cl: (df.loc[df["c"] == cl, "v"] == cat).mean() for cl in clusters}
        hi, lo = max(props, key=props.get), min(props, key=props.get)
        spread = props[hi] - props[lo]
        if spread > best_spread:
            best_spread, best_pair = spread, (hi, lo)
    if best_pair is None:
        return np.nan
    hi, lo = best_pair
    cats = sorted(df["v"].unique())
    ch = df.loc[df["c"] == hi, "v"]
    cl_vals = df.loc[df["c"] == lo, "v"]
    table = np.array([[int((ch == k).sum()) for k in cats],
                      [int((cl_vals == k).sum()) for k in cats]])
    table = table[:, table.sum(axis=0) > 0]  # drop categories absent from both
    if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
        return np.nan
    try:
        return round(float(chi2_contingency(table).pvalue), 6)
    except ValueError:
        return np.nan


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


def onevsall_gap_cat(cluster_vals, rest_vals):
    """Winning category behind a multicat Detail gap: the category with the
    largest |cluster - rest| proportion difference (the one onevsall_gap()
    reports). NaN if either side is empty. Only meaningful for multicat
    features."""
    cs, rs = pd.Series(cluster_vals).dropna(), pd.Series(rest_vals).dropna()
    if len(cs) == 0 or len(rs) == 0:
        return np.nan
    best_cat, best_abs = np.nan, -1.0
    for cat in sorted(pd.concat([cs, rs]).unique()):
        d = abs((cs == cat).mean() - (rs == cat).mean())
        if d > best_abs:
            best_abs, best_cat = d, cat
    return best_cat


def omnibus_separability_p(values, labels, kind):
    """Overview *_sep: one omnibus p across all clusters. Categorical -> chi2
    on (categories x clusters); numeric -> one-way ANOVA (per spec). NaN on
    degenerate input (preserves current 'don't crash' behavior; 0-cell policy
    deferred)."""
    df = pd.DataFrame({"v": pd.Series(values).values, "c": np.asarray(labels)})
    df = df[df["c"] != -1].dropna(subset=["v"])
    if df["c"].nunique() < 2:
        return np.nan
    if kind == "numeric":
        groups = [g["v"].values for _, g in df.groupby("c") if len(g) > 0]
        if len(groups) < 2:
            return np.nan
        try:
            return round(float(stats.f_oneway(*groups).pvalue), 6)
        except (ValueError, FloatingPointError):
            return np.nan
    table = pd.crosstab(df["v"], df["c"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    try:
        return round(float(chi2_contingency(table).pvalue), 6)
    except ValueError:
        return np.nan


# R's fisher.test (via rpy2) handles r x c tables, which scipy's fisher_exact
# (2x2 only) cannot. rpy2 is a required dependency; the import is lazy so the rest
# of this module stays importable when only the non-Fisher paths are exercised.
_R_FISHER = None


def _r_fisher_test():
    global _R_FISHER
    if _R_FISHER is None:
        import rpy2.robjects as ro
        _R_FISHER = (ro, ro.r["fisher.test"])
    return _R_FISHER


def fisher_rxc_p(table):
    """r x c Fisher exact test p-value via R's fisher.test (rpy2). NaN if the
    table is smaller than 2x2 after construction."""
    table = np.asarray(table, dtype=int)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    ro, fisher = _r_fisher_test()
    rmat = ro.r["matrix"](ro.IntVector(table.flatten(order="C")),
                          nrow=table.shape[0], byrow=True)
    return round(float(fisher(rmat).rx2("p.value")[0]), 6)


def omnibus_error_sep_p(values, labels, error_kind):
    """Overview error 'sep.': one omnibus p across all clusters. Continuous error
    -> one-way ANOVA; binary / multi-class error -> r x c Fisher exact (R via
    rpy2) on the (error-value x cluster) contingency table. NaN on degenerate
    input."""
    df = pd.DataFrame({"v": pd.Series(values).values, "c": np.asarray(labels)})
    df = df[df["c"] != -1].dropna(subset=["v"])
    if df["c"].nunique() < 2:
        return np.nan
    if error_kind == "numeric":
        groups = [g["v"].values for _, g in df.groupby("c") if len(g) > 0]
        if len(groups) < 2:
            return np.nan
        try:
            return round(float(stats.f_oneway(*groups).pvalue), 6)
        except (ValueError, FloatingPointError):
            return np.nan
    return fisher_rxc_p(pd.crosstab(df["v"], df["c"]).values)


def error_kind_for(error_type, multiclass_option=None):
    """Metric kind used to analyze the error column, mirroring feature_kind():
    'numeric' for regression, 'multicat' for the multi-class 'per_class' / 'per_cell'
    options (a categorical error label), and 'binary' otherwise (binary
    classification, and the multi-class 'accuracy' / 'onehot' options whose error
    columns are 0/1 indicators)."""
    if error_type == "regression":
        return "numeric"
    if error_type == "multiclass" and multiclass_option in ("per_class", "per_cell"):
        return "multicat"
    return "binary"


def multiclass_error_types(y_true, y_pred, option):
    """Derive a per-row error column (or columns) from multi-class predictions.

    Returns a DataFrame so the result plugs into the existing single-error pipeline
    (one column) or the one-hot pipeline (one column per class). Options:

    - 'accuracy' : single 'error' column, 1 if misclassified else 0 (binary error).
    - 'per_class': single 'error' column, the true-class label (as str) for
      misclassified rows and 'correct' for correct rows (multicat error — *which*
      true class is being gotten wrong).
    - 'per_cell' : single 'error' column, the confusion cell "true→pred" for
      misclassified rows and 'correct' for correct rows (multicat error).
    - 'onehot'   : one binary 'error=<c>' column per true class c (sorted), 1 where
      a row of true class c was misclassified (one-vs-all error indicator).
    """
    yt = pd.Series(np.asarray(y_true)).reset_index(drop=True)
    yp = pd.Series(np.asarray(y_pred)).reset_index(drop=True)
    wrong = (yt != yp)

    if option == "accuracy":
        return pd.DataFrame({"error": wrong.astype(int)})
    if option == "per_class":
        col = np.where(wrong, yt.astype(str), "correct")
        return pd.DataFrame({"error": col})
    if option == "per_cell":
        cell = yt.astype(str) + "→" + yp.astype(str)
        col = np.where(wrong, cell, "correct")
        return pd.DataFrame({"error": col})
    if option == "onehot":
        classes = sorted(pd.unique(yt))
        return pd.DataFrame({
            f"error={c}": ((yt == c) & wrong).astype(int) for c in classes
        })
    raise ValueError(
        f"Unknown multiclass error option '{option}'. "
        "Choose from: accuracy, per_class, per_cell, onehot."
    )


def onevsall_gap_p(cluster_vals, rest_vals, kind):
    """Detail *_gap_sig: significance of the one-vs-all (cluster vs rest) gap.

    Per the spec test mapping: numeric -> Mann-Whitney U; binary -> Fisher exact
    2x2 on the positive value; multicat -> Fisher exact 2x2 on the most divergent
    category ([winning-cat, other] x [cluster, rest]), i.e. the same category
    onevsall_gap_cat() reports. NaN if either side is empty. (Binary switched from
    poisson_means_test to Fisher; multicat from r x c chi2 to a 2x2 Fisher on the
    winning category, per the redesign spec.)"""
    cs, rs = pd.Series(cluster_vals).dropna(), pd.Series(rest_vals).dropna()
    if len(cs) == 0 or len(rs) == 0:
        return np.nan
    if kind == "numeric":
        return one_vs_all_p_continuous(cs.values, rs.values)
    if kind == "binary":
        pos = pd.concat([cs, rs]).max()
        cp, rp = int((cs == pos).sum()), int((rs == pos).sum())
        return _fisher_2x2(cp, len(cs) - cp, rp, len(rs) - rp)
    best_cat, best_abs = None, -1.0  # multicat: most divergent category
    for cat in sorted(pd.concat([cs, rs]).unique()):
        d = abs((cs == cat).mean() - (rs == cat).mean())
        if d > best_abs:
            best_abs, best_cat = d, cat
    if best_cat is None:
        return np.nan
    cp, rp = int((cs == best_cat).sum()), int((rs == best_cat).sum())
    return _fisher_2x2(cp, len(cs) - cp, rp, len(rs) - rp)


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
