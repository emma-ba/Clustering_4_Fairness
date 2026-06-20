"""
Experiment utilities for clustering fairness analysis.

This module provides functions for:
- Creating result recap tables for each experimental condition
- Omnibus separability tests (Fisher r x c / chi2 / ANOVA) for cluster quality
- Quality metrics summary
- Running batch experiments with the generic cluster() function
"""

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.metrics import silhouette_samples
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, false_discovery_control
from itertools import combinations
from .clustering import cluster
from .fairness_metrics import (
    mean_diff,
    feature_kind, cluster_value, cluster_value_cat, overview_gap, overview_gap_cat,
    onevsall_gap, onevsall_gap_cat, onevsall_gap_p,
    omnibus_separability_p,
    size_metrics, extreme_pair_gap_p, omnibus_error_sep_p,
    error_kind_for,
)


# Per-chunk RAM budget (MiB) for silhouette pairwise-distance computation.
# Silhouette on large datasets streams distances in chunks of shape (chunk_rows, n).
# Lower this if you hit MemoryError; raise it on machines with more RAM for fewer,
# larger chunks. Result is bit-identical regardless of value.
SILHOUETTE_WORKING_MEMORY_MIB = 128


# =============================================================================
# Utils for Results - Recap
# =============================================================================

def make_recap(data_result, feature_set, sensitive_cols=None, error_col='errors', error_type='binary', feature_matrix=None, distance_matrix=None, original_sensitive_cols=None, error_label=None, continuous_sensitive_cols=None, multiclass_option=None, error_cols=None, error_cols_kind='binary'):
  """
  Create a per-cluster Detail recap of cluster info, error stats, and per-feature
  one-vs-all fairness metrics.

  Emits one row per non-noise cluster with columns:
    c, count, proportion, silh, <error value col(s)>, error_gap, error_gap_sig,
    and for each sensitive feature F: F_value, [F_cat], F_gap, [F_gap_cat], F_gap_sig.
    (F_cat / F_gap_cat are emitted only for multi-categorical features; a multi-class
    error column likewise adds error_value / error_cat / error_gap_cat.)

  Each sensitive feature is classified ONCE (binary / multicat / numeric) via
  feature_kind() and represented by a SINGLE column group — no one-hot expansion.
  The one-vs-all gap significance (F_gap_sig / error_gap_sig) uses Fisher 2x2
  (binary / multicat winning category) or Mann-Whitney (numeric).

  Parameters
  ----------
  data_result : pd.DataFrame
      Clustered data with 'clusters' column.
  feature_set : list
      Feature columns used for clustering (for silhouette computation).
  sensitive_cols : list, optional
      Sensitive columns to report. One F_value/F_gap/F_gap_sig group per column.
  error_col : str
      Name of the error column. Default 'errors'.
  error_type : str
      'binary' for classification errors (0/1), 'regression' for continuous errors.
  continuous_sensitive_cols : list, optional
      Columns to treat as numeric (median-based) regardless of cardinality.
  error_cols : list, optional
      onehot multi-class error: one binary error column per class. When given,
      each emits its own [<ec>, <ec>_gap, <ec>_gap_sig] set and the single-error
      columns are omitted (error_col is then unused for the error tables).
  """
  if sensitive_cols is None:
    sensitive_cols = []
  continuous_set = set(continuous_sensitive_cols or [])
  error_kind = error_kind_for(error_type, multiclass_option)
  # onehot multi-class error: one binary error column per class, each producing its
  # own [value, gap, gap_sig] set instead of a single error group.
  onehot = bool(error_cols)
  error_cols = list(error_cols or [])

  # Exclude noise points (cluster label -1 from DBSCAN/HDBSCAN) before any computation
  noise_mask = data_result['clusters'] != -1
  data_result = data_result[noise_mask].copy()
  if feature_matrix is not None:
    feature_matrix = feature_matrix[noise_mask.values]

  check_cols = error_cols if onehot else [error_col]
  for ec in check_cols:
    if ec not in data_result.columns:
      raise ValueError(f"error column '{ec}' not found in data. Available columns: {list(data_result.columns)}")

  # Classify each sensitive feature ONCE (declaration + cardinality, NOT dtype).
  kinds = {F: feature_kind(data_result[F], F in continuous_set) for F in sensitive_cols}

  # ...with cluster size
  temp = data_result[['clusters']].copy()
  temp['count'] = 1
  recap = temp.groupby(['clusters'], as_index=False).sum()
  recap = recap.set_index('clusters', drop=False)

  # ...with proportion of total (non-noise) population
  recap['proportion'] = recap['count'] / recap['count'].sum()

  if onehot:
    pass  # per-class binary error sets are accumulated in the loop below
  elif error_kind == 'numeric':
    # Regression path: signed mean (bias direction) + absolute mean (magnitude)
    res = data_result[['clusters', error_col]]
    recap['error_mean'] = res.groupby(['clusters'])[error_col].mean().values
    recap['abs_error_mean'] = res.groupby(['clusters'])[error_col].apply(lambda x: x.abs().mean()).values
  elif error_kind == 'binary':
    # Binary path: per-cluster error count and error rate
    res = data_result[['clusters', error_col]]
    recap['n_error'] = res.groupby(['clusters']).sum().astype(int)
    recap['error_value'] = res.groupby(['clusters']).mean()
  # multicat error (per_class / per_cell): error_value (modal-category proportion)
  # and error_cat (modal label) are accumulated per cluster in the loop below,
  # mirroring the multi-categorical feature path.

  # Per-cluster accumulators. Category columns (winning category label) are only
  # emitted for multi-categorical features, so accumulate them for those only.
  multicat_cols = [F for F in sensitive_cols if kinds[F] == 'multicat']
  # Per-class error columns (onehot / classwise) carry their own kind: 'binary'
  # one-vs-all indicators, or 'multicat' TP/FN/FP/TN confusion cells.
  oh_kind = error_cols_kind if onehot else 'binary'
  # Global negative label per binary column (the proportion-positive denominator),
  # so an all-negative cluster reads 0 rather than 100% (see cluster_value).
  binary_neg = {F: data_result[F].min() for F in sensitive_cols if kinds[F] == 'binary'}
  oh_neg = {ec: data_result[ec].min() for ec in error_cols} if oh_kind == 'binary' else {}
  error_gap_sig = []
  error_gap = []
  # multicat-error-only accumulators (mirror the multicat feature columns).
  error_value_acc = []
  error_cat_acc = []
  error_gap_cat_acc = []
  # per-class error accumulators: one [value, gap, gap_sig] set per class, plus
  # cat / gap_cat when the columns are multi-categorical (classwise).
  oh_value = {ec: [] for ec in error_cols}
  oh_gap = {ec: [] for ec in error_cols}
  oh_gap_sig = {ec: [] for ec in error_cols}
  oh_cat = {ec: [] for ec in error_cols}
  oh_gap_cat = {ec: [] for ec in error_cols}
  feat_value = {F: [] for F in sensitive_cols}
  feat_gap_sig = {F: [] for F in sensitive_cols}
  feat_gap = {F: [] for F in sensitive_cols}
  feat_cat = {F: [] for F in multicat_cols}
  feat_gap_cat = {F: [] for F in multicat_cols}
  silhouette = []

  # Get individual silhouette scores
  clusters = data_result['clusters']
  if(len(recap['clusters'].unique()) > 1):
    # Use scaled feature_matrix if provided (matches the space clustering was done in)
    # Otherwise fall back to raw feature columns (less accurate silhouette)
    with config_context(working_memory=SILHOUETTE_WORKING_MEMORY_MIB):
      if distance_matrix is not None:
        # Gower clustering: use precomputed distance matrix (metric="precomputed")
        silhouette_val = silhouette_samples(distance_matrix, clusters, metric="precomputed")
      else:
        X_for_silhouette = feature_matrix if feature_matrix is not None else data_result[feature_set].values
        silhouette_val = silhouette_samples(X_for_silhouette, clusters)

  for c in recap['clusters']:
    # Get in-cluster data
    c_data = data_result.loc[data_result['clusters'] == c]

    # F_value is always computable (no one-vs-all needed)
    for F in sensitive_cols:
      feat_value[F].append(cluster_value(c_data[F], kinds[F], neg=binary_neg.get(F)))
      if kinds[F] == 'multicat':
        feat_cat[F].append(cluster_value_cat(c_data[F], kinds[F]))

    # Multicat error value (modal-category proportion) + label, also always computable.
    if error_kind == 'multicat':
      error_value_acc.append(cluster_value(c_data[error_col], 'multicat'))
      error_cat_acc.append(cluster_value_cat(c_data[error_col], 'multicat'))
    # per-class error value (positive rate, or modal-cell proportion + label),
    # always computable.
    if onehot:
      for ec in error_cols:
        oh_value[ec].append(cluster_value(c_data[ec], oh_kind, neg=oh_neg.get(ec)))
        if oh_kind == 'multicat':
          oh_cat[ec].append(cluster_value_cat(c_data[ec], 'multicat'))

    # Get out-of-cluster data
    rest_data = data_result.loc[data_result['clusters'] != c]
    # Single-cluster guard: no one-vs-all possible -> *_gap / *_gap_sig are NaN.
    if(len(rest_data) == 0):
      error_gap_sig.append(np.nan)
      error_gap.append(np.nan)
      if error_kind == 'multicat':
        error_gap_cat_acc.append(np.nan)
      for ec in error_cols:
        oh_gap[ec].append(np.nan)
        oh_gap_sig[ec].append(np.nan)
        if oh_kind == 'multicat':
          oh_gap_cat[ec].append(np.nan)
      for F in sensitive_cols:
        feat_gap_sig[F].append(np.nan)
        feat_gap[F].append(np.nan)
      for F in multicat_cols:
        feat_gap_cat[F].append(np.nan)
      silhouette.append(np.nan)
      break

    # Add silhouette score
    silhouette.append(silhouette_val[clusters == c].mean())

    rest_recap = recap.loc[recap['clusters'] != c]
    rest_count = rest_recap['count'].sum()

    # Error — one-vs-all signed gap, then a family-appropriate gap-significance
    # test (numeric: Mann-Whitney; binary/multicat: Fisher 2x2 on the positive /
    # most-divergent category) via onevsall_gap_p. onehot: one binary set per class.
    if onehot:
      for ec in error_cols:
        oh_gap[ec].append(round(onevsall_gap(c_data[ec], rest_data[ec], oh_kind), 4))
        oh_gap_sig[ec].append(onevsall_gap_p(c_data[ec], rest_data[ec], oh_kind))
        if oh_kind == 'multicat':
          oh_gap_cat[ec].append(onevsall_gap_cat(c_data[ec], rest_data[ec]))
    else:
      if error_kind == 'numeric':
        c_errors = c_data[error_col].values
        rest_errors = rest_data[error_col].values
        error_gap.append(round(mean_diff(c_errors, rest_errors), 6))
      elif error_kind == 'multicat':
        c_err = c_data[error_col]
        rest_err = rest_data[error_col]
        error_gap.append(round(onevsall_gap(c_err, rest_err, 'multicat'), 4))
        error_gap_cat_acc.append(onevsall_gap_cat(c_err, rest_err))
      else:
        rest_n_error = rest_recap['n_error'].sum()
        error_gap.append(recap['error_value'][c] - rest_n_error / rest_count)
      error_gap_sig.append(onevsall_gap_p(c_data[error_col], rest_data[error_col], error_kind))

    # Per-feature one-vs-all signed gap and gap-significance.
    for F in sensitive_cols:
      kind = kinds[F]
      c_vals = c_data[F]
      rest_vals = rest_data[F]
      feat_gap[F].append(round(onevsall_gap(c_vals, rest_vals, kind), 4))
      if kind == 'multicat':
        feat_gap_cat[F].append(onevsall_gap_cat(c_vals, rest_vals))
      feat_gap_sig[F].append(onevsall_gap_p(c_vals, rest_vals, kind))

  # Collect all new columns into a dict then concat once to avoid fragmentation.
  # Multicat error emits value/cat (modal) alongside the gap/gap-cat, mirroring a
  # multi-categorical feature.
  new_cols = {}
  if onehot:
    # One [value, (cat), gap, (gap_cat), gap_sig] set per class; no single-error cols.
    for ec in error_cols:
      new_cols[ec] = np.around(np.asarray(oh_value[ec], dtype=float), 3) if oh_kind == 'binary' else oh_value[ec]
      if oh_kind == 'multicat':
        new_cols[f'{ec}_cat'] = oh_cat[ec]
      new_cols[f'{ec}_gap'] = np.around(np.asarray(oh_gap[ec], dtype=float), 3)
      if oh_kind == 'multicat':
        new_cols[f'{ec}_gap_cat'] = oh_gap_cat[ec]
      new_cols[f'{ec}_gap_sig'] = oh_gap_sig[ec]
  else:
    if error_kind == 'multicat':
      new_cols['error_value'] = error_value_acc
      new_cols['error_cat'] = error_cat_acc
    new_cols['error_gap'] = np.around(error_gap, 3)
    if error_kind == 'multicat':
      new_cols['error_gap_cat'] = error_gap_cat_acc
    new_cols['error_gap_sig'] = error_gap_sig
  for F in sensitive_cols:
    new_cols[f'{F}_value'] = feat_value[F]
    if F in feat_cat:
      new_cols[f'{F}_cat'] = feat_cat[F]
    new_cols[f'{F}_gap'] = feat_gap[F]
    if F in feat_gap_cat:
      new_cols[f'{F}_gap_cat'] = feat_gap_cat[F]
    new_cols[f'{F}_gap_sig'] = feat_gap_sig[F]
  new_cols['silh'] = silhouette

  recap = pd.concat([recap, pd.DataFrame(new_cols, index=recap.index)], axis=1)

  if onehot:
    pass  # onehot error values already rounded above
  elif error_kind == 'numeric':
    recap['error_mean'] = np.around(recap['error_mean'], 3)
    recap['abs_error_mean'] = np.around(recap['abs_error_mean'], 3)
  else:
    recap['error_value'] = np.around(recap['error_value'], 3)

  recap = recap.reset_index(drop=True)
  recap.rename(columns={'clusters': 'c'}, inplace=True)

  return recap


# =============================================================================
# Utils for Results - Separability Check (Chi-squared / Kruskal-Wallis)
# =============================================================================

def separability_check(data, labels, columns):
    """
    Test if clusters are significantly different across features.

    Uses appropriate statistical test based on data type and cluster count:
    - Categorical (object, category, bool): Chi-squared test
    - Numeric, 2 clusters: Mann-Whitney U test
    - Numeric, 3+ clusters: Kruskal-Wallis test

    Parameters
    ----------
    data : pd.DataFrame
        Data with features to test.
    labels : np.ndarray
        Cluster labels for each row.
    columns : list
        Column names to test.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: test, statistic, p_value
        Index is column names.
    """
    results = {}
    unique_labels = [l for l in np.unique(labels) if l != -1]

    # Filter to non-noise points
    mask = labels != -1
    data_filtered = data[mask]
    labels_filtered = labels[mask]

    if len(unique_labels) < 2:
        # Need at least 2 clusters for comparison
        return pd.DataFrame(columns=['test', 'statistic', 'p_value'])

    for col in columns:
        if col not in data.columns:
            continue

        col_data = data_filtered[col]

        if col_data.dtype in ['object', 'category', 'bool'] or col_data.dtype.name == 'category':
            # Chi-squared for categorical
            try:
                contingency = pd.crosstab(col_data, labels_filtered)
                stat, p, dof, expected = chi2_contingency(contingency)
                results[col] = {'test': 'chi2', 'statistic': round(stat, 4), 'p_value': round(p, 6)}
            except Exception:
                results[col] = {'test': 'chi2', 'statistic': np.nan, 'p_value': np.nan}
        else:
            # Numeric: Mann-Whitney U (2 clusters) or Kruskal-Wallis (3+)
            try:
                groups = [data_filtered[labels_filtered == l][col].dropna().values for l in unique_labels]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) == 2:
                    stat, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    results[col] = {'test': 'mannwhitneyu', 'statistic': round(stat, 4), 'p_value': round(p, 6)}
                elif len(groups) >= 3:
                    stat, p = kruskal(*groups)
                    results[col] = {'test': 'kruskal', 'statistic': round(stat, 4), 'p_value': round(p, 6)}
                else:
                    results[col] = {'test': 'n/a', 'statistic': np.nan, 'p_value': np.nan}
            except Exception:
                results[col] = {'test': 'kruskal', 'statistic': np.nan, 'p_value': np.nan}

    return pd.DataFrame(results).T


# =============================================================================
# Utils for Results - Chi-Square Tests
# =============================================================================

def make_chi_tests(results, sensitive_cols=None, error_type='binary', error_col='errors', error_label=None, continuous_sensitive_cols=None, multiclass_option=None, error_cols=None):
  """
  Compute one omnibus separability p-value per condition for the error column
  and for each sensitive feature.

  Each sensitive feature is classified ONCE (binary / multicat / numeric) via
  feature_kind() and gets a single `<F>_sep` column (chi2 for categorical, one-way
  ANOVA for numeric). The error column gets an `error_sep` column: one-way ANOVA
  for regression, otherwise an r x c Fisher exact test (R via rpy2) on the
  (error-value x cluster) table. Benjamini-Hochberg FDR correction is applied
  ACROSS the `<F>_sep` columns within each row.

  Parameters
  ----------
  results : dict
      Results from run_experiments_generic().
  sensitive_cols : list, optional
      Sensitive column names. One `<F>_sep` column per feature.
  error_type : str
      'binary' (error treated as binary) or 'regression' (error treated as numeric).
  error_col : str
      Name of the error column in the per-condition result DataFrames.
  error_label : str, optional
      Unused for column naming (kept for signature compatibility).
  continuous_sensitive_cols : list, optional
      Subset of sensitive_cols to treat as numeric regardless of cardinality.

  Returns
  -------
  pd.DataFrame
      Columns: cond_descr, cond_name, error_sep, <F>_sep (one per sensitive feature).
  """
  if sensitive_cols is None:
    sensitive_cols = []
  continuous_set = set(continuous_sensitive_cols or [])

  error_kind = error_kind_for(error_type, multiclass_option)
  onehot = bool(error_cols)
  error_cols = list(error_cols or [])
  sep_cols = [f'{F}_sep' for F in sensitive_cols]
  # onehot: one omnibus error_sep per class; otherwise a single 'error_sep'.
  err_sep_cols = [f'{ec}_sep' for ec in error_cols] if onehot else ['error_sep']

  chi_res = {'cond_descr': [], 'cond_name': []}
  for sc in err_sep_cols + sep_cols:
    chi_res[sc] = []

  for i in range(0, len(results['cond_name'])):
    chi_res['cond_descr'].append(results['cond_descr'][i])
    chi_res['cond_name'].append(results['cond_name'][i])

    res_df = results['cond_res'][i]
    labels = res_df['clusters'].values

    # Single-cluster guard: < 2 non-noise clusters -> all-NaN row.
    if len(set(labels) - {-1}) < 2:
      for sc in err_sep_cols + sep_cols:
        chi_res[sc].append(np.nan)
      continue

    if onehot:
      for ec in error_cols:
        chi_res[f'{ec}_sep'].append(omnibus_error_sep_p(res_df[ec], labels, 'binary'))
    else:
      chi_res['error_sep'].append(omnibus_error_sep_p(res_df[error_col], labels, error_kind))
    for F in sensitive_cols:
      kind = feature_kind(res_df[F], F in continuous_set)
      chi_res[f'{F}_sep'].append(omnibus_separability_p(res_df[F], labels, kind))

  chi_df = pd.DataFrame(chi_res)

  # Benjamini-Hochberg FDR correction across the <F>_sep columns of each row.
  if sep_cols and len(chi_df) > 0:
    for i in chi_df.index:
      row_p = chi_df.loc[i, sep_cols].values.astype(float)
      valid_mask = ~np.isnan(row_p)
      if valid_mask.sum() > 1:
        corrected = false_discovery_control(row_p[valid_mask], method='bh')
        row_p[valid_mask] = np.round(corrected, 6)
        chi_df.loc[i, sep_cols] = row_p

  return chi_df


# =============================================================================
# Utils for Results - All Quality Metrics
# =============================================================================

def recap_quali_metrics(chi_res, results, exp_condition, sensitive_cols=None, original_sensitive_cols=None, error_label=None, continuous_sensitive_cols=None, error_col='errors', error_type='binary', multiclass_option=None, error_cols=None, error_cols_kind='binary'):
  """
  Build the Overview frame: per-condition silhouette, cluster-size summary, error
  separability / gap / gap-significance, and per-feature gap / gap-significance.

  Output columns (in order):
    cond_descr, cond_name, silh,
    min_size, min_prop, max_prop,
    error_sep, error_gap, error_gap_sig,
    <F>_gap, <F>_gap_cat*, <F>_gap_sig  (per sensitive feature;
    *_gap_cat only for multi-categorical features)

  `error_sep` is the omnibus separability across all clusters (copied from
  chi_res). `error_gap` is the cross-cluster spread; `error_gap_sig` is the
  significance of that spread tested on ONLY the two extreme clusters that define
  it (extreme-pair test). Each sensitive feature follows the same gap /
  extreme-pair-gap-sig pattern. Per the spec, the Overview omits a per-feature
  omnibus `<F>_sep` column (that lives only in the separability-test output).

  Parameters
  ----------
  chi_res : pd.DataFrame
      Output of make_chi_tests (provides error_sep).
  results : dict
      Results from run_experiments_generic().
  exp_condition : pd.DataFrame
      Experimental conditions (unused, kept for signature compatibility).
  sensitive_cols : list, optional
      Sensitive column names. One <F>_gap/<F>_gap_sig pair per feature.
  continuous_sensitive_cols : list, optional
      Subset of sensitive_cols to treat as numeric regardless of cardinality.
  error_col : str
      Name of the error column in the per-condition result DataFrames.
  error_type : str
      'binary', 'regression', or 'multiclass' (controls the error gap kind).
  """
  if sensitive_cols is None:
    sensitive_cols = []
  continuous_set = set(continuous_sensitive_cols or [])
  error_kind = error_kind_for(error_type, multiclass_option)
  onehot = bool(error_cols)
  error_cols = list(error_cols or [])
  oh_kind = error_cols_kind if onehot else 'binary'

  # Feature kinds are constant across conditions (they depend only on the column
  # data, not the cluster labels), so classify once from the first condition.
  # The winning-category column is emitted only for multi-categorical features.
  kinds = {}
  if len(chi_res['cond_name']) > 0:
    ref_df = results['cond_res'][0]
    kinds = {F: feature_kind(ref_df[F], F in continuous_set) for F in sensitive_cols}
  multicat_cols = [F for F in sensitive_cols if kinds.get(F) == 'multicat']

  silh = []
  min_size, min_prop, max_prop = [], [], []
  error_gap, error_gap_sig = [], []
  error_gap_class = []  # multicat error only: winning error class behind the gap
  oh_gap = {ec: [] for ec in error_cols}        # per-class overview gap
  oh_gap_sig = {ec: [] for ec in error_cols}    # per-class extreme-pair sig
  oh_gap_class = {ec: [] for ec in error_cols}  # classwise: winning cell behind gap
  feat_gap = {F: [] for F in sensitive_cols}
  feat_gap_cat = {F: [] for F in multicat_cols}
  feat_gap_sig = {F: [] for F in sensitive_cols}

  for i in range(0, len(chi_res['cond_name'])):
    recap = results['cond_recap'][i]
    res_df = results['cond_res'][i]
    labels = res_df['clusters'].values

    # Size summary is well-defined even for a single cluster.
    sizes = size_metrics(labels)
    min_size.append(sizes['min_size'])
    min_prop.append(sizes['min_prop'])
    max_prop.append(sizes['max_prop'])

    # Gaps / extreme-pair significance need >= 2 clusters.
    if len(recap) == 1:
      silh.append(np.nan)
      error_gap.append(np.nan)
      error_gap_sig.append(np.nan)
      if error_kind == 'multicat':
        error_gap_class.append(np.nan)
      for ec in error_cols:
        oh_gap[ec].append(np.nan)
        oh_gap_sig[ec].append(np.nan)
        if oh_kind == 'multicat':
          oh_gap_class[ec].append(np.nan)
      for F in sensitive_cols:
        feat_gap[F].append(np.nan)
        feat_gap_sig[F].append(np.nan)
      for F in multicat_cols:
        feat_gap_cat[F].append(np.nan)
      continue

    silh.append(recap['silh'].mean())
    if onehot:
      for ec in error_cols:
        oh_gap[ec].append(overview_gap(res_df[ec], labels, oh_kind))
        oh_gap_sig[ec].append(extreme_pair_gap_p(res_df[ec], labels, oh_kind))
        if oh_kind == 'multicat':
          oh_gap_class[ec].append(overview_gap_cat(res_df[ec], labels))
    else:
      error_gap.append(overview_gap(res_df[error_col], labels, error_kind))
      error_gap_sig.append(extreme_pair_gap_p(res_df[error_col], labels, error_kind))
      if error_kind == 'multicat':
        error_gap_class.append(overview_gap_cat(res_df[error_col], labels))
    for F in sensitive_cols:
      kind = kinds[F]
      feat_gap[F].append(overview_gap(res_df[F], labels, kind))
      feat_gap_sig[F].append(extreme_pair_gap_p(res_df[F], labels, kind))
      if kind == 'multicat':
        feat_gap_cat[F].append(overview_gap_cat(res_df[F], labels))

  all_quali = {
      'cond_descr': chi_res['cond_descr'].values,
      'cond_name': chi_res['cond_name'].values,
      'silh': silh,
      'min_size': min_size,
      'min_prop': min_prop,
      'max_prop': max_prop,
  }
  if onehot:
    # One [sep, gap, (gap_class), gap_sig] set per class; sep copied from chi_res.
    for ec in error_cols:
      all_quali[f'{ec}_sep'] = chi_res[f'{ec}_sep'].values
      all_quali[f'{ec}_gap'] = oh_gap[ec]
      if oh_kind == 'multicat':
        all_quali[f'{ec}_gap_class'] = oh_gap_class[ec]
      all_quali[f'{ec}_gap_sig'] = oh_gap_sig[ec]
  else:
    all_quali['error_sep'] = chi_res['error_sep'].values
    all_quali['error_gap'] = error_gap
    if error_kind == 'multicat':
      all_quali['error_gap_class'] = error_gap_class
    all_quali['error_gap_sig'] = error_gap_sig
  for F in sensitive_cols:
    all_quali[f'{F}_gap'] = feat_gap[F]
    if F in feat_gap_cat:
      all_quali[f'{F}_gap_cat'] = feat_gap_cat[F]
    all_quali[f'{F}_gap_sig'] = feat_gap_sig[F]

  return pd.DataFrame(all_quali)


# =============================================================================
# Visualization — result-table heatmaps (see src/result_viz.py)
# =============================================================================
# Re-exported so existing callers (main.py, tests) keep importing these from
# src.experiments; the implementations live in result_viz to keep this module
# focused on table building.
from .result_viz import (  # noqa: E402,F401
    classify_column, order_result_columns,
    render_result_heatmap, plot_quality_heatmap, plot_cluster_recap_heatmap,
)



# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiments_generic(data, exp_condition, algorithm, distance,
                            n_clusters=None, n_min=None, n_max=None,
                            max_iter=300, seed=42,
                            scoring_fn=None, sensitive_cols=None, error_col='errors',
                            min_cluster_size=15, min_samples=5, eps=0.5,
                            min_datapoints=None, feature_weights=None,
                            error_type='binary', categorical_col_names=None,
                            standardize=True, error_label=None,
                            original_sensitive_cols=None,
                            continuous_sensitive_cols=None,
                            ohe_col_names=None, multiclass_option=None,
                            error_cols=None, error_cols_kind='binary'):
  """
  Run all experimental conditions using the generic cluster() function.

  Works with any algorithm supported by cluster() (kmeans, bisectingkmeans,
  kmedoids, kprototypes, dbscan, hdbscan). Returns a dict that downstream
  code (make_chi_tests, recap_quali_metrics, heatmaps) consumes.

  Parameters
  ----------
  data : pd.DataFrame
      Input data with features and error columns.
  exp_condition : pd.DataFrame
      DataFrame with columns: feature_set_descr, feature_set_name, feature_set
  algorithm : str
      Clustering algorithm name.
  distance : str
      Distance metric.
  n_clusters : int, optional
      Fixed number of clusters.
  n_min, n_max : int, optional
      Range for k-search.
  max_iter : int
      Maximum iterations.
  seed : int
      Random seed.
  scoring_fn : callable, optional
      Scoring function for k-selection.
  sensitive_cols : list, optional
      Sensitive columns for recap.
  error_col : str
      Name of the error column.
  min_cluster_size : int
      HDBSCAN min_cluster_size.
  min_samples : int
      HDBSCAN min_samples.
  eps : float
      DBSCAN eps.
  min_datapoints : int, optional
      Minimum datapoints per cluster.
  feature_weights : dict, optional
      Feature weights for clustering.
  error_type : str
      'binary', 'regression', or 'multiclass'. Default 'binary'.

  Returns
  -------
  dict
      Results dictionary with keys: cond_name, cond_descr, cond_res, cond_recap
  """
  np.random.seed(seed)

  results = {'cond_name': [],
            'cond_descr': [],
            'cond_res': [],
            'cond_recap': []}

  cat_names_set = set(categorical_col_names) if categorical_col_names else set()
  ohe_col_set = set(ohe_col_names) if ohe_col_names else set()

  n_conditions = len(exp_condition)
  for i in range(n_conditions):
    feature_set = exp_condition['feature_set'][i]
    cond_name = exp_condition['feature_set_name'][i].strip()
    print(f"  [{i+1}/{n_conditions}] {cond_name} ...", flush=True)

    cat_features = [j for j, c in enumerate(feature_set) if c in cat_names_set] or None
    ohe_feature_indices = [j for j, c in enumerate(feature_set) if c in ohe_col_set] or None

    result = cluster(
        features=data[feature_set],
        algorithm=algorithm,
        distance=distance,
        n_clusters=n_clusters,
        n_min=n_min,
        n_max=n_max,
        max_iter=max_iter,
        random_state=seed,
        scoring_fn=scoring_fn,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        eps=eps,
        min_datapoints=min_datapoints,
        feature_weights=feature_weights,
        categorical_features=cat_features,
        standardize=standardize,
        ohe_features=ohe_feature_indices,
    )

    sil_str = f", silhouette={result.silhouette:.3f}" if result.silhouette is not None else ""
    noise_str = f", noise={result.n_noise}" if result.n_noise > 0 else ""
    print(f"         k={result.n_clusters}{sil_str}{noise_str}")

    # Build result DataFrame: original data + 'clusters' column
    res_df = data.copy()
    if result.mask is not None:
      # Subset was applied: assign -1 to excluded rows, labels to included
      res_df['clusters'] = -1
      res_df.loc[result.mask, 'clusters'] = result.labels
    else:
      res_df['clusters'] = result.labels

    recap = make_recap(res_df, feature_set,
                       sensitive_cols=sensitive_cols, error_col=error_col,
                       error_type=error_type,
                       feature_matrix=result.feature_matrix,
                       distance_matrix=result.distance_matrix,
                       original_sensitive_cols=original_sensitive_cols,
                       error_label=error_label,
                       continuous_sensitive_cols=continuous_sensitive_cols,
                       multiclass_option=multiclass_option,
                       error_cols=error_cols, error_cols_kind=error_cols_kind)

    results['cond_name'].append(exp_condition['feature_set_name'][i])
    results['cond_descr'].append(exp_condition['feature_set_descr'][i])
    results['cond_res'].append(res_df)
    results['cond_recap'].append(recap)

  return results


# =============================================================================
# Experimental Conditions Setup
# =============================================================================

def create_exp_conditions(groups):
  """
  Generate all experimental conditions from named feature groups.

  Generates all non-empty subsets of groups, excluding subsets where the
  only group present is 'ERR'. Each condition is named like
  '+REG +SEN -ERR -SPECIAL' (uppercase = included, lowercase = excluded).

  Parameters
  ----------
  groups : dict
      Mapping of group_name -> list of column names.
      Example: {'REG': ['age_scaled', ...], 'SEN': ['sex_Female', ...],
                'ERR': ['errors'], 'SPECIAL': ['Shap_age_scaled', ...]}

  Returns
  -------
  pd.DataFrame with columns: feature_set_descr, feature_set_name, feature_set
  """
  group_names = list(groups.keys())
  n = len(group_names)

  feature_set_name = []
  feature_set_descr = []
  feature_set = []

  # Generate all non-empty subsets
  for r in range(1, n + 1):
    for subset in combinations(range(n), r):
      included = set(subset)
      included_names = [group_names[i] for i in included]

      # Skip if the only group is 'ERR'
      if included_names == ['ERR']:
        continue

      # Build name: +REG +SEN -ERR (uppercase=included, lowercase=excluded)
      name_parts = []
      for i, gname in enumerate(group_names):
        if i in included:
          name_parts.append(f'+{gname.upper()}')
        else:
          name_parts.append(f'-{gname.lower()}')
      name = ' '.join(name_parts)

      # Build description
      descr = ' + '.join(included_names)

      # Build feature set: concatenation of included groups' columns
      cols = []
      for i in included:
        cols.extend(groups[group_names[i]])

      feature_set_name.append(name)
      feature_set_descr.append(descr)
      feature_set.append(cols)

  exp_condition = pd.DataFrame({'feature_set_descr': feature_set_descr,
                                'feature_set_name': feature_set_name,
                                'feature_set': feature_set})
  return exp_condition
