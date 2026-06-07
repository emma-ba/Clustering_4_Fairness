"""
Experiment utilities for clustering fairness analysis.

This module provides functions for:
- Creating result recap tables for each experimental condition
- Chi-square / Kruskal-Wallis tests for cluster quality
- Quality metrics summary
- Running batch experiments with the generic cluster() function
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import re
from sklearn import config_context
from sklearn.metrics import silhouette_samples
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, false_discovery_control
from itertools import combinations
from .clustering import cluster
from .fairness_metrics import (
    cluster_proportion, one_vs_all_p_binary, one_vs_all_p_continuous, mean_diff,
    feature_kind, cluster_value, overview_gap, onevsall_gap,
    omnibus_separability_p, onevsall_categorical_p,
)


# Per-chunk RAM budget (MiB) for silhouette pairwise-distance computation.
# Silhouette on large datasets streams distances in chunks of shape (chunk_rows, n).
# Lower this if you hit MemoryError; raise it on machines with more RAM for fewer,
# larger chunks. Result is bit-identical regardless of value.
SILHOUETTE_WORKING_MEMORY_MIB = 128


# =============================================================================
# Utils for Results - Recap
# =============================================================================

def make_recap(data_result, feature_set, sensitive_cols=None, error_col='errors', error_type='binary', feature_matrix=None, distance_matrix=None, original_sensitive_cols=None, error_label=None, continuous_sensitive_cols=None):
  """
  Create a per-cluster Detail recap of cluster info, error stats, and per-feature
  one-vs-all fairness metrics.

  Emits one row per non-noise cluster with columns:
    c, count, proportion, silh, <error value col(s)>, error_sep, error_gap,
    and for each sensitive feature F: F_value, F_sep, F_gap.

  Each sensitive feature is classified ONCE (binary / multicat / numeric) via
  feature_kind() and represented by a SINGLE column trio — no one-hot expansion.

  Parameters
  ----------
  data_result : pd.DataFrame
      Clustered data with 'clusters' column.
  feature_set : list
      Feature columns used for clustering (for silhouette computation).
  sensitive_cols : list, optional
      Sensitive columns to report. One F_value/F_sep/F_gap trio per column.
  error_col : str
      Name of the error column. Default 'errors'.
  error_type : str
      'binary' for classification errors (0/1), 'regression' for continuous errors.
  continuous_sensitive_cols : list, optional
      Columns to treat as numeric (median-based) regardless of cardinality.
  """
  if sensitive_cols is None:
    sensitive_cols = []
  continuous_set = set(continuous_sensitive_cols or [])

  # Exclude noise points (cluster label -1 from DBSCAN/HDBSCAN) before any computation
  noise_mask = data_result['clusters'] != -1
  data_result = data_result[noise_mask].copy()
  if feature_matrix is not None:
    feature_matrix = feature_matrix[noise_mask.values]

  if error_col not in data_result.columns:
    raise ValueError(f"error_col '{error_col}' not found in data. Available columns: {list(data_result.columns)}")

  # Classify each sensitive feature ONCE (declaration + cardinality, NOT dtype).
  kinds = {F: feature_kind(data_result[F], F in continuous_set) for F in sensitive_cols}

  res = data_result[['clusters', error_col]]

  # ...with cluster size
  temp = data_result[['clusters']].copy()
  temp['count'] = 1
  recap = temp.groupby(['clusters'], as_index=False).sum()
  recap = recap.set_index('clusters', drop=False)

  # ...with proportion of total (non-noise) population
  recap['proportion'] = recap['count'] / recap['count'].sum()

  if error_type == 'regression':
    # Regression path: signed mean (bias direction) + absolute mean (magnitude)
    recap['error_mean'] = res.groupby(['clusters'])[error_col].mean().values
    recap['abs_error_mean'] = res.groupby(['clusters'])[error_col].apply(lambda x: x.abs().mean()).values
  else:
    # Binary path: per-cluster error count and error rate
    recap['n_error'] = res.groupby(['clusters']).sum().astype(int)
    recap['error_value'] = res.groupby(['clusters']).mean()

  # Per-feature global positive value (binary): use the GLOBAL max so that an
  # all-zero cluster doesn't miscount which value is "positive".
  global_pos = {F: data_result[F].max() for F in sensitive_cols if kinds[F] == 'binary'}

  # Per-cluster accumulators
  error_sep = []
  error_gap = []
  feat_value = {F: [] for F in sensitive_cols}
  feat_sep = {F: [] for F in sensitive_cols}
  feat_gap = {F: [] for F in sensitive_cols}
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
    c_count = recap['count'][c]

    # F_value is always computable (no one-vs-all needed)
    for F in sensitive_cols:
      feat_value[F].append(cluster_value(c_data[F], kinds[F]))

    # Get out-of-cluster data
    rest_data = data_result.loc[data_result['clusters'] != c]
    # Single-cluster guard: no one-vs-all possible -> *_sep / *_gap are NaN.
    if(len(rest_data) == 0):
      error_sep.append(np.nan)
      error_gap.append(np.nan)
      for F in sensitive_cols:
        feat_sep[F].append(np.nan)
        feat_gap[F].append(np.nan)
      silhouette.append(np.nan)
      break

    # Add silhouette score
    silhouette.append(silhouette_val[clusters == c].mean())

    rest_recap = recap.loc[recap['clusters'] != c]
    rest_count = rest_recap['count'].sum()

    # Error — one-vs-all signed gap and separability p-value
    if error_type == 'regression':
      c_errors = c_data[error_col].values
      rest_errors = rest_data[error_col].values
      error_gap.append(round(mean_diff(c_errors, rest_errors), 6))
      error_sep.append(one_vs_all_p_continuous(c_errors, rest_errors))
    else:
      rest_n_error = rest_recap['n_error'].sum()
      error_gap.append(recap['error_value'][c] - rest_n_error / rest_count)
      error_sep.append(one_vs_all_p_binary(
          recap['n_error'][c], recap['count'][c], rest_n_error, rest_count
      ))

    # Per-feature one-vs-all separability (sep) and signed gap.
    for F in sensitive_cols:
      kind = kinds[F]
      c_vals = c_data[F]
      rest_vals = rest_data[F]
      feat_gap[F].append(round(onevsall_gap(c_vals, rest_vals, kind), 4))
      if kind == 'binary':
        pos = global_pos[F]
        c_pos = int((c_vals == pos).sum())
        rest_pos = int((rest_vals == pos).sum())
        feat_sep[F].append(one_vs_all_p_binary(c_pos, len(c_vals), rest_pos, len(rest_vals)))
      elif kind == 'multicat':
        feat_sep[F].append(onevsall_categorical_p(c_vals, rest_vals))
      else:  # numeric
        feat_sep[F].append(one_vs_all_p_continuous(c_vals.dropna().values, rest_vals.dropna().values))

  # Collect all new columns into a dict then concat once to avoid fragmentation
  new_cols = {
      'error_sep': error_sep,
      'error_gap': np.around(error_gap, 3),
  }
  for F in sensitive_cols:
    new_cols[f'{F}_value'] = feat_value[F]
    new_cols[f'{F}_sep'] = feat_sep[F]
    new_cols[f'{F}_gap'] = feat_gap[F]
  new_cols['silh'] = silhouette

  recap = pd.concat([recap, pd.DataFrame(new_cols, index=recap.index)], axis=1)

  if error_type == 'regression':
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

def make_chi_tests(results, sensitive_cols=None, error_type='binary', error_col='errors', error_label=None, continuous_sensitive_cols=None):
  """
  Compute one omnibus separability p-value per condition for the error column
  and for each sensitive feature.

  Each sensitive feature is classified ONCE (binary / multicat / numeric) via
  feature_kind() and gets a single `<F>_sep` column. The error column gets an
  `error_sep` column (Kruskal-Wallis for regression, otherwise treated as
  binary/categorical via chi2). Benjamini-Hochberg FDR correction is applied
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

  error_kind = 'numeric' if error_type == 'regression' else 'binary'
  sep_cols = [f'{F}_sep' for F in sensitive_cols]

  chi_res = {'cond_descr': [], 'cond_name': [], 'error_sep': []}
  for sc in sep_cols:
    chi_res[sc] = []

  for i in range(0, len(results['cond_name'])):
    chi_res['cond_descr'].append(results['cond_descr'][i])
    chi_res['cond_name'].append(results['cond_name'][i])

    res_df = results['cond_res'][i]
    labels = res_df['clusters'].values

    # Single-cluster guard: < 2 non-noise clusters -> all-NaN row.
    if len(set(labels) - {-1}) < 2:
      chi_res['error_sep'].append(np.nan)
      for sc in sep_cols:
        chi_res[sc].append(np.nan)
      continue

    chi_res['error_sep'].append(
        omnibus_separability_p(res_df[error_col], labels, error_kind)
    )
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

def recap_quali_metrics(chi_res, results, exp_condition, sensitive_cols=None, original_sensitive_cols=None, error_label=None, continuous_sensitive_cols=None, error_col='errors', error_type='binary'):
  """
  Build the Overview frame: per-condition silhouette, separability p-values
  (copied from chi_res) and across-cluster gaps (spread of the error column and
  each sensitive feature).

  Output columns (in order):
    cond_descr, cond_name, silh, error_sep, error_gap, <F>_sep, <F>_gap

  Parameters
  ----------
  chi_res : pd.DataFrame
      Output of make_chi_tests (provides error_sep and <F>_sep).
  results : dict
      Results from run_experiments_generic().
  exp_condition : pd.DataFrame
      Experimental conditions (unused, kept for signature compatibility).
  sensitive_cols : list, optional
      Sensitive column names. One <F>_sep/<F>_gap pair per feature.
  continuous_sensitive_cols : list, optional
      Subset of sensitive_cols to treat as numeric regardless of cardinality.
  error_col : str
      Name of the error column in the per-condition result DataFrames.
  error_type : str
      'binary' or 'regression' (controls the error gap kind).
  """
  if sensitive_cols is None:
    sensitive_cols = []
  continuous_set = set(continuous_sensitive_cols or [])
  error_kind = 'numeric' if error_type == 'regression' else 'binary'

  silh = []
  error_gap = []
  feat_gap = {F: [] for F in sensitive_cols}

  for i in range(0, len(chi_res['cond_name'])):
    recap = results['cond_recap'][i]
    res_df = results['cond_res'][i]
    labels = res_df['clusters'].values
    single_cluster = len(recap) == 1

    if single_cluster:
      silh.append(np.nan)
      error_gap.append(np.nan)
      for F in sensitive_cols:
        feat_gap[F].append(np.nan)
      continue

    silh.append(recap['silh'].mean())
    error_gap.append(overview_gap(res_df[error_col], labels, error_kind))
    for F in sensitive_cols:
      kind = feature_kind(res_df[F], F in continuous_set)
      feat_gap[F].append(overview_gap(res_df[F], labels, kind))

  all_quali = {
      'cond_descr': chi_res['cond_descr'].values,
      'cond_name': chi_res['cond_name'].values,
      'silh': silh,
      'error_sep': chi_res['error_sep'].values,
      'error_gap': error_gap,
  }
  for F in sensitive_cols:
    all_quali[f'{F}_sep'] = chi_res[f'{F}_sep'].values
    all_quali[f'{F}_gap'] = feat_gap[F]

  return pd.DataFrame(all_quali)


# =============================================================================
# Visualization
# =============================================================================

def plot_quality_heatmap(all_quali_viz, output_path, figsize=None, error_label='error'):
  """
  Plot quality metrics heatmap with color-coded column groups.

  - Error column (error_label): Reds_r  (lower p = more significant = redder)
  - Sensitive p-value columns: Greens_r
  - Silhouette: Blues              (higher = better = darker blue)
  - Balance/entropy columns: Greens
  """
  df = all_quali_viz.copy()

  # Classify columns
  error_cols = [c for c in df.columns if c == error_label]
  sil_cols = [c for c in df.columns if c == 'silhouette']
  sensitive_cols = [c for c in df.columns if c not in error_cols + sil_cols]

  groups = []
  if error_cols:
    groups.append((error_cols, 'Reds_r', None, None))
  if sensitive_cols:
    groups.append((sensitive_cols, 'Greens_r', None, None))
  if sil_cols:
    groups.append((sil_cols, 'Blues', None, None))

  if not groups:
    return

  n_groups = len(groups)
  col_counts = [len(g[0]) for g in groups]
  total_cols = sum(col_counts)

  if figsize is None:
    n_rows = len(df)
    figsize = (max(6, total_cols * 1.2), max(4, n_rows * 0.6))

  fig, axes = plt.subplots(1, n_groups, figsize=figsize,
                           gridspec_kw={'width_ratios': col_counts, 'wspace': 0})
  if n_groups == 1:
    axes = [axes]

  for ax, (cols, cmap, vmin, vmax) in zip(axes, groups):
    kw = dict(annot=True, fmt='.4g', cbar=False, ax=ax, robust=True)
    if vmin is not None:
      kw.update(vmin=vmin, vmax=vmax)
      kw.pop('robust')
    sns.heatmap(df[cols], cmap=cmap, **kw)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='major', length=0, rotation=45)
    ax.tick_params(axis='y', which='major', length=0)
    ax.set_xticklabels(ax.get_xticklabels(), ha='left', rotation=45, rotation_mode='anchor')
    ax.set(xlabel='', ylabel='')

  # Only show y-tick labels on the first subplot
  for ax in axes[1:]:
    ax.set_yticklabels([])

  plt.tight_layout()
  plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)


def plot_cluster_recap_heatmap(recap, cond_name, output_dir, multiclass_dummies=None):
  """Plot one-vs-all cluster comparison heatmap with color-coded column groups."""
  recap = recap.sort_values(by=['diff_vs_rest'], ascending=False)
  recap = recap.copy()
  recap['count'] = recap['count'] / recap['count'].sum()
  recap = recap.rename(columns={"count": "size_prop"})
  drop_cols = ['c']
  if 'n_error' in recap.columns:
    drop_cols.append('n_error')
  drop_cols.extend([c for c in recap.columns if c.endswith('_n')])
  recap = recap.drop(drop_cols, axis=1)

  # Custom black→white→blue colormap for *_diff columns (neg=black, zero=white, pos=blue)
  bwb_cmap = mcolors.LinearSegmentedColormap.from_list(
      'bwb', [(0.0, 'black'), (0.5, 'white'), (1.0, '#1f77b4')]
  )

  suppressed_bases = set()
  if multiclass_dummies:
    for dummy_list in multiclass_dummies.values():
      suppressed_bases.update(dummy_list)

  def _is_per_value_stat(col):
    for suffix in ('_prop', '_diff', '_p'):
      if col.endswith(suffix):
        base = col[:-len(suffix)]
        if base in suppressed_bases or '=' in base:
          return True
    return False

  cols = [c for c in recap.columns if not _is_per_value_stat(c)]

  # Classify columns into groups (order matters for display)
  error_rate_cols = [c for c in cols if c.endswith('_rate') or c.endswith('_mean')]
  mw_cols = [c for c in cols if c == 'mannwhitney_p']
  dvr_cols = [c for c in cols if c == 'diff_vs_rest']
  prop_cols = [c for c in cols if c.endswith('_prop') and c != 'size_prop']
  diff_cols = [c for c in cols if c.endswith('_diff') and c not in dvr_cols]
  p_cols = [c for c in cols if c.endswith('_p') and c not in mw_cols]
  sil_cols = [c for c in cols if c == 'silhouette']
  size_cols = [c for c in cols if c == 'size_prop']
  # Anything else (regression error stats)
  other_cols = [c for c in cols if c not in (
      error_rate_cols + mw_cols + dvr_cols + prop_cols + diff_cols + p_cols + sil_cols + size_cols
  )]

  groups = []
  if error_rate_cols + mw_cols:
    groups.append((error_rate_cols + mw_cols, 'Reds', None, None, '.3g'))
  if dvr_cols:
    groups.append((dvr_cols, 'Blues', None, None, '.3g'))
  if prop_cols:
    groups.append((prop_cols, 'Greens', None, None, '.3g'))
  if diff_cols:
    groups.append((diff_cols, bwb_cmap, -1.0, 1.0, '.3g'))
  if p_cols:
    groups.append((p_cols, 'Greens_r', None, None, '.3g'))
  if sil_cols:
    groups.append((sil_cols, 'Blues', None, None, '.3g'))
  if size_cols:
    groups.append((size_cols, 'Greys', None, None, '.3g'))
  if other_cols:
    groups.append((other_cols, 'vlag', None, None, '.3g'))

  # Filter out empty groups and groups where columns aren't in recap
  groups = [(c, cmap, vmin, vmax, fmt) for c, cmap, vmin, vmax, fmt in groups
             if c and all(col in recap.columns for col in c)]

  if not groups:
    return

  col_counts = [len(g[0]) for g in groups]
  n_rows = len(recap)
  fig_width = max(10, sum(col_counts) * 0.9)
  fig_height = max(4, n_rows * 1.2)

  fig, axes = plt.subplots(1, len(groups), figsize=(fig_width, fig_height),
                           gridspec_kw={'width_ratios': col_counts, 'wspace': 0})
  if len(groups) == 1:
    axes = [axes]

  for ax, (g_cols, cmap, vmin, vmax, fmt) in zip(axes, groups):
    kw = dict(annot=True, fmt=fmt, cbar=False, ax=ax, robust=True)
    if vmin is not None:
      kw.update(vmin=vmin, vmax=vmax, center=0.0)
      kw.pop('robust')
    sns.heatmap(recap[g_cols], cmap=cmap, **kw)
    ax.xaxis.tick_top()
    ax.set(xlabel='', ylabel='')
    ax.tick_params(axis='x', which='major', length=0)
    ax.tick_params(axis='y', which='major', length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

  # Show y-tick labels only on first subplot
  for ax in axes[1:]:
    ax.set_yticklabels([])

  fig.suptitle(re.sub(' +', ' ', cond_name), y=1.01)
  plt.tight_layout()
  plt.savefig(f'{output_dir}/' + re.sub(' +', '', cond_name) + '.png',
              dpi=300, bbox_inches='tight', pad_inches=0)


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
                            ohe_col_names=None):
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
      'binary' or 'regression'. Default 'binary'.

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
                       continuous_sensitive_cols=continuous_sensitive_cols)

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
