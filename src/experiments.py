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
from .fairness_metrics import cluster_proportion, one_vs_all_p_binary, one_vs_all_p_continuous, mean_diff


# Per-chunk RAM budget (MiB) for silhouette pairwise-distance computation.
# Silhouette on large datasets streams distances in chunks of shape (chunk_rows, n).
# Lower this if you hit MemoryError; raise it on machines with more RAM for fewer,
# larger chunks. Result is bit-identical regardless of value.
SILHOUETTE_WORKING_MEMORY_MIB = 128


# =============================================================================
# Utils for Results - Recap
# =============================================================================

def _expand_multiclass_cols(data, sensitive_cols):
  """
  Expand non-binary sensitive columns into per-value binary indicators.

  For columns with more than 2 unique values, creates binary columns
  named '{col}={value}' for each unique value. Binary columns are kept as-is.

  Returns (data_copy, expanded_cols) where expanded_cols replaces the
  original multi-class columns with their indicator names.
  """
  data = data.copy()
  expanded_cols = []
  for col in sensitive_cols:
    unique_vals = sorted(data[col].dropna().unique())
    if len(unique_vals) <= 2:
      # Binary or single-value column — keep as-is
      expanded_cols.append(col)
    else:
      print(f"  Expanded sensitive col '{col}' into {len(unique_vals)} categorical indicators.")
      # Multi-class: create per-value binary indicators
      indicator_names = []
      for val in unique_vals:
        indicator_name = f'{col}={val}'
        data[indicator_name] = (data[col] == val).astype(int)
        expanded_cols.append(indicator_name)
        indicator_names.append(indicator_name)
      # OOD check: each row should belong to at most one category
      row_sums = data[indicator_names].sum(axis=1)
      if (row_sums > 1).any():
        raise ValueError(
          f"Sensitive column '{col}' has rows assigned to multiple categories "
          f"after expansion. Check for overlapping values in the data."
        )
      n_nan = (row_sums == 0).sum()
      if n_nan > 0:
        print(f"  Warning: {n_nan} rows with missing value in '{col}' — assigned 0 for all indicators.")
  return data, expanded_cols


def make_recap(data_result, feature_set, sensitive_cols=None, error_col='errors', error_type='binary', feature_matrix=None, distance_matrix=None, original_sensitive_cols=None, error_label=None, continuous_sensitive_cols=None):
  """
  Create recap of cluster info with error rates and sensitive feature proportions.

  Parameters
  ----------
  data_result : pd.DataFrame
      Clustered data with 'clusters' column.
  feature_set : list
      Feature columns used for clustering (for silhouette computation).
  sensitive_cols : list, optional
      Sensitive columns to compute proportions for. Both binary (0/1) and
      multi-class columns are supported. Multi-class columns are auto-expanded
      into per-value binary indicators.
  error_col : str
      Name of the error column. Default 'errors'.
  error_type : str
      'binary' for classification errors (0/1), 'regression' for continuous errors.
  """
  if sensitive_cols is None:
    sensitive_cols = []
  continuous_sensitive_cols = set(continuous_sensitive_cols or [])

  _auto_continuous = []
  for col in sensitive_cols:
    if col in continuous_sensitive_cols:
      continue
    if col in data_result.columns and data_result[col].dtype.kind == 'f' and data_result[col].nunique() > 2:
      continuous_sensitive_cols.add(col)
      _auto_continuous.append(col)
  if _auto_continuous:
    print(f"  Auto-detected as continuous (float dtype): {', '.join(_auto_continuous)}")

  numeric_sensitive_cols = [c for c in sensitive_cols if c in continuous_sensitive_cols]
  categorical_sensitive_cols = [c for c in sensitive_cols if c not in continuous_sensitive_cols]

  # Exclude noise points (cluster label -1 from DBSCAN/HDBSCAN) before any computation
  noise_mask = data_result['clusters'] != -1
  data_result = data_result[noise_mask].copy()
  if feature_matrix is not None:
    feature_matrix = feature_matrix[noise_mask.values]

  # Expand multi-class sensitive columns into binary indicators
  data_result, sensitive_cols_expanded = _expand_multiclass_cols(data_result, categorical_sensitive_cols)

  # MAKE RECAP of cluster info
  # ...with error rates
  if error_col not in data_result.columns:
    raise ValueError(f"error_col '{error_col}' not found in data. Available columns: {list(data_result.columns)}")
  res = data_result[['clusters', error_col]]

  # ...with cluster size
  temp = data_result[['clusters']].copy()
  temp['count'] = 1
  recap = temp.groupby(['clusters'], as_index=False).sum()
  recap = recap.set_index('clusters', drop=False)

  if error_type == 'regression':
    # Regression path: signed error stats (bias direction)
    recap['error_mean'] = res.groupby(['clusters'])[error_col].mean().values
    recap['error_std'] = res.groupby(['clusters'])[error_col].std().values
    recap['error_median'] = res.groupby(['clusters'])[error_col].median().values
    # Absolute error stats (accuracy magnitude)
    recap['abs_error_mean'] = res.groupby(['clusters'])[error_col].apply(lambda x: x.abs().mean()).values
    recap['abs_error_median'] = res.groupby(['clusters'])[error_col].apply(lambda x: x.abs().median()).values
  else:
    # Binary path: count-based error stats
    # ...with number of error
    recap['n_error'] = res.groupby(['clusters']).sum().astype(int)

    # ...with 1-vs-All error diff
    recap['error_rate'] = res.groupby(['clusters']).mean()

  # Prepare Quality metrics
  diff_vs_rest = []
  diff_p = []

  # Dynamic sensitive column tracking (using expanded columns for multi-class support)
  sensitive_data = {col: {'prop': [], 'diff': [], 'p': []} for col in sensitive_cols_expanded}
  numeric_data = {col: {'avg': [], 'avg_delta': [], 'p': []} for col in numeric_sensitive_cols}

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

    # Get out-of-cluster data
    rest_data = data_result.loc[data_result['clusters'] != c]
    # Check if no other cluster
    if(len(rest_data) == 0):
      diff_vs_rest.append(np.nan)
      diff_p.append(np.nan)
      for col in sensitive_cols_expanded:
        sensitive_data[col]['prop'].append(np.nan)
        sensitive_data[col]['diff'].append(np.nan)
        sensitive_data[col]['p'].append(np.nan)
      for col in numeric_sensitive_cols:
        numeric_data[col]['avg'].append(np.nan)
        numeric_data[col]['avg_delta'].append(np.nan)
        numeric_data[col]['p'].append(np.nan)
      silhouette.append(np.nan)
      break

    # Add silhouette score
    silhouette.append(silhouette_val[clusters == c].mean())

    rest_recap = recap.loc[recap['clusters'] != c]
    rest_count = rest_recap['count'].sum()

    # Error — one-vs-all diff and p-value
    if error_type == 'regression':
      c_errors = c_data[error_col].values
      rest_errors = rest_data[error_col].values
      diff_vs_rest.append(round(mean_diff(c_errors, rest_errors), 6))
      diff_p.append(one_vs_all_p_continuous(c_errors, rest_errors))
    else:
      rest_n_error = rest_recap['n_error'].sum()
      diff_vs_rest.append(recap['error_rate'][c] - rest_n_error / rest_count)
      diff_p.append(one_vs_all_p_binary(
          recap['n_error'][c], recap['count'][c], rest_n_error, rest_count
      ))

    # Binary sensitive features — one-vs-all proportion, diff, p-value
    for col in sensitive_cols_expanded:
      c_n = c_data[col].sum()
      rest_n = rest_data[col].sum()
      c_prop = cluster_proportion(c_n, c_count)
      rest_prop = cluster_proportion(rest_n, rest_count)
      sensitive_data[col]['prop'].append(round(c_prop, 4))
      sensitive_data[col]['diff'].append(round(c_prop - rest_prop, 4))
      sensitive_data[col]['p'].append(one_vs_all_p_binary(c_n, c_count, rest_n, rest_count))

    # Numeric sensitive features — one-vs-all mean, diff, p-value
    for col in numeric_sensitive_cols:
      c_vals = c_data[col].dropna().values
      rest_vals = rest_data[col].dropna().values
      if len(c_vals) == 0 or len(rest_vals) == 0:
        numeric_data[col]['avg'].append(np.nan)
        numeric_data[col]['avg_delta'].append(np.nan)
        numeric_data[col]['p'].append(np.nan)
        continue
      numeric_data[col]['avg'].append(round(float(c_vals.mean()), 4))
      numeric_data[col]['avg_delta'].append(round(mean_diff(c_vals, rest_vals), 4))
      numeric_data[col]['p'].append(one_vs_all_p_continuous(c_vals, rest_vals))

  # Collect all new columns into a dict then concat once to avoid fragmentation
  new_cols = {
      'diff_vs_rest': np.around(diff_vs_rest, 3),
      'mannwhitney_p': diff_p,
  }

  for col in sensitive_cols_expanded:
    new_cols[f'{col}_prop'] = sensitive_data[col]['prop']
    new_cols[f'{col}_diff'] = sensitive_data[col]['diff']
    new_cols[f'{col}_p'] = sensitive_data[col]['p']

  for col in numeric_sensitive_cols:
    new_cols[f'{col}_avg'] = numeric_data[col]['avg']
    new_cols[f'{col}_avg_delta'] = numeric_data[col]['avg_delta']
    new_cols[f'{col}_p'] = numeric_data[col]['p']

  new_cols['silhouette'] = silhouette

  recap = pd.concat([recap, pd.DataFrame(new_cols, index=recap.index)], axis=1)

  if error_type == 'regression':
    recap['error_mean'] = np.around(recap['error_mean'], 3)
    recap['error_std'] = np.around(recap['error_std'], 3)
    recap['error_median'] = np.around(recap['error_median'], 3)
    recap['abs_error_mean'] = np.around(recap['abs_error_mean'], 3)
    recap['abs_error_median'] = np.around(recap['abs_error_median'], 3)
  else:
    recap['error_rate'] = np.around(recap['error_rate'], 3)

  recap = recap.reset_index(drop=True)
  recap.rename(columns={'clusters': 'c'}, inplace=True)

  # Rename error columns after all processing is done
  if error_label is not None:
    rename_map = {}
    for src in ['error_rate', 'error_mean', 'error_std', 'error_median',
                'abs_error_mean', 'abs_error_median']:
      if src in recap.columns:
        new_name = src.replace('error', error_label, 1)
        rename_map[src] = new_name
    if rename_map:
      recap = recap.rename(columns=rename_map)

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

def _get_sensitive_cols_from_recap(recap, sensitive_cols):
  """
  Determine the actual sensitive column names present in the recap.

  If make_recap expanded multi-class columns (e.g., 'race' -> 'race=0', 'race=1'),
  find those expanded names. Otherwise return the original column names.
  """
  actual_cols = []
  for col in sensitive_cols:
    if f'{col}_prop' in recap.columns:
      actual_cols.append(col)
    else:
      # Look for expanded multi-class indicators (col=value pattern)
      expanded = [c.replace('_prop', '') for c in recap.columns
                  if c.startswith(f'{col}=') and c.endswith('_prop')]
      actual_cols.extend(expanded)
  return actual_cols


def make_chi_tests(results, sensitive_cols=None, error_type='binary', error_col='errors', error_label=None, continuous_sensitive_cols=None):
  """
  Run chi-squared / Kruskal-Wallis tests on cluster recaps for error and sensitive columns.

  Supports both binary and multi-class sensitive columns. For multi-class
  columns that were expanded by make_recap(), builds a full multi-row
  contingency table across all values.

  For regression errors, uses Kruskal-Wallis H-test on raw error values
  instead of chi-squared on contingency tables.

  Parameters
  ----------
  results : dict
      Results from run_experiments_generic().
  sensitive_cols : list, optional
      Original sensitive column names.
  error_type : str
      'binary' for chi-squared on error counts, 'regression' for Kruskal-Wallis on raw errors.
  error_col : str
      Name of the error column in the data. Used for regression path.
  error_label : str, optional
      Display name for the error column in output tables. Defaults to 'error'.
  """
  if sensitive_cols is None:
    sensitive_cols = []
  if error_label is None:
    error_label = 'error'
  continuous_sensitive_cols = set(continuous_sensitive_cols or [])

  categorical_input = [c for c in sensitive_cols if c not in continuous_sensitive_cols]
  numeric_input = [c for c in sensitive_cols if c in continuous_sensitive_cols]

  # Determine actual categorical columns from first recap (handles multi-class expansion)
  if len(results['cond_recap']) > 0:
    actual_categorical = _get_sensitive_cols_from_recap(results['cond_recap'][0], categorical_input)
  else:
    actual_categorical = categorical_input
  actual_sensitive = actual_categorical + numeric_input

  chi_res = {'cond_descr': [],
            'cond_name': [],
            error_label: []}
  for col in actual_sensitive:
    chi_res[col] = []

  for i in range(0, len(results['cond_name'])):
    chi_res['cond_descr'].append(results['cond_descr'][i])
    chi_res['cond_name'].append(results['cond_name'][i])
    recap = results['cond_recap'][i]

    if(len(recap['mannwhitney_p']) == 1):
      chi_res[error_label].append(np.nan)
      for col in actual_sensitive:
        chi_res[col].append(np.nan)
      continue

    # Test error differences
    if error_type == 'regression':
      # Kruskal-Wallis on raw continuous error values grouped by cluster
      res_df = results['cond_res'][i]
      cluster_labels = res_df['clusters'].values
      unique_clusters = sorted(set(cluster_labels) - {-1})
      groups = [res_df.loc[res_df['clusters'] == cl, error_col].values for cl in unique_clusters]
      groups = [g for g in groups if len(g) > 0]
      if len(groups) >= 2:
        try:
          _, p = kruskal(*groups)
          chi_res[error_label].append(round(p, 6))
        except ValueError:
          chi_res[error_label].append(np.nan)
      else:
        chi_res[error_label].append(np.nan)
    else:
      # Binary: chi-squared on [n_correct, n_error] contingency table
      test_data = recap[['count', 'n_error']].copy(deep=True)
      test_data['count'] = test_data['count'] - test_data['n_error']
      test_data = test_data.rename(columns={"count": "n_correct"})
      test_data = test_data.transpose()
      
      if (test_data.sum(axis=1) == 0).any():
        # Zero row in contingency table — fall back to Kruskal-Wallis on raw error values
        res_df = results['cond_res'][i]
        if error_col not in res_df.columns:
          chi_res[error_label].append(np.nan)
          for col in actual_sensitive:
            chi_res[col].append(np.nan)
          continue
        cluster_labels = res_df['clusters'].values
        unique_clusters = sorted(set(cluster_labels) - {-1})
        groups = [res_df.loc[res_df['clusters'] == cl, error_col].values for cl in unique_clusters]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
          try:
            _, p = kruskal(*groups)
            chi_res[error_label].append(round(p, 6))
          except ValueError:
            chi_res[error_label].append(np.nan)
        else:
          chi_res[error_label].append(np.nan)
      else:
        test_res = chi2_contingency(test_data)
        chi_res[error_label].append(round(test_res.pvalue, 6))

    numeric_set = set(numeric_input)
    res_df = results['cond_res'][i]
    for col in actual_sensitive:
      if col in numeric_set:
        if col not in res_df.columns:
          chi_res[col].append(np.nan)
          continue
        cluster_labels = res_df['clusters'].values
        unique_clusters = sorted(set(cluster_labels) - {-1})
        groups = [res_df.loc[res_df['clusters'] == cl, col].dropna().values
                  for cl in unique_clusters]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
          try:
            _, p = kruskal(*groups)
            chi_res[col].append(round(float(p), 6))
          except ValueError:
            chi_res[col].append(np.nan)
        else:
          chi_res[col].append(np.nan)
        continue

      n_col = f'{col}_n'
      if n_col not in recap.columns:
        chi_res[col].append(np.nan)
        continue
      test_data = recap[['count', n_col]].copy(deep=True).astype(int)
      test_data['count'] = test_data['count'] - test_data[n_col]
      test_data = test_data.rename(columns={'count': f'not_{col}_n'})
      test_data = test_data.transpose()
      if (test_data.sum(axis=1) == 0).any():
        chi_res[col].append(np.nan)
      else:
        test_res = chi2_contingency(test_data)
        chi_res[col].append(round(test_res.pvalue, 6))

  chi_df = pd.DataFrame(chi_res)

  if actual_sensitive and len(chi_df) > 0:
    for i in chi_df.index:
      row_p = chi_df.loc[i, actual_sensitive].values.astype(float)
      valid_mask = ~np.isnan(row_p)
      if valid_mask.sum() > 1:
        corrected = false_discovery_control(row_p[valid_mask], method='bh')
        row_p[valid_mask] = np.round(corrected, 6)
        chi_df.loc[i, actual_sensitive] = row_p

  return chi_df


# =============================================================================
# Utils for Results - All Quality Metrics
# =============================================================================

def recap_quali_metrics(chi_res, results, exp_condition, sensitive_cols=None, original_sensitive_cols=None, error_label=None, continuous_sensitive_cols=None):
  """
  Combine chi-squared results with silhouette scores, entropy, and balance scores.

  Parameters
  ----------
  chi_res : pd.DataFrame
      Chi-squared test results.
  results : dict
      Results from run_experiments_generic().
  exp_condition : pd.DataFrame
      Experimental conditions.
  sensitive_cols : list, optional
      Original sensitive column names. Actual columns used are inferred from
      chi_res (which may contain expanded multi-class indicator names).
  original_sensitive_cols : list, optional
      Pre-encoding sensitive column names.
  error_label : str, optional
      Display name for the error column (used as key in chi_res).
  """
  if error_label is None:
    error_label = 'error'
  continuous_sensitive_cols = set(continuous_sensitive_cols or [])

  skip_cols = {'cond_descr', 'cond_name', error_label}
  actual_sensitive = [c for c in chi_res.columns if c not in skip_cols]

  all_quali = {'cond_descr': chi_res['cond_descr'],
               'cond_name': chi_res['cond_name'],
               error_label: chi_res[error_label]}
  for col in actual_sensitive:
    all_quali[col] = chi_res[col]
  all_quali['silhouette'] = []

  orig_cols = original_sensitive_cols or sensitive_cols or []
  categorical_orig = [c for c in orig_cols if c not in continuous_sensitive_cols]
  numeric_orig = [c for c in orig_cols if c in continuous_sensitive_cols]
  for col in numeric_orig:
    all_quali[f'{col}_avg_range'] = []

  for i in range(0, len(chi_res['cond_name'])):
    recap = results['cond_recap'][i]
    res_df = results['cond_res'][i]
    single_cluster = len(recap['mannwhitney_p']) == 1

    if single_cluster:
      all_quali['silhouette'].append(np.nan)
      for col in numeric_orig:
        all_quali[f'{col}_avg_range'].append(np.nan)
      continue

    all_quali['silhouette'].append(recap['silhouette'].mean())

    labels = res_df['clusters'].values
    for col in numeric_orig:
      avg_col = f'{col}_avg'
      if avg_col in recap.columns:
        vals = recap[avg_col].dropna().values
        if len(vals) >= 1:
          all_quali[f'{col}_avg_range'].append(round(float(vals.max() - vals.min()), 4))
        else:
          all_quali[f'{col}_avg_range'].append(np.nan)
      else:
        all_quali[f'{col}_avg_range'].append(np.nan)


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
