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
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import silhouette_samples
from scipy import stats
from scipy.stats import chi2_contingency


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
      # Multi-class: create per-value binary indicators
      for val in unique_vals:
        indicator_name = f'{col}={val}'
        data[indicator_name] = (data[col] == val).astype(int)
        expanded_cols.append(indicator_name)
  return data, expanded_cols


def make_recap(data_result, feature_set, sensitive_cols=None, error_col='errors', error_type='binary'):
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

  # Expand multi-class sensitive columns into binary indicators
  data_result, sensitive_cols_expanded = _expand_multiclass_cols(data_result, sensitive_cols)

  # MAKE RECAP of cluster info
  # ...with error rates
  res = data_result[['clusters', error_col]]

  # ...with cluster size
  temp = data_result[['clusters']].copy()
  temp['count'] = 1
  recap = temp.groupby(['clusters'], as_index=False).sum()

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

  silhouette = []

  # Get individual silhouette scores
  clusters = data_result['clusters']
  if(len(recap['clusters'].unique()) > 1):
    silhouette_val = silhouette_samples(data_result[feature_set], clusters)

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
      silhouette.append(np.nan)
      break

    # Add silhouette score
    silhouette.append(silhouette_val[clusters == c].mean())

    rest_recap = recap.loc[recap['clusters'] != c]
    rest_count = rest_recap['count'].sum()

    #### Quick test: differences in error rates
    if error_type == 'regression':
      # Regression: diff of means + Mann-Whitney U test
      c_errors = c_data[error_col].values
      rest_errors = rest_data[error_col].values
      diff_vs_rest.append(round(c_errors.mean() - rest_errors.mean(), 6))
      try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(c_errors, rest_errors, alternative='two-sided')
        diff_p.append(round(p, 3))
      except ValueError:
        diff_p.append(np.nan)
    else:
      # Binary: Get error rate difference 1-vs-rest
      rest_n_error = rest_recap['n_error'].sum()
      rest_rate = rest_n_error / rest_count
      diff_vs_rest.append(recap['error_rate'][c] - rest_rate)

      # ...with Poisson stat test
      # Deal with splits with 0 error
      if((recap['n_error'][c] < 1) | (recap['count'][c] < 1) | (rest_n_error < 1) | (rest_count < 1)):
        res = stats.poisson_means_test(recap['count'][c] - recap['n_error'][c], recap['count'][c], rest_count - rest_n_error, rest_count)
        diff_p.append(round(res.pvalue, 3))
      else:
        res = stats.poisson_means_test(recap['n_error'][c], recap['count'][c], rest_n_error, rest_count)
        diff_p.append(round(res.pvalue, 3))

    ##### Sensitive features — dynamic loop (uses expanded columns)
    for col in sensitive_cols_expanded:
      rest_n = rest_data[col].sum()
      rest_prop = rest_n / rest_count

      c_n = c_data[col].sum()
      c_prop = c_n / c_count

      sensitive_data[col]['prop'].append(c_prop)
      sensitive_data[col]['diff'].append(c_prop - rest_prop)

      # Poisson means test (handle zero counts)
      if((c_n < 1) | (c_count < 1) | (rest_n < 1) | (rest_count < 1)):
        res = stats.poisson_means_test(c_count - c_n, c_count, rest_count - rest_n, rest_count)
        sensitive_data[col]['p'].append(round(res.pvalue, 3))
      else:
        res = stats.poisson_means_test(c_n, c_count, rest_n, rest_count)
        sensitive_data[col]['p'].append(round(res.pvalue, 3))

  recap['diff_vs_rest'] = np.around(diff_vs_rest, 3)
  recap['diff_p'] = diff_p

  for col in sensitive_cols_expanded:
    recap[f'{col}_prop'] = np.around(sensitive_data[col]['prop'], 3)
    recap[f'{col}_diff'] = np.around(sensitive_data[col]['diff'], 3)
    recap[f'{col}_p'] = sensitive_data[col]['p']

  recap['silhouette'] = silhouette

  if error_type == 'regression':
    recap['error_mean'] = np.around(recap['error_mean'], 3)
    recap['error_std'] = np.around(recap['error_std'], 3)
    recap['error_median'] = np.around(recap['error_median'], 3)
    recap['abs_error_mean'] = np.around(recap['abs_error_mean'], 3)
    recap['abs_error_median'] = np.around(recap['abs_error_median'], 3)
  else:
    recap['error_rate'] = np.around(recap['error_rate'] , 3)

  recap.rename(columns={'clusters':'c'}, inplace=True)

  return(recap)


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
    from scipy.stats import kruskal, mannwhitneyu

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


def make_chi_tests(results, sensitive_cols=None, error_type='binary', error_col='errors'):
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
  """
  if sensitive_cols is None:
    sensitive_cols = []

  # Determine actual columns from first recap
  if len(results['cond_recap']) > 0:
    actual_sensitive = _get_sensitive_cols_from_recap(results['cond_recap'][0], sensitive_cols)
  else:
    actual_sensitive = sensitive_cols

  chi_res = {'cond_descr': [],
            'cond_name': [],
            'error': []}
  for col in actual_sensitive:
    chi_res[col] = []

  for i in range(0, len(results['cond_name'])):
    chi_res['cond_descr'].append(results['cond_descr'][i])
    chi_res['cond_name'].append(results['cond_name'][i])
    recap = results['cond_recap'][i]

    if(len(recap['diff_p']) == 1):
      chi_res['error'].append(np.nan)
      for col in actual_sensitive:
        chi_res[col].append(np.nan)
      continue

    # Test error differences
    if error_type == 'regression':
      # Kruskal-Wallis on raw continuous error values grouped by cluster
      from scipy.stats import kruskal as kruskal_test
      res_df = results['cond_res'][i]
      cluster_labels = res_df['clusters'].values
      unique_clusters = sorted(set(cluster_labels) - {-1})
      groups = [res_df.loc[res_df['clusters'] == cl, error_col].values for cl in unique_clusters]
      groups = [g for g in groups if len(g) > 0]
      if len(groups) >= 2:
        try:
          _, p = kruskal_test(*groups)
          chi_res['error'].append(round(p, 6))
        except ValueError:
          chi_res['error'].append(np.nan)
      else:
        chi_res['error'].append(np.nan)
    else:
      # Binary: chi-squared on [n_correct, n_error] contingency table
      test_data = recap[['count', 'n_error']].copy(deep=True)
      test_data['count'] = test_data['count'] - test_data['n_error']
      test_data = test_data.rename(columns={"count": "n_correct"})
      test_data = test_data.transpose()
      test_res = chi2_contingency(test_data)
      chi_res['error'].append(round(test_res.pvalue, 6))

    # Test each sensitive column (binary: 2x2 table, multi-class indicators: 2xN each)
    for col in actual_sensitive:
      prop_col = f'{col}_prop'
      test_data = recap[['count', prop_col]].copy(deep=True)
      test_data[prop_col] = round(test_data['count'] * test_data[prop_col])
      test_data = test_data.rename(columns={prop_col: f'{col}_n'}).astype(int)
      test_data['count'] = test_data['count'] - test_data[f'{col}_n']
      test_data = test_data.rename(columns={"count": f'not_{col}_n'})
      test_data = test_data.transpose()
      test_res = chi2_contingency(test_data)
      chi_res[col].append(round(test_res.pvalue, 6))

  return(pd.DataFrame(chi_res))


# =============================================================================
# Utils for Results - All Quality Metrics
# =============================================================================

def recap_quali_metrics(chi_res, results, exp_condition, sensitive_cols=None):
  """
  Combine chi-squared results with silhouette scores.

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
  """
  # Use whatever sensitive columns are actually in chi_res
  # (may be expanded multi-class names like 'race=0', 'race=1')
  skip_cols = {'cond_descr', 'cond_name', 'error'}
  actual_sensitive = [c for c in chi_res.columns if c not in skip_cols]

  all_quali = {'cond_descr': chi_res['cond_descr'],
            'cond_name': chi_res['cond_name'],
            'error': chi_res['error']}
  for col in actual_sensitive:
    all_quali[col] = chi_res[col]
  all_quali['silhouette'] = []

  for i in range(0, len(chi_res['cond_name'])):
    data = results['cond_res'][i]
    feature_set = exp_condition['feature_set'][i]
    clusters = data['clusters']
    recap = results['cond_recap'][i]
    if(len(recap['diff_p']) == 1):
      all_quali['silhouette'].append(np.nan)
      continue
    silhouette_indiv = silhouette_samples(data[feature_set], clusters)
    silhouette_avg = silhouette_indiv.mean()
    all_quali['silhouette'].append(silhouette_avg)

  return(pd.DataFrame(all_quali))


# =============================================================================
# Visualization
# =============================================================================

def plot_quality_heatmap(all_quali_viz, output_path, figsize=(4,4)):
  """
  Plot quality metrics heatmap.

  For silhouette, higher=better, so color should be inverse (blue instead of red).
  """
  plt.figure(figsize=figsize)
  ax = sns.heatmap(all_quali_viz, annot=True, center=0, cbar=False,
                  cmap=sns.color_palette("vlag", as_cmap=True), robust=True)
  ax.set(xlabel="", ylabel="")
  ax.xaxis.tick_top()
  ax.tick_params(axis='x', which='major', length=0)
  ax.tick_params(axis='y', which='major', pad=150, length=0)
  plt.yticks(ha='left')
  plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)


def plot_cluster_recap_heatmap(recap, cond_name, output_dir):
  """Plot one-vs-all cluster comparison heatmap."""
  recap = recap.sort_values(by=['diff_vs_rest'], ascending=False)
  recap['count'] = recap['count']/recap['count'].sum()
  recap = recap.rename(columns={"count": "size_prop"})
  drop_cols = ['c']
  if 'n_error' in recap.columns:
    drop_cols.append('n_error')
  recap = recap.drop(drop_cols, axis=1)

  n_cols = len(recap.columns)
  n_rows = len(recap)
  fig_width = max(10, n_cols * 0.9)
  fig_height = max(4, n_rows * 1.2)
  plt.figure(figsize=(fig_width, fig_height))
  ax = sns.heatmap(recap, annot=True, fmt='.3g', center=0, cbar=False,
                   cmap=sns.color_palette("vlag", as_cmap=True), robust=True)
  ax.set_title(re.sub(' +', ' ', cond_name))
  ax.xaxis.tick_top()
  ax.set(xlabel="", ylabel="")
  ax.tick_params(axis='x', which='major', length=0)
  ax.tick_params(axis='y', which='major', length=0)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
  plt.yticks(rotation='horizontal')
  plt.savefig(f'{output_dir}/'+re.sub(' +', '', cond_name)+'.png', dpi=300, bbox_inches='tight', pad_inches=0)


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiments_generic(data, exp_condition, algorithm, distance,
                            n_clusters=None, n_min=None, n_max=None,
                            max_iter=300, seed=42,
                            scoring_fn=None, sensitive_cols=None, error_col='errors',
                            min_cluster_size=15, min_samples=5, eps=0.5,
                            min_datapoints=None, feature_weights=None,
                            error_type='binary'):
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
  from .clustering import cluster

  np.random.seed(seed)

  results = {'cond_name': [],
            'cond_descr': [],
            'cond_res': [],
            'cond_recap': []}

  for i in range(len(exp_condition)):
    feature_set = exp_condition['feature_set'][i]

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
    )

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
                       error_type=error_type)

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
  from itertools import combinations

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
