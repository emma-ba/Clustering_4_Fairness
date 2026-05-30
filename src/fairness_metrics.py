"""
Fairness metrics for clustering analysis.
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats
from scipy.stats import mannwhitneyu


@dataclass
class FairnessMetrics:
    """Container for fairness evaluation metrics."""

    demographic_parity: dict
    representation_ratio: dict
    balance_score: float
    entropy_per_cluster: dict
    overall_entropy: float


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
