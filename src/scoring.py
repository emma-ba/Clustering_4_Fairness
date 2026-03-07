"""
Scoring functions for fairness-aware k-selection.

Provides pluggable scoring functions that can be used during k-search
to optimize cluster assignments for different objectives:
- silhouette: standard cluster quality (cohesion + separation)
- chi2_error: how well clusters separate error types
- chi2_sensitive: how fairly clusters distribute sensitive attributes
- composite: weighted combination of all three
"""

import numpy as np
from typing import Callable, Optional
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency, kruskal

# Type alias: takes (X, labels) -> float, higher = better
ScoringFn = Callable[[np.ndarray, np.ndarray], float]


def silhouette_scorer(X: np.ndarray, labels: np.ndarray) -> float:
    """Standard silhouette score. Higher = better cluster quality."""
    n_clusters = len(set(labels) - {-1})
    if n_clusters < 2:
        return -1.0
    non_noise = labels != -1
    if non_noise.sum() <= n_clusters:
        return -1.0
    return silhouette_score(X[non_noise], labels[non_noise])


def make_chi2_error_scorer(
    error_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> ScoringFn:
    """
    Factory: returns a scorer that measures how well clusters separate errors.

    Builds a contingency table [error, correct] x clusters and returns
    1 - p_value (higher = better error separation).

    Parameters
    ----------
    error_data : np.ndarray
        Binary error column (0/1) for all rows.
    mask : np.ndarray, optional
        Boolean mask applied during clustering (subset filtering).
    """
    data = error_data[mask] if mask is not None else error_data

    def scorer(X: np.ndarray, labels: np.ndarray) -> float:
        n_clusters = len(set(labels) - {-1})
        if n_clusters < 2:
            return 0.0
        non_noise = labels != -1
        err = data[non_noise] if len(data) == len(labels) else data[:len(labels)][non_noise]
        lab = labels[non_noise]
        # Build contingency: rows = [correct, error], cols = clusters
        unique_labels = sorted(set(lab))
        table = np.zeros((2, len(unique_labels)), dtype=int)
        for j, cl in enumerate(unique_labels):
            cl_mask = lab == cl
            table[1, j] = int(err[cl_mask].sum())
            table[0, j] = int(cl_mask.sum()) - table[1, j]
        # Avoid degenerate tables
        if table.sum() == 0 or (table == 0).all(axis=1).any():
            return 0.0
        try:
            _, p, _, _ = chi2_contingency(table)
            return 1.0 - p  # higher = better error separation
        except ValueError:
            return 0.0

    return scorer


def make_kruskal_error_scorer(
    error_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> ScoringFn:
    """
    Factory: returns a scorer that measures how well clusters separate continuous errors.

    Uses Kruskal-Wallis H-test on continuous error values grouped by cluster.
    Returns 1 - p_value (same interface as chi2 scorer: higher = better separation).

    Parameters
    ----------
    error_data : np.ndarray
        Continuous error column (e.g. absolute residuals) for all rows.
    mask : np.ndarray, optional
        Boolean mask applied during clustering (subset filtering).
    """
    data = error_data[mask] if mask is not None else error_data

    def scorer(X: np.ndarray, labels: np.ndarray) -> float:
        n_clusters = len(set(labels) - {-1})
        if n_clusters < 2:
            return 0.0
        non_noise = labels != -1
        err = data[non_noise] if len(data) == len(labels) else data[:len(labels)][non_noise]
        lab = labels[non_noise]
        unique_labels = sorted(set(lab))
        groups = [err[lab == cl] for cl in unique_labels]
        # Need at least 2 non-empty groups
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return 0.0
        try:
            _, p = kruskal(*groups)
            return 1.0 - p  # higher = better error separation
        except ValueError:
            return 0.0

    return scorer


def make_chi2_sensitive_scorer(
    sensitive_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> ScoringFn:
    """
    Factory: returns a scorer that measures cluster fairness w.r.t. a sensitive attribute.

    Returns p_value directly (higher = clusters do NOT separate by sensitive
    attribute = fairer).

    Parameters
    ----------
    sensitive_data : np.ndarray
        Sensitive attribute column for all rows.
    mask : np.ndarray, optional
        Boolean mask applied during clustering (subset filtering).
    """
    data = sensitive_data[mask] if mask is not None else sensitive_data

    def scorer(X: np.ndarray, labels: np.ndarray) -> float:
        n_clusters = len(set(labels) - {-1})
        if n_clusters < 2:
            return 0.0
        non_noise = labels != -1
        sens = data[non_noise] if len(data) == len(labels) else data[:len(labels)][non_noise]
        lab = labels[non_noise]
        unique_vals = sorted(set(sens))
        unique_labels = sorted(set(lab))
        if len(unique_vals) < 2:
            return 1.0  # perfectly fair (no variation)
        table = np.zeros((len(unique_vals), len(unique_labels)), dtype=int)
        for i, v in enumerate(unique_vals):
            for j, cl in enumerate(unique_labels):
                table[i, j] = int(((sens == v) & (lab == cl)).sum())
        if table.sum() == 0 or (table == 0).all(axis=1).any():
            return 0.0
        try:
            _, p, _, _ = chi2_contingency(table)
            return p  # higher p = less separation = fairer
        except ValueError:
            return 0.0

    return scorer


def make_composite_scorer(
    error_data: np.ndarray,
    sensitive_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    silhouette_weight: float = 0.3,
    error_weight: float = 0.5,
    fairness_weight: float = 0.2,
    error_type: str = 'binary',
) -> ScoringFn:
    """
    Factory: weighted combination of silhouette, error separation, and fairness.

    All components are normalized to [0, 1] before weighting.
    Higher composite score = better.

    Parameters
    ----------
    error_data : np.ndarray
        Binary error column (0/1) or continuous error column (regression).
    sensitive_data : np.ndarray
        Sensitive attribute column.
    mask : np.ndarray, optional
        Boolean mask from subset filtering.
    silhouette_weight : float
        Weight for silhouette component (default 0.3).
    error_weight : float
        Weight for error separation component (default 0.5).
    fairness_weight : float
        Weight for fairness component (default 0.2).
    error_type : str
        'binary' for chi2-based error scorer, 'regression' for Kruskal-Wallis.
    """
    if error_type == 'regression':
        error_scorer = make_kruskal_error_scorer(error_data, mask)
    else:
        error_scorer = make_chi2_error_scorer(error_data, mask)
    sensitive_scorer = make_chi2_sensitive_scorer(sensitive_data, mask)

    def scorer(X: np.ndarray, labels: np.ndarray) -> float:
        # Silhouette: range [-1, 1] -> normalize to [0, 1]
        sil = silhouette_scorer(X, labels)
        sil_norm = (sil + 1.0) / 2.0

        # Error separation: already [0, 1]
        err_score = error_scorer(X, labels)

        # Fairness: already [0, 1]
        fair_score = sensitive_scorer(X, labels)

        return (silhouette_weight * sil_norm +
                error_weight * err_score +
                fairness_weight * fair_score)

    return scorer
