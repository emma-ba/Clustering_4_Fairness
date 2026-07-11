"""
Fairness clustering package.

Modules:
- clustering: Main clustering function with support for multiple algorithms and distances
- scoring: Scoring functions for fairness-aware k-selection
- visualization: Plotting functions for cluster analysis
- fairness_metrics: Metrics for evaluating demographic representation in clusters
- experiments: Experiment utilities for batch clustering and result analysis
"""

from .clustering import cluster, ClusteringResult, gower_distance
from .scoring import (
    ScoringFn,
    silhouette_scorer,
    make_chi2_error_scorer,
    make_kruskal_error_scorer,
    make_chi2_sensitive_scorer,
    make_composite_scorer,
)
from .visualization import (
    plot_clusters,
    plot_clusters_by_attribute,
    plot_cluster_composition,
    visualize_clustering_result,
    reduce_dimensions,
    plot_silhouette_heatmap,
    plot_quality_metrics_heatmap,
)
from .fairness_metrics import (
    one_vs_all_p_continuous,
    mean_diff,
    size_metrics,
    extreme_pair_gap_p,
    omnibus_separability_p,
    error_sep_p,
    fisher_rxc_p,
)
from .experiments import (
    make_recap,
    make_chi_tests,
    recap_quali_metrics,
    run_experiments_generic,
    create_exp_conditions,
    separability_check,
)

__all__ = [
    # clustering
    "cluster",
    "ClusteringResult",
    "gower_distance",
    # scoring
    "ScoringFn",
    "silhouette_scorer",
    "make_chi2_error_scorer",
    "make_kruskal_error_scorer",
    "make_chi2_sensitive_scorer",
    "make_composite_scorer",
    # visualization
    "plot_clusters",
    "plot_clusters_by_attribute",
    "plot_cluster_composition",
    "visualize_clustering_result",
    "reduce_dimensions",
    "plot_silhouette_heatmap",
    "plot_quality_metrics_heatmap",
    # fairness_metrics
    "one_vs_all_p_continuous",
    "mean_diff",
    "size_metrics",
    "extreme_pair_gap_p",
    "omnibus_separability_p",
    "error_sep_p",
    "fisher_rxc_p",
    # experiments
    "make_recap",
    "make_chi_tests",
    "recap_quali_metrics",
    "run_experiments_generic",
    "create_exp_conditions",
    "separability_check",
]
