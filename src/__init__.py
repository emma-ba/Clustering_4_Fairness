"""
Fairness clustering package.

Modules:
- clustering: Main clustering function with support for multiple algorithms and distances
- visualization: Plotting functions for cluster analysis
- fairness_metrics: Metrics for evaluating demographic representation in clusters
- experiments: Experiment utilities for HBAC clustering and result analysis
"""

from .clustering import cluster, ClusteringResult, gower_distance
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
    evaluate_fairness,
    FairnessMetrics,
    compute_demographic_parity,
    compute_representation_ratio,
    compute_balance_score,
    print_fairness_report,
)
from .experiments import (
    hbac_dbscan,
    make_recap,
    make_chi_tests,
    recap_quali_metrics,
    run_experiments,
    run_experiments_multiple_seeds,
    create_default_exp_conditions,
    get_error_rate,
    subset_TP_FN,
    subset_TN_FP,
)

__all__ = [
    # clustering
    "cluster",
    "ClusteringResult",
    "gower_distance",
    # visualization
    "plot_clusters",
    "plot_clusters_by_attribute",
    "plot_cluster_composition",
    "visualize_clustering_result",
    "reduce_dimensions",
    "plot_silhouette_heatmap",
    "plot_quality_metrics_heatmap",
    # fairness_metrics
    "evaluate_fairness",
    "FairnessMetrics",
    "compute_demographic_parity",
    "compute_representation_ratio",
    "compute_balance_score",
    "print_fairness_report",
    # experiments
    "hbac_dbscan",
    "make_recap",
    "make_chi_tests",
    "recap_quali_metrics",
    "run_experiments",
    "run_experiments_multiple_seeds",
    "create_default_exp_conditions",
    "get_error_rate",
    "subset_TP_FN",
    "subset_TN_FP",
]
