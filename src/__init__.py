"""
Fairness clustering package.

Modules:
- clustering: Main clustering function with support for multiple algorithms and distances
- visualization: Plotting functions for cluster analysis
- fairness_metrics: Metrics for evaluating demographic representation in clusters
"""

from .clustering import cluster, ClusteringResult, gower_distance
from .visualization import (
    plot_clusters,
    plot_clusters_by_attribute,
    plot_cluster_composition,
    visualize_clustering_result,
    reduce_dimensions,
)
from .fairness_metrics import (
    evaluate_fairness,
    FairnessMetrics,
    compute_demographic_parity,
    compute_representation_ratio,
    compute_balance_score,
    print_fairness_report,
)

__all__ = [
    "cluster",
    "ClusteringResult",
    "gower_distance",
    "plot_clusters",
    "plot_clusters_by_attribute",
    "plot_cluster_composition",
    "visualize_clustering_result",
    "reduce_dimensions",
    "evaluate_fairness",
    "FairnessMetrics",
    "compute_demographic_parity",
    "compute_representation_ratio",
    "compute_balance_score",
    "print_fairness_report",
]
