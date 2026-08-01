"""Fairness clustering package. See the README for what each module does."""

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
    plot_cluster_composition,
    reduce_dimensions,
)
from .fairness_metrics import (
    one_vs_all_p_continuous,
    mean_diff,
    size_metrics,
    extreme_pair_gap_p,
    omnibus_separability_p,
    error_sep_p,
    fisher_rxc_p,
    binary_error_rate_column,
)
from .experiments import (
    make_recap,
    make_chi_tests,
    recap_quali_metrics,
    run_experiments_generic,
    create_exp_conditions,
    separability_check,
)
from .preprocessing import encode_categoricals
from .cli import build_sensitive_analysis_list, apply_salient_reconstruction

__all__ = [
    "cluster",
    "ClusteringResult",
    "gower_distance",
    "ScoringFn",
    "silhouette_scorer",
    "make_chi2_error_scorer",
    "make_kruskal_error_scorer",
    "make_chi2_sensitive_scorer",
    "make_composite_scorer",
    "plot_clusters",
    "plot_cluster_composition",
    "reduce_dimensions",
    "one_vs_all_p_continuous",
    "mean_diff",
    "size_metrics",
    "extreme_pair_gap_p",
    "omnibus_separability_p",
    "error_sep_p",
    "fisher_rxc_p",
    "binary_error_rate_column",
    "make_recap",
    "make_chi_tests",
    "recap_quali_metrics",
    "run_experiments_generic",
    "create_exp_conditions",
    "separability_check",
    "encode_categoricals",
    "build_sensitive_analysis_list",
    "apply_salient_reconstruction",
]
