"""CLI argument parsing and shared column/weight helpers for c4f."""

import os
import argparse
import pandas as pd
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "clustering_results", SESSION_DATE)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_column_list(col_string):
    """Parse comma-separated column names into a list."""
    if col_string is None or col_string.strip() == "":
        return []
    return [c.strip() for c in col_string.split(",")]


def parse_feature_weights(
    weight_str, regular_cols, sensitive_cols, special_cols, all_cols
):
    """
    Parse feature weights from CLI string.

    Supports two formats:
    1. Group weights: 'regular:1.5,sensitive:0.5,special:2.0'
    2. Individual column weights: 'age:2.0,income:0.5'
    3. Mixed: 'regular:1.0,age:2.0' (individual overrides group)

    Returns dict mapping column name -> weight
    """
    if not weight_str:
        return None

    weights = {}
    for pair in weight_str.split(","):
        parts = pair.strip().split(":")
        if len(parts) != 2:
            continue
        name, w = parts[0].strip(), float(parts[1].strip())

        # Check if it's a group name
        if name == "regular":
            for col in regular_cols:
                weights[col] = w
        elif name == "sensitive":
            for col in sensitive_cols:
                weights[col] = w
        elif name == "special":
            for col in special_cols:
                weights[col] = w
        else:
            # Individual column
            if name in all_cols:
                weights[name] = w

    return weights if weights else None


def parse_label_map(s):
    """'col:Label,col2:Label2' -> {col: Label} for display-label overrides. Empty -> {}."""
    out = {}
    for pair in (s or "").split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def parse_projection_list(s):
    """Comma-separated projection methods (e.g. 'pca,tsne'). 'none' anywhere -> skip all."""
    methods = [m.strip() for m in s.split(",") if m.strip()]
    return [] if "none" in methods else methods


def _build_sensitive_analysis_list(
    sensitive_cols, multiclass_dummies, original_sensitive_cols, option="onehot"
):
    """Which sensitive columns the result tables analyze.

    option='onehot' (default): one binary column per category — the readable multi-class
    dummies (factorized originals dropped). Deduplicated: under Euclidean/Manhattan the
    dummies already sit in sensitive_cols (clustered); under Gower sensitive_cols holds
    the factorized original and the dummies are added back here.

    option='salient': one multi-categorical column per feature (winning category). The
    dummies/factorized originals are dropped and the readable original name is used;
    apply_salient_reconstruction() must rebuild that column into the recap frame."""
    if option == "salient":
        dummy_set = {d for ds in multiclass_dummies.values() for d in ds}
        analysis = [c for c in sensitive_cols
                    if c not in multiclass_dummies and c not in dummy_set]
        for orig_col in multiclass_dummies:
            if orig_col in original_sensitive_cols and orig_col not in analysis:
                analysis.append(orig_col)
        return analysis
    analysis = [c for c in sensitive_cols if c not in multiclass_dummies]
    for orig_col, dummies in multiclass_dummies.items():
        if orig_col in original_sensitive_cols:
            for dc in dummies:
                if dc not in analysis:
                    analysis.append(dc)
    return analysis


def _reconstruct_multicat(df, orig, dummies):
    """Readable category per row from its all-K one-hot dummies (exactly one 1 per row).
    Rebuilds a multi-categorical sensitive column for the 'salient' table option — the
    original was replaced by dummies (Euclidean) or factorized to codes (Gower)."""
    present = [d for d in dummies if d in df.columns]
    labels = [d[len(orig) + 1:] for d in present]  # strip 'orig_' prefix
    idx = df[present].to_numpy(dtype=float).argmax(axis=1)
    return pd.Series([labels[i] for i in idx], index=df.index)


def apply_salient_reconstruction(df, multiclass_dummies, original_sensitive_cols):
    """For the 'salient' table option: rebuild each readable multi-categorical sensitive
    column from its dummies, in place. Use on the POST-clustering recap frame so the Gower
    code column used for clustering is left untouched."""
    for orig, dummies in multiclass_dummies.items():
        if orig in original_sensitive_cols:
            df[orig] = _reconstruct_multicat(df, orig, dummies)


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering for fairness analysis")

    parser.add_argument(
        "--algorithm",
        type=str,
        default="hdbscan",
        choices=[
            "dbscan",
            "hdbscan",
            "kmeans",
            "bisectingkmeans",
            "kmedoids",
            "kprototypes",
        ],
        help="Clustering algorithm",
    )

    # Distance metric
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "gower"],
        help="Distance metric",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Exact number of clusters (mutually exclusive with n_min/n_max). Defaults to 5 if neither n_min/n_max is given.",
    )
    parser.add_argument(
        "--n_min",
        type=int,
        default=None,
        help="Minimum number of clusters (for range-based k search)",
    )
    parser.add_argument(
        "--n_max",
        type=int,
        default=None,
        help="Maximum number of clusters (for range-based k search)",
    )

    # DBSCAN parameters
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="Maximum distance between samples for neighborhood (DBSCAN)",
    )

    # HDBSCAN parameters
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="HDBSCAN/DBSCAN only: minimum samples in a neighborhood for a point to be a core point.",
    )
    # General parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (used by KMeans, BisectingKMeans, KMedoids, KPrototypes, and experiment mode)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed experiments (e.g., '42,123,456'). Mutually exclusive with --seed in experiment mode.",
    )

    # KMeans parameters
    parser.add_argument(
        "--max_iter",
        type=int,
        default=300,
        help="Maximum iterations for KMeans/BisectingKMeans",
    )

    # Scoring method for k-selection
    parser.add_argument(
        "--scoring",
        type=str,
        default="composite",
        choices=["silhouette", "chi2_error", "chi2_sensitive", "composite"],
        help="Scoring method for k-search: composite (default, weighted silhouette+error+fairness), silhouette (cluster quality only), chi2_error (error separation), chi2_sensitive (fairness)",
    )
    parser.add_argument(
        "--composite_weights",
        type=str,
        default="silhouette:0.3,error:0.5,fairness:0.2",
        help="Weights for composite scoring as 'silhouette:W,error:W,fairness:W'. Accepts any value in [0, Inf); weights are normalized to sum to 1.",
    )

    # Feature weights
    parser.add_argument(
        "--feature_weights",
        type=str,
        default=None,
        help="Feature weights as 'col:weight' pairs. Groups: 'regular:1.5,sensitive:0.5'. Individual: 'age:2.0'. Mixed: 'regular:1.0,age:2.0'",
    )

    parser.add_argument(
        "--min_datapoints",
        type=int,
        default=None,
        help="Minimum cluster size. For HDBSCAN: enforced natively during extraction. For all other algorithms: post-hoc filter (small clusters become noise).",
    )

    # Statistical tests
    parser.add_argument(
        "--separability_check",
        action="store_true",
        help="Run chi-squared tests on clusters",
    )

    parser.add_argument(
        "--y_true_col",
        type=str,
        default=None,
        help="Column name for ground truth labels (for subset filtering)",
    )
    parser.add_argument(
        "--y_pred_col",
        type=str,
        default=None,
        help="Column name for predicted labels (for subset filtering)",
    )
    # Subset analysis
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=["TP", "TN", "FP", "FN", "TP_TN", "FP_FN"],
        help="Analyze only this confusion matrix subset (TP_TN=correct predictions, FP_FN=errors)",
    )

    # Projection method
    parser.add_argument(
        "--projection",
        type=str,
        default="tsne",
        help="Projection method(s) for visualization: comma-separated from pca, tsne, mds "
        "(e.g. 'pca,tsne' emits one plot each). t-SNE uses the same distance as clustering "
        "(precomputed Gower matrix under --distance gower, Manhattan under --distance manhattan). "
        "Use 'none' to skip. Default: tsne.",
    )

    parser.add_argument(
        "--regular_cols",
        type=str,
        default=None,
        help="Regular features for clustering (comma-separated column names)",
    )
    # TODO: Side-by-side comparison of Euclidean vs Gower clustering results — for the same k, show cluster proportions, error separation (chi2/KW), and sensitive feature distribution per cluster for both distances. Helps assess whether Gower adds value over standard Euclidean.
    # In progress: Finish package
    # TODO: For mixed data, when is it better to run zhich data. Test by running exp with these 3 options. & try it on a bunch of datasets, such as the ones that we already have for testing purposes. See if we have consistent results & if it depends on the balance between acategorigal & numerical features.
    # TODO: Try clustering iteratively.

    # TODO: Look into journals that take research artifacts. Or a DEMO at a conference.
    # TODO: Documentation
    # TODO: ACM Badge
    # TODO: Publish: Look for open science journals - 1 v all
    # NOTE: On a besoin juste d'un datapoint pour le ndcg. Meme system que pour regression.
    # NOTE: Ranking/recommender system: need P & Recall as error measures for clustering that considers multiple error forms.
    # TODO: site web ou on peut uploader le dataset, confirmer les colommes a utilier, sensitives. Penser un peu aux tests qu'on peut applquer.
    # TODO: On peut faire un clustering qui considere +ieurs formes d'erruer. Pour pb de ranking, on a P & Recall - pour + tard.
    # TODO: Look into finding hte number of clusters if it works or not. Should wokr

    # TODO: K-centroid clustering variant - have including the fair-centroid version.

    parser.add_argument(
        "--sensitive_cols",
        type=str,
        default=None,
        help="Sensitive/protected attributes (comma-separated column names). Both binary (0/1) and multi-class columns are supported.",
    )
    parser.add_argument(
        "--continuous_sensitive_cols",
        type=str,
        default=None,
        help="Subset of --sensitive_cols to treat as continuous (numeric). For these columns: per-cluster mean / mean-delta / Mann-Whitney p in the recap, Kruskal-Wallis across clusters in chi_res, and mean-range in all_quali. Default: none (all sensitive cols treated as categorical).",
    )
    parser.add_argument(
        "--proxy_cols",
        type=str,
        default=None,
        help="Proxy features for sensitive attributes (comma-separated column names)",
    )
    parser.add_argument(
        "--special_cols",
        type=str,
        default=None,
        help="Special features like SHAP values (comma-separated column names)",
    )
    parser.add_argument(
        "--categorical_cols",
        type=str,
        default=None,
        help="Columns to treat as categorical (comma-separated). String/category dtype columns are detected automatically; use this to force-mark additional columns.",
    )
    parser.add_argument(
        "--error_col",
        type=str,
        default=None,
        help="Error column for analysis. Binary (0/1) for classification, continuous for regression.",
    )
    parser.add_argument(
        "--error_label",
        type=str,
        default=None,
        help="Display name for the error column in output tables and heatmaps. Defaults to the value of --error_col.",
    )
    parser.add_argument(
        "--sensitive_labels",
        type=str,
        default=None,
        help="Display labels for sensitive features in heatmaps, as 'col:Label,col2:Label2'. "
        "Keys are the analysis-column names as they appear in the recap (e.g. a binary "
        "string feature 'sex' becomes 'sex_Male'). Display-only; CSVs keep raw names.",
    )
    parser.add_argument(
        "--error_type",
        type=str,
        default="binary",
        choices=["binary", "regression", "multiclass"],
        help="Type of error column: 'binary' (classification 0/1), 'regression' (continuous), or 'multiclass' (3+ class predictions; the error column is derived from --y_true_col / --y_pred_col). Default: binary",
    )
    parser.add_argument(
        "--binary_error_metric",
        type=str,
        default="raw",
        choices=["raw", "fpr", "fnr", "precision", "prec_neg"],
        help="Binary error definition for the result tables (only with --error_type binary). "
        "'raw' (default): use --error_col as-is (proportion positive). The rate options are "
        "derived from --y_true_col / --y_pred_col per cluster: 'fpr' FP/(FP+TN), 'fnr' "
        "FN/(FN+TP), 'precision' 1-Precision = FP/(FP+TP), 'prec_neg' 1-Precision-for-Negative "
        "= FN/(FN+TN). Positive class = the larger label.",
    )
    parser.add_argument(
        "--multicat_table_option",
        type=str,
        default="onehot",
        choices=["onehot", "salient"],
        help="How multi-categorical SENSITIVE features appear in the result tables. "
        "'onehot' (default): one binary column per category. 'salient': one column per "
        "feature showing the winning category and its value. Independent of the clustering "
        "distance/encoding.",
    )
    parser.add_argument(
        "--sensitive_gap_test",
        type=str,
        default="chi2",
        choices=["chi2", "fisher"],
        help="Significance test for the sensitive-feature gap columns (<F>_gap_sig). "
        "'chi2' (default, per spec) or 'fisher' (exact 2x2). Error gap significance stays "
        "Fisher; numeric sensitive features use Mann-Whitney/ANOVA regardless.",
    )
    parser.add_argument(
        "--multicat_sig",
        type=str,
        default="auto",
        choices=["auto", "fisher_rxc", "chi2", "fisher_ova"],
        help="Omnibus separability test for multi-class errors / multi-categorical "
        "sensitive features (Overview error_sep / <F>_sep). 'auto' (default): exact "
        "r x c Fisher via R when available (needs 'c4fairness[r]' + R >= 4.5), else "
        "scipy (chi-square when cell counts are adequate, else one-vs-all 2x2 Fisher). "
        "'fisher_rxc' forces R (errors if absent); 'chi2' / 'fisher_ova' force scipy.",
    )
    parser.add_argument(
        "--error_multiclass_option",
        type=str,
        default="per_class",
        choices=[
            "accuracy",
            "per_class",
            "precision",
            "per_cell",
            "binary_cells",
            "onehot",
            "classwise",
        ],
        help="How to derive the multi-class error column (only with --error_type multiclass): "
        "'accuracy' (binary correct/incorrect), "
        "'per_class' (default; true-class label of each error, 'correct' otherwise), "
        "'precision' (predicted-class label of each error, 'correct' otherwise), "
        "'per_cell' (confusion cell 'true→pred' of each error, 'correct' otherwise), "
        "'binary_cells' (TP/TN/FP/FN confusion cell; 2-class problems), "
        "'onehot' (one binary one-vs-all error column per class), "
        "'classwise' (one TP/FN/FP/TN one-vs-all column per class).",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input CSV file"
    )
    # Output
    parser.add_argument(
        "--no_standardize",
        action="store_true",
        help="Disable automatic standardization of numeric features before clustering. Use this if your data is already normalized.",
    )
    parser.add_argument(
        "--no_plots", action="store_true", help="Skip saving visualization plots"
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR, help="Output directory for plots"
    )

    # Batch experiment mode
    parser.add_argument(
        "--experiment",
        nargs="?",
        const="",
        default=None,
        help="Run batch experiment. Optionally pass comma-separated groups to exclude (e.g. --experiment SPECIAL or --experiment SPECIAL,ERR). Available: REG, SEN, ERR, SPECIAL.",
    )

    return parser.parse_args()
