import os, argparse, re
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from c4f.clustering import cluster, gower_distance
from c4f.scoring import (
    make_chi2_error_scorer,
    make_kruskal_error_scorer,
    make_categorical_error_scorer,
    make_chi2_sensitive_scorer, make_composite_scorer,
)
from c4f.visualization import reduce_dimensions, plot_clusters, plot_cluster_composition
from c4f.experiments import (
    create_exp_conditions,
    run_experiments_generic, make_recap, make_chi_tests,
    recap_quali_metrics, plot_quality_heatmap, plot_cluster_recap_heatmap,
    separability_check
)
from c4f.preprocessing import encode_categoricals
from c4f.fairness_metrics import multiclass_error_types
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_DATE = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = os.path.join(PROJECT_DIR, "clustering_results", SESSION_DATE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_column_list(col_string):
      """Parse comma-separated column names into a list."""
      if col_string is None or col_string.strip() == "":
          return []
      return [c.strip() for c in col_string.split(",")]


def parse_feature_weights(weight_str, regular_cols, sensitive_cols, special_cols, all_cols):
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
    for pair in weight_str.split(','):
        parts = pair.strip().split(':')
        if len(parts) != 2:
            continue
        name, w = parts[0].strip(), float(parts[1].strip())

        # Check if it's a group name
        if name == 'regular':
            for col in regular_cols:
                weights[col] = w
        elif name == 'sensitive':
            for col in sensitive_cols:
                weights[col] = w
        elif name == 'special':
            for col in special_cols:
                weights[col] = w
        else:
            # Individual column
            if name in all_cols:
                weights[name] = w

    return weights if weights else None


def _encode_multiclass_categoricals(df, col_lists, categorical_cols_arg, algorithm, distance='euclidean'):
    """Thin wrapper around encode_categoricals.

    Multi-class one-hot dummies (including sensitive and proxy) are inserted into
    their col_lists so they enter the clustering feature matrix. Binary columns
    are unchanged. Under Gower the original is factorized in place instead.
    """
    return encode_categoricals(df, col_lists, categorical_cols_arg, algorithm,
                               distance=distance)


def _build_sensitive_analysis_list(sensitive_cols, multiclass_dummies, original_sensitive_cols):
    """Binary/numeric sensitives + readable multi-class dummies, factorized
    originals dropped. Deduplicated: under Euclidean/Manhattan the dummies already
    sit in sensitive_cols (clustered); under Gower sensitive_cols holds the
    factorized original and the dummies are added back here."""
    analysis = [c for c in sensitive_cols if c not in multiclass_dummies]
    for orig_col, dummies in multiclass_dummies.items():
        if orig_col in original_sensitive_cols:
            for dc in dummies:
                if dc not in analysis:
                    analysis.append(dc)
    return analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering for fairness analysis")
    
    parser.add_argument("--algorithm", type=str, default="hdbscan",
                        choices=["dbscan", "hdbscan", "kmeans", "bisectingkmeans", "kmedoids", "kprototypes"],
                        help="Clustering algorithm")

    # Distance metric
    parser.add_argument("--distance", type=str, default="euclidean",
                        choices=["euclidean", "manhattan", "gower"],
                        help="Distance metric")
    
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Exact number of clusters (mutually exclusive with n_min/n_max). Defaults to 5 if neither n_min/n_max is given.")
    parser.add_argument("--n_min", type=int, default=None,
                        help="Minimum number of clusters (for range-based k search)")
    parser.add_argument("--n_max", type=int, default=None,
                        help="Maximum number of clusters (for range-based k search)")

    #DBSCAN parameters
    parser.add_argument("--eps", type=float, default=0.5,
                        help="Maximum distance between samples for neighborhood (DBSCAN)")
    
    # HDBSCAN parameters
    parser.add_argument("--min_samples", type=int, default=5,
                        help="HDBSCAN/DBSCAN only: minimum samples in a neighborhood for a point to be a core point.")
    # General parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (used by KMeans, BisectingKMeans, KMedoids, KPrototypes, and experiment mode)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed experiments (e.g., '42,123,456'). Mutually exclusive with --seed in experiment mode.")

    # KMeans parameters
    parser.add_argument("--max_iter", type=int, default=300,
                        help="Maximum iterations for KMeans/BisectingKMeans")

    # Scoring method for k-selection
    parser.add_argument("--scoring", type=str, default="composite",
                        choices=["silhouette", "chi2_error", "chi2_sensitive", "composite"],
                        help="Scoring method for k-search: composite (default, weighted silhouette+error+fairness), silhouette (cluster quality only), chi2_error (error separation), chi2_sensitive (fairness)")
    parser.add_argument("--composite_weights", type=str, default="silhouette:0.3,error:0.5,fairness:0.2",
                        help="Weights for composite scoring as 'silhouette:W,error:W,fairness:W'. Accepts any value in [0, Inf); weights are normalized to sum to 1.")

    # Feature weights
    parser.add_argument("--feature_weights", type=str, default=None,
                        help="Feature weights as 'col:weight' pairs. Groups: 'regular:1.5,sensitive:0.5'. Individual: 'age:2.0'. Mixed: 'regular:1.0,age:2.0'")

    parser.add_argument("--min_datapoints", type=int, default=None,
                        help="Minimum cluster size. For HDBSCAN: enforced natively during extraction. For all other algorithms: post-hoc filter (small clusters become noise).")

    # Statistical tests
    parser.add_argument("--separability_check", action="store_true",
                        help="Run chi-squared tests on clusters")

    parser.add_argument("--y_true_col", type=str, default=None,                                                                                                                         
                          help="Column name for ground truth labels (for subset filtering)")                                                                                              
    parser.add_argument("--y_pred_col", type=str, default=None,                                                                                                                         
                        help="Column name for predicted labels (for subset filtering)") 
    # Subset analysis
    parser.add_argument("--subset", type=str, default=None,
                        choices=["TP", "TN", "FP", "FN", "TP_TN", "FP_FN"],
                        help="Analyze only this confusion matrix subset (TP_TN=correct predictions, FP_FN=errors)")

    # Projection method
    parser.add_argument("--projection", type=str, default="tsne",
                        choices=["pca", "tsne", "mds", "none"],
                        help="Projection method for visualization. When --distance gower is used, MDS is applied automatically with the precomputed Gower matrix regardless of this flag. Use 'none' to skip.")

    parser.add_argument("--regular_cols", type=str, default=None, help="Regular features for clustering (comma-separated column names)")     
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
    
    
    
    parser.add_argument("--sensitive_cols", type=str, default=None,
                        help="Sensitive/protected attributes (comma-separated column names). Both binary (0/1) and multi-class columns are supported.")
    parser.add_argument("--continuous_sensitive_cols", type=str, default=None,
                        help="Subset of --sensitive_cols to treat as continuous (numeric). For these columns: per-cluster mean / mean-delta / Mann-Whitney p in the recap, Kruskal-Wallis across clusters in chi_res, and mean-range in all_quali. Default: none (all sensitive cols treated as categorical).")
    parser.add_argument("--proxy_cols", type=str, default=None, help="Proxy features for sensitive attributes (comma-separated column names)")                                                                                  
    parser.add_argument("--special_cols", type=str, default=None,
                          help="Special features like SHAP values (comma-separated column names)")
    parser.add_argument("--categorical_cols", type=str, default=None,
                        help="Columns to treat as categorical (comma-separated). String/category dtype columns are detected automatically; use this to force-mark additional columns.")
    parser.add_argument("--error_col", type=str, default=None,
                        help="Error column for analysis. Binary (0/1) for classification, continuous for regression.")
    parser.add_argument("--error_label", type=str, default=None,
                        help="Display name for the error column in output tables and heatmaps. Defaults to the value of --error_col.")
    parser.add_argument("--error_type", type=str, default="binary",
                        choices=["binary", "regression", "multiclass"],
                        help="Type of error column: 'binary' (classification 0/1), 'regression' (continuous), or 'multiclass' (3+ class predictions; the error column is derived from --y_true_col / --y_pred_col). Default: binary")
    parser.add_argument("--error_multiclass_option", type=str, default="per_class",
                        choices=["accuracy", "per_class", "precision", "per_cell",
                                 "binary_cells", "onehot", "classwise"],
                        help="How to derive the multi-class error column (only with --error_type multiclass): "
                             "'accuracy' (binary correct/incorrect), "
                             "'per_class' (default; true-class label of each error, 'correct' otherwise), "
                             "'precision' (predicted-class label of each error, 'correct' otherwise), "
                             "'per_cell' (confusion cell 'true→pred' of each error, 'correct' otherwise), "
                             "'binary_cells' (TP/TN/FP/FN confusion cell; 2-class problems), "
                             "'onehot' (one binary one-vs-all error column per class), "
                             "'classwise' (one TP/FN/FP/TN one-vs-all column per class).")
    parser.add_argument("--data_path", type=str, required=True,
                          help="Path to input CSV file")
    # Output
    parser.add_argument("--no_standardize", action="store_true",
                        help="Disable automatic standardization of numeric features before clustering. Use this if your data is already normalized.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip saving visualization plots")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for plots")

    # Batch experiment mode
    parser.add_argument("--experiment", nargs="?", const="", default=None,
                        help="Run batch experiment. Optionally pass comma-separated groups to exclude (e.g. --experiment SPECIAL or --experiment SPECIAL,ERR). Available: REG, SEN, ERR, SPECIAL.")

    return parser.parse_args()


def run_batch_experiment(df, args, output_dir, metadata=None):#
    """
    Run all experimental conditions and generate outputs.

    Generates conditions from CLI column groups (generic, dataset-agnostic).

    Parameters
    ----------
    df : pd.DataFrame
        Input data with all required columns.
    args : argparse.Namespace
        CLI arguments (must include error_col, sensitive_cols, etc.).
    output_dir : str
        Directory to save outputs.

    Returns
    -------
    dict
        Results dictionary with experiment data.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    print("Running batch experiment...")
    print(f"  Dataset: {os.path.basename(args.data_path)}")

    # Parse column groups from CLI
    regular_cols = parse_column_list(args.regular_cols)
    sensitive_cols = parse_column_list(args.sensitive_cols)
    continuous_sensitive_cols = set(parse_column_list(getattr(args, 'continuous_sensitive_cols', None)) or [])
    proxy_cols = parse_column_list(args.proxy_cols)
    special_cols = parse_column_list(args.special_cols)
    error_col = args.error_col

    unknown_continuous = continuous_sensitive_cols - set(sensitive_cols or [])
    if unknown_continuous:
        raise ValueError(
            f"--continuous_sensitive_cols entries not found in --sensitive_cols: {sorted(unknown_continuous)}"
        )

    # Validate: need error_col and at least one feature group
    if not error_col:
        raise ValueError("--error_col is required in experiment mode")
    if not regular_cols and not sensitive_cols and not special_cols:
        raise ValueError("At least one feature group (--regular_cols, --sensitive_cols, or --special_cols) is required in experiment mode")
    if not sensitive_cols:
        raise ValueError("--sensitive_cols is required in experiment mode (for proportion analysis)")

    # Preserve original (pre-encoding) sensitive column names for fairness analysis
    original_sensitive_cols = parse_column_list(args.sensitive_cols)

    # Resolve error_label
    error_label = getattr(args, 'error_label', None) or error_col or 'error'

    # Encode categorical columns (one-hot for non-kprototypes; detect names for kprototypes).
    # Multi-class dummies from sensitive cols are excluded from col_lists['sensitive']
    # (so they don't go into the feature matrix), but kept in the DataFrame and tracked
    # via `multiclass_dummies` for fairness analysis.
    categorical_cols_arg = parse_column_list(getattr(args, 'categorical_cols', None))
    col_lists = {'regular': regular_cols, 'sensitive': sensitive_cols, 'proxy': proxy_cols, 'special': special_cols}
    df, col_lists, categorical_col_names, multiclass_dummies, ohe_col_names = _encode_multiclass_categoricals(
        df, col_lists, categorical_cols_arg, args.algorithm, distance=args.distance
    )
    regular_cols = col_lists['regular']
    sensitive_cols = col_lists['sensitive']
    proxy_cols = col_lists['proxy']
    special_cols = col_lists['special']

    # Fairness-analysis sensitive list: binary/numeric sensitives + readable
    # multi-class dummies (factorized originals dropped), deduplicated across the
    # Euclidean (dummies already in sensitive_cols) and Gower (factorized original
    # in sensitive_cols) paths.
    sensitive_cols_analysis = _build_sensitive_analysis_list(
        sensitive_cols, multiclass_dummies, original_sensitive_cols
    )

    # Build groups dict for condition generation
    groups = {}
    if regular_cols:
        groups['REG'] = regular_cols
    if sensitive_cols:
        groups['SEN'] = sensitive_cols
    groups['ERR'] = [error_col]
    if proxy_cols:
        groups['PROXY'] = proxy_cols
    if special_cols:
        groups['SPECIAL'] = special_cols

    # Apply group exclusions passed to --experiment (e.g. --experiment REG,SPECIAL).
    # Excluded columns are still used for scoring/fairness evaluation — only removed from the condition matrix.
    if args.experiment:
        excluded = {g.strip().upper() for g in args.experiment.split(',')}
        unknown = excluded - set(groups.keys())
        if unknown:
            print(f"  Warning: unknown groups to exclude: {unknown}. Available: {set(groups.keys())}")
        groups = {k: v for k, v in groups.items() if k not in excluded}
        print(f"  Excluded groups: {excluded - unknown}")

    # Create experimental conditions
    exp_condition = create_exp_conditions(groups)
    print(f"  Conditions: {len(exp_condition)}")
    print(f"  Groups: {list(groups.keys())}")

    # Save experimental conditions table
    exp_condition_save = exp_condition[['feature_set_descr', 'feature_set_name']].copy()
    exp_condition_save['feature_set'] = exp_condition['feature_set'].apply(lambda x: ', '.join(x))
    exp_condition_save.to_csv(f"{output_dir}/exp_condition.csv", index=False)
    print(f"\nSaved: exp_condition.csv")

    # Build scoring function for k-selection (same logic as single-run mode)
    scoring_fn = None
    if args.scoring == "chi2_error":
        if not error_col:
            raise ValueError("--error_col required for chi2_error scoring")
        if args.error_type == 'regression':
            scoring_fn = make_kruskal_error_scorer(df[error_col].values)
        elif args.error_type == 'multiclass':
            scoring_fn = make_categorical_error_scorer(df[error_col].values)
        else:
            scoring_fn = make_chi2_error_scorer(df[error_col].values)
    elif args.scoring == "chi2_sensitive":
        if not sensitive_cols:
            raise ValueError("--sensitive_cols required for chi2_sensitive scoring")
        scoring_fn = make_chi2_sensitive_scorer(df[sensitive_cols[0]].values)
    elif args.scoring == "composite":
        if error_col or sensitive_cols:
            cw = {}
            for pair in args.composite_weights.split(','):
                name, w = pair.strip().split(':')
                cw[name.strip()] = float(w.strip())
            scoring_fn = make_composite_scorer(
                error_data=df[error_col].values if error_col else None,
                sensitive_data=df[sensitive_cols[0]].values if sensitive_cols else None,
                silhouette_weight=cw.get('silhouette', 0.3),
                error_weight=cw.get('error', 0.5),
                fairness_weight=cw.get('fairness', 0.2),
                error_type=args.error_type,
            )
        # else: no error_col or sensitive_cols -> scoring_fn stays None -> silhouette fallback

    # Parse feature weights (include sensitive_cols — they are part of clustering)
    all_clustering_cols = regular_cols + sensitive_cols + proxy_cols + special_cols
    feature_weights = parse_feature_weights(
        args.feature_weights, regular_cols, sensitive_cols, special_cols, all_clustering_cols
    )

    # Run all experiments.
    # Pass sensitive_cols_analysis (binary + multi-class dummies) so fairness analysis
    # inside make_recap sees the multi-class dummies that were excluded from the feature matrix.
    results = run_experiments_generic(
        df,
        exp_condition,
        algorithm=args.algorithm,
        distance=args.distance,
        n_clusters=args.n_clusters,
        n_min=args.n_min,
        n_max=args.n_max,
        max_iter=args.max_iter,
        seed=args.seed,
        scoring_fn=scoring_fn,
        sensitive_cols=sensitive_cols_analysis,
        error_col=error_col,
        min_samples=args.min_samples,
        eps=args.eps,
        min_datapoints=args.min_datapoints,
        error_type=args.error_type,
        feature_weights=feature_weights,
        categorical_col_names=categorical_col_names,
        standardize=not args.no_standardize,
        continuous_sensitive_cols=continuous_sensitive_cols,
        ohe_col_names=ohe_col_names,
        multiclass_option=args.error_multiclass_option,
        error_cols=args.error_cols,
        error_cols_kind=args.error_cols_kind,
    )

    # Print progress for each condition
    print()
    for i, cond_name in enumerate(results['cond_name']):
        recap = results['cond_recap'][i]
        n_clusters = len(recap)
        silhouette_avg = recap['silh'].mean() if 'silh' in recap.columns else np.nan
        print(f"Condition {i+1}/{len(results['cond_name'])}: {cond_name.strip()}")
        print(f"  Clusters: {n_clusters}, Silhouette: {silhouette_avg:.3f}" if not np.isnan(silhouette_avg) else f"  Clusters: {n_clusters}")

    # Generate chi-squared / Kruskal-Wallis test results
    chi_res = make_chi_tests(results, sensitive_cols=sensitive_cols_analysis,
                             error_type=args.error_type, error_col=error_col,
                             continuous_sensitive_cols=continuous_sensitive_cols,
                             multiclass_option=args.error_multiclass_option,
                             error_cols=args.error_cols,
                             error_cols_kind=args.error_cols_kind)
    chi_res.to_csv(f"{output_dir}/chi_res.csv", index=False)
    print(f"\nSaved: chi_res.csv")

    # Print separability test results summary.
    # chi_res columns: cond_descr, cond_name, <error sep col(s)>, <F>_sep per feature.
    # onehot has one 'error=<class>_sep' per class instead of a single 'error_sep'.
    err_sep_cols = [c for c in chi_res.columns
                    if c == 'error_sep' or (c.startswith('error=') and c.endswith('_sep'))]
    skip_meta = {'cond_descr', 'cond_name'} | set(err_sep_cols)
    sep_cols = [c for c in chi_res.columns if c not in skip_meta]
    print("\nSeparability test results (p-values):")
    chi_display_cols = ['cond_name'] + err_sep_cols + sep_cols
    chi_display = chi_res[chi_display_cols].copy()
    chi_display.columns = ['Condition'] + err_sep_cols + sep_cols
    print(chi_display.to_string(index=False))

    # Generate quality metrics
    all_quali = recap_quali_metrics(chi_res, results,
                                    sensitive_cols=sensitive_cols_analysis,
                                    continuous_sensitive_cols=continuous_sensitive_cols,
                                    error_col=error_col,
                                    error_type=args.error_type,
                                    multiclass_option=args.error_multiclass_option,
                             error_cols=args.error_cols,
                             error_cols_kind=args.error_cols_kind)

    if not args.no_plots:
      # Separability test heatmap (omnibus p-values) — same blue/red/violet styling
      # and slanted labels as the Overview heatmap (error=red, sensitive=violet).
      chi_viz_cols = err_sep_cols + sep_cols
      chi_res_viz = chi_res[chi_viz_cols].copy()
      chi_res_viz.index = chi_res['cond_name'].str.strip()
      plot_quality_heatmap(chi_res_viz, f"{output_dir}/chi_res_heatmap.png",
                           error_label=error_label,
                           title="Separability Test Results (p-values)")
      print(f"Saved: chi_res_heatmap.png")

      # Create quality metrics heatmap
      skip_meta_quali = {'cond_descr', 'cond_name'}
      quali_viz_cols = [c for c in all_quali.columns if c not in skip_meta_quali]
      all_quali_viz = all_quali[quali_viz_cols].copy()
      all_quali_viz.index = all_quali['cond_name'].str.strip()
      plot_quality_heatmap(all_quali_viz, f"{output_dir}/all_quali_heatmap.png",
                           error_label=error_label)
      plt.close()
      print(f"Saved: all_quali_heatmap.png")

    # Generate per-condition recap heatmaps and composition plots
    if not args.no_plots:
        print(f"\nGenerating {len(results['cond_name'])} recap heatmaps...")
        for i, cond_name in enumerate(results['cond_name']):
            recap = results['cond_recap'][i].copy()
            if len(recap) > 1:  # Only plot if there are multiple clusters
                plot_cluster_recap_heatmap(recap, cond_name, output_dir, error_label=error_label)
                plt.close()
        print(f"Saved: {len(results['cond_name'])} recap heatmaps")

        # Per-condition cluster scatter plots
        if args.projection != "none":
            print(f"Generating cluster scatter plots...")
            cat_names_set_viz = set(categorical_col_names or [])
            for i, cond_name in enumerate(results['cond_name']):
                res_df = results['cond_res'][i]
                labels = res_df['clusters'].values
                feature_set = exp_condition['feature_set'][i]
                if len(set(labels) - {-1}) > 1:
                    cond_clean = re.sub(r'\s+', '', cond_name)
                    non_noise = labels != -1
                    if args.distance == "gower" and args.projection == "mds":
                        # MDS on per-condition Gower matrix (non-noise rows only)
                        X_raw = res_df[feature_set].values[non_noise].astype(float)
                        cat_idx = [j for j, c in enumerate(feature_set) if c in cat_names_set_viz] or None
                        D = gower_distance(X_raw, cat_idx)
                        X_2d = reduce_dimensions(D, method="mds", precomputed=True)
                        plot_clusters(X_2d, labels[non_noise],
                                      title=f"Clusters ({cond_name}, gower+MDS)",
                                      out_path=f"{output_dir}/{cond_clean}_clusters.png")
                    else:
                        X_vals = res_df[feature_set].values[non_noise].astype(float)
                        X = StandardScaler().fit_transform(X_vals)
                        X_2d = reduce_dimensions(X, method=args.projection)
                        plot_clusters(X_2d, labels[non_noise],
                                      title=f"Clusters ({cond_name})",
                                      out_path=f"{output_dir}/{cond_clean}_clusters.png")
                    plt.close()
            print(f"Saved: cluster scatter plots")

        # Composition bar plots per condition x sensitive attribute
        print(f"Generating composition plots...")
        for i, cond_name in enumerate(results['cond_name']):
            res_df = results['cond_res'][i]
            labels = res_df['clusters'].values
            if len(set(labels) - {-1}) > 1:
                cond_clean = re.sub(r'\s+', '', cond_name)
                for attr in sensitive_cols_analysis:
                    if attr in continuous_sensitive_cols:
                        continue
                    plot_cluster_composition(labels, res_df[attr].values, attr,
                        out_path=f"{output_dir}/{cond_clean}_composition_{attr}.png")
                    plt.close()
        print(f"Saved: composition plots")

    # --- CSV outputs ---

    # Global summary CSV (Overview): one row per condition.
    # Columns: condition, <metadata>, cond_name, n_clusters, silh,
    #          error_sep, error_gap, <F>_sep, <F>_gap.
    # silh/<gap>/<sep> are pulled from all_quali (which copies sep from chi_res).
    summary_rows = []
    chi_res_lookup = chi_res.set_index('cond_name') if chi_res is not None else None
    all_quali_lookup = all_quali.set_index('cond_name') if all_quali is not None else None
    # Overview metric columns from all_quali, preserving its column order.
    quali_metric_cols = [c for c in all_quali.columns if c not in ('cond_descr', 'cond_name')]
    for i, cond_name in enumerate(results['cond_name']):
        recap = results['cond_recap'][i]

        row = {'condition': results['cond_descr'][i]}
        if metadata:
            row.update(metadata)
        row['cond_name'] = cond_name
        row['n_clusters'] = len(recap)

        # Pull silh / *_sep / *_gap from all_quali (order: silh, error_sep,
        # error_gap, then per-feature sep/gap pairs).
        if all_quali_lookup is not None and cond_name in all_quali_lookup.index:
            q = all_quali_lookup.loc[cond_name]
            for c in quali_metric_cols:
                row[c] = q[c]
        else:
            for c in quali_metric_cols:
                row[c] = np.nan

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    # Ensure 'condition' is the first column
    front_cols = ['condition'] + [c for c in summary_df.columns if c != 'condition']
    summary_df = summary_df[front_cols]
    summary_df.to_csv(f"{output_dir}/results_summary.csv", index=False)
    print(f"\nSaved: results_summary.csv")

    # Per-condition result CSVs (individual level)
    # One CSV per condition at root level:
    #   - 1 row per cluster with a 'rule' (most distinctive sensitive feature)
    #   - 1 OVERALL row with the KW stat test result
    all_cols_to_test = list(set(
        parse_column_list(args.regular_cols)
        + sensitive_cols
        + parse_column_list(args.special_cols)
    ))
    print(f"\nSaving per-condition result CSVs...")
    for i, cond_name in enumerate(results['cond_name']):
        cond_clean = re.sub(r'\s+', '', cond_name)
        recap_i = results['cond_recap'][i].copy()

        # Add rule: most distinctive sensitive feature per cluster (largest |gap|).
        # Derived from the per-feature <F>_gap columns (signed cluster-vs-rest gap);
        # exclude the error column's own gap.
        gap_cols = [c for c in recap_i.columns if c.endswith('_gap')
                    and not c.startswith('error_') and not c.startswith('error=')]
        if gap_cols:
            def _make_rule(row):
                vals = row[gap_cols].abs()
                if vals.isna().all():
                    return ''
                best_col = vals.idxmax()
                if pd.isna(best_col):
                    return ''
                best_val = row[best_col]
                feature = best_col[:-len('_gap')]
                direction = '+' if best_val > 0 else ''
                return f"{feature} ({direction}{round(best_val, 3)})"
            recap_i.insert(1, 'rule', recap_i.apply(_make_rule, axis=1))
        else:
            recap_i.insert(1, 'rule', '')

        # Add OVERALL row with the omnibus error separability p-value(s).
        # onehot has one 'error=<class>_sep' per class instead of a single 'error_sep'.
        overall_row = {col: '' for col in recap_i.columns}
        overall_row['c'] = 'OVERALL'
        if chi_res_lookup is not None and cond_name in chi_res_lookup.index:
            row = chi_res_lookup.loc[cond_name]
            es_cols = [c for c in chi_res_lookup.columns
                       if c == 'error_sep' or (c.startswith('error=') and c.endswith('_sep'))]
            parts = []
            for c in es_cols:
                v = row[c]
                parts.append(f"{c}: {round(v, 4) if not (v is None or pd.isna(v)) else 'n/a'}")
            overall_row['rule'] = '; '.join(parts) if parts else 'error_sep: n/a'
        else:
            overall_row['rule'] = 'error_sep: n/a'
        recap_i = pd.concat([recap_i, pd.DataFrame([overall_row])], ignore_index=True)

        # Append separability test results (feature-level stat tests across clusters)
        res_df = results['cond_res'][i]
        labels = res_df['clusters'].values
        if len(set(labels) - {-1}) > 1:
            sep_result = separability_check(res_df, labels, all_cols_to_test)
            if not sep_result.empty:
                for feat, sep_row in sep_result.iterrows():
                    sep_entry = {col: '' for col in recap_i.columns}
                    sep_entry['c'] = f'SEP:{feat}'
                    sep_entry['rule'] = f"{sep_row.get('test', '')} p={round(sep_row.get('p_value', np.nan), 4)}"
                    recap_i = pd.concat([recap_i, pd.DataFrame([sep_entry])], ignore_index=True)

        recap_i.to_csv(f"{output_dir}/{cond_clean}.csv", index=False)
    print(f"Saved: {len(results['cond_name'])} per-condition CSVs")

    print(f"\nAll outputs saved to: {output_dir}/")
    print(f"  - results_summary.csv (Overview: 1 row per condition, silh/error_sep/error_gap + per-feature sep/gap)")
    print(f"  - {len(results['cond_name'])} per-condition CSVs (1 row per cluster + OVERALL error_sep + SEP rows)")
    print("  - chi_res.csv")
    print("  - exp_condition.csv")
    if not args.no_plots:
        print("  - chi_res_heatmap.png / all_quali_heatmap.png")
        print(f"  - {len(results['cond_name'])} recap heatmaps + cluster scatter plots")
        print(f"  - composition plots")

    return results


def main():
    args = parse_args()

    # Resolve n_clusters / n_min / n_max defaults:
    # If n_min or n_max given - range search (fill in the other if missing)
    # If neither -  default to n_clusters=5
    if args.n_min is not None or args.n_max is not None:
        if args.n_clusters is not None:
            print("Warning: --n_clusters ignored when --n_min/--n_max are provided")
            args.n_clusters = None
        if args.n_min is None:
            args.n_min = 2
        if args.n_max is None:
            args.n_max = 10
    elif args.n_clusters is None:
        args.n_clusters = 5

    session_date = datetime.now().strftime('%Y-%m-%d')
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    # Block subset for regression (TP/TN/FP/FN doesn't apply)
    if args.error_type == 'regression' and args.subset:
        raise ValueError("--subset (TP/TN/FP/FN) is not compatible with --error_type regression. "
                         "Confusion matrix subsets only apply to binary classification.")

    print(f"Loading data...")
    df = pd.read_csv(args.data_path)

    if args.error_type == 'regression' and not args.error_col:
        if args.y_true_col and args.y_pred_col:
            df['_regression_error'] = df[args.y_true_col] - df[args.y_pred_col]
            args.error_col = '_regression_error'
            print(f"  Auto-computed signed regression error: {args.y_true_col} - {args.y_pred_col}")
        else:
            raise ValueError("--error_type regression requires either --error_col or both --y_true_col and --y_pred_col")

    # Multi-class error: derive a categorical/indicator error column from y_true/y_pred.
    # error_cols (+ error_cols_kind) is set for the per-class multi-column options
    # (onehot = binary indicators, classwise = TP/FN/FP/TN multi-categorical).
    args.error_cols = None
    args.error_cols_kind = 'binary'
    if args.error_type == 'multiclass':
        if not (args.y_true_col and args.y_pred_col):
            raise ValueError("--error_type multiclass requires both --y_true_col and --y_pred_col")
        err_df = multiclass_error_types(df[args.y_true_col], df[args.y_pred_col],
                                        args.error_multiclass_option)
        if args.error_multiclass_option in ('onehot', 'classwise'):
            # One error column per class -> one result-table set each. A single
            # 'any error' indicator drives clustering's ERR group + scoring.
            for col in err_df.columns:
                df[col] = err_df[col].values
            args.error_cols = list(err_df.columns)
            args.error_cols_kind = 'binary' if args.error_multiclass_option == 'onehot' else 'multicat'
            df['_multiclass_error'] = (df[args.y_true_col] != df[args.y_pred_col]).astype(int)
            args.error_col = '_multiclass_error'
        else:
            df['_multiclass_error'] = err_df['error'].values
            args.error_col = '_multiclass_error'
        print(f"  Derived multi-class error ('{args.error_multiclass_option}') "
              f"from {args.y_true_col} vs {args.y_pred_col}")

    # Experiment mode: run all conditions
    if args.experiment is not None:
        full_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Multi-seed experiment mode
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
            seeds_str = "_".join(f"s{s}" for s in seeds)
            weight_suffix = "_w_" + args.feature_weights.replace(":", "").replace(",", "_") if args.feature_weights else ""
            base_output_dir = os.path.join(args.output_dir, f"{full_timestamp}_experiment_{dataset_name}_{seeds_str}{weight_suffix}")
            os.makedirs(base_output_dir, exist_ok=True)

            all_chi_res = []

            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"Running experiment with seed={seed}")
                print(f"{'='*60}")
                seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
                os.makedirs(seed_dir, exist_ok=True)
                # Save per-seed metadata
                metadata = pd.DataFrame([{
                    'seed': seed,
                    'algorithm': args.algorithm,
                    'distance': args.distance,
                    'dataset': dataset_name,
                    'timestamp': full_timestamp,
                    'scoring_method': args.scoring,
                }])
                metadata.to_csv(os.path.join(seed_dir, 'metadata.csv'), index=False)
                args.seed = seed
                run_batch_experiment(df, args, seed_dir)

                # Collect chi_res for summary
                chi_path = os.path.join(seed_dir, 'chi_res.csv')
                if os.path.exists(chi_path):
                    chi_df = pd.read_csv(chi_path)
                    chi_df['seed'] = seed
                    all_chi_res.append(chi_df)

            # Generate cross-seed summary
            if all_chi_res:
                combined = pd.concat(all_chi_res, ignore_index=True)
                p_value_cols = [c for c in combined.columns if c not in ('cond_descr', 'cond_name', 'seed')]
                summary_rows = []
                for cond_name in combined['cond_name'].unique():
                    cond_data = combined[combined['cond_name'] == cond_name]
                    row = {'cond_name': cond_name}
                    for col in p_value_cols:
                        vals = cond_data[col].dropna().values
                        if len(vals) == 0:
                            row[f'{col}_fisher_p'] = np.nan
                            row[f'{col}_std'] = np.nan
                            row[f'{col}_n_sig'] = 0
                        else:
                            clipped = np.clip(vals, np.finfo(float).tiny, 1.0)
                            _, fisher_p = combine_pvalues(clipped, method='fisher')
                            row[f'{col}_fisher_p'] = round(float(fisher_p), 6)
                            row[f'{col}_std'] = round(float(vals.std()), 6) if len(vals) > 1 else np.nan
                            row[f'{col}_n_sig'] = int((vals < 0.05).sum())
                    summary_rows.append(row)
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(os.path.join(base_output_dir, 'cross_seed_summary.csv'), index=False)
                print(f"\nCross-seed summary saved to: {base_output_dir}/cross_seed_summary.csv")

            print("\nDone (multi-seed).")
            return

        # Single-seed experiment mode
        weight_suffix = "_w_" + args.feature_weights.replace(":", "").replace(",", "_") if args.feature_weights else ""
        output_dir = os.path.join(args.output_dir, f"{full_timestamp}_experiment_{dataset_name}_{args.algorithm}_{args.distance}_s{args.seed}{weight_suffix}")
        os.makedirs(output_dir, exist_ok=True)
        metadata = {
            'dataset': dataset_name,
            'algorithm': args.algorithm,
            'distance': args.distance,
            'seed': args.seed,
            'scoring_method': args.scoring,
            'timestamp': full_timestamp,
        }
        run_batch_experiment(df, args, output_dir, metadata=metadata)
        print("\nDone.")
        return

    # Single run mode
    full_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    weight_suffix = "_w_" + args.feature_weights.replace(":", "").replace(",", "_") if args.feature_weights else ""
    run_id = f"{full_timestamp}_{dataset_name}_{args.algorithm}_{args.distance}_s{args.seed}{weight_suffix}"
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    scoring_method = getattr(args, 'scoring', 'silhouette')
    metadata = pd.DataFrame([{
        'seed': args.seed,
        'algorithm': args.algorithm,
        'distance': args.distance,
        'dataset': dataset_name,
        'timestamp': full_timestamp,
        'scoring_method': scoring_method,
    }])
    metadata.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

    regular_cols = parse_column_list(args.regular_cols)
    sensitive_cols = parse_column_list(args.sensitive_cols)
    continuous_sensitive_cols = set(parse_column_list(getattr(args, 'continuous_sensitive_cols', None)) or [])
    proxy_cols = parse_column_list(args.proxy_cols)
    special_cols = parse_column_list(args.special_cols)
    original_sensitive_cols = list(sensitive_cols)

    unknown_continuous = continuous_sensitive_cols - set(sensitive_cols or [])
    if unknown_continuous:
        raise ValueError(
            f"--continuous_sensitive_cols entries not found in --sensitive_cols: {sorted(unknown_continuous)}"
        )

    # Encode categorical columns (one-hot for non-kprototypes; detect names for kprototypes).
    # Multi-class sensitive dummies stay in the DataFrame for fairness analysis but are
    # excluded from col_lists['sensitive'] so they don't inflate the feature matrix.
    categorical_cols_arg = parse_column_list(getattr(args, 'categorical_cols', None))
    col_lists = {'regular': regular_cols, 'sensitive': sensitive_cols,
                 'proxy': proxy_cols, 'special': special_cols}
    df, col_lists, categorical_col_names, multiclass_dummies, ohe_col_names = _encode_multiclass_categoricals(
        df, col_lists, categorical_cols_arg, args.algorithm, distance=args.distance
    )
    regular_cols = col_lists['regular']
    sensitive_cols = col_lists['sensitive']
    proxy_cols = col_lists['proxy']
    special_cols = col_lists['special']

    # Fairness-analysis sensitive list (see _build_sensitive_analysis_list).
    sensitive_cols_analysis = _build_sensitive_analysis_list(
        sensitive_cols, multiclass_dummies, original_sensitive_cols
    )

    # Build clustering features
    clustering_cols = regular_cols + sensitive_cols + proxy_cols + special_cols
    features = df[clustering_cols] if clustering_cols else df

    categorical_features = [i for i, c in enumerate(clustering_cols) if c in categorical_col_names] or None
    ohe_col_set = set(ohe_col_names)
    ohe_feature_indices = [i for i, c in enumerate(clustering_cols) if c in ohe_col_set] or None

    # Parse feature weights
    feature_weights = parse_feature_weights(
        args.feature_weights, regular_cols, sensitive_cols, special_cols, clustering_cols
    )                                                                                            
                                                                                                                                                                                        
    # Get y_true/y_pred from DataFrame if subset is requested                                                                                                                           
    y_true, y_pred = None, None                                                                                                                                                         
    if args.subset:                                                                                                                                                                     
        if args.y_true_col and args.y_pred_col:                                                                                                                             
            y_true = df[args.y_true_col].values                                                                                                                                         
            y_pred = df[args.y_pred_col].values                                                                                                                                         
        else:                                                                                                                                                                           
            raise ValueError("--y_true_col and --y_pred_col required when using --subset")                                                                                              
                                                                                                                                                                                        
    # Build scoring function for k-selection
    scoring_fn = None
    # Compute subset mask for scorer (same logic as cluster() uses internally)
    scorer_mask = None
    if args.subset and y_true is not None and y_pred is not None:
        if args.subset == "TP":
            scorer_mask = (y_true == 1) & (y_pred == 1)
        elif args.subset == "TN":
            scorer_mask = (y_true == 0) & (y_pred == 0)
        elif args.subset == "FP":
            scorer_mask = (y_true == 0) & (y_pred == 1)
        elif args.subset == "FN":
            scorer_mask = (y_true == 1) & (y_pred == 0)
        elif args.subset == "TP_TN":
            scorer_mask = y_true == y_pred
        elif args.subset == "FP_FN":
            scorer_mask = y_true != y_pred

    if args.scoring == "chi2_error":
        if not args.error_col:
            raise ValueError("--error_col required for chi2_error scoring")
        if args.error_type == 'regression':
            scoring_fn = make_kruskal_error_scorer(df[args.error_col].values, mask=scorer_mask)
        elif args.error_type == 'multiclass':
            scoring_fn = make_categorical_error_scorer(df[args.error_col].values, mask=scorer_mask)
        else:
            scoring_fn = make_chi2_error_scorer(df[args.error_col].values, mask=scorer_mask)
    elif args.scoring == "chi2_sensitive":
        if not sensitive_cols:
            raise ValueError("--sensitive_cols required for chi2_sensitive scoring")
        scoring_fn = make_chi2_sensitive_scorer(df[sensitive_cols[0]].values, mask=scorer_mask)
    elif args.scoring == "composite":
        if args.error_col or sensitive_cols:
            cw = {}
            for pair in args.composite_weights.split(','):
                name, w = pair.strip().split(':')
                cw[name.strip()] = float(w.strip())
            scoring_fn = make_composite_scorer(
                error_data=df[args.error_col].values if args.error_col else None,
                sensitive_data=df[sensitive_cols[0]].values if sensitive_cols else None,
                mask=scorer_mask,
                silhouette_weight=cw.get('silhouette', 0.3),
                error_weight=cw.get('error', 0.5),
                fairness_weight=cw.get('fairness', 0.2),
                error_type=args.error_type,
            )
        # else: no error_col or sensitive_cols -> scoring_fn stays None -> silhouette fallback

    # Run clustering
    print(f"\nClustering...")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Distance: {args.distance}")
    print(f"  Scoring: {args.scoring}")

    # Validate algorithm + distance combinations
    if args.algorithm == 'kprototypes' and args.distance == 'gower':
        print("Warning: KPrototypes uses its own distance metric. --distance gower is ignored.")
        print("For Gower-based clustering, use DBSCAN or HDBSCAN instead.")

    result = cluster(
        features=features,
        y_true=y_true,
        y_pred=y_pred,
        subset=args.subset,
        algorithm=args.algorithm,
        distance=args.distance,
        categorical_features=categorical_features if categorical_features else None,
        feature_weights=feature_weights,
        eps=args.eps,
        min_samples=args.min_samples,
        n_clusters=args.n_clusters,
        n_min=args.n_min,
        n_max=args.n_max,
        max_iter=args.max_iter,
        random_state=args.seed,
        min_datapoints=args.min_datapoints,
        scoring_fn=scoring_fn,
        standardize=not args.no_standardize,
        ohe_features=ohe_feature_indices,
    )

    # Results
    print(f"\nResults:")
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Noise: {result.n_noise}")
    if result.silhouette is not None:
        print(f"  Silhouette: {result.silhouette:.3f}")
    if result.calinski_harabasz is not None:
        print(f"  Calinski-Harabasz: {result.calinski_harabasz:.1f}")
    print(f"  Cluster sizes: {result.cluster_sizes}")

    # Build recap table (error stats, sensitive proportions, diff_vs_rest, p-values)
    if args.error_col and result.n_clusters > 1:
        res_df = df.copy()
        if result.mask is not None:
            res_df = res_df[result.mask].copy()
        res_df['clusters'] = result.labels

        recap = make_recap(res_df, clustering_cols,
                           sensitive_cols=sensitive_cols_analysis,
                           error_col=args.error_col,
                           error_type=args.error_type,
                           feature_matrix=result.feature_matrix,
                           distance_matrix=result.distance_matrix,
                           continuous_sensitive_cols=continuous_sensitive_cols,
                           multiclass_option=args.error_multiclass_option,
                           error_cols=args.error_cols,
                           error_cols_kind=args.error_cols_kind)

        # Save recap CSV
        recap_dir = os.path.join(output_dir, "recap")
        os.makedirs(recap_dir, exist_ok=True)
        run_name = f"{args.algorithm}_{args.distance}_k{result.n_clusters}"
        recap.to_csv(os.path.join(recap_dir, f"{run_name}.csv"), index=False)
        print(f"\nSaved: recap/{run_name}.csv")

        # Save recap heatmap
        if not args.no_plots and len(recap) > 1:
            error_label = getattr(args, 'error_label', None) or args.error_col or 'error'
            plot_cluster_recap_heatmap(recap.copy(), run_name, output_dir, error_label=error_label)
            print(f"Saved: {run_name}.png")

    # Separability check (chi-squared for categorical, Kruskal-Wallis for numeric)
    df_for_sep = df if result.mask is None else df[result.mask]
    all_cols_to_test = list(dict.fromkeys(clustering_cols + sensitive_cols))
    if result.n_clusters > 1:
        sep_results = separability_check(df_for_sep, result.labels, all_cols_to_test)
        if not sep_results.empty:
            sep_dir = os.path.join(output_dir, "separability")
            os.makedirs(sep_dir, exist_ok=True)
            sep_name = f"{args.algorithm}_{args.distance}_k{result.n_clusters}"
            sep_results.to_csv(os.path.join(sep_dir, f"{sep_name}.csv"))
            print(f"Saved: separability/{sep_name}.csv")
            if args.separability_check:
                print(f"\nSeparability check:")
                print(sep_results.to_string())
    elif args.separability_check:
        print("\nSeparability check:")
        print("  Not enough clusters for separability analysis")

    # Visualization
    if not args.no_plots:
        print(f"\nGenerating visualizations ({args.projection})...")

        if args.projection != "none":
            if args.distance == "gower" and result.distance_matrix is not None:
                # MDS on precomputed Gower matrix — only non-noise points have a distance entry
                non_noise = result.labels != -1
                X_2d = reduce_dimensions(result.distance_matrix, method="mds", precomputed=True)
                plot_clusters(X_2d, result.labels[non_noise],
                              title=f"Clusters ({args.algorithm}, gower+MDS)",
                              out_path=f"{output_dir}/clusters.png")
            else:
                # Standard Euclidean projection; drop categorical columns for kprototypes
                if categorical_features and args.algorithm == "kprototypes":
                    numeric_mask = [i for i in range(result.feature_matrix.shape[1]) if i not in categorical_features]
                    X_for_viz = result.feature_matrix[:, numeric_mask].astype(float)
                else:
                    X_for_viz = result.feature_matrix
                X_2d = reduce_dimensions(X_for_viz, method=args.projection)
                plot_clusters(X_2d, result.labels,
                              title=f"Clusters ({args.algorithm}, {args.distance})",
                              out_path=f"{output_dir}/clusters.png")                                                                                                                       
                                                                                                                                                                                        
        # Plot composition for each sensitive attribute                                                                                                                                 
        if sensitive_cols:                                                                                                                                                              
            for attr_name in sensitive_cols:                                                                                                                                            
                attr_for_eval = df[attr_name].values                                                                                                                                    
                if result.mask is not None:                                                                                                                                             
                    attr_for_eval = attr_for_eval[result.mask]                                                                                                                          
                plot_cluster_composition(result.labels, attr_for_eval, attr_name,                                                                                                       
                                        out_path=f"{output_dir}/composition_{attr_name}.png")                                                                                     
                                                                                                                                                                                          
        print(f"  Saved to {args.output_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
