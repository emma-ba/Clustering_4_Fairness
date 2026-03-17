import os, argparse, sys, re
import numpy as np
import pandas as pd
from src.clustering import cluster
from src.scoring import (
    make_chi2_error_scorer,
    make_kruskal_error_scorer,
    make_chi2_sensitive_scorer, make_composite_scorer,
)
from src.visualization import reduce_dimensions, plot_clusters, plot_cluster_composition
from src.fairness_metrics import evaluate_fairness, print_fairness_report
from src.experiments import (
    create_exp_conditions,
    run_experiments_generic, make_recap, make_chi_tests,
    recap_quali_metrics, plot_quality_heatmap, plot_cluster_recap_heatmap,
    separability_check
)
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_DATE = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = os.path.join(PROJECT_DIR, "clustering_results", SESSION_DATE)
DATA_DIR = "Data"

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
    # Note: For a general minimum cluster size filter across all algorithms, use --min_datapoints (post-hoc filter).
    # min_cluster_size is HDBSCAN-specific (built-in parameter controlling hierarchical extraction).
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="Minimum cluster size (HDBSCAN)")
    
    parser.add_argument("--min_samples", type=int, default=5,
                        help="Minimum samples (HDBSCAN)")
    # General parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (used by KMeans, BisectingKMeans, KMedoids, KPrototypes, and experiment mode)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed experiments (e.g., '42,123,456'). Mutually exclusive with --seed in experiment mode.")

    # KMeans parameters
    parser.add_argument("--max_iter", type=int, default=300,
                        help="Maximum iterations for KMeans/BisectingKMeans")

    # Scoring method for k-selection
    parser.add_argument("--scoring", type=str, default="silhouette",
                        choices=["silhouette", "chi2_error", "chi2_sensitive", "composite"],
                        help="Scoring method for k-search: silhouette (cluster quality), chi2_error (error separation), chi2_sensitive (fairness), composite (weighted combination)")
    parser.add_argument("--composite_weights", type=str, default="silhouette:0.3,error:0.5,fairness:0.2",
                        help="Weights for composite scoring as 'silhouette:W,error:W,fairness:W'")

    # Feature weights
    parser.add_argument("--feature_weights", type=str, default=None,
                        help="Feature weights as 'col:weight' pairs. Groups: 'regular:1.5,sensitive:0.5'. Individual: 'age:2.0'. Mixed: 'regular:1.0,age:2.0'")

    # Cluster filtering
    parser.add_argument("--min_datapoints", type=int, default=None,
                        help="Minimum datapoints per cluster (smaller clusters become noise)")

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
                        choices=["pca", "tsne", "none"],
                        help="Projection method for visualization (use 'none' to skip)")

    parser.add_argument("--regular_cols", type=str, default=None,                                                                                                                       
                          help="Regular features for clustering (comma-separated column names)")              
    # TODODONE: DHIGH PRIO. Statistical tests for numeric sensitive features. When 2 groups: Mann-Whitney. For more than 2 groups: Kruskal-Wallis.                                                                        
    #TODODONE : Save results at the global level. 1 table for each exp condition, 1 assessment metric
    # TODODONE - to review: Save results at individual level, have 1 set of results for each exp condition, where we have this in 1 csv, with 1 rule per cluste & 1vall stat test     
    # TODODONE: Do a PCA for the result too for each exp condition. 
    # TODOto review: Assess the feasibility of using the error as an input feature to the clustering or oversample the datapints. e.g. SMOTE for binary features
    # TODODONE: prepare demo of how to use the code now for testing, and present to others in an easy way. Make a README showing how it works and go through it during the meeting.
    # TODO: Experiment with Health Data. Not a priority. 
    # TODO: Parse crimes columns percentages by groups to make sense
    # TODO Create a new environment
    # TODO: Do the IQR for the student age, instead of having all ages. For numeric snesitive features ,we have 2 columns, the iqr and the P-value. 
    # TODO: Improve descriptions of arguments in READMe so that they are more self-explanatory.
    # TODO: Look into journales that take research artifacts. Or a DEMO at a conference.
    # TODO: Look into finding hte number of clusters if it works or not. Should wokr
    # TODO : Try clustering iteratively. 
    # TODO: Try rescaling the data. 
    # TODO: Try resampling the data, to keep the sample 50% errors, 50% non-errors. 
    # TODO: 
    parser.add_argument("--sensitive_cols", type=str, default=None,
                        help="Sensitive/protected attributes (comma-separated column names). Both binary (0/1) and multi-class columns are supported.")                                                                                           
    parser.add_argument("--proxy_cols", type=str, default=None,                                                                                                                         
                        help="Proxy features for sensitive attributes (comma-separated column names)")                                                                                  
    parser.add_argument("--special_cols", type=str, default=None,
                          help="Special features like SHAP values (comma-separated column names)")
    parser.add_argument("--error_col", type=str, default=None,
                        help="Error column for analysis. Binary (0/1) for classification, continuous for regression.")
    parser.add_argument("--error_type", type=str, default="binary",
                        choices=["binary", "regression"],
                        help="Type of error column: 'binary' (classification 0/1) or 'regression' (continuous). Default: binary")

    parser.add_argument("--data_path", type=str, required=True,
                          help="Path to input CSV file")
    # Output
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip saving visualization plots")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for plots")

    # Batch experiment mode
    parser.add_argument("--experiment", action="store_true",
                        help="Run batch experiment with all feature group combinations")

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
    special_cols = parse_column_list(args.special_cols)
    error_col = args.error_col

    # Validate: need error_col and at least one feature group
    if not error_col:
        raise ValueError("--error_col is required in experiment mode")
    if not regular_cols and not sensitive_cols and not special_cols:
        raise ValueError("At least one feature group (--regular_cols, --sensitive_cols, or --special_cols) is required in experiment mode")
    if not sensitive_cols:
        raise ValueError("--sensitive_cols is required in experiment mode (for proportion analysis)")

    # Build groups dict for condition generation
    groups = {}
    if regular_cols:
        groups['REG'] = regular_cols
    if sensitive_cols:
        groups['SEN'] = sensitive_cols
    groups['ERR'] = [error_col]
    if special_cols:
        groups['SPECIAL'] = special_cols

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
    if args.scoring != "silhouette":
        if args.scoring == "chi2_error":
            if not error_col:
                raise ValueError("--error_col required for chi2_error scoring")
            if args.error_type == 'regression':
                scoring_fn = make_kruskal_error_scorer(df[error_col].values)
            else:
                scoring_fn = make_chi2_error_scorer(df[error_col].values)
        elif args.scoring == "chi2_sensitive":
            if not sensitive_cols:
                raise ValueError("--sensitive_cols required for chi2_sensitive scoring")
            scoring_fn = make_chi2_sensitive_scorer(df[sensitive_cols[0]].values)
        elif args.scoring == "composite":
            if not error_col or not sensitive_cols:
                raise ValueError("--error_col and --sensitive_cols required for composite scoring")
            cw = {}
            for pair in args.composite_weights.split(','):
                name, w = pair.strip().split(':')
                cw[name.strip()] = float(w.strip())
            scoring_fn = make_composite_scorer(
                df[error_col].values,
                df[sensitive_cols[0]].values,
                silhouette_weight=cw.get('silhouette', 0.3),
                error_weight=cw.get('error', 0.5),
                fairness_weight=cw.get('fairness', 0.2),
                error_type=args.error_type,
            )

    # Parse feature weights
    all_clustering_cols = regular_cols + special_cols
    feature_weights = parse_feature_weights(
        args.feature_weights, regular_cols, sensitive_cols, special_cols, all_clustering_cols
    )

    # Run all experiments
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
        sensitive_cols=sensitive_cols,
        error_col=error_col,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        eps=args.eps,
        min_datapoints=args.min_datapoints,
        error_type=args.error_type,
        feature_weights=feature_weights,
    )

    # Print progress for each condition
    print()
    for i, cond_name in enumerate(results['cond_name']):
        recap = results['cond_recap'][i]
        n_clusters = len(recap)
        silhouette_avg = recap['silhouette'].mean() if 'silhouette' in recap.columns else np.nan
        print(f"Condition {i+1}/{len(results['cond_name'])}: {cond_name.strip()}")
        print(f"  Clusters: {n_clusters}, Silhouette: {silhouette_avg:.3f}" if not np.isnan(silhouette_avg) else f"  Clusters: {n_clusters}")

    # Generate chi-squared / Kruskal-Wallis test results
    chi_res = make_chi_tests(results, sensitive_cols=sensitive_cols,
                             error_type=args.error_type, error_col=error_col)
    chi_res.to_csv(f"{output_dir}/chi_res.csv", index=False)
    print(f"\nSaved: chi_res.csv")

    # Print chi-squared results summary
    # Use actual sensitive columns from chi_res (may be expanded for multi-class)
    actual_sensitive_cols = [c for c in chi_res.columns if c not in ('cond_descr', 'cond_name', 'error')]
    print("\nChi-squared test results:")
    chi_display_cols = ['cond_name', 'error'] + actual_sensitive_cols
    chi_display = chi_res[chi_display_cols].copy()
    chi_display.columns = ['Condition', 'error'] + actual_sensitive_cols
    print(chi_display.to_string(index=False))

    # Generate quality metrics
    all_quali = recap_quali_metrics(chi_res, results, exp_condition, sensitive_cols=sensitive_cols)

    # Create chi-squared heatmap visualization
    chi_viz_cols = ['error'] + actual_sensitive_cols
    chi_res_viz = chi_res[chi_viz_cols].copy()
    chi_res_viz.index = chi_res['cond_name'].str.strip()

    plt.figure(figsize=(max(4, len(chi_viz_cols) + 2), max(6, len(chi_res_viz) * 0.6)))
    ax = sns.heatmap(chi_res_viz, annot=True, center=0.05, cbar=False,
                     cmap=sns.color_palette("vlag", as_cmap=True), robust=True)
    ax.set_title("Chi-squared Test Results (p-values)")
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='major', length=0)
    ax.tick_params(axis='y', which='major', length=0)
    plt.yticks(rotation=0, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chi_res_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: chi_res_heatmap.png")

    # Create quality metrics heatmap
    quali_viz_cols = ['error'] + actual_sensitive_cols + ['silhouette']
    all_quali_viz = all_quali[quali_viz_cols].copy()
    all_quali_viz.index = all_quali['cond_name'].str.strip()
    plot_quality_heatmap(all_quali_viz, f"{output_dir}/all_quali_heatmap.png",
                         figsize=(max(4, len(quali_viz_cols) + 2), max(6, len(all_quali_viz) * 0.6)))
    plt.close()
    print(f"Saved: all_quali_heatmap.png")

    # Generate per-condition recap heatmaps and composition plots
    if not args.no_plots:
        print(f"\nGenerating {len(results['cond_name'])} recap heatmaps...")
        for i, cond_name in enumerate(results['cond_name']):
            recap = results['cond_recap'][i].copy()
            if len(recap) > 1:  # Only plot if there are multiple clusters
                plot_cluster_recap_heatmap(recap, cond_name, output_dir)
                plt.close()
        print(f"Saved: {len(results['cond_name'])} recap heatmaps")

        # Per-condition cluster scatter plots (tsne/pca)
        if args.projection != "none":
            print(f"Generating cluster scatter plots...")
            for i, cond_name in enumerate(results['cond_name']):
                res_df = results['cond_res'][i]
                labels = res_df['clusters'].values
                feature_set = exp_condition['feature_set'][i]
                if len(set(labels) - {-1}) > 1:
                    cond_clean = re.sub(r'\s+', '', cond_name)
                    X = res_df[feature_set].values.astype(float)
                    X_2d = reduce_dimensions(X, method=args.projection)
                    plot_clusters(X_2d, labels,
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
                for attr in sensitive_cols:
                    plot_cluster_composition(labels, res_df[attr].values, attr,
                        out_path=f"{output_dir}/{cond_clean}_composition_{attr}.png")
                    plt.close()
        print(f"Saved: composition plots")

    # --- CSV outputs ---

    # Global summary CSV: one row per condition, one primary assessment metric (KW p-value for error)
    # Metadata columns prepended so the file is self-contained for analysis
    summary_rows = []
    chi_res_lookup = chi_res.set_index('cond_name') if chi_res is not None else None
    for i, cond_name in enumerate(results['cond_name']):
        recap = results['cond_recap'][i]
        kw_p = chi_res_lookup.loc[cond_name, 'error'] if chi_res_lookup is not None and cond_name in chi_res_lookup.index else np.nan
        row = {} if metadata is None else dict(metadata)
        row.update({
            'cond_name': cond_name,
            'cond_descr': results['cond_descr'][i],
            'n_clusters': len(recap),
            'kw_p_error': round(kw_p, 4) if not np.isnan(kw_p) else np.nan,
            'silhouette_avg': round(recap['silhouette'].mean(), 4) if 'silhouette' in recap.columns else np.nan,
            'error_rate_avg': round(recap['error_rate'].mean(), 4) if 'error_rate' in recap.columns else np.nan,
            'error_mean_avg': round(recap['error_mean'].mean(), 4) if 'error_mean' in recap.columns else np.nan,
            'abs_error_mean_avg': round(recap['abs_error_mean'].mean(), 4) if 'abs_error_mean' in recap.columns else np.nan,
        })
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
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

        # Add rule: most distinctive sensitive feature per cluster (highest abs _diff)
        diff_cols = [c for c in recap_i.columns if c.endswith('_diff')]
        if diff_cols:
            def _make_rule(row):
                best_col = row[diff_cols].abs().idxmax()
                best_val = row[best_col]
                feature = best_col.replace('_diff', '')
                direction = '+' if best_val > 0 else ''
                return f"{feature} ({direction}{round(best_val, 3)})"
            recap_i.insert(1, 'rule', recap_i.apply(_make_rule, axis=1))
        else:
            recap_i.insert(1, 'rule', '')

        # Add OVERALL row with Kruskal-Wallis stat test
        kw_p = chi_res_lookup.loc[cond_name, 'error'] if chi_res_lookup is not None and cond_name in chi_res_lookup.index else np.nan
        overall_row = {col: '' for col in recap_i.columns}
        overall_row['c'] = 'OVERALL'
        overall_row['rule'] = f"kruskallwallis_p (error): {round(kw_p, 4) if not np.isnan(kw_p) else 'n/a'}"
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
    print("  - results_summary.csv (global: 1 row per condition, kw_p_error as key metric)")
    print(f"  - {len(results['cond_name'])} per-condition CSVs (1 row per cluster + OVERALL stat test)")
    print("  - chi_res.csv / chi_res_heatmap.png")
    print("  - all_quali_heatmap.png")
    print("  - exp_condition.csv")
    if not args.no_plots:
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

    # Experiment mode: run all conditions
    if args.experiment:
        full_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Multi-seed experiment mode
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
            base_output_dir = os.path.join(args.output_dir, f"{full_timestamp}_experiment_{dataset_name}_multi_seed")
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
                # Compute mean/std of numeric columns grouped by condition
                numeric_cols = [c for c in combined.columns if c not in ('cond_descr', 'cond_name', 'seed')]
                summary_rows = []
                for cond_name in combined['cond_name'].unique():
                    cond_data = combined[combined['cond_name'] == cond_name]
                    row = {'cond_name': cond_name}
                    for col in numeric_cols:
                        vals = cond_data[col].dropna()
                        row[f'{col}_mean'] = vals.mean() if len(vals) > 0 else np.nan
                        row[f'{col}_std'] = vals.std() if len(vals) > 1 else np.nan
                    summary_rows.append(row)
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(os.path.join(base_output_dir, 'cross_seed_summary.csv'), index=False)
                print(f"\nCross-seed summary saved to: {base_output_dir}/cross_seed_summary.csv")

            print("\nDone (multi-seed).")
            return

        # Single-seed experiment mode
        output_dir = os.path.join(args.output_dir, f"{full_timestamp}_experiment_{dataset_name}_{args.algorithm}_{args.distance}_s{args.seed}")
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
    run_id = f"{full_timestamp}_{dataset_name}_{args.algorithm}_{args.distance}_s{args.seed}"
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
    proxy_cols = parse_column_list(args.proxy_cols)
    special_cols = parse_column_list(args.special_cols)                                                                                                                                                    
                                                                                                                                                                                        
    # Build clustering features 
    clustering_cols = regular_cols + sensitive_cols + proxy_cols + special_cols
    features = df[clustering_cols] if clustering_cols else df

    # Identify categorical columns (string-like or category dtype)
    categorical_features = [i for i, col in enumerate(clustering_cols)
                            if df[col].dtype.kind in ('O', 'U', 'S') or df[col].dtype.name == 'category'
                            or str(df[col].dtype) in ('string', 'str')]

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
    if args.scoring != "silhouette":
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
            else:
                scoring_fn = make_chi2_error_scorer(df[args.error_col].values, mask=scorer_mask)
        elif args.scoring == "chi2_sensitive":
            if not sensitive_cols:
                raise ValueError("--sensitive_cols required for chi2_sensitive scoring")
            scoring_fn = make_chi2_sensitive_scorer(df[sensitive_cols[0]].values, mask=scorer_mask)
        elif args.scoring == "composite":
            if not args.error_col or not sensitive_cols:
                raise ValueError("--error_col and --sensitive_cols required for composite scoring")
            # Parse composite weights
            cw = {}
            for pair in args.composite_weights.split(','):
                name, w = pair.strip().split(':')
                cw[name.strip()] = float(w.strip())
            scoring_fn = make_composite_scorer(
                df[args.error_col].values,
                df[sensitive_cols[0]].values,
                mask=scorer_mask,
                silhouette_weight=cw.get('silhouette', 0.3),
                error_weight=cw.get('error', 0.5),
                fairness_weight=cw.get('fairness', 0.2),
                error_type=args.error_type,
            )

    # Run clustering
    print(f"\nClustering...")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Distance: {args.distance}")
    if args.scoring != "silhouette":
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
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        n_clusters=args.n_clusters,
        n_min=args.n_min,
        n_max=args.n_max,
        max_iter=args.max_iter,
        random_state=args.seed,
        min_datapoints=args.min_datapoints,
        scoring_fn=scoring_fn,
    )

    # Results
    print(f"\nResults:")
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Noise: {result.n_noise}")
    if result.silhouette:
        print(f"  Silhouette: {result.silhouette:.3f}")
    if result.calinski_harabasz:
        print(f"  Calinski-Harabasz: {result.calinski_harabasz:.1f}")
    print(f"  Cluster sizes: {result.cluster_sizes}")

    # Fairness evaluation
    if sensitive_cols:
        print(f"\nFairness evaluation:")
        for attr_name in sensitive_cols:
            attr_for_eval = df[attr_name].values
            if result.mask is not None:
                attr_for_eval = attr_for_eval[result.mask]
            metrics = evaluate_fairness(result, attr_for_eval, attr_name)
            print(print_fairness_report(metrics, attr_name, attribute_labels=None))

    # Build recap table (error stats, sensitive proportions, diff_vs_rest, p-values)
    if args.error_col and result.n_clusters > 1:
        res_df = df.copy()
        if result.mask is not None:
            res_df = res_df[result.mask].copy()
        res_df['clusters'] = result.labels

        recap = make_recap(res_df, clustering_cols,
                           sensitive_cols=sensitive_cols,
                           error_col=args.error_col,
                           error_type=args.error_type)

        # Save recap CSV
        recap_dir = os.path.join(output_dir, "recap")
        os.makedirs(recap_dir, exist_ok=True)
        run_name = f"{args.algorithm}_{args.distance}_k{result.n_clusters}"
        recap.to_csv(os.path.join(recap_dir, f"{run_name}.csv"), index=False)
        print(f"\nSaved: recap/{run_name}.csv")

        # Save recap heatmap
        if not args.no_plots and len(recap) > 1:
            plot_cluster_recap_heatmap(recap.copy(), run_name, output_dir)
            print(f"Saved: {run_name}.png")

    # Separability check (chi-squared for categorical, Kruskal-Wallis for numeric)
    df_for_sep = df if result.mask is None else df[result.mask]
    all_cols_to_test = clustering_cols + sensitive_cols
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
            # For mixed-type data (kprototypes or gower), use only numeric columns for projection
            if categorical_features and (args.algorithm == "kprototypes" or args.distance == "gower"):
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
