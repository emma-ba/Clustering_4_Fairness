import os, argparse, sys
import numpy as np
import pandas as pd
from src.clustering import cluster
from src.visualization import reduce_dimensions, plot_clusters, plot_cluster_composition
from src.fairness_metrics import evaluate_fairness, print_fairness_report
from src.experiments import (
    create_default_exp_conditions, run_experiments, make_chi_tests,
    recap_quali_metrics, plot_quality_heatmap, plot_cluster_recap_heatmap
)
from datetime import datetime

SESSION_DATE = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = f"visualization/clustering_results/{SESSION_DATE}"
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
                        choices=["dbscan", "hdbscan", "kmeans", "bisectingkmeans", "kprototypes"],
                        help="Clustering algorithm")

    # Distance metric
    parser.add_argument("--distance", type=str, default="euclidean",
                        choices=["euclidean", "manhattan", "gower"],
                        help="Distance metric")
    
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Exact number of clusters (mutually exclusive with n_min/n_max)")
    parser.add_argument("--n_min", type=int, default=2,
                        help="Minimum number of clusters (for range-based search)")
    parser.add_argument("--n_max", type=int, default=10,
                        help="Maximum number of clusters (for range-based search")

    # DBSCAN parameters                                                                                                                                                                 
    parser.add_argument("--eps", type=float, default=0.5,                                                                                                                               
                        help="Maximum distance between samples for neighborhood (DBSCAN)")
    
    # HDBSCAN parameters
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="Minimum cluster size (HDBSCAN)")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="Minimum samples (HDBSCAN)")

    parser.add_argument("--max_iter", type=int, default=300,
                        help="Maximum iterations for KMeans/BisectingKMeans")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Feature weights
    parser.add_argument("--feature_weights", type=str, default=None,
                        help="Feature weights as 'col:weight' pairs. Groups: 'regular:1.5,sensitive:0.5'. Individual: 'age:2.0'. Mixed: 'regular:1.0,age:2.0'")

    # Cluster filtering
    parser.add_argument("--min_datapoints", type=int, default=None,
                        help="Minimum datapoints per cluster (smaller clusters become noise)")

    # Statistical tests
    parser.add_argument("--separability_check", action="store_true",
                        help="Run chi-squared/Kruskal-Wallis tests on clusters")

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
                        choices=["pca", "tsne"],
                        help="Projection method for visualization (UMAP disabled)")

    # What to cluster
    # : Remove for now (we'll see)
    # parser.add_argument("--target", type=str, default="users",
    #                     choices=["users", "items"],
    #                     help="Cluster users or items")

    #  Faudrait que les gens puissent donner une liste avec les noms de colomnes pour chaque categorie qu'on a Regular/sensitive/proxy et toutes peuvent etre vides. (special on met dans les exp_condition)
    parser.add_argument("--regular_cols", type=str, default=None,                                                                                                                       
                          help="Regular features for clustering (comma-separated column names)")                                                                                          
    parser.add_argument("--sensitive_cols", type=str, default=None,                                                                                                                     
                        help="Sensitive/protected attributes (comma-separated column names)")                                                                                           
    parser.add_argument("--proxy_cols", type=str, default=None,                                                                                                                         
                        help="Proxy features for sensitive attributes (comma-separated column names)")                                                                                  
    parser.add_argument("--special_cols", type=str, default=None,                                                                                                                       
                          help="Special features like SHAP values (comma-separated column names)")


    parser.add_argument("--data_path", type=str, required=True,
                          help="Path to input CSV file")
    # Output
    parser.add_argument("--save_plots", action="store_true",
                        help="Save visualization plots")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for plots")

    # Batch experiment mode
    parser.add_argument("--experiment", action="store_true",
                        help="Run batch experiment with all 18 COMPAS feature combinations")

    return parser.parse_args()


def run_batch_experiment(df, args, output_dir):
    """
    Run all 18 experimental conditions and generate outputs.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with all required columns.
    args : argparse.Namespace
        CLI arguments.
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

    # Create experimental conditions
    exp_condition = create_default_exp_conditions()
    print(f"  Conditions: {len(exp_condition)}")

    # Save experimental conditions table
    exp_condition_save = exp_condition[['feature_set_descr', 'feature_set_name']].copy()
    exp_condition_save['feature_set'] = exp_condition['feature_set'].apply(lambda x: ', '.join(x))
    exp_condition_save.to_csv(f"{output_dir}/exp_condition.csv", index=False)
    print(f"\nSaved: exp_condition.csv")

    # Run all experiments
    results = run_experiments(
        df,
        exp_condition,
        min_splittable_cluster_prop=0.05,
        min_acceptable_cluster_prop=0.05,
        min_acceptable_error_diff=0.005,
        max_iter=100,
        eps=1,
        seed=args.seed
    )

    # Print progress for each condition
    print()
    for i, cond_name in enumerate(results['cond_name']):
        recap = results['cond_recap'][i]
        n_clusters = len(recap)
        silhouette_avg = recap['silhouette'].mean() if 'silhouette' in recap.columns else np.nan
        print(f"Condition {i+1}/{len(results['cond_name'])}: {cond_name.strip()}")
        print(f"  Clusters: {n_clusters}, Silhouette: {silhouette_avg:.3f}" if not np.isnan(silhouette_avg) else f"  Clusters: {n_clusters}")

    # Generate chi-squared test results
    chi_res = make_chi_tests(results)
    chi_res.to_csv(f"{output_dir}/chi_res.csv", index=False)
    print(f"\nSaved: chi_res.csv")

    # Print chi-squared results summary
    print("\nChi-squared test results:")
    chi_display = chi_res[['cond_name', 'error', 'race_aa', 'race_c', 'gender']].copy()
    chi_display.columns = ['Condition', 'error', 'race_aa', 'race_c', 'gender']
    print(chi_display.to_string(index=False))

    # Generate quality metrics
    all_quali = recap_quali_metrics(chi_res, results, exp_condition)

    # Create chi-squared heatmap visualization
    chi_res_viz = chi_res[['error', 'race_aa', 'race_c', 'gender']].copy()
    chi_res_viz.index = chi_res['cond_name'].str.strip()

    plt.figure(figsize=(6, 10))
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
    all_quali_viz = all_quali[['error', 'race_aa', 'race_c', 'gender', 'silhouette']].copy()
    all_quali_viz.index = all_quali['cond_name'].str.strip()
    plot_quality_heatmap(all_quali_viz, f"{output_dir}/all_quali_heatmap.png", figsize=(6, 10))
    plt.close()
    print(f"Saved: all_quali_heatmap.png")

    # Generate per-condition recap heatmaps
    if args.save_plots:
        print(f"\nGenerating {len(results['cond_name'])} recap heatmaps...")
        for i, cond_name in enumerate(results['cond_name']):
            recap = results['cond_recap'][i].copy()
            if len(recap) > 1:  # Only plot if there are multiple clusters
                plot_cluster_recap_heatmap(recap, cond_name, output_dir)
                plt.close()
        print(f"Saved: {len(results['cond_name'])} recap heatmaps")

    print(f"\nAll outputs saved to: {output_dir}/")
    print("  - exp_condition.csv")
    print("  - chi_res.csv")
    print("  - chi_res_heatmap.png")
    print("  - all_quali_heatmap.png")
    if args.save_plots:
        print(f"  - {len(results['cond_name'])} recap heatmaps")

    return results


def main():
    args = parse_args()
    session_date = datetime.now().strftime('%Y-%m-%d')
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    print(f"Loading data...")
    df = pd.read_csv(args.data_path)

    # Experiment mode: run all 18 conditions
    if args.experiment:
        full_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = os.path.join(args.output_dir, f"{full_timestamp}_experiment_{dataset_name}")
        os.makedirs(output_dir, exist_ok=True)
        run_batch_experiment(df, args, output_dir)
        print("\nDone.")
        return

    # Single run mode
    full_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_id = f"{full_timestamp}_{dataset_name}_{args.algorithm}_{args.distance}"
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Parse column lists
    regular_cols = parse_column_list(args.regular_cols)
    sensitive_cols = parse_column_list(args.sensitive_cols)
    proxy_cols = parse_column_list(args.proxy_cols)
    special_cols = parse_column_list(args.special_cols)                                                                                                                                                    
                                                                                                                                                                                        
    # Build clustering features
    clustering_cols = regular_cols + proxy_cols + special_cols
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
                                                                                                                                                                                        
    # Run clustering                                                                                                                                                                    
    print(f"\nClustering...")                                                                                                                                                           
    print(f"  Algorithm: {args.algorithm}")                                                                                                                                             
    print(f"  Distance: {args.distance}") 

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

    # Separability check (chi-squared for categorical, Kruskal-Wallis for numeric)
    if args.separability_check:
        from src.experiments import separability_check
        print(f"\nSeparability check:")
        # Get the data subset if applicable
        df_for_sep = df if result.mask is None else df[result.mask]
        all_cols_to_test = clustering_cols + sensitive_cols
        sep_results = separability_check(df_for_sep, result.labels, all_cols_to_test)
        if not sep_results.empty:
            print(sep_results.to_string())
        else:
            print("  Not enough clusters for separability analysis")

    # Visualization
    if args.save_plots:
        print(f"\nGenerating visualizations ({args.projection})...")

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
