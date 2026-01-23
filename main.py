import os, argparse, sys
import numpy as np
import pandas as pd
from src.clustering import cluster
from src.visualization import reduce_dimensions, plot_clusters, plot_cluster_composition
from src.fairness_metrics import evaluate_fairness, print_fairness_report
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


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering for fairness analysis")
    
    parser.add_argument("--algorithm", type=str, default="hdbscan",
                        choices=["dbscan", "hdbscan", "kmeans", "bisectingkmeans", "kprototype"],
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
    # TODO also add k_prototype (somethign else, like K-means, but doens't work well on a mix of cat/num)
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="Minimum cluster size (HDBSCAN)")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="Minimum samples (HDBSCAN)")

    # Feature weights
    parser.add_argument("--weight_gender", type=float, default=1.0,
                        help="Weight for gender feature")
    parser.add_argument("--weight_age", type=float, default=1.0,
                        help="Weight for age feature")

    
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
                        choices=["umap", "pca", "tsne"],
                        help="Projection method for visualization")

    # What to cluster
    # TODO: Remove for now (we'll see)
    # parser.add_argument("--target", type=str, default="users",
    #                     choices=["users", "items"],
    #                     help="Cluster users or items")

    # TODO Faudrait que les gens puissent donner une liste avec les noms de colomnes pour chaque categorie qu'on a Regular/sensitive/proxy et toutes peuvent etre vides. (special on met dans les exp_condition)
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

    return parser.parse_args()

def main():                                                                                                                                                                             
    args = parse_args()                                                                                                                                                                 
    session_date = datetime.now().strftime('%Y-%m-%d')                                                                        
    run_id = f"{session_date}_{args.algorithm}_{args.distance}"                                                               
    output_dir = os.path.join(args.output_dir, run_id)                                                                        
    os.makedirs(output_dir, exist_ok=True)                                                                                                                                              
    # Parse column lists                                                                                                                                                                
    regular_cols = parse_column_list(args.regular_cols)                                                                                                                                 
    sensitive_cols = parse_column_list(args.sensitive_cols)                                                                                                                             
    proxy_cols = parse_column_list(args.proxy_cols)                                                                                                                                     
    special_cols = parse_column_list(args.special_cols)                                                                                                                                 
                                                                                                                                                                                        
    print(f"Loading data...")                                                                                                                                                           
    df = pd.read_csv(args.data_path)                                                                                                                                                    
                                                                                                                                                                                        
    # Build clustering features                                                                                                                                                         
    clustering_cols = regular_cols + proxy_cols                                                                                                                                         
    features = df[clustering_cols] if clustering_cols else df                                                                                                                           
                                                                                                                                                                                        
    # Identify categorical columns                                                                                                                                                      
    categorical_features = [i for i, col in enumerate(clustering_cols)                                                                                                                  
                            if df[col].dtype == 'object' or df[col].dtype.name == 'category']                                                                                            
                                                                                                                                                                                        
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

    result = cluster(
        features=features,
        y_true=y_true,
        y_pred=y_pred,
        subset=args.subset,
        algorithm=args.algorithm,
        distance=args.distance,
        categorical_features=categorical_features if categorical_features else None,                                                                                                    
        eps=args.eps,                                                                                                                                                                   
        min_cluster_size=args.min_cluster_size,                                                                                                                                         
        min_samples=args.min_samples,                                                                                                                                                   
        n_clusters=args.n_clusters,                                                                                                                                                     
        n_min=args.n_min,                                                                                                                                                               
        n_max=args.n_max,
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
              print(print_fairness_report(metrics, attr_name, attr_labels=None))
    # Visualization
    if args.save_plots:
        print(f"\nGenerating visualizations ({args.projection})...")                                                                                                                    
                                                                                                                                                                                          
        X_2d = reduce_dimensions(result.feature_matrix, method=args.projection)                                                                                                         
                                                                                                                                                                                        
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
