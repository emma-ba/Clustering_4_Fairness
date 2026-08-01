"""c4fairness entry point: dispatch single-run / experiment / multi-seed modes."""

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
from datetime import datetime
from c4fairness.clustering import cluster
from c4fairness.scoring import (
    make_chi2_error_scorer,
    make_kruskal_error_scorer,
    make_categorical_error_scorer,
    make_chi2_sensitive_scorer,
    make_composite_scorer,
)
from c4fairness.visualization import reduce_dimensions, plot_clusters, plot_cluster_composition
from c4fairness.experiments import make_recap, separability_check, plot_cluster_recap_heatmap
from c4fairness.preprocessing import encode_categoricals
from c4fairness.fairness_metrics import multiclass_error_types, binary_error_rate_column
from c4fairness.cli import (
    parse_args,
    parse_column_list,
    parse_feature_weights,
    build_sensitive_analysis_list,
    apply_salient_reconstruction,
    parse_projection_list,
    parse_label_map,
)
from c4fairness.experiment import run_batch_experiment


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

    session_date = datetime.now().strftime("%Y-%m-%d")
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    if args.error_type == "regression" and args.subset:
        raise ValueError(
            "--subset (TP/TN/FP/FN) is not compatible with --error_type regression. "
            "Confusion matrix subsets only apply to binary classification."
        )

    print(f"Loading data...")
    df = pd.read_csv(args.data_path)

    if args.error_type == "regression" and not args.error_col:
        if args.y_true_col and args.y_pred_col:
            df["_regression_error"] = df[args.y_true_col] - df[args.y_pred_col]
            args.error_col = "_regression_error"
            print(
                f"  Auto-computed signed regression error: {args.y_true_col} - {args.y_pred_col}"
            )
        else:
            raise ValueError(
                "--error_type regression requires either --error_col or both --y_true_col and --y_pred_col"
            )

    # Binary error rate (fpr/fnr/precision/prec_neg): a masked {1,0,NaN} column whose
    # per-cluster mean is the rate. Recap-only; clustering/scoring keep the raw 0/1 signal.
    args.error_analysis_col = None
    if args.error_type == "binary" and args.binary_error_metric != "raw":
        if not (args.y_true_col and args.y_pred_col):
            raise ValueError(
                f"--binary_error_metric {args.binary_error_metric} requires "
                "--y_true_col and --y_pred_col."
            )
        df["_binary_error_rate"] = binary_error_rate_column(
            df[args.y_true_col], df[args.y_pred_col], args.binary_error_metric
        ).values
        args.error_analysis_col = "_binary_error_rate"
        if not args.error_col:
            df["_binary_misclassified"] = (
                df[args.y_true_col] != df[args.y_pred_col]
            ).astype(int)
            args.error_col = "_binary_misclassified"
        if not getattr(args, "error_label", None):
            args.error_label = {
                "fpr": "FP Rate", "fnr": "FN Rate",
                "precision": "1 - Precision", "prec_neg": "1 - Prec. (neg)",
            }[args.binary_error_metric]
        print(
            f"  Binary error metric '{args.binary_error_metric}' "
            f"from {args.y_true_col} vs {args.y_pred_col}"
        )

    # Multi-class error: derive a categorical/indicator error column from y_true/y_pred.
    # error_cols (+ error_cols_kind) is set for the per-class multi-column options
    # (onehot = binary indicators, classwise = TP/FN/FP/TN multi-categorical).
    args.error_cols = None
    args.error_cols_kind = "binary"
    if args.error_type == "multiclass":
        if not (args.y_true_col and args.y_pred_col):
            raise ValueError(
                "--error_type multiclass requires both --y_true_col and --y_pred_col"
            )
        err_df = multiclass_error_types(
            df[args.y_true_col], df[args.y_pred_col], args.error_multiclass_option
        )
        # ERR clustering needs a numeric feature: per_class/per_cell error columns are
        # categorical labels. error_cluster_col (0/1) is the ERR group; the derived
        # error_col drives scoring + the result tables.
        df["_multiclass_error_ind"] = (
            df[args.y_true_col] != df[args.y_pred_col]
        ).astype(int)
        args.error_cluster_col = "_multiclass_error_ind"
        if args.error_multiclass_option in ("onehot", "classwise"):
            # One error column per class -> one result-table set each.
            for col in err_df.columns:
                df[col] = err_df[col].values
            args.error_cols = list(err_df.columns)
            args.error_cols_kind = (
                "binary" if args.error_multiclass_option == "onehot" else "multicat"
            )
            df["_multiclass_error"] = df["_multiclass_error_ind"]
            args.error_col = "_multiclass_error"
        else:
            df["_multiclass_error"] = err_df["error"].values
            args.error_col = "_multiclass_error"
        print(
            f"  Derived multi-class error ('{args.error_multiclass_option}') "
            f"from {args.y_true_col} vs {args.y_pred_col}"
        )

    # Experiment mode: run all conditions
    if args.experiment is not None:
        full_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Multi-seed experiment mode
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
            seeds_str = "_".join(f"s{s}" for s in seeds)
            weight_suffix = (
                "_w_" + args.feature_weights.replace(":", "").replace(",", "_")
                if args.feature_weights
                else ""
            )
            base_output_dir = os.path.join(
                args.output_dir,
                f"{full_timestamp}_experiment_{dataset_name}_{seeds_str}{weight_suffix}",
            )
            os.makedirs(base_output_dir, exist_ok=True)

            all_chi_res = []

            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"Running experiment with seed={seed}")
                print(f"{'='*60}")
                seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
                os.makedirs(seed_dir, exist_ok=True)
                metadata = pd.DataFrame(
                    [
                        {
                            "seed": seed,
                            "algorithm": args.algorithm,
                            "distance": args.distance,
                            "dataset": dataset_name,
                            "timestamp": full_timestamp,
                            "scoring_method": args.scoring,
                        }
                    ]
                )
                metadata.to_csv(os.path.join(seed_dir, "metadata.csv"), index=False)
                args.seed = seed
                run_batch_experiment(df, args, seed_dir)

                chi_path = os.path.join(seed_dir, "chi_res.csv")
                if os.path.exists(chi_path):
                    chi_df = pd.read_csv(chi_path)
                    chi_df["seed"] = seed
                    all_chi_res.append(chi_df)

            if all_chi_res:
                combined = pd.concat(all_chi_res, ignore_index=True)
                p_value_cols = [
                    c
                    for c in combined.columns
                    if c not in ("cond_descr", "cond_name", "seed")
                ]
                summary_rows = []
                for cond_name in combined["cond_name"].unique():
                    cond_data = combined[combined["cond_name"] == cond_name]
                    row = {"cond_name": cond_name}
                    for col in p_value_cols:
                        vals = cond_data[col].dropna().values
                        if len(vals) == 0:
                            row[f"{col}_fisher_p"] = np.nan
                            row[f"{col}_std"] = np.nan
                            row[f"{col}_n_sig"] = 0
                        else:
                            clipped = np.clip(vals, np.finfo(float).tiny, 1.0)
                            _, fisher_p = combine_pvalues(clipped, method="fisher")
                            row[f"{col}_fisher_p"] = round(float(fisher_p), 6)
                            row[f"{col}_std"] = (
                                round(float(vals.std()), 6) if len(vals) > 1 else np.nan
                            )
                            row[f"{col}_n_sig"] = int((vals < 0.05).sum())
                    summary_rows.append(row)
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(
                    os.path.join(base_output_dir, "cross_seed_summary.csv"), index=False
                )
                print(
                    f"\nCross-seed summary saved to: {base_output_dir}/cross_seed_summary.csv"
                )

            print("\nDone (multi-seed).")
            return

        # Single-seed experiment mode
        weight_suffix = (
            "_w_" + args.feature_weights.replace(":", "").replace(",", "_")
            if args.feature_weights
            else ""
        )
        output_dir = os.path.join(
            args.output_dir,
            f"{full_timestamp}_experiment_{dataset_name}_{args.algorithm}_{args.distance}_s{args.seed}{weight_suffix}",
        )
        os.makedirs(output_dir, exist_ok=True)
        metadata = {
            "dataset": dataset_name,
            "algorithm": args.algorithm,
            "distance": args.distance,
            "seed": args.seed,
            "scoring_method": args.scoring,
            "timestamp": full_timestamp,
        }
        run_batch_experiment(df, args, output_dir, metadata=metadata)
        print("\nDone.")
        return

    # Single run mode
    full_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    weight_suffix = (
        "_w_" + args.feature_weights.replace(":", "").replace(",", "_")
        if args.feature_weights
        else ""
    )
    run_id = f"{full_timestamp}_{dataset_name}_{args.algorithm}_{args.distance}_s{args.seed}{weight_suffix}"
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    scoring_method = getattr(args, "scoring", "silhouette")
    metadata = pd.DataFrame(
        [
            {
                "seed": args.seed,
                "algorithm": args.algorithm,
                "distance": args.distance,
                "dataset": dataset_name,
                "timestamp": full_timestamp,
                "scoring_method": scoring_method,
            }
        ]
    )
    metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    regular_cols = parse_column_list(args.regular_cols)
    sensitive_cols = parse_column_list(args.sensitive_cols)
    continuous_sensitive_cols = set(
        parse_column_list(getattr(args, "continuous_sensitive_cols", None)) or []
    )
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
    categorical_cols_arg = parse_column_list(getattr(args, "categorical_cols", None))
    col_lists = {
        "regular": regular_cols,
        "sensitive": sensitive_cols,
        "proxy": proxy_cols,
        "special": special_cols,
    }
    df, col_lists, categorical_col_names, multiclass_dummies, ohe_col_names = (
        encode_categoricals(
            df, col_lists, categorical_cols_arg, args.algorithm, distance=args.distance
        )
    )
    regular_cols = col_lists["regular"]
    sensitive_cols = col_lists["sensitive"]
    proxy_cols = col_lists["proxy"]
    special_cols = col_lists["special"]

    # Fairness-analysis sensitive list (see build_sensitive_analysis_list).
    sensitive_cols_analysis = build_sensitive_analysis_list(
        sensitive_cols, multiclass_dummies, original_sensitive_cols,
        option=args.multicat_table_option,
    )

    clustering_cols = regular_cols + sensitive_cols + proxy_cols + special_cols
    features = df[clustering_cols] if clustering_cols else df

    categorical_features = [
        i for i, c in enumerate(clustering_cols) if c in categorical_col_names
    ] or None
    ohe_col_set = set(ohe_col_names)
    ohe_feature_indices = [
        i for i, c in enumerate(clustering_cols) if c in ohe_col_set
    ] or None

    feature_weights = parse_feature_weights(
        args.feature_weights,
        regular_cols,
        sensitive_cols,
        special_cols,
        clustering_cols,
    )
    y_true, y_pred = None, None
    if args.subset:
        if args.y_true_col and args.y_pred_col:
            y_true = df[args.y_true_col].values
            y_pred = df[args.y_pred_col].values
        else:
            raise ValueError(
                "--y_true_col and --y_pred_col required when using --subset"
            )
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
        if args.error_type == "regression":
            scoring_fn = make_kruskal_error_scorer(
                df[args.error_col].values, mask=scorer_mask
            )
        elif args.error_type == "multiclass":
            scoring_fn = make_categorical_error_scorer(
                df[args.error_col].values, mask=scorer_mask
            )
        else:
            scoring_fn = make_chi2_error_scorer(
                df[args.error_col].values, mask=scorer_mask
            )
    elif args.scoring == "chi2_sensitive":
        if not sensitive_cols:
            raise ValueError("--sensitive_cols required for chi2_sensitive scoring")
        scoring_fn = make_chi2_sensitive_scorer(
            df[sensitive_cols[0]].values, mask=scorer_mask
        )
    elif args.scoring == "composite":
        if args.error_col or sensitive_cols:
            cw = {}
            for pair in args.composite_weights.split(","):
                name, w = pair.strip().split(":")
                cw[name.strip()] = float(w.strip())
            scoring_fn = make_composite_scorer(
                error_data=df[args.error_col].values if args.error_col else None,
                sensitive_data=df[sensitive_cols[0]].values if sensitive_cols else None,
                mask=scorer_mask,
                silhouette_weight=cw.get("silhouette", 0.3),
                error_weight=cw.get("error", 0.5),
                fairness_weight=cw.get("fairness", 0.2),
                error_type=args.error_type,
            )
        # else: no error_col or sensitive_cols -> scoring_fn stays None -> silhouette fallback

    print(f"\nClustering...")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Distance: {args.distance}")
    print(f"  Scoring: {args.scoring}")

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
        res_df["clusters"] = result.labels
        # 'salient': rebuild readable multicat columns post-clustering (Gower's code
        # column, used for clustering, stays untouched).
        if args.multicat_table_option == "salient":
            apply_salient_reconstruction(
                res_df, multiclass_dummies, original_sensitive_cols
            )

        recap = make_recap(
            res_df,
            clustering_cols,
            sensitive_cols=sensitive_cols_analysis,
            error_col=args.error_analysis_col or args.error_col,
            error_type=args.error_type,
            feature_matrix=result.feature_matrix,
            distance_matrix=result.distance_matrix,
            continuous_sensitive_cols=continuous_sensitive_cols,
            multiclass_option=args.error_multiclass_option,
            error_cols=args.error_cols,
            error_cols_kind=args.error_cols_kind,
            sensitive_gap_test=args.sensitive_gap_test,
        )

        recap_dir = os.path.join(output_dir, "recap")
        os.makedirs(recap_dir, exist_ok=True)
        run_name = f"{args.algorithm}_{args.distance}_k{result.n_clusters}"
        recap.to_csv(os.path.join(recap_dir, f"{run_name}.csv"), index=False)
        print(f"\nSaved: recap/{run_name}.csv")

        if not args.no_plots and len(recap) > 1:
            error_label = (
                getattr(args, "error_label", None) or args.error_col or "error"
            )
            plot_cluster_recap_heatmap(
                recap.copy(), run_name, output_dir, error_label=error_label,
                sensitive_labels=parse_label_map(getattr(args, "sensitive_labels", None)),
            )
            print(f"Saved: {run_name}.png")

    # chi-square for categorical; Mann-Whitney (2 clusters) / Kruskal-Wallis (3+) for numeric
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

    if not args.no_plots:
        methods = parse_projection_list(args.projection)
        print(f"\nGenerating visualizations ({', '.join(methods) or 'none'})...")

        for method in methods:
            # t-SNE / MDS on the precomputed Gower matrix respect the cluster distance;
            # only non-noise points have a distance-matrix entry.
            if args.distance == "gower" and result.distance_matrix is not None \
                    and method in ("tsne", "mds"):
                X_2d = reduce_dimensions(
                    result.distance_matrix, method=method, precomputed=True
                )
                labels_2d = result.labels[result.labels != -1]
                title = f"Clusters ({args.algorithm}, gower+{method})"
            else:
                # Feature-space projection; drop categorical columns for kprototypes.
                if categorical_features and args.algorithm == "kprototypes":
                    numeric_mask = [
                        i
                        for i in range(result.feature_matrix.shape[1])
                        if i not in categorical_features
                    ]
                    X_for_viz = result.feature_matrix[:, numeric_mask].astype(float)
                else:
                    X_for_viz = result.feature_matrix
                # t-SNE uses the same distance as clustering (Manhattan); pca/euclidean default.
                metric = "manhattan" if (method == "tsne" and args.distance == "manhattan") \
                    else "euclidean"
                X_2d = reduce_dimensions(X_for_viz, method=method, metric=metric)
                labels_2d = result.labels
                title = f"Clusters ({args.algorithm}, {args.distance}, {method})"
            plot_clusters(
                X_2d, labels_2d, title=title,
                out_path=f"{output_dir}/clusters_{method}.png",
            )

        # Composition bar chart per sensitive attribute. Continuous attributes are
        # skipped: one stacked band per distinct value turns e.g. `age` into a
        # 60-entry legend that costs more to lay out than the rest of the run.
        if sensitive_cols:
            for attr_name in sensitive_cols:
                if attr_name in continuous_sensitive_cols:
                    continue
                attr_for_eval = df[attr_name].values
                if result.mask is not None:
                    attr_for_eval = attr_for_eval[result.mask]
                plot_cluster_composition(
                    result.labels,
                    attr_for_eval,
                    attr_name,
                    out_path=f"{output_dir}/composition_{attr_name}.png",
                )

        print(f"  Saved to {args.output_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
