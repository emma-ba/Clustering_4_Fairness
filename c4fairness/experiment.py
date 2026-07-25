"""Batch experiment mode for c4fairness: run all feature-group conditions and write
the Overview/Detailed tables, heatmaps and CSVs."""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from c4fairness.clustering import gower_distance
from c4fairness.scoring import (
    make_chi2_error_scorer,
    make_kruskal_error_scorer,
    make_categorical_error_scorer,
    make_chi2_sensitive_scorer, make_composite_scorer,
)
from c4fairness.visualization import reduce_dimensions, plot_clusters, plot_cluster_composition
from c4fairness.experiments import (
    create_exp_conditions,
    run_experiments_generic, make_chi_tests,
    recap_quali_metrics, plot_quality_heatmap, plot_cluster_recap_heatmap,
    separability_check,
)
from c4fairness.preprocessing import encode_categoricals
from c4fairness.cli import parse_column_list, parse_feature_weights, _build_sensitive_analysis_list, parse_label_map


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
    print("Running batch experiment...")
    print(f"  Dataset: {os.path.basename(args.data_path)}")

    # Parse column groups from CLI
    regular_cols = parse_column_list(args.regular_cols)
    sensitive_cols = parse_column_list(args.sensitive_cols)
    continuous_sensitive_cols = set(parse_column_list(getattr(args, 'continuous_sensitive_cols', None)) or [])
    proxy_cols = parse_column_list(args.proxy_cols)
    special_cols = parse_column_list(args.special_cols)
    error_col = args.error_col
    # Binary-rate mode: recap/omnibus read the masked rate column; ERR + scoring keep raw.
    error_analysis_col = getattr(args, 'error_analysis_col', None) or error_col

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
    df, col_lists, categorical_col_names, multiclass_dummies, ohe_col_names = encode_categoricals(
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
        sensitive_cols, multiclass_dummies, original_sensitive_cols,
        option=args.multicat_table_option,
    )

    # Build groups dict for condition generation
    groups = {}
    if regular_cols:
        groups['REG'] = regular_cols
    if sensitive_cols:
        groups['SEN'] = sensitive_cols
    # ERR clusters on a numeric 0/1 indicator (categorical per_class/per_cell error
    # columns can't be scaled); error_col still drives scoring + tables.
    groups['ERR'] = [getattr(args, 'error_cluster_col', None) or error_col]
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
        error_col=error_analysis_col,
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
        multiclass_dummies=multiclass_dummies,
        original_sensitive_cols=original_sensitive_cols,
        multicat_table_option=args.multicat_table_option,
        sensitive_gap_test=args.sensitive_gap_test,
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
                             error_type=args.error_type, error_col=error_analysis_col,
                             continuous_sensitive_cols=continuous_sensitive_cols,
                             multiclass_option=args.error_multiclass_option,
                             error_cols=args.error_cols,
                             sig=getattr(args, 'multicat_sig', 'auto'))
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
                             error_cols_kind=args.error_cols_kind,
                             sensitive_gap_test=args.sensitive_gap_test)

    slabels = parse_label_map(getattr(args, 'sensitive_labels', None))
    if not args.no_plots:
      # Separability test heatmap
      chi_viz_cols = err_sep_cols + sep_cols
      chi_res_viz = chi_res[chi_viz_cols].copy()
      chi_res_viz.index = chi_res['cond_name'].str.strip()
      plot_quality_heatmap(chi_res_viz, f"{output_dir}/chi_res_heatmap.png",
                           error_label=error_label, sensitive_labels=slabels,
                           title="Separability Test Results (p-values)")
      print(f"Saved: chi_res_heatmap.png")

      # Create quality metrics heatmap
      skip_meta_quali = {'cond_descr', 'cond_name'}
      quali_viz_cols = [c for c in all_quali.columns if c not in skip_meta_quali]
      all_quali_viz = all_quali[quali_viz_cols].copy()
      all_quali_viz.index = all_quali['cond_name'].str.strip()
      plot_quality_heatmap(all_quali_viz, f"{output_dir}/all_quali_heatmap.png",
                           error_label=error_label, sensitive_labels=slabels)
      plt.close()
      print(f"Saved: all_quali_heatmap.png")

    # Generate per-condition recap heatmaps and composition plots
    if not args.no_plots:
        print(f"\nGenerating {len(results['cond_name'])} recap heatmaps...")
        for i, cond_name in enumerate(results['cond_name']):
            recap = results['cond_recap'][i].copy()
            if len(recap) > 1:  # Only plot if there are multiple clusters
                plot_cluster_recap_heatmap(recap, cond_name, output_dir, error_label=error_label,
                                           sensitive_labels=slabels)
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
