# CLI reference

Every `c4fairness` option. Run `c4fairness --help` for the same list in the terminal.

## Parameters

Run `c4fairness --help` for the full list. Key options:

### Data & feature columns
| Parameter | Description |
|---|---|
| `--data_path` | **Required.** Path to input CSV. |
| `--regular_cols` | Features to cluster on (comma-separated). |
| `--sensitive_cols` | Protected attributes to audit. Binary, multi-categorical (auto one-hot), or numeric. |
| `--continuous_sensitive_cols` | Subset of `--sensitive_cols` to analyse as numbers (median), e.g. `age`. Otherwise treated as categories. |
| `--proxy_cols` / `--special_cols` | Proxies for sensitive attributes / extra features (e.g. SHAP) — clustered but reported separately. |
| `--categorical_cols` | Force-mark integer-coded columns as categorical (string/object columns are detected automatically). |

### Error definition
| Parameter | Description |
|---|---|
| `--error_type` | `binary` (default), `regression`, or `multiclass`. |
| `--error_col` | Pre-computed error column: 0/1 for binary, signed float for regression. |
| `--y_true_col` / `--y_pred_col` | Derive the error from ground truth + prediction (required for multiclass and for binary rate metrics). |
| `--binary_error_metric` | Binary error definition for the tables: `raw` (default, use `--error_col`), `fpr`, `fnr`, `precision` (=1−Precision), `prec_neg`. Rate options are derived per cluster from `y_true`/`y_pred`. |
| `--error_multiclass_option` | How to type multi-class errors: `per_class` (default), `accuracy`, `precision`, `per_cell` (confusion cell), `binary_cells`, `onehot`, `classwise`. |
| `--error_label` | Display label for the error in tables/heatmaps (e.g. `"FP Rate"`). |

### Sensitive-feature reporting
| Parameter | Description |
|---|---|
| `--multicat_table_option` | Multi-categorical sensitive display: `onehot` (default, one column per category) or `salient` (winning category + value). |
| `--sensitive_labels` | Display labels for sensitive features in heatmaps, as `col:Label,col2:Label2` (display-only; CSVs keep raw names). |
| `--sensitive_gap_test` | Significance test for `<F>_gap_sig`: `chi2` (default) or `fisher`. |

### Significance
| Parameter | Description |
|---|---|
| `--multicat_sig` | Omnibus test for multi-class error / multi-categorical `*_sep`: `auto` (default; exact r×c Fisher if R present, else scipy), `fisher_rxc` (force R), `chi2`, `fisher_ova`. |

### Algorithm
| Parameter | Description |
|---|---|
| `--algorithm` | `kmeans` (default), `bisectingkmeans`, `kmedoids`, `kprototypes`, `dbscan`, `hdbscan`. |
| `--distance` | `euclidean` (default), `manhattan`, `gower` (mixed types; required for kmedoids on categoricals). |
| `--n_clusters` | Fixed k — use this or `--n_min`/`--n_max`. |
| `--n_min` / `--n_max` | k range for automatic selection. |
| `--eps` | DBSCAN neighborhood radius. |
| `--min_samples` | Core-point threshold (DBSCAN / HDBSCAN). |
| `--max_iter` | Iterations for KMeans / BisectingKMeans / KMedoids. |
| `--scoring` | k-selection objective: `composite` (default), `silhouette`, `chi2_error`, `chi2_sensitive`. |
| `--composite_weights` | Composite scorer weights, e.g. `silhouette:0.3,error:0.5,fairness:0.2`. |
| `--feature_weights` | Per-column clustering weights, e.g. `age:2.0,income:0.5`. Recorded in the output directory name (`_w_...`). |
| `--seed` / `--seeds` | Random seed / comma-separated seeds for multi-seed experiment runs. |

### Analysis & output
| Parameter | Description |
|---|---|
| `--subset` | Restrict to a confusion-matrix subset: `TP`, `TN`, `FP`, `FN`, `TP_TN`, `FP_FN`. |
| `--min_datapoints` | Drop clusters smaller than this before analysis. |
| `--separability_check` | Also print feature separability tests to the console. |
| `--projection` | Scatter-plot projection(s), comma-separated: `pca`, `tsne`, `mds`, `none` (e.g. `pca,tsne` emits one plot each). t-SNE uses the **same distance as clustering** (precomputed Gower, Manhattan, or Euclidean). |
| `--no_standardize` | Skip StandardScaler on numeric features. |
| `--no_plots` | Skip all plots. |
| `--output_dir` | Output directory (default `clustering_results/<date>/`). |
| `--experiment` | Run experiment mode over feature-group combinations (REG / SEN / ERR); optionally exclude groups, e.g. `--experiment SPECIAL,ERR`. |
