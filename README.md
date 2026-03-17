# Clustering 4 Fairness

Cluster data points and analyze whether prediction errors or sensitive attributes are distributed unevenly across clusters. Support (for now) both classification (binary error) and regression (signed error) tasks.

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py --data_path <path> [options]
```

---

## Parameters

### Required
| Parameter | Description |
|---|---|
| `--data_path` | Path to input CSV file |

### Feature columns
| Parameter | Description |
|---|---|
| `--regular_cols` | Regular/socioeconomic features for clustering |
| `--sensitive_cols` | Protected attributes (binary 0/1 or multi-class) |
| `--proxy_cols` | Proxy features for sensitive attributes |
| `--special_cols` | Special features e.g. SHAP values |
| `--error_col` | Pre-computed error column (binary 0/1 or continuous) |
| `--error_type` | `binary` (classification) or `regression`. Default: `binary` |
| `--y_true_col` | Ground truth column — used to auto-compute error |
| `--y_pred_col` | Predicted values column — used to auto-compute error |

### Algorithm
| Parameter | Description |
|---|---|
| `--algorithm` | `kmeans` (default), `bisectingkmeans`, `kmedoids`, `kprototypes`, `dbscan`, `hdbscan` |
| `--distance` | `euclidean` (default), `manhattan`, `gower` |
| `--n_clusters` | Fixed number of clusters |
| `--n_min` / `--n_max` | Range for automatic k selection |
| `--scoring` | k-selection scoring: `silhouette`, `chi2_error`, `chi2_sensitive`, `composite` |
| `--feature_weights` | Column weights e.g. `regular:1.0,age:2.0` |
| `--seed` | Random seed for reproducibility |
| `--seeds` | Multiple seeds for multi-seed experiments e.g. `42,123,456` |
| `--max_iter` | Max iterations for KMeans/BisectingKMeans |

### DBSCAN / HDBSCAN specific
| Parameter | Description |
|---|---|
| `--eps` | Max distance between samples (DBSCAN) |
| `--min_cluster_size` | Minimum cluster size (HDBSCAN) |
| `--min_samples` | Minimum samples in neighborhood (HDBSCAN) |

### Analysis & output
| Parameter | Description |
|---|---|
| `--subset` | Analyze only a confusion matrix subset: `TP`, `TN`, `FP`, `FN`, `TP_TN`, `FP_FN` |
| `--min_datapoints` | Drop clusters smaller than this |
| `--separability_check` | Print separability test results to console |
| `--projection` | Visualization projection: `tsne` (default), `pca`, `none` |
| `--no_plots` | Skip saving visualization plots |
| `--output_dir` | Output directory (default: `clustering_results/<date>/`) |
| `--experiment` | Run all feature group combinations as separate conditions |

#TODO decile_score is leaking your ground truth, better to remove it. 

---

## Modes

### Single run
Cluster once with a specific feature set and analyze results.

### Experiment mode (`--experiment`)
Automatically generates all combinations of feature groups (REG, SEN, ERR) and runs each as a separate condition. Produces a comparative summary across conditions.

### Multi-seed (`--seeds`)
Runs experiment mode across multiple seeds and aggregates results.

---

## Examples

### Classification 
```bash
python main.py \
  --data_path Data/Compas_error_shap.csv \
  --regular_cols age,decile_score,priors_count \
  --sensitive_cols sex_Female,race_African-American,race_Caucasian \
  --error_col errors \
  --algorithm kmeans --n_clusters 5 --seed 42
```

```bash
python main.py \
  --data_path Data/Compas_error_shap.csv \
  --regular_cols age,decile_score,priors_count \
  --sensitive_cols sex_Female,race_African-American,race_Caucasian \
  --error_col errors \
  --algorithm kmeans --n_min 2 --n_max 6 --seed 42 \
  --scoring chi2_error --experiment
```

```bash
python main.py \
  --data_path Data/Compas_error_shap.csv \
  --regular_cols age,decile_score,priors_count \
  --sensitive_cols sex_Female,race_African-American,race_Caucasian \
  --y_true_col true_class --y_pred_col predicted_class \
  --subset FP_FN \
  --algorithm kmeans --n_clusters 5 --seed 42
```

### Regression

```bash
python main.py \
  --data_path Data/student_performance.csv \
  --regular_cols Medu,Fedu,studytime,failures,absences,G1,G2 \
  --sensitive_cols sex_F,age \
  --y_true_col y_true --y_pred_col y_pred --error_type regression \
  --algorithm kmeans --n_clusters 4 --seed 42 \
 
```

```bash
python main.py \
  --data_path Data/student_performance.csv \
  --regular_cols Medu,Fedu,studytime,failures,absences,G1,G2 \
  --sensitive_cols sex_F,age \
  --y_true_col y_true --y_pred_col y_pred --error_type regression \
  --algorithm kmeans --n_min 3 --n_max 6 --seed 42 \
  --scoring chi2_error --experiment
```

```bash
python main.py \
  --data_path Data/student_performance.csv \
  --regular_cols Medu,Fedu,studytime,failures,absences,G1,G2 \
  --sensitive_cols sex_F,age \
  --error_col regression_error --error_type regression \
  --algorithm kmedoids --distance gower --n_clusters 4 --seed 42 \
 
```

---

## Output structure

### Single run
```
clustering_results/<date>/<timestamp>_<dataset>_<algorithm>_<distance>_s<seed>/
  recap/<run_name>.csv              # per-cluster stats (error, sensitive proportions, mannwhitney_p)
  separability/<run_name>.csv       # feature separability tests across clusters
  <run_name>.png                    # recap heatmap
  clusters.png                      # projection scatter plot (tsne/pca)
  composition_<attr>.png            # cluster composition per sensitive attribute
  metadata.csv
```

### Experiment mode
```
clustering_results/<date>/<timestamp>_experiment_<dataset>_<algorithm>_<distance>_s<seed>/
  results_summary.csv               # one row per condition + metadata (key metric: kw_p_error)
  <condition>.csv                   # per-cluster rows with rule + mannwhitney_p,
                                    # OVERALL row with kruskallwallis_p,
                                    # SEP: rows with feature separability tests
  <condition>.png                   # recap heatmap
  <condition>_clusters.png          # projection scatter plot (tsne/pca)
  <condition>_composition_<attr>.png
  all_quali_heatmap.png             # overview heatmap across all conditions
  chi_res.csv / chi_res_heatmap.png # KW/chi2 p-values per condition
  exp_condition.csv                 # feature set per condition
```

---

## Datasets used

| Dataset | Task | Sensitive attributes | N |
|---|---|---|---|
| Communities & Crime | Regression | `Black`, `racepctblack` | 1994 |
| Student Performance | Regression | `sex_F`, `age` | 670 |
| COMPAS | Classification | `sex_Female`, `race_African-American`, `race_Caucasian` | 5050 |
| German Credit | Classification | `Gender`, `Age`, `ForeignWorker` | 1000 |
