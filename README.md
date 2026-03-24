# Clustering 4 Fairness

Cluster data points and analyze whether prediction errors or sensitive attributes are distributed unevenly across clusters. Support (for now) both classification (binary error) and regression (signed error) tasks.

---

## Requirements

Python 3.10+ is recommended. Setting up a virtual environment before installing is strongly recommended to avoid dependency conflicts.

### Option A — venv (built-in)

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To deactivate when done:
```bash
deactivate
```

### Option B — conda

See https://docs.conda.io/projects/conda/en/latest/user-guide/install/ for installation, then `pip install -r requirements.txt`.

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
| `--regular_cols` | Comma-separated list of socioeconomic or behavioral features to cluster on (e.g. `age,income,priors_count`) |
| `--sensitive_cols` | Comma-separated protected attributes to include in clustering and analyze for bias (e.g. `sex_Female,race_African-American`). Binary 0/1 or multi-class — multi-class columns are auto-expanded into per-value indicators |
| `--proxy_cols` | Features that act as proxies for sensitive attributes (included in clustering but treated separately in analysis) |
| `--special_cols` | Special features such as SHAP values — included in clustering but not in separability tests |
| `--error_col` | Name of the pre-computed error column. For classification: binary 0 (correct) / 1 (wrong). For regression: signed float (`y_true - y_pred`) |
| `--error_type` | `binary` for classification tasks (default), `regression` for continuous signed error |
| `--y_true_col` | Ground truth column name — if provided alongside `--y_pred_col`, error is auto-computed (no need for `--error_col`) |
| `--y_pred_col` | Model prediction column name — used together with `--y_true_col` to auto-compute error |

### Algorithm
| Parameter | Description |
|---|---|
| `--algorithm` | Clustering algorithm: `kmeans` (default), `bisectingkmeans`, `kmedoids`, `kprototypes`, `dbscan`, `hdbscan`. kprototypes handles mixed numeric/categorical; DBSCAN/HDBSCAN are density-based and do not require specifying k |
| `--distance` | Distance metric: `euclidean` (default), `manhattan`, `gower`. Gower handles mixed types and is required for kmedoids on categorical data |
| `--n_clusters` | Fixed number of clusters — use this or `--n_min`/`--n_max`, not both |
| `--n_min` / `--n_max` | Range of k values to try when doing automatic k selection (e.g. `--n_min 2 --n_max 8`) |
| `--scoring` | Scoring function used to pick the best k: `silhouette` (cluster tightness), `chi2_error` (picks k where error is most unevenly distributed), `chi2_sensitive` (same but for sensitive attributes), `composite` (combination) |
| `--feature_weights` | Per-column weights applied before clustering, as `col:weight` pairs e.g. `age:2.0,income:0.5`. Useful to emphasize certain features |
| `--seed` | Random seed for reproducibility |
| `--seeds` | Comma-separated list of seeds to run experiment mode multiple times and aggregate (e.g. `42,123,456`) |
| `--max_iter` | Maximum number of iterations for KMeans and BisectingKMeans (default: 300) |

### DBSCAN / HDBSCAN specific
| Parameter | Description |
|---|---|
| `--eps` | Maximum distance between two points for them to be considered neighbors (DBSCAN only) — smaller = tighter clusters |
| `--min_cluster_size` | Minimum number of points required to form a cluster (HDBSCAN) |
| `--min_samples` | Minimum number of points in a neighborhood for a point to be a core point (HDBSCAN) — higher = more conservative, more noise |

### Analysis & output
| Parameter | Description |
|---|---|
| `--subset` | Restrict analysis to a confusion matrix subset: `TP`, `TN`, `FP`, `FN`, `TP_TN`, `FP_FN`. Useful for e.g. clustering only false positives to find systematic patterns |
| `--min_datapoints` | Drop clusters smaller than this threshold before analysis — avoids noisy small clusters |
| `--separability_check` | Print feature separability test results to the console in addition to saving them |
| `--projection` | Dimensionality reduction method for the scatter plot: `tsne` (default), `pca`, `none` |
| `--no_plots` | Skip generating and saving all visualization plots (recap heatmap, scatter, composition bars) |
| `--output_dir` | Custom output directory. Default is `clustering_results/<date>/` |
| `--experiment` | Run in experiment mode: automatically generates all combinations of feature groups (REG, SEN, ERR) and runs each as a separate condition, then produces a comparative summary |

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
  recap/<run_name>.csv              # per-cluster stats (error rate/mean, sensitive proportions, mannwhitney_p)
  separability/<run_name>.csv       # Kruskal-Wallis separability tests per feature across clusters
  <run_name>.png                    # recap heatmap
  clusters.png                      # projection scatter plot (tsne/pca)
  composition_<attr>.png            # cluster composition bar chart per sensitive attribute
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
| Open University (binary) | Classification | `gender`, `age_band_35-55`, `age_band_55_and_older`, `disability` | 32593 |
