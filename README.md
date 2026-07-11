# Clustering 4 Fairness

Cluster data points and analyze whether prediction errors or sensitive attributes are distributed unevenly across clusters. Supports both classification (binary error) and regression (signed error) tasks.

---

## Quick start

Install once from the repo root, then call `c4f` from anywhere:

```bash
pip install -e .
```

```bash
c4f --data_path Data/compas/Compas_error_shap.csv \
    --regular_cols age_scaled,priors_count_scaled \
    --sensitive_cols sex_Female,race_African-American \
    --error_col errors \
    --algorithm kmeans --n_clusters 5 --seed 42
```

`c4f` and `python main.py` are identical — same arguments, same output. The installed package is an editable install pointing at this repo, so any local changes take effect immediately without reinstalling.

> **Note:** This setup (`pip install -e .`) is intended for local research use. Publishing to PyPI would require renaming `src/` to avoid package name conflicts and making the CLI entry point self-contained.

---

## Project structure

```
Clustering_4_Fairness/
├── main.py                  # CLI entry point — run directly or via `c4f`
├── pyproject.toml           # package definition for pip install -e .
├── requirements.txt
│
├── src/                     # core library
│   ├── clustering.py        # cluster(), gower_distance(), ClusteringResult
│   ├── scoring.py           # silhouette, chi2, kruskal, composite scorers
│   ├── experiments.py       # make_recap(), run_experiments_generic(), chi tests
│   ├── visualization.py     # scatter plots, composition bars, heatmaps
│   ├── fairness_metrics.py  # demographic parity, representation ratio
│   └── preprocessing.py     # encode_categoricals()
│
├── c4f/                     # package entry point (re-exports src, bridges to main.py)
│
├── Data/                    # datasets (not versioned)
├── scripts/                 # preprocessing scripts
└── run_tests.sh             # test suite
```

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

### Experiment mode — R + rpy2

Experiment mode (`--experiment`) runs the error separability test with R's `fisher.test` via `rpy2`, so it needs **R ≥ 4.5** and `rpy2` installed (`pip install rpy2`). Single-run mode needs neither.

On macOS the bundled R framework (often 4.3.x) is too old and `rpy2` picks it up by default, failing with `symbol 'R_getVar' not found`. Install a modern R (`brew install r`) and point `rpy2` at it before running:

```bash
export R_HOME="$(/usr/local/opt/r/bin/R RHOME)"   # Apple Silicon: /opt/homebrew/opt/r/bin/R
export RPY2_CFFI_MODE=ABI
```

Add those two lines to `~/.zshrc` to make them permanent.

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
| `--continuous_sensitive_cols` | Subset of `--sensitive_cols` to treat as continuous rather than categorical (e.g. `age`). For these columns the recap shows per-cluster mean, mean delta vs. the rest, and a Mann-Whitney U p-value instead of proportions. The global test in `chi_res` uses Kruskal-Wallis instead of chi-square. Use this for any sensitive attribute that is a genuine numeric quantity — age, income, score — where computing a proportion would be meaningless. |
| `--proxy_cols` | Features that act as proxies for sensitive attributes (included in clustering but treated separately in analysis) |
| `--special_cols` | Special features such as SHAP values — included in clustering but not in separability tests |
| `--categorical_cols` | Columns to treat as categorical, comma-separated. String/category dtype columns are detected and one-hot encoded automatically — use this to force-mark additional columns (e.g. integer-encoded labels). For `kprototypes`, columns are passed directly to the algorithm without encoding. For `--distance gower` with binary 0/1 integer columns, one-hot encoding is a no-op and Gower distances are identical either way. Multi-class integer columns with Gower are currently one-hot encoded and treated as numeric; all pairwise distances end up equal which is approximately correct but not identical to true categorical treatment |
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
| `--scoring` | Scoring function used to pick the best k: `silhouette` (cluster tightness), `chi2_error` (picks k where error is most unevenly distributed), `chi2_sensitive` (same but for sensitive attributes), `composite` (combination, default) |
| `--composite_weights` | Weights for the composite scorer as `component:weight` pairs (default: `silhouette:0.3,error:0.5,fairness:0.2`). Only used when `--scoring composite` |
| `--feature_weights` | Per-column weights applied before clustering, as `col:weight` pairs e.g. `age:2.0,income:0.5`. Useful to emphasize certain features |
| `--seed` | Random seed for reproducibility |
| `--seeds` | Comma-separated list of seeds to run experiment mode multiple times and aggregate (e.g. `42,123,456`) |
| `--max_iter` | Maximum number of iterations for KMeans and BisectingKMeans (default: 300) |

### kprototypes — silhouette score

Standard silhouette cannot be computed directly for kprototypes because it uses a mixed distance (numeric + categorical) that sklearn's silhouette implementation does not support. Instead, the pipeline precomputes the full pairwise distance matrix using the same distance as the algorithm and passes it to `silhouette_score` with `metric='precomputed'`. See [Silhouette Coefficient for K-Modes and K-Prototypes Clustering](https://codinginfinite.com/silhouette-coefficient-for-k-modes-and-k-prototypes-clustering/) for the approach this implementation follows.

> **Note:** The distance matrix combines squared Euclidean distance (numeric features) and Hamming distance (categorical features), weighted by gamma — the same gamma the algorithm uses internally during fitting (`fitted_model.gamma_`). Using plain Euclidean distance would ignore the categorical part of the space entirely, producing a silhouette score that does not reflect the actual clustering.

### DBSCAN / HDBSCAN specific
| Parameter | Description |
|---|---|
| `--eps` | Maximum distance between two points for them to be considered neighbors (DBSCAN only) — smaller = tighter clusters |
| `--min_samples` | Minimum number of points in a neighborhood for a point to be a core point (HDBSCAN) — higher = more conservative, more noise |

### Analysis & output
| Parameter | Description |
|---|---|
| `--subset` | Restrict analysis to a confusion matrix subset: `TP`, `TN`, `FP`, `FN`, `TP_TN`, `FP_FN`. Useful for e.g. clustering only false positives to find systematic patterns |
| `--min_datapoints` | Drop clusters smaller than this threshold before analysis — avoids noisy small clusters |
| `--separability_check` | Print feature separability test results to the console in addition to saving them |
| `--projection` | Dimensionality reduction method for the scatter plot: `tsne` (default), `pca`, `mds`, `none`. When `--distance gower` is used, MDS is applied automatically regardless of this flag |
| `--no_standardize` | Disable automatic standardization of numeric features before clustering. Use this if your data is already normalized |
| `--no_plots` | Skip generating and saving all visualization plots (recap heatmap, scatter, composition bars) |
| `--output_dir` | Custom output directory. Default is `clustering_results/<date>/` |
| `--experiment` | Run in experiment mode: automatically generates all combinations of feature groups (REG, SEN, ERR) and runs each as a separate condition, then produces a comparative summary |


---

## Input data format notes

### Categorical columns

The pipeline auto-detects string/object/category dtype columns and one-hot encodes them. One-hot encoded columns are kept as 0/1 and excluded from `StandardScaler`. Applying StandardScaler to binary OHE columns distorts Euclidean distances: rarer categories get scaled to larger values than common ones, so they end up contributing more to the distance between points regardless of their actual importance. The rarer the category, the worse the distortion. Skipping StandardScaler for OHE columns avoids this entirely. See [Cross Validated — Bias towards categorical data when one-hot encoding and standardizing](https://stats.stackexchange.com/questions/612809/bias-towards-categorical-data-when-one-hot-encoding-and-standardizing-for-machi) for a full discussion.

Use `--categorical_cols` to force-mark additional columns that look numeric but are semantically categorical (e.g. integer-encoded labels, ordinal bands).

## Data already one-hot encoded

If your CSV has columns that were one-hot encoded externally (e.g. `gender_F = 0/1`, `region_NorthWest = 0/1`), the pipeline cannot detect them as categorical by dtype — they are integers. Without declaring them, `StandardScaler` will be applied and rare categories will be over-weighted in clustering.

Pass those columns via `--categorical_cols`. The pipeline will re-encode them (harmlessly, since a binary 0/1 column re-encodes to an identical column), add them to the protected set, and exclude them from scaling.

If all features in your dataset are already on comparable scales, use `--no_standardize` instead.

> **Planned:** A dedicated `--ohe_cols` argument will allow declaring pre-encoded columns directly without going through re-encoding.

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
  --regular_cols age,priors_count \
  --sensitive_cols sex_Female,race_African-American,race_Caucasian \
  --error_col errors \
  --algorithm kmeans --n_clusters 5 --seed 42
```

```bash
python main.py \
  --data_path Data/Compas_error_shap.csv \
  --regular_cols age,priors_count \
  --sensitive_cols sex_Female,race_African-American,race_Caucasian \
  --error_col errors \
  --algorithm kmeans --n_min 2 --n_max 6 --seed 42 \
  --scoring chi2_error --experiment
```

```bash
python main.py \
  --data_path Data/Compas_error_shap.csv \
  --regular_cols age,priors_count \
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

## Fairness metrics

Each sensitive column produces the following metrics per cluster in the recap CSV and heatmap. All metric functions are in `src/fairness_metrics.py` and are called from `make_recap` in `src/experiments.py`.

| Column suffix | What it measures | Reference |
|---|---|---|
| `_prop` | Proportion of that group in the cluster (demographic parity) | [Fairlearn — common fairness metrics](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html) |
| `_repr_ratio` | Cluster proportion divided by the overall population proportion. >1 = over-represented, <1 = under-represented (disparate impact ratio) | [Fairlearn — common fairness metrics](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html) |
| `_entropy` | Shannon entropy of group proportions within the cluster. Higher = more evenly mixed | [scipy.stats.entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html) |
| `_max_prop_diff` | Max minus min proportion across all values of a multi-class attribute — how spread the representation is within a cluster | — |
| `_p` | Poisson means test p-value comparing the group count in this cluster against all other clusters combined | — |

The `balance_score` (1 − mean absolute deviation from population proportions; 1.0 = perfectly balanced) and per-condition entropy averages appear in `all_quali_heatmap.png` and `results_summary.csv` in experiment mode. These are defined as custom metrics following the pattern described in [Fairlearn — custom fairness metrics](https://fairlearn.org/main/user_guide/assessment/custom_fairness_metrics.html).

### Multiple comparison correction

In experiment mode, p-values across sensitive columns are corrected per condition using the Benjamini-Hochberg (BH) procedure (`scipy.stats.false_discovery_control`), which controls the false discovery rate. This runs on a small array of floats after clustering is complete and adds no measurable overhead to the pipeline.

---

## Datasets used

| Dataset | Task | Sensitive attributes | N |
|---|---|---|---|
| Communities & Crime | Regression | `Black`, `racepctblack` | 1994 |
| Student Performance | Regression | `sex_F`, `age` | 670 |
| COMPAS | Classification | `sex_Female`, `race_African-American`, `race_Caucasian` | 5050 |
| German Credit | Classification | `Gender`, `Age`, `ForeignWorker` | 1000 |
| Open University (binary) | Classification | `gender`, `age_band_35-55`, `age_band_55_and_older`, `disability` | 32593 |
