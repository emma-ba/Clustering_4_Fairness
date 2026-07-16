# Clustering 4 Fairness

[![PyPI version](https://img.shields.io/pypi/v/c4fairness.svg)](https://pypi.org/project/c4fairness/)
[![Python versions](https://img.shields.io/pypi/pyversions/c4fairness.svg)](https://pypi.org/project/c4fairness/)

**Discover where a model's errors fall unevenly.** `c4fairness` clusters the rows of a
model's test set and reports how prediction-error disparities and sensitive-attribute
composition vary across the discovered clusters — surfacing under-served subgroups
*without* pre-specifying the protected group. Works for **binary**, **multi-class**, and
**regression** tasks.

Made at **Vrije Universiteit Amsterdam (VU)**, in collaboration with the **University of
Twente (UT)**.

---

## Install

```bash
pip install c4fairness              # from PyPI
pip install "c4fairness[web]"       # + the Gradio web UI
pip install "c4fairness[r]"         # + rpy2 (exact r×c Fisher; also needs a system R ≥ 4.5)
```

Or from a local checkout (editable, for development):

```bash
pip install -e .
```

The import name is `c4fairness`; the CLI command is `c4fairness` (equivalently
`python -m c4fairness.main`).

## Quick start

```bash
c4fairness --data_path Data/compas/Compas_error_shap.csv \
    --regular_cols age,priors_count \
    --sensitive_cols sex,race --continuous_sensitive_cols age \
    --error_col errors --error_type binary \
    --algorithm kmeans --n_clusters 4 --seed 42
```

This clusters the test set on `age`/`priors_count` (+ the sensitive columns), then writes a
per-cluster recap and heatmap showing each cluster's error rate and its `sex`/`race`/`age`
make-up. String sensitive columns (`sex`, `race`) are one-hot encoded automatically.

---

## Web UI

A [Gradio](https://www.gradio.app/) web app wraps experiment mode: upload a CSV, assign
column roles, and run from the browser. Results (heatmaps, an overview table, downloadable
CSVs) render in tabs, with **Home**, **Documentation**, and **About** pages.

```bash
pip install "c4fairness[web]"
c4fairness-web
```

Then open the printed local URL (default http://localhost:7860). The form exposes every CLI
option; algorithm-specific fields (`eps`, `min_samples`, `max_iter`) appear only for the
relevant algorithm.

---

## Requirements

Python 3.10+. A virtual environment is recommended.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .                # or: pip install c4fairness
```

### R (optional) — exact multi-categorical Fisher

The omnibus significance test for multi-class errors and multi-categorical sensitive
features can use R's `fisher.test` (exact r×c Fisher–Freeman–Halton) via `rpy2`. This is
**optional**: without R, `c4fairness` falls back to scipy automatically (chi-square with
0-cell handling, or one-vs-all 2×2 Fisher). Install the extra + a system R ≥ 4.5 only if you
need the exact test on sparse tables (`--multicat_sig auto`/`fisher_rxc`):

```bash
pip install "c4fairness[r]"
```

On macOS the bundled R framework is often too old and `rpy2` picks it up, failing with
`symbol 'R_getVar' not found`. Install a modern R (`brew install r`) and point `rpy2` at it:

```bash
export R_HOME="$(/opt/homebrew/opt/r/bin/R RHOME)"   # Intel: /usr/local/opt/r/bin/R
export RPY2_CFFI_MODE=ABI
```

---

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

---

## What the result tables contain

Each run writes a **Detailed** recap (one row per cluster) and, in experiment mode, an
**Overview** (one row per condition). Columns are coloured by family in the heatmaps — blue =
size, red = error, violet = sensitive; p-value columns render darker when more significant.

| Column | Meaning | Significance test |
|---|---|---|
| `silh` | Silhouette | — |
| `count` / `proportion` (`min_size`/`min_prop`/`max_prop` in Overview) | Cluster size / share | — |
| `error_value` (or `error_mean`, `abs_error_mean` for regression) | Cluster error magnitude / rate | — |
| `error_cat` / `error_gap_class` | Winning error type (multi-class, `salient`/Option 3) | — |
| `error_gap` | Error gap vs. the rest (detailed, one-vs-all) / max cross-cluster spread (overview) | — |
| `error_sep` | Omnibus error separability | binary → **Fisher**; multiclass → `--multicat_sig`; regression → ANOVA |
| `error_gap_sig` | Error-gap significance | detailed → one-vs-all **Fisher** (binary/multiclass) / **Mann-Whitney** (regression); overview → extreme-pair Fisher / ANOVA |
| `<F>_value` / `<F>_cat` | Sensitive value per cluster (positive proportion / median / winning category) | — |
| `<F>_gap` / `<F>_gap_cat` | Sensitive gap vs. rest + winning category | — |
| `<F>_gap_sig` | Sensitive separability | binary/multicat → **Chi-square** (`--sensitive_gap_test`, default); numeric → Mann-Whitney / ANOVA |

Multi-class errors and multi-categorical sensitive features can be shown either **one-hot**
(one column set per class/category) or **salient** (a single winning-category column) — see
`--error_multiclass_option`, `--multicat_table_option`. In experiment mode, per-condition
omnibus `*_sep` p-values are collected in `chi_res.csv` and Benjamini-Hochberg corrected
across sensitive features.

---

## Input data notes — one-hot encoding & StandardScaler

The pipeline auto-detects string/object/category columns and one-hot encodes them, keeping
the 0/1 dummies **out of `StandardScaler`**. Applying StandardScaler to binary OHE columns
distorts Euclidean distances — rarer categories get scaled to larger values and contribute
more to distances regardless of importance. Skipping it for OHE columns avoids this. See
[Cross Validated — bias when one-hot encoding and standardizing](https://stats.stackexchange.com/questions/612809/bias-towards-categorical-data-when-one-hot-encoding-and-standardizing-for-machi).

If your CSV already has externally one-hot-encoded 0/1 columns (integers, so not detected by
dtype), pass them via `--categorical_cols` so they are excluded from scaling — or use
`--no_standardize` if all features are already on comparable scales.

### kprototypes silhouette

Standard silhouette can't be computed directly for kprototypes (mixed numeric + categorical
distance). The pipeline precomputes the full pairwise distance matrix with the same distance
the algorithm uses (squared Euclidean + Hamming, weighted by the fitted gamma) and passes it
to `silhouette_score(metric='precomputed')`.

---

## Output structure

### Single run
```
clustering_results/<date>/<timestamp>_<dataset>_<algorithm>_<distance>_s<seed>[_w_<weights>]/
  recap/<run_name>.csv          # per-cluster: error_value/gap/gap_sig, <F>_value/gap/gap_sig, silh
  separability/<run_name>.csv   # per-feature separability tests across clusters
  <run_name>.png                # recap heatmap
  clusters_<method>.png         # one scatter per --projection method
  composition_<attr>.png        # cluster composition per sensitive attribute
  metadata.csv
```

### Experiment mode
```
clustering_results/<date>/<timestamp>_experiment_<dataset>_<algorithm>_<distance>_s<seed>[_w_<weights>]/
  results_summary.csv           # Overview: one row per condition
  <condition>.csv               # per-cluster detailed recap + OVERALL / SEP rows
  <condition>.png               # detailed heatmap
  all_quali_heatmap.png         # Overview heatmap across conditions
  chi_res.csv / chi_res_heatmap.png   # omnibus separability p-values per condition
  exp_condition.csv             # feature set per condition
```

The `_w_<weights>` suffix records `--feature_weights` (feature + weight, e.g.
`_w_age2.0_priors_count0.5`).

---

## Examples

**Binary classification (COMPAS), auditing the false-positive rate:**
```bash
c4fairness --data_path Data/compas/Compas_error_shap.csv \
  --regular_cols age,priors_count --sensitive_cols sex,race --continuous_sensitive_cols age \
  --y_true_col true_class --y_pred_col predicted_class --binary_error_metric fpr \
  --multicat_table_option salient --algorithm kmeans --n_clusters 4 --seed 42
```

**Experiment mode with automatic k selection:**
```bash
c4fairness --data_path Data/compas/Compas_error_shap.csv \
  --regular_cols age,priors_count --sensitive_cols sex,race \
  --error_col errors --error_type binary \
  --algorithm kmeans --n_min 2 --n_max 6 --scoring chi2_error --experiment
```

**Regression (student grades):**
```bash
c4fairness --data_path Data/student_performance.csv \
  --regular_cols G1,G2,studytime,absences --sensitive_cols sex_F,Medu,age \
  --continuous_sensitive_cols age --categorical_cols Medu \
  --y_true_col y_true --y_pred_col y_pred --error_type regression \
  --algorithm kmeans --n_clusters 3 --seed 42
```

**Mixed types with Gower + kmedoids (real-world batch style):**
```bash
c4fairness --data_path Data/binary_student_dataset/binary_student_dataset_both_with_preds.csv \
  --regular_cols num_of_prev_attempts,studied_credits,module_presentation_length \
  --sensitive_cols gender,region,imd_band,disability,age_band \
  --categorical_cols imd_band,code_module,region,age_band \
  --error_col error --error_type binary \
  --algorithm kmedoids --distance gower --n_min 2 --n_max 25 --scoring silhouette \
  --no_standardize --projection pca --experiment
```

Worked, narrated notebooks: [`docs/example_binary.ipynb`](docs/example_binary.ipynb) and
[`docs/example_regression.ipynb`](docs/example_regression.ipynb). Research directions and a
publishing plan: [`docs/RESEARCH.md`](docs/RESEARCH.md).

---

## Datasets used

| Dataset | Task | Sensitive attributes | N |
|---|---|---|---|
| COMPAS | Classification | `sex`, `race`, `age` | 5050 |
| Student Performance | Regression | `sex_F`, `Medu`, `age` | 670 |
| Open University (binary) | Classification | `gender`, `region`, `imd_band`, `disability`, `age_band` | 32593 |
| German Credit | Classification | `Gender`, `Age`, `ForeignWorker` | 1000 |
| Communities & Crime | Regression | `racepctblack` | 1994 |

---

## Project structure

```
Clustering_4_Fairness/
├── c4fairness/               # the package
│   ├── main.py               # CLI entry point (`c4fairness` / python -m c4fairness.main)
│   ├── cli.py                # argument parsing + column-role helpers
│   ├── clustering.py         # cluster(), gower_distance(), ClusteringResult
│   ├── scoring.py            # silhouette / chi2 / kruskal / composite scorers
│   ├── preprocessing.py      # encode_categoricals()
│   ├── fairness_metrics.py   # per-cluster error/sensitive metrics + significance tests
│   ├── experiments.py        # make_recap(), make_chi_tests(), recap_quali_metrics()
│   ├── experiment.py         # run_batch_experiment() (experiment mode)
│   ├── result_viz.py         # result-table heatmaps
│   ├── visualization.py      # scatter/projection + composition plots
│   └── webapp.py             # Gradio web UI (`c4fairness-web`)
├── docs/                     # example notebooks + RESEARCH.md
├── tests/
├── Data/                     # datasets (not versioned)
└── pyproject.toml
```
