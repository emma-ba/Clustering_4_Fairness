# Outputs

What a run writes to disk and what each column in the result tables means.

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
