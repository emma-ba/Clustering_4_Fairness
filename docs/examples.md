# Examples and datasets

Commands are run from the repository root.

## Examples

All four run against the datasets in [`docs/datasets/`](datasets/).

**Binary classification (COMPAS), auditing the false-positive rate:**
```bash
c4fairness --data_path docs/datasets/compas_audit.csv \
  --regular_cols age,priors_count --sensitive_cols sex,race,age --continuous_sensitive_cols age \
  --y_true_col true_class --y_pred_col predicted_class --binary_error_metric fpr \
  --multicat_table_option salient --algorithm kmeans --n_clusters 4 --seed 42
```

**Experiment mode with automatic k selection:**
```bash
c4fairness --data_path docs/datasets/compas_audit.csv \
  --regular_cols age,priors_count --sensitive_cols sex,race \
  --error_col errors --error_type binary \
  --algorithm kmeans --n_min 2 --n_max 6 --scoring chi2_error --experiment
```

**Regression (student grades):**
```bash
c4fairness --data_path docs/datasets/student_grades.csv \
  --regular_cols G1,G2,studytime,absences --sensitive_cols sex_F,Medu,age \
  --continuous_sensitive_cols age --categorical_cols Medu \
  --y_true_col y_true --y_pred_col y_pred --error_type regression \
  --algorithm kmeans --n_clusters 3 --seed 42
```

**Mixed numeric/categorical types with Gower distance:**
```bash
c4fairness --data_path docs/datasets/student_grades.csv \
  --regular_cols G1,G2,studytime,absences --sensitive_cols sex_F,Medu,age \
  --continuous_sensitive_cols age --categorical_cols Medu,sex_F \
  --y_true_col y_true --y_pred_col y_pred --error_type regression \
  --algorithm hdbscan --distance gower --min_samples 15 --min_datapoints 25 \
  --no_standardize --projection pca --experiment
```

Worked, narrated notebooks: [`docs/example_binary.ipynb`](example_binary.ipynb) and
[`docs/example_regression.ipynb`](example_regression.ipynb).

---

## Datasets

### Bundled

Two extracts ship with the repository, in [`docs/datasets/`](datasets/). Each row is
one test-set example carrying the features, the ground truth, and a model's prediction, so
they can be audited as-is.

| File | Task | Sensitive attributes | N | Source |
|---|---|---|---|---|
| `compas_audit.csv` | Classification | `sex`, `race`, `age` | 5050 | [ProPublica, *Machine Bias*](https://github.com/propublica/compas-analysis) (`compas-scores-two-years.csv`) |
| `student_grades.csv` | Regression | `sex_F`, `Medu`, `age` | 670 | [UCI — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) |

### Also evaluated

Not redistributed here — fetch them from the source and add your model's predictions as a
column.

| Dataset | Task | Sensitive attributes | N | Source |
|---|---|---|---|---|
| Open University | Classification | `gender`, `region`, `imd_band`, `disability`, `age_band` | 32593 | [UCI — OULAD](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset) |
| German Credit | Classification | `Gender`, `Age`, `ForeignWorker` | 1000 | [UCI — Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Communities & Crime | Regression | `racepctblack` | 1994 | [UCI — Communities and Crime](https://archive.ics.uci.edu/dataset/183/communities+and+crime) |

The bundled extracts are derived from their sources: columns are subset, and `y_pred` /
`predicted_class` come from a model trained for these examples. Both originals carry their
own licence and citation requirements — check the source before redistributing either.
