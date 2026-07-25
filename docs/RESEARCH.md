# Research directions & publishing plan for `c4fairness`

`c4fairness` clusters the rows of a model's test set and reports how **error
disparities** and **sensitive-attribute composition** vary across the discovered
clusters — surfacing subgroups where a model underperforms *without* pre-specifying
the protected group. That capability supports several concrete studies. Each below
names a hypothesis, datasets, the metrics the package already computes, and a baseline
to beat.

> Deadlines below are indicative windows from past editions — **verify against each
> venue's current CFP** (links in the Publishing plan).

---

## Experiment 1 — Cross-dataset subgroup-discovery benchmark

**Hypothesis.** Unsupervised clustering of test-set rows recovers error-disparate
subgroups that align with known protected attributes *and* exposes intersectional
subgroups that single-attribute audits miss.

**Datasets.** COMPAS, Adult/Census income, German Credit, and `folktables`
(ACSIncome/ACSPublicCoverage across US states) — all have documented protected
attributes and published disparity results.

**Method / metrics.** For each dataset: train a standard classifier, run `c4fairness`
(overview `error_sep` / `error_gap` + per-feature `*_gap` / `*_sep`). Measure how often
a discovered cluster's high `error_gap` coincides with a significant sensitive-feature
`*_gap_sig`. Report intersectional clusters (two sensitive features jointly extreme).

**Baseline.** Per-attribute group fairness (subgroup error rates over each protected
attribute individually) and Slice Finder / FairVis-style slice discovery.

---

## Experiment 2 — Sensitivity to clustering choices

**Hypothesis.** The discovered disparities are stable across reasonable clustering
configurations; where they are *not*, the instability is itself diagnostic.

**Method.** Grid over `--algorithm` (kmeans, hdbscan, kprototypes), `--distance`
(euclidean, manhattan, gower), and `k`. For each config record the top error-disparate
cluster and its sensitive composition. Quantify agreement (cluster-membership ARI,
rank correlation of `error_gap`) across configs.

**Metrics.** Package outputs (`silh`, `error_gap`, `error_gap_sig`) + ARI between
label sets. **Baseline:** a single default config (what a naïve user would run).

**Why it matters for the tool.** Establishes recommended defaults per data profile and
warns when conclusions are configuration-dependent.

---

## Experiment 3 — Distance metric and mixed-type data

**Hypothesis.** For mixed numeric/categorical data, **Gower** distance discovers
subgroups that Euclidean-on-one-hot misses, because one-hot + StandardScaler distorts
categorical geometry.

**Method.** On datasets with many categorical features (German Credit, Adult), compare
`--distance gower` vs `--distance euclidean`. Compare the sensitive composition and
`error_gap` of the top clusters; measure how many high-disparity rows each grouping
isolates.

**Baseline.** Euclidean-on-one-hot (the common default). This directly tests the
README's OHE/StandardScaler caveat.

---

## Experiment 4 — Error definition changes the story

**Hypothesis.** Which confusion-matrix rate you audit (`--binary_error_metric`
fpr / fnr / precision / prec_neg) surfaces *different* disparate subgroups; a single
"accuracy gap" hides direction-specific harm.

**Method.** Hold clustering fixed; recompute the recap under each binary error metric.
Report how the top error-disparate cluster and its protected composition change between
FPR-audit and FNR-audit (e.g. over-policing vs under-service).

**Metrics.** Per-cluster `error_value` / `error_gap` under each metric; overlap of the
flagged subgroups. **Baseline:** overall accuracy gap.

---

## Experiment 5 — Multi-class error decomposition

**Hypothesis.** For multi-class models, per-confusion-cell error typing
(`--error_multiclass_option per_cell` / `per_class`) localises *which* class-confusions
drive a cluster's disparity, beyond a scalar "is-error" rate.

**Method.** On a multi-class task, cluster and inspect `error_cat` (winning error type
per cluster) and `error_gap_class`. Compare exact r×c Fisher (`--multicat_sig
fisher_rxc`, with R) vs the scipy fallback on sparse cells.

**Metrics.** Winning error type per cluster + `error_sep`. **Baseline:** binary
correct/incorrect clustering. Doubles as an evaluation of the WS2 significance stack.

---

## A supporting artifact — hosted demo

The bundled Gradio app (`c4f-web`) can be deployed (e.g. Hugging Face Spaces) as an
interactive companion: upload a test set + predictions, pick sensitive columns, and get
the overview/detailed heatmaps live. A public demo strengthens a tool/demo submission
and lowers the barrier for reviewers and practitioners to reproduce the analysis.

---

## Publishing plan

| Venue | Fit | Indicative timing (verify CFP) | Angle |
|---|---|---|---|
| **ACM FAccT** — <https://facctconference.org/> | Primary fairness/accountability venue | Abstracts ~late Jan, papers ~early Feb | Experiments 1 + 3 + 4 as a full paper: "clustering-based discovery of error-disparate subgroups". |
| **AIES 2026** — <https://www.aies-conference.com/2026/call-for-papers/> | AI ethics & society | ~Feb–Mar | Experiment 4/5 framing on harm direction and error typing; or a shorter methods paper. |
| **BNAIC 2026** — <https://www.maastrichtuniversity.nl/bnaic2026/call-papers> | Benelux AI, tool/demo-friendly | ~autumn 2026 | Tool/demo paper: the package + hosted demo (Experiment 2 as the empirical core). Good first outlet / fast feedback. |

**Sequencing suggestion.** Land the tool/demo at **BNAIC** first (fast, regional, demo
track) to get the artifact cited and reviewed, then target **FAccT** with the full
cross-dataset study (Experiments 1, 3, 4). Use **AIES** as the alternate/parallel venue
for the harm-direction framing if FAccT timing slips.

**Reproducibility checklist for any submission.** Pin `c4fairness[r]` + R for exact
Fisher; ship the exact CLI invocations and seeds; include the two example notebooks
(`docs/example_binary.ipynb`, `docs/example_regression.ipynb`) as the minimal
reproduction; report results under ≥2 clustering configs (Experiment 2) so conclusions
are not configuration artifacts.

> Note on exact Fisher: multi-categorical `error_sep` / `<F>_sep` use exact r×c Fisher
> only when R is present (`c4fairness[r]` + R ≥ 4.5); otherwise the scipy fallback
> (chi-square with 0-cell handling, or one-vs-all 2×2 Fisher) is **approximate on sparse
> tables**. State which was used in the paper.
