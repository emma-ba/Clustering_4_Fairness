# c4fairness — notebook and web app improvement plan

Status: drafted and implemented 2026-08-01. Phases 0-6 done except where marked.
Scope: the two example notebooks in `docs/`, and the Gradio web app in `c4fairness/webapp.py`.

Section 1 is the original diagnosis and is left as written, so the measurements the
plan was built on stay legible. Section 9 records the audit that followed, including
the places where the diagnosis turned out to be wrong.

---

## 1. Findings the plan is based on

### 1.1 Notebooks

|                        | `example_binary.ipynb` | `example_regression.ipynb` |
| ---------------------- | ---------------------- | -------------------------- |
| cells                  | 19 (8 code)            | 20 (9 code)                |
| executes cleanly       | yes                    | **no — dies at cell 15**   |
| stored outputs         | none                   | none                       |
| markdown words / code lines | 673 / 61          | 2134 / 67                  |

**Blocking errors**

1. Regression §7 and §8 never run. `cluster(..., algorithm="kmedoids")` imports `sklearn_extra.cluster.KMedoids`, which fails on numpy ≥ 2:

   ```
   ImportError: numpy.core.multiarray failed to import
   ```

   `scikit-learn-extra` has no numpy-2 wheels (already noted in `pyproject.toml`). Cells 15, 16 and 18 are dead, and the takeaway's CLI snippet recommends the same broken `--algorithm kmedoids --distance gower`.

2. Neither notebook stores outputs. A reader on GitHub or PyPI sees code and prose but no tables and no heatmaps — the heatmap being the package's main visual argument.

3. Both read `../Data/…`, which is gitignored, so neither notebook runs for anyone who clones the repo.

4. `example_binary.ipynb` §7 contradicts its own numbers: the highest-FPR cluster is cluster 1 with n=2628 of 5050, yet the takeaway calls it "a disparity localised to a subgroup".

5. Both import unexported API. `_build_sensitive_analysis_list` is private; `encode_categoricals`, `apply_salient_reconstruction` and `plot_cluster_recap_heatmap` are absent from `__all__`.

**Verified working replacements** (tested against the project data on numpy 2):

- Gower on mixed data: `hdbscan` + `distance="gower"`, `min_samples=15`, `min_datapoints=25` → k=8, silhouette 0.493. The partition splits on `sex_F × Medu`, which is an honest and teachable Gower result.
- k-search: `kmeans` + `n_min=2, n_max=8` → k=5, silhouette 0.304. `n_min`/`n_max` exists only in the kmeans-family branch of `cluster()`.
- `kprototypes` accepts `distance="gower"` and **silently ignores it** (`clustering.py:376` documents this but never raises). It must not be used to illustrate Gower.

**Coverage.** The package is 5,399 LOC with 40+ exported names; the notebooks exercise about seven functions. Untouched: the whole `scoring` module, `subset=` confusion-matrix clustering, `feature_weights`, `min_datapoints`, all of `visualization`, `separability_check`, multiclass error typing, proxy/special column roles, `--experiment` output, and the `multicat_sig` R-vs-scipy tradeoff.

**Prose.** Measured AI-writing markers: 18 em-dashes in 673 words (binary) and 42 in 2134 (regression); 9 and 16 bolded bullet lead-ins; a recurring setup → reveal → moral rhetorical shape; reflex pairs and triads; hedge-then-reassure constructions; coaching second person. The regression notebook runs 3:1 prose to code; the binary notebook is too thin to serve as a reference.

### 1.2 Web app

Measured on an M-series Mac. Hugging Face free CPU is slower.

| dataset             | rows | conditions | projection | wall time |
| ------------------- | ---- | ---------- | ---------- | --------- |
| student_performance | 670  | 6          | tsne       | 72 s      |
| COMPAS              | 5050 | 6          | none       | 201 s     |
| COMPAS              | 5050 | 6          | tsne       | 467 s     |

cProfile of the 201 s run:

- `run_experiments_generic` (clustering, recap, all significance tests): **18.7 s**
- everything else, ≈ **180 s**, is matplotlib: `draw_text` 59.6 s tottime, `_update_ticks` 76.6 s cumulative, PNG encoding 26.9 s for 50 images.

**The app is not slow at statistics. It is slow at drawing.** One run writes 50 PNGs / 7.5 MB: 6 recap heatmaps, 2 quality heatmaps, and 42 composition plots (one per sensitive dummy per condition — `race` alone expands to six).

Causes of the latency:

- `webapp.py:143` hardcodes `--experiment`, so every click sweeps 2ⁿ−2 conditions. `main.py` has a single-run mode the UI cannot reach.
- Composition plots scale as `conditions × sensitive_dummies`, uncapped.
- `--projection tsne` is on by default; t-SNE is O(n²) and runs per condition (+266 s on COMPAS).
- `_run` blocks with no streaming, so the UI shows a bare spinner for minutes.
- `RUN_TIMEOUT_S = 3000` (50 minutes) is not a usable cap.

Causes of the errors:

- `pyproject.toml` has `web = ["gradio", "rpy2"]`. rpy2 needs system R ≥ 4.5; without it every run logs a dlopen `ImportError` traceback. Harmless (scipy fallback works) but reads as a crash.
- `webapp.py:379` still claims experiment mode needs R ≥ 4.5 + rpy2. Stale since R became optional.
- `webapp.py:305` rmtrees every `c4f_web_*` directory at the start of each run, including one whose PNGs the browser is still serving.
- `gradio` is unpinned; `demo.route()` (`webapp.py:641`) requires Gradio ≥ 5.6.
- `projection` is a free-text Textbox (`webapp.py:538`); a typo reaches argparse and kills the run after all the compute.
- `_run` validates error columns but never `sensitive`, so omitting it surfaces `ValueError: --sensitive_cols is required` as a raw traceback.
- `experiment.py:216` prints `recap['silh'].mean()` under the same "Silhouette" label the clusterer used for its own score (0.409 vs 0.474 for the same condition).

---

## 2. Decisions already made

- Demo datasets are **committed to the repo**, not linked. Combined size 217 KB (84 KB compressed) against a 125 MB `.git`; a URL would add a network dependency, rot risk, and a blocking prereq on notebook execution, and would save nothing measurable.
- The same two files are **also** to be published to a Hugging Face dataset repo, so the Space can offer a one-click "Load example dataset" button. That is a UX fix, not a hosting decision.
- Web app defaults to **single run**; the full sweep moves behind a toggle.

### Open decision

**D1 — what to do about `kmedoids`.** `scikit-learn-extra` is unmaintained and broken on numpy ≥ 2.

- **Option A (recommended):** keep `kmedoids` in the signature, but raise a clear error naming the dependency and the numpy constraint instead of the opaque `ImportError`. Notebooks and CLI examples stop using it.
- **Option B:** reimplement PAM / alternating medoids on a precomputed distance matrix (~40 lines), drop the dependency, restore fixed-k Gower clustering.

A is a one-line error message; B means owning a clustering implementation. B only pays off if fixed-k Gower is load-bearing for the research. HDBSCAN + Gower already covers the mixed-data story in the notebook.

---

## 3. Phase 0 — Unblock

- [x] **0.1** Write trimmed demo datasets to `docs/datasets/`:
  - `compas_audit.csv` — 5050 × `age, priors_count, sex, race, true_class, predicted_class, errors` (147 KB)
  - `student_grades.csv` — 670 × 29 (64 KB)

  `docs/datasets/` deliberately avoids the `Data` entry in `.gitignore` (git patterns are case-sensitive; APFS is not).
- [x] **0.2** Resolve D1: Option A — keep `kmedoids` in the signature, but raise a clear error naming the dependency and the numpy constraint instead of the opaque `ImportError`. Notebooks and CLI examples stop using it.
- [x] **0.3** Pin `gradio>=5.6` in `pyproject.toml`. `demo.route()` does not exist before that version.
- [x] **0.4** Changed the web extra to `web = ["gradio>=5.6"]`. R is optional; keeping rpy2 here was what made `pip install c4fairness[web]` fragile on Spaces and what produced the dlopen traceback in every run log.

---

## 4. Phase 1 — Notebooks

### 4.1 Fix the errors

- [x] Regression §7: `kmedoids` + gower → `hdbscan` + gower (`min_samples=15`, `min_datapoints=25`). Rewrote the surrounding prose for a density-based algorithm; noted that the Gower partition splits on the categorical axes.
- [x] Regression §8: k-search → `kmeans` + euclidean, `n_min=2`, `n_max=8`. Stated plainly that `n_min`/`n_max` is kmeans-family only.
- [x] Regression takeaway CLI: `--algorithm kmedoids --distance gower` → `--algorithm hdbscan --distance gower --min_samples 15`.
- [x] Binary §7: dropped the "localised subgroup" claim and added the cluster's share of all rows to the output so the reader can judge for themselves.
- [x] Repointed both notebooks at `datasets/…` relative to `docs/`.
- [x] Stripped stray leading blank lines in binary cells 15 and 17.

### 4.2 Rewrite the prose

Target voice: technical documentation, not essay.

- [x] Em-dashes: binary 18 → **1**, regression 42 → **0**.
- [x] Removed the setup → reveal → moral shape. State the fact, then the consequence.
- [x] Bolded bullet lead-ins: 9 and 16 → **0**. The recap column references are plain lists with the column names in backticks.
- [x] Cut the coaching second person and the hedge-then-reassure tics.
- [x] Word counts: regression 2134 → **1362**, binary 673 → **735**. Both titles follow the same `# <Task> example with c4fairness` form.

### 4.3 Extend coverage

Both notebooks get the same section skeleton so they read as a pair. Every addition below has been run against the project data.

**`example_binary.ipynb` (COMPAS false-positive rate), 19 → ~28 cells**

- [ ] k-selection compared: silhouette vs error separation vs composite — `make_chi2_error_scorer`, `make_composite_scorer`. Measured k = 6 / 2 / 6.
- [ ] Projection scatter — `reduce_dimensions`, `plot_clusters`.
- [ ] Clustering only the mistakes — `cluster(subset="FP_FN")`, 1948 of 5050 rows, and what `res.mask` is for.
- [ ] `onehot` vs `salient` sensitive display — `_build_sensitive_analysis_list(option=…)`.
- [ ] All four rate metrics side by side — `binary_error_rate_column` with `fpr`/`fnr`/`precision`/`prec_neg`.
- [ ] CLI and `--experiment` equivalent, showing the actual Overview table.

**`example_regression.ipynb` (student grades), 20 → ~26 cells**

- [ ] Gower on mixed data — fixed, per 4.1.
- [ ] k-search — fixed, per 4.1.
- [ ] k-selection for continuous error — `make_kruskal_error_scorer`. Measured k = 2 vs silhouette k = 5 vs composite k = 8.
- [ ] Weighting sensitive features — `feature_weights={col: 3.0}` (the dict form resolves against DataFrame column names).
- [ ] Noise handling — `min_datapoints` and the `-1` label.
- [ ] CLI and `--experiment` equivalent.

### 4.4 Execute and commit with outputs

- [ ] Add `ipykernel` to the dev environment (`ipython` and `nbclient` are already installed).
- [ ] Run both notebooks end to end with `jupyter nbconvert --to notebook --execute --inplace`, no `--allow-errors`.
- [ ] Commit the stored outputs. Let the notebook output carry the heatmap images; do not add loose PNGs to the tree.

---

## 5. Phase 2 — Web app speed

Plotting is ~90% of wall time, so the fix is about what gets drawn, not about faster maths.

- [x] **2.1** Added a *Single run* / *Full sweep* radio, defaulting to single. `_build_cmd` now takes `experiment=True|False` and only appends `--experiment` (and the flags that only apply to the sweep) when sweeping.
- [x] **2.2** Added `--max_composition_plots` (`cli.py`), applied in `experiment.py` as a total cap across conditions x attributes. The web UI passes 6.
- [x] **2.3** Web UI defaults to `projection="none"`, and the control is now a Dropdown (`none` / `pca` / `tsne` / `pca,tsne`) instead of a Textbox, so a typo can no longer kill a run after all the compute.
- [x] **2.4** **Not done, and should not be.** The premise was wrong. Profiling the single-run path found the real cost was `main.py` emitting a composition plot for *continuous* sensitive attributes: `age` becomes one stacked band per distinct value, a ~60-entry legend whose text layout cost more than everything else combined. `experiment.py` already skipped continuous attributes here; `main.py` did not. Adding that skip took single-run COMPAS from **283 s to 42 s**. A re-profile of the remaining 42 s shows no hotspot at all — the largest single entry is 1.6 s. Rasterizing the mesh would now save 1–2 s of 42 and is not worth the change.
- [x] **2.5** `_run` is a generator; it folds stderr into stdout and yields the accumulated log line by line, with the timeout enforced against a monotonic deadline inside the read loop.
- [x] **2.6** `RUN_TIMEOUT_S` 3000 → 600.

Measured on COMPAS (5050 rows), `--projection none`:

| path | before | after |
| ---- | ------ | ----- |
| single run | 283 s | **42 s** |
| full sweep, composition plots capped at 6 | 201 s | **99 s** |
| the web app's default path (was sweep + t-SNE, now single + none) | 467 s | **42 s** |

---

## 6. Phase 3 — Web app correctness

- [x] **3.1** Run directories are now sorted by mtime and the newest `KEEP_RUNS` (2) are left alone, so a new run no longer deletes the images the browser is still fetching.
- [x] **3.2** Validation moved into a `_validate` helper covering `sensitive`, `regular`/`special` and the error columns, checked before any subprocess is spawned. Covered by the module self-check.
- [x] **3.3** Replaced with `(no outputs — check the log above.)`; the R claim is gone.
- [x] **3.4** `_has_r` mutes the `rpy2` logger and redirects stderr for the duration of the probe, restoring both in a `finally`. rpy2 narrates its own API→ABI import fallback to stderr, which read like a crash in every run log.
- [x] **3.5** Now prints `mean per-cluster silhouette`, with a comment noting it is deliberately a different quantity from the clusterer's own score printed above it.
- [x] **3.6** Added a *Load example dataset (COMPAS)* button. It prefers `docs/datasets/compas_audit.csv` from a checkout and falls back to fetching the same file over HTTP, since `docs/` is not part of the installed package. The fallback URL points at `main`, so it will 404 until this branch merges.

---

## 7. Phase 4 — API hygiene

Small, and it unblocks Phase 1 — do it before the notebooks so they are not rewritten twice.

- [x] **4.1** Export what the tutorials use from `c4fairness/__init__.py`: `encode_categoricals`, `plot_cluster_recap_heatmap`, `binary_error_rate_column`, plus `_build_sensitive_analysis_list` and `apply_salient_reconstruction`.
- [x] **4.2** Make `kprototypes` reject `distance="gower"` (`clustering.py:376`), the way `kmeans` already does at `clustering.py:384`.
- [x] **4.3** Give a clear error for a missing `scikit-learn-extra` at `clustering.py:411`, per decision D1. (Done in Phase 0.)
- [x] **4.4** Guard the `mask=` footgun in `scoring.py`. `data[: len(labels)][non_noise]` silently truncated when lengths disagreed, then raised `IndexError`. Now raises a clear `ValueError` instead.

---

## 8. Phase 5 — Verification

- [x] Both notebooks execute clean under `nbconvert --to notebook --execute --inplace`, no `--allow-errors`. 13 and 19 stored outputs, zero error cells.
- [x] `python -m c4fairness.webapp` self-check passes, extended to cover single-run vs sweep, `--max_composition_plots`, and every `_validate` branch.
- [x] End-to-end smoke test of `_run` as a generator against `docs/datasets/compas_audit.csv`: 23 streamed updates, 8 PNGs, 3 CSVs, populated preview table.
- [x] `pytest tests/` — 126 passed, 3 failed. **All three failures pre-date this work**, confirmed by stashing every change and re-running: `test_experiment_pipeline_runs[extra2]` and two `test_multiclass_error` cases, all numpy-2 dtype errors in the multiclass path. Not caused here, not fixed here.
- [ ] `pip install -e ".[web]"` in a clean venv with no system R. **Not run** — needs a throwaway environment.

---

## 9. Phase 6 — Audit

Three parallel audits: over-engineering in the web-app path, correctness of the working diff, and every comment and docstring in the package.

### 9.1 Bugs found and fixed

- [x] **`--projection pca,tsne` crashed the sweep.** `experiment.py` passed the raw string to `reduce_dimensions`, which only accepts one method, so a multi-method projection raised `Unknown method: 'pca,tsne'` *after* every condition had been clustered. Single-run mode parsed it correctly. `experiment.py` now uses `parse_projection_list` and loops, which also restores Gower-aware t-SNE (it previously took the precomputed matrix only for `mds`, a branch the UI could never reach). Introduced into the UI by 2.3.
- [x] **The run timeout could not fire.** The deadline was only checked after a line arrived on `proc.stdout`, but that read blocks — and the child goes minutes without printing during the plot loops and per-condition t-SNE. A run that hung produced no line, so the check never ran and nothing was killed. Replaced with a `threading.Timer` watchdog that kills the process group regardless of output.
- [x] **Multi-seed runs came back with an empty gallery.** `--seeds` writes into `<run>/seed_<n>/`, one level below where the PNG glob looked. The gallery and the Overview table were both empty while the Downloads tab (recursive) was full. Both globs now recurse, and `cross_seed_summary.csv` joins the summary fallback chain.
- [x] **`main.py` warned that `--distance gower` is "ignored" for kprototypes, then crashed.** `cluster()` raises for that combination now. Deleted the warning.

### 9.2 Dead code removed

- [x] `plot_clusters_by_attribute`, `visualize_clustering_result`, `plot_silhouette_heatmap`, `plot_quality_metrics_heatmap` — four exported plotting functions with zero callers anywhere in the package, tests or notebooks. `visualization.py` 506 → 222 lines. Their invisibility is why `plot_quality_metrics_heatmap`'s docstring still advertised a `umap` option that is not a dependency and was never accepted. `plot_quality_metrics_heatmap` was superseded by `result_viz.plot_quality_heatmap`, which is what both run paths actually call. Removed from `__all__` too — a public-API change, deliberate at 0.1.x.
- [x] Unused imports in `main.py` (`plt`, `StandardScaler`, `gower_distance`, `OUTPUT_DIR`) and `visualization.py` (`sns`, `Union`, `ClusteringResult`).

### 9.3 Comments

96 comment lines removed, 249 remain.

- [x] The one `ponytail:` agent-tool tag, rewritten rather than deleted: it held the only written record of the single-user concurrency assumption behind `KEEP_RUNS`.
- [x] ~82 comments that only restated the line beneath them, plus 14 orphaned `# =====` banner rules in `experiments.py`.
- [x] The stale ones, which were the point of the exercise. Every churn area produced hits: `experiments.py` still presented rpy2 as the only significance path; `cli.py` offered `kmedoids` with no hint it raises on numpy 2; `webapp.py`'s module docstring still said "experiment mode"; `clustering.py`'s docstring listed `"bisecting"` and `"agglomerative"` as valid algorithms and gave the wrong `min_cluster_size` default; `preprocessing.py` documented a return contract the Gower branch does not honour; two comments described Kruskal-Wallis tests that the code does not run.
- [x] 14 research TODOs and French notes moved out of the middle of the argparse block into `docs/RESEARCH_NOTES.md`. One of them (`site web ou on peut uploader le dataset`) was already shipped as `webapp.py`.
- [x] `_build_sensitive_analysis_list` renamed to `build_sensitive_analysis_list`. It was exported under its private name, which promised a wrapper that did not exist.

Net across the package: **+378 / −592**.

### 9.4 Audit findings deliberately not acted on

- **Collapsing `_build_cmd`'s 40-parameter signature.** The same 39 values are written out five times (signature, body, `_run` signature, call site, `inputs=[...]`), so adding one form field costs five edits. Gradio 6 supports `inputs={component, ...}` with a dict handler, which would delete two of those five and remove the silent-corruption-on-reorder hazard. Roughly −190 lines, but it touches the whole file, and this branch already carries a lot of web-app change.
- **Extracting `prepare_columns` and `build_scoring_fn`** from the duplicated ~70 lines shared by `main.py` and `experiment.py`. This is the structural cause of the divergences below; worth doing, not worth doing now.
- **Four CLI flags that silently do nothing in one mode.** `--subset`, `--separability_check` and `--seeds` are ignored by the sweep or by the single run, and the web UI shows all of them in both modes. `--max_composition_plots` is honoured only in the sweep. Either wire them up or hide the controls on `mode.change`. Currently they lie to the user.
- **The three pre-existing multiclass test failures.**

### 9.5 `error_analysis_col` — fixed

Listed above as the next thing to fix, now done. `recap_quali_metrics` received the raw 0/1 misclassification column while `make_chi_tests` and `run_experiments_generic` received the masked rate, so with `--binary_error_metric` the Overview's `error_gap` and its `error_sep` described different quantities in the same row. `error_sep` is copied straight from `chi_res`, so it was always on the rate; `error_gap` was computed locally on the raw column.

The Overview understated the disparity substantially. On `docs/datasets/compas_audit.csv` with `--binary_error_metric fpr`:

| condition | `error_gap` before | after | per-cluster FPR range in the Detailed table |
| --------- | ------------------ | ----- | ------------------------------------------- |
| `+REG +SEN -err` | 0.0350 | **0.6260** | 0.2240 – 0.8500 |
| `+REG -sen -err` | 0.0534 | **0.6537** | 0.1970 – 0.8510 |

Roughly an 18x understatement: the old number was the spread of the overall misclassification rate, which barely moves across clusters, rather than the spread of the false-positive rate, which is the thing being audited. The gap significance was affected the same way (`error_gap_sig` 0.16 → 0.0 for `+REG +SEN -err`).

Both gap functions already `dropna` on the value column, so the masked rate's NaN rows (the actual positives, outside FPR's denominator) fall out correctly — the same mechanism the Detailed tables rely on.

Regression test: `tests/test_experiment_pipeline.py::test_overview_error_gap_uses_the_rate_column` asserts the Overview `error_gap` equals the spread of the per-cluster rates in that condition's Detailed table. Confirmed to fail against the old code and pass against the new.

---

## 10. Sequencing

```
D1 (decision) ───┐
0.1, 0.3, 0.4 ───┼──→ Phase 4 ──→ Phase 1 (notebooks) ──┐
                 └──→ Phase 2, 3 (web app) ─────────────┴──→ Phase 5
```

Phase 4 precedes Phase 1 so the notebooks import the public API. Phases 2 and 3 are independent of the notebook work and can proceed in parallel.

---

## 11. Explicitly out of scope

- Per-run TTL cleanup, a job queue, caching, or a results database for the web app. It is a single-user demo; `mkdtemp` plus fix 3.1 covers it.
- Any abstraction over the plotting backend. The fix is drawing less, not drawing differently.

---

## Appendix — environment note

The project venv carried a stale editable install of the former package name (`c4f-0.1.0`), so `import c4fairness` only resolved from the repository root. It has been replaced with `pip install -e .`, and `ipython` plus `nbclient` were installed to execute the notebooks.
