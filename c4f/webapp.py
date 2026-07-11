"""
Gradio web UI for c4f experiment mode.

ponytail: drives the existing `c4f` CLI via subprocess instead of an in-process
API — the CLI already does all validation + output writing, so the UI is just a
form + a results gallery. Swap to in-process if subprocess startup ever matters.

NOTE: experiment mode's omnibus error test needs rpy2 + R (>= 4.5) on the host.
Install the extra:  pip install c4f[web]   (and have R available).
Launch:  c4f-web
"""

import os
import sys
import glob
import tempfile
import subprocess
import pandas as pd

# gradio imported lazily inside launch() so _build_cmd stays testable without it.


HOME_MD = """
# c4fairness — Clustering for Fairness

**Discover where a model's errors fall unevenly.** `c4fairness` clusters the rows of a
model's test set and reports how prediction-error disparities and sensitive-attribute
composition vary across the discovered clusters — surfacing under-served subgroups
*without* pre-specifying the protected group.

**Motivation.** Aggregate accuracy hides localized harm: a model can look fair on average
yet fail a specific intersection of groups. Unsupervised clustering of the test set exposes
those pockets and quantifies the disparity with proper significance tests.

**Made at Vrije Universiteit Amsterdam (VU), in collaboration with the University of
Twente (UT).**

<!-- Add further information here (authors, funding, links, citation). -->

---

Upload a CSV, choose your columns, and run. Results (heatmaps + tables) appear in the tabs
below. See the **Documentation** tab for column roles and options.

*Exact multi-categorical Fisher needs R ≥ 4.5 + `c4fairness[r]`; otherwise a scipy fallback
is used automatically.*
"""


DOC_MD = """
## Documentation

### Workflow
1. **Upload** a CSV where each row is one test-set example (features + the model's outputs).
2. **Columns** — assign roles (below).
3. **Error** — tell the app how to read the model's mistakes.
4. **Clustering options** — algorithm + its parameters (only the relevant ones are shown).
5. **Run** — heatmaps and tables appear in the tabs.

### Column roles
- **Regular features** — numeric/other features the clustering groups on.
- **Sensitive features** — protected attributes to audit (binary or multi-categorical).
- **Numeric sensitive features** — the subset of sensitive columns to analyse as
  numbers (median), e.g. `age`. Leave a sensitive column out of here and it is treated as
  categories.
- **Proxy / Special features** — proxies for sensitive attributes, and extra columns
  (e.g. SHAP values) to include.

### Error types
- **binary** — a 0/1 error column (or a confusion-matrix rate).
- **regression** — a continuous target; error is the signed residual (median per cluster).
- **multiclass** — pick `y_true` / `y_pred`; the *Multi-class error option* controls how the
  error is typed (per-class, per-cell confusion, one-hot, …).

### Multi-categorical options
- **Sensitive display** — `onehot` (one column per category) or `salient` (winning category
  + its value).
- **Significance test** — `auto` uses exact r×c Fisher when R is available, else scipy
  (chi-square with 0-cell handling, or one-vs-all 2×2 Fisher). `fisher_rxc` forces R;
  `chi2` / `fisher_ova` force scipy.

### Reading the heatmaps
Blue = cluster size, red = error, violet = sensitive; darker = higher (for p-value columns,
darker = more significant). A cluster with a high error gap and a significant `gap sig.` is
where the model most misfires — read its sensitive columns to see which group it concentrates.

### More
Worked examples: `docs/example_binary.ipynb`, `docs/example_regression.ipynb`.
Research directions & publishing plan: `docs/RESEARCH.md`.
"""


def _build_cmd(
    data_path,
    out_dir,
    regular,
    sensitive,
    error_type,
    error_col,
    y_true,
    y_pred,
    mc_option,
    algorithm,
    distance,
    n_clusters,
    seed,
    exclude,
    eps=0.5,
    min_samples=5,
    max_iter=300,
    multicat_sig="auto",
    multicat_table_option="onehot",
    continuous_sensitive="",
    proxy="",
    special="",
    error_label="",
    n_min="",
    n_max="",
    seeds="",
    scoring="composite",
    composite_weights="",
    feature_weights="",
    min_datapoints="",
    subset="none",
    separability_check=False,
    categorical_cols="",
    no_standardize=False,
    projection="tsne",
):
    """Assemble the `c4f --experiment ...` argv from form inputs."""
    cmd = [sys.executable, "-m", "c4f.main", "--data_path", data_path, "--experiment"]
    if exclude:
        cmd.append(exclude)
    cmd += [
        "--algorithm",
        algorithm,
        "--distance",
        distance,
        "--seed",
        str(int(seed)),
        "--output_dir",
        out_dir,
        "--error_type",
        error_type,
        "--scoring",
        scoring,
        "--multicat_sig",
        multicat_sig,
        "--multicat_table_option",
        multicat_table_option,
        "--projection",
        projection,
    ]
    # Cluster count: a range (n_min/n_max) triggers a k-search; otherwise fixed k.
    if n_min and n_max:
        cmd += ["--n_min", str(int(n_min)), "--n_max", str(int(n_max))]
    else:
        cmd += ["--n_clusters", str(int(n_clusters))]
    # Algorithm-specific parameters — only the flags the chosen algorithm actually uses.
    if algorithm == "dbscan":
        cmd += ["--eps", str(eps)]
    if algorithm in ("dbscan", "hdbscan"):
        cmd += ["--min_samples", str(int(min_samples))]
    if algorithm in ("kmeans", "bisectingkmeans", "kmedoids"):
        cmd += ["--max_iter", str(int(max_iter))]
    if seeds:
        cmd += ["--seeds", seeds]
    if composite_weights:
        cmd += ["--composite_weights", composite_weights]
    if feature_weights:
        cmd += ["--feature_weights", feature_weights]
    if min_datapoints not in ("", None):
        cmd += ["--min_datapoints", str(int(min_datapoints))]
    if subset and subset != "none":
        cmd += ["--subset", subset]
    if categorical_cols:
        cmd += ["--categorical_cols", categorical_cols]
    if separability_check:
        cmd += ["--separability_check"]
    if no_standardize:
        cmd += ["--no_standardize"]
    if regular:
        cmd += ["--regular_cols", regular]
    if sensitive:
        cmd += ["--sensitive_cols", sensitive]
    if continuous_sensitive:
        cmd += ["--continuous_sensitive_cols", continuous_sensitive]
    if proxy:
        cmd += ["--proxy_cols", proxy]
    if special:
        cmd += ["--special_cols", special]
    if error_label:
        cmd += ["--error_label", error_label]
    if error_type == "binary":
        cmd += ["--error_col", error_col]
    elif error_type == "regression":
        if error_col:
            cmd += ["--error_col", error_col]
        else:
            cmd += ["--y_true_col", y_true, "--y_pred_col", y_pred]
    else:  # multiclass
        cmd += [
            "--y_true_col",
            y_true,
            "--y_pred_col",
            y_pred,
            "--error_multiclass_option",
            mc_option,
        ]
    return cmd


def _cols(path):
    """Header columns of an uploaded CSV (empty list if unreadable)."""
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _csv1(v):
    """Coerce a dropdown value (list / str / None) to a comma string."""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return v or ""


def _run(
    file,
    regular,
    sensitive,
    error_type,
    error_col,
    y_true,
    y_pred,
    mc_option,
    algorithm,
    distance,
    n_clusters,
    seed,
    exclude,
    eps,
    min_samples,
    max_iter,
    multicat_table_option,
    multicat_sig,
    continuous_sensitive,
    proxy,
    special,
    error_label,
    n_min,
    n_max,
    seeds,
    scoring,
    composite_weights,
    feature_weights,
    min_datapoints,
    subset,
    separability_check,
    categorical_cols,
    no_standardize,
    projection,
):
    if file is None:
        return "Upload a CSV first.", [], [], None
    out_dir = tempfile.mkdtemp(prefix="c4f_web_")
    cmd = _build_cmd(
        file.name,
        out_dir,
        _csv1(regular),
        _csv1(sensitive),
        error_type,
        _csv1(error_col),
        _csv1(y_true),
        _csv1(y_pred),
        mc_option,
        algorithm,
        distance,
        n_clusters,
        seed,
        _csv1(exclude),
        eps=eps,
        min_samples=min_samples,
        max_iter=max_iter,
        multicat_sig=multicat_sig,
        multicat_table_option=multicat_table_option,
        continuous_sensitive=_csv1(continuous_sensitive),
        proxy=_csv1(proxy),
        special=_csv1(special),
        error_label=error_label or "",
        n_min=n_min or "",
        n_max=n_max or "",
        seeds=seeds or "",
        scoring=scoring,
        composite_weights=composite_weights or "",
        feature_weights=feature_weights or "",
        min_datapoints=(int(min_datapoints) if min_datapoints else ""),
        subset=subset,
        separability_check=bool(separability_check),
        categorical_cols=_csv1(categorical_cols),
        no_standardize=bool(no_standardize),
        projection=(projection or "tsne"),
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    # run_batch_experiment writes into out_dir/<timestamp>_experiment_.../
    runs = sorted(glob.glob(os.path.join(out_dir, "*experiment*")))
    base = runs[-1] if runs else out_dir
    pngs = sorted(glob.glob(os.path.join(base, "*.png")))
    csvs = sorted(glob.glob(os.path.join(base, "*.csv")))
    summary = os.path.join(base, "results_summary.csv")
    preview = pd.read_csv(summary) if os.path.exists(summary) else None
    if not pngs and not csvs:
        log += "\n\n(no outputs — check the log above; experiment mode needs R >= 4.5 + rpy2.)"
    return log, pngs, csvs, preview


def _build():
    import gradio as gr

    MC_OPTS = [
        "accuracy",
        "per_class",
        "precision",
        "per_cell",
        "binary_cells",
        "onehot",
        "classwise",
    ]

    with gr.Blocks(title="c4f — Clustering for Fairness") as demo:
        gr.Markdown(HOME_MD)
        file = gr.File(label="1. Dataset CSV", file_types=[".csv"])

        with gr.Group():
            gr.Markdown("### 2. Columns  *(populated from your CSV)*")
            with gr.Row():
                regular = gr.Dropdown(
                    label="Regular features",
                    multiselect=True,
                    info="Numeric/other features to cluster on",
                )
                sensitive = gr.Dropdown(
                    label="Sensitive features",
                    multiselect=True,
                    info="Protected attributes (binary or multi-class)",
                )
            with gr.Row():
                continuous_sensitive = gr.Dropdown(
                    label="Numeric sensitive features",
                    multiselect=True,
                    info="Subset of sensitive to analyse as numeric/median (e.g. age) — "
                    "otherwise treated as categories",
                )
                proxy = gr.Dropdown(
                    label="Proxy features", multiselect=True,
                    info="Proxies for the sensitive attributes",
                )
                special = gr.Dropdown(
                    label="Special features", multiselect=True,
                    info="e.g. SHAP values",
                )

        with gr.Group():
            gr.Markdown("### 3. Error")
            error_type = gr.Radio(
                ["binary", "regression", "multiclass"],
                value="binary",
                label="Error type",
            )
            error_col = gr.Dropdown(
                label="Error column", info="0/1 for binary, numeric for regression"
            )
            error_label = gr.Textbox(
                label="Error display label (optional)", placeholder="e.g. FP Rate"
            )
            with gr.Row():
                y_true = gr.Dropdown(label="y_true column", visible=False)
                y_pred = gr.Dropdown(label="y_pred column", visible=False)
            mc_option = gr.Dropdown(
                MC_OPTS,
                value="per_class",
                label="Multi-class error option",
                visible=False,
            )

        with gr.Accordion("4. Clustering options", open=False):
            with gr.Row():
                algorithm = gr.Dropdown(
                    [
                        "kmeans",
                        "hdbscan",
                        "dbscan",
                        "bisectingkmeans",
                        "kmedoids",
                        "kprototypes",
                    ],
                    value="kmeans",
                    label="Algorithm",
                )
                distance = gr.Dropdown(
                    ["euclidean", "manhattan", "gower"],
                    value="euclidean",
                    label="Distance",
                )
                n_clusters = gr.Number(value=4, label="n_clusters", precision=0)
                seed = gr.Number(value=42, label="seed", precision=0)
            with gr.Row():
                # Algorithm-specific — only the ones the default (kmeans) uses start visible.
                eps = gr.Number(value=0.5, label="eps (DBSCAN)", visible=False)
                min_samples = gr.Number(
                    value=5, label="min_samples (HDBSCAN/DBSCAN)", precision=0,
                    visible=False,
                )
                max_iter = gr.Number(
                    value=300, label="max_iter (KMeans family)", precision=0,
                    visible=True,
                )
            with gr.Row():
                multicat_table_option = gr.Dropdown(
                    ["onehot", "salient"], value="onehot",
                    label="Multi-categorical sensitive display",
                )
                multicat_sig = gr.Dropdown(
                    ["auto", "fisher_rxc", "chi2", "fisher_ova"], value="auto",
                    label="Multi-cat significance test",
                    info="'auto'/'fisher_rxc' use R if installed; others are scipy-only",
                )
            exclude = gr.Textbox(
                label="Exclude groups from conditions", placeholder="e.g. SPECIAL,ERR"
            )

        with gr.Accordion("5. Advanced options", open=False):
            with gr.Row():
                n_min = gr.Number(value=0, label="n_min (0 = fixed k)", precision=0,
                                  info="Search cluster count over [n_min, n_max] instead of fixed k")
                n_max = gr.Number(value=0, label="n_max (0 = fixed k)", precision=0)
                seeds = gr.Textbox(label="Seeds (multi-run)", placeholder="e.g. 1,2,3")
            with gr.Row():
                scoring = gr.Dropdown(
                    ["composite", "silhouette", "chi2_error", "chi2_sensitive"],
                    value="composite", label="Scoring (k-selection objective)",
                )
                composite_weights = gr.Textbox(
                    label="Composite weights",
                    placeholder="silhouette:0.3,error:0.5,fairness:0.2",
                )
                feature_weights = gr.Textbox(
                    label="Feature weights", placeholder="sensitive:2.0  or  age:1.5",
                )
            with gr.Row():
                subset = gr.Dropdown(
                    ["none", "TP", "TN", "FP", "FN", "TP_TN", "FP_FN"], value="none",
                    label="Confusion-matrix subset",
                )
                min_datapoints = gr.Number(value=0, label="min_datapoints (0 = off)", precision=0)
                projection = gr.Textbox(value="tsne", label="Projection(s)",
                                        placeholder="pca,tsne  or  none")
            with gr.Row():
                categorical_cols = gr.Dropdown(
                    label="Force categorical", multiselect=True,
                    info="Columns to treat as categorical",
                )
                separability_check = gr.Checkbox(label="Separability check", value=False)
                no_standardize = gr.Checkbox(label="Skip StandardScaler", value=False)

        run_btn = gr.Button("▶ Run experiment", variant="primary", size="lg")

        with gr.Tab("Heatmaps"):
            gallery = gr.Gallery(label="Heatmaps & plots", columns=2, height=520)
        with gr.Tab("Overview table"):
            preview = gr.Dataframe(label="results_summary.csv", wrap=True)
        with gr.Tab("Downloads"):
            files = gr.File(label="All CSV outputs", file_count="multiple")
        with gr.Tab("Log"):
            log = gr.Textbox(label="Run log", lines=18)
        with gr.Tab("Documentation"):
            gr.Markdown(DOC_MD)

        # Populate every column picker from the uploaded CSV header.
        _pickers = [regular, sensitive, continuous_sensitive, proxy, special,
                    categorical_cols, error_col, y_true, y_pred]

        def _fill(f):
            cols = _cols(f.name) if f else []
            return [gr.update(choices=cols) for _ in range(len(_pickers))]

        file.change(_fill, inputs=file, outputs=_pickers)

        # Show only the fields the chosen error type uses.
        def _toggle(t):
            return (
                gr.update(visible=t in ("binary", "regression")),  # error_col
                gr.update(visible=t in ("regression", "multiclass")),  # y_true
                gr.update(visible=t in ("regression", "multiclass")),  # y_pred
                gr.update(visible=t == "multiclass"),
            )  # mc_option

        error_type.change(
            _toggle, inputs=error_type, outputs=[error_col, y_true, y_pred, mc_option]
        )

        # Show only the clustering params the chosen algorithm actually uses.
        def _algo_fields(a):
            return (
                gr.update(visible=a == "dbscan"),                          # eps
                gr.update(visible=a in ("dbscan", "hdbscan")),             # min_samples
                gr.update(visible=a in ("kmeans", "bisectingkmeans", "kmedoids")),  # max_iter
            )

        algorithm.change(
            _algo_fields, inputs=algorithm, outputs=[eps, min_samples, max_iter]
        )

        run_btn.click(
            _run,
            inputs=[
                file,
                regular,
                sensitive,
                error_type,
                error_col,
                y_true,
                y_pred,
                mc_option,
                algorithm,
                distance,
                n_clusters,
                seed,
                exclude,
                eps,
                min_samples,
                max_iter,
                multicat_table_option,
                multicat_sig,
                continuous_sensitive,
                proxy,
                special,
                error_label,
                n_min,
                n_max,
                seeds,
                scoring,
                composite_weights,
                feature_weights,
                min_datapoints,
                subset,
                separability_check,
                categorical_cols,
                no_standardize,
                projection,
            ],
            outputs=[log, gallery, files, preview],
        )
    return demo


def launch(**kwargs):
    _build().launch(**kwargs)


def _selfcheck():
    # binary path
    c = _build_cmd(
        "d.csv",
        "/out",
        "age",
        "race,sex",
        "binary",
        "err",
        "",
        "",
        "per_class",
        "kmeans",
        "euclidean",
        4,
        42,
        "",
    )
    assert c[:4] == [sys.executable, "-m", "c4f.main", "--data_path"]
    assert "--experiment" in c and "--error_col" in c and "err" in c
    assert "--y_true_col" not in c
    # multiclass path adds y_true/y_pred + option, not error_col
    c = _build_cmd(
        "d.csv",
        "/out",
        "",
        "race",
        "multiclass",
        "",
        "yt",
        "yp",
        "classwise",
        "kmeans",
        "gower",
        5,
        1,
        "SPECIAL",
    )
    assert "--y_true_col" in c and "yt" in c and "classwise" in c
    assert "--error_col" not in c
    assert c[c.index("--experiment") + 1] == "SPECIAL"
    print("webapp selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
