"""
Gradio web UI for c4fairness: single-run audits and full feature-group sweeps.


Install the extra:  pip install "c4fairness[web]"   (R >= 4.5 optional; scipy fallback otherwise).
Launch:  c4fairness-web
"""

import os
import sys
import glob
import signal
import shutil
import tempfile
import threading
import subprocess
import pandas as pd

# gradio imported lazily inside _build() so _build_cmd stays testable without it.

RUN_TIMEOUT_S = int(os.environ.get("C4F_WEB_TIMEOUT", "600"))  # per-run cap; override via env
KEEP_RUNS = 2  # previous run dirs left on disk so their images keep resolving in the browser


HOME_MD = """
# c4fairness — Clustering for Fairness

Upload a CSV, choose your columns, and run. Results (heatmaps + tables) appear in the tabs
below. See the **Documentation** page (top navbar) for column roles and options.
"""


DOC_MD = """
## Documentation



1. **Upload** a CSV where each row is one test-set example (features + the model's
   outputs), or press *Load example dataset* to try it on COMPAS.
2. **Columns** — assign roles (below).
3. **Error** — tell the app how to read the model's mistakes.
4. **Clustering options** — algorithm + its parameters (only the relevant ones are shown).
5. **Run** — heatmaps and tables appear in the tabs.

### Run modes
- **Single run** audits exactly the configuration on the form. This is the default.
- **Full sweep** repeats that audit for every combination of the feature groups (regular,
  sensitive, error), which shows whether a finding survives the choice of what to cluster
  on. It runs one audit per combination, so it takes several times longer.

Both modes get slower with more rows, a wider k range, and the `tsne` projection, which is
quadratic in the number of rows and runs once per condition.

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
Worked examples: `docs/example_binary.ipynb`, `docs/example_regression.ipynb` are available in the [Github repository](https://github.com/emma-ba/Clustering_4_Fairness)
"""


ABOUT_MD = """
## About

**Discover where a model's errors fall unevenly.** `c4fairness` clusters the rows of a
model's test set and reports how prediction-error disparities and sensitive-attribute
composition vary across the discovered clusters — surfacing under-served subgroups
*without* pre-specifying the protected group.

**Motivation.** Aggregate accuracy hides localized harm: a model can look fair on average
yet fail a specific intersection of groups. Unsupervised clustering of the test set exposes
those pockets and quantifies the disparity with proper significance tests.

**Made at Vrije Universiteit Amsterdam (VU), in collaboration with the University of
Twente (UT).**

### Project
- **Package:** [`c4fairness`](https://pypi.org/project/c4fairness/) (v0.1.1) — `pip install c4fairness`
- **Source:** <https://github.com/emma-ba/Clustering_4_Fairness>

### Authors
- Filip Muntean — <filip.mihai.muntean@gmail.com>
- Emma Beauxis-Aussalet — <e.m.a.l.beauxisaussalet@vu.nl>
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
    sensitive_gap_test="chi2",
    continuous_sensitive="",
    proxy="",
    special="",
    error_label="",
    sensitive_labels="",
    binary_error_metric="raw",
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
    projection="none",
    experiment=True,
    max_composition_plots=6,
):
    """Assemble the `c4fairness ...` argv from form inputs.

    `experiment=True` sweeps every feature-group combination (2^n - 2 conditions);
    `False` runs the single configuration the form describes.
    """
    cmd = [sys.executable, "-m", "c4fairness.main", "--data_path", data_path]
    if experiment:
        cmd.append("--experiment")
        if exclude:
            cmd.append(exclude)
        if max_composition_plots:
            cmd += ["--max_composition_plots", str(int(max_composition_plots))]
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
        "--sensitive_gap_test",
        sensitive_gap_test,
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
    if algorithm in ("kmeans", "bisectingkmeans", "kmedoids", "kprototypes"):
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
    if sensitive_labels:
        cmd += ["--sensitive_labels", sensitive_labels]
    if error_type == "binary":
        if binary_error_metric != "raw":
            # FPR/FNR/... are derived per cluster from y_true/y_pred, not a fixed column.
            cmd += ["--binary_error_metric", binary_error_metric,
                    "--y_true_col", y_true, "--y_pred_col", y_pred]
        else:
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


EXAMPLE_CSV = "compas_audit.csv"
EXAMPLE_URL = (
    "https://raw.githubusercontent.com/emma-ba/Clustering_4_Fairness/main/"
    f"docs/datasets/{EXAMPLE_CSV}"
)


def _example_dataset():
    """Path to the bundled COMPAS extract, so a first visit can run something.

    Prefers a repo checkout; falls back to fetching it, because `docs/` is not part
    of the installed package.
    """
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "docs", "datasets", EXAMPLE_CSV)
    if os.path.exists(local):
        return local
    import urllib.request
    dest = os.path.join(tempfile.mkdtemp(prefix="c4f_example_"), EXAMPLE_CSV)
    urllib.request.urlretrieve(EXAMPLE_URL, dest)
    return dest


def _csv1(v):
    """Coerce a dropdown value (list / str / None) to a comma string."""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return v or ""


def _validate(file, regular, sensitive, special, error_type, error_col,
              y_true, y_pred, binary_error_metric):
    """Form-level checks. Returns an error message, or None when the run can proceed.

    Without these the empty flag reaches the subprocess as e.g. `--error_col ""` and
    fails minutes later with a traceback in the log.
    """
    if file is None:
        return "Upload a CSV first."
    if not _csv1(sensitive):
        return "Pick at least one Sensitive feature — the audit is defined against it."
    if not (_csv1(regular) or _csv1(special)):
        return "Pick at least one Regular (or Special) feature to cluster on."
    if error_type == "binary":
        if binary_error_metric == "raw" and not _csv1(error_col):
            return "Pick an Error column (or a metric that derives from y_true/y_pred)."
        if binary_error_metric != "raw" and not (_csv1(y_true) and _csv1(y_pred)):
            return f"'{binary_error_metric}' needs both y_true and y_pred columns."
    elif error_type == "regression":
        if not _csv1(error_col) and not (_csv1(y_true) and _csv1(y_pred)):
            return "Regression needs an Error column, or both y_true and y_pred."
    else:  # multiclass
        if not (_csv1(y_true) and _csv1(y_pred)):
            return "Multiclass needs both y_true and y_pred columns."
    return None


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
    sensitive_gap_test,
    continuous_sensitive,
    proxy,
    special,
    error_label,
    sensitive_labels,
    binary_error_metric,
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
    mode,
    max_composition_plots,
):
    err = _validate(file, regular, sensitive, special, error_type, error_col,
                    y_true, y_pred, binary_error_metric)
    if err:
        yield err, [], [], None
        return
    # Deleting a run whose images the browser is still fetching breaks the gallery,
    # so the newest few survive. Assumes one user at a time; key the dirs by session
    # if that changes.
    old = sorted(glob.glob(os.path.join(tempfile.gettempdir(), "c4f_web_*")),
                 key=os.path.getmtime)
    for d in old[:-KEEP_RUNS] if KEEP_RUNS else old:
        shutil.rmtree(d, ignore_errors=True)
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
        sensitive_gap_test=sensitive_gap_test,
        continuous_sensitive=_csv1(continuous_sensitive),
        proxy=_csv1(proxy),
        special=_csv1(special),
        error_label=error_label or "",
        sensitive_labels=sensitive_labels or "",
        binary_error_metric=binary_error_metric,
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
        projection=(projection or "none"),
        experiment=(mode != "Single run"),
        max_composition_plots=max_composition_plots,
    )
    # start_new_session puts the child in its own process group so on timeout we can
    # SIGKILL the whole group — otherwise grandchildren (matplotlib, etc.) can orphan.
    # stderr is folded into stdout so one stream can be tailed line by line.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, start_new_session=True,
    )
    # The timeout runs on its own timer rather than being checked inside the read
    # loop: `for line in proc.stdout` blocks until the child writes a newline, and
    # the child goes minutes at a time without printing (per-condition t-SNE, the
    # plot loops). Checking the clock only on arrival of a line would never fire
    # for the runs that most need stopping.
    killed = threading.Event()

    def _kill():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            killed.set()
        except (ProcessLookupError, PermissionError):
            pass

    watchdog = threading.Timer(RUN_TIMEOUT_S, _kill)
    watchdog.daemon = True
    watchdog.start()
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line)
            # Stream the log; the result components stay empty until the run finishes.
            yield "".join(lines), [], [], None
        proc.wait()
    finally:
        watchdog.cancel()
    if killed.is_set():
        yield (
            "".join(lines) + f"\n\nRun exceeded {RUN_TIMEOUT_S // 60} min and was stopped. "
            "Try fewer rows, a smaller k range, or a simpler algorithm.",
            [], [], None,
        )
        return
    log = "".join(lines)
    if proc.returncode != 0:
        log = f"⚠ run exited with code {proc.returncode}\n\n" + log
    # Every mode writes into out_dir/<timestamp>_.../, but at different depths: a
    # single run and a single-seed sweep put their files there directly, while
    # --seeds adds a seed_<n>/ level underneath. Both globs recurse so multi-seed
    # runs don't come back with an empty gallery.
    runs = sorted(d for d in glob.glob(os.path.join(out_dir, "*")) if os.path.isdir(d))
    base = runs[-1] if runs else out_dir
    pngs = sorted(glob.glob(os.path.join(base, "**", "*.png"), recursive=True))
    csvs = sorted(glob.glob(os.path.join(base, "**", "*.csv"), recursive=True))
    # The summary table each mode produces: Overview for a sweep, the cross-seed
    # table for --seeds, the per-cluster recap for a single run.
    preview = None
    for candidate in (os.path.join(base, "results_summary.csv"),
                      os.path.join(base, "cross_seed_summary.csv"),
                      *sorted(glob.glob(os.path.join(base, "recap", "*.csv")))):
        if os.path.exists(candidate):
            preview = pd.read_csv(candidate)
            break
    if not pngs and not csvs:
        log += "\n\n(no outputs — check the log above.)"
    yield log, pngs, csvs, preview


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

    with gr.Blocks(title="c4fairness — Clustering for Fairness") as demo:
        gr.Markdown(HOME_MD)
        file = gr.File(label="1. Dataset CSV", file_types=[".csv"])
        example_btn = gr.Button("Load example dataset (COMPAS)", size="sm")

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
            sensitive_labels = gr.Textbox(
                label="Sensitive display labels (optional)",
                placeholder="race:Ethnicity,sex_Male:Male",
                info="Heatmap labels per analysis-column name; CSVs keep raw names",
            )
            binary_error_metric = gr.Dropdown(
                ["raw", "fpr", "fnr", "precision", "prec_neg"], value="raw",
                label="Binary error metric",
                info="raw = use the error column; fpr/fnr/… are derived per cluster from "
                "y_true / y_pred",
            )
            with gr.Row():
                y_true = gr.Dropdown(label="y_true column")
                y_pred = gr.Dropdown(label="y_pred column")
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
                sensitive_gap_test = gr.Dropdown(
                    ["chi2", "fisher"], value="chi2",
                    label="Sensitive gap-sig test",
                    info="Test for <F>_gap_sig; error gap-sig stays Fisher",
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
                # A dropdown, not free text: a typo used to reach argparse and kill the
                # run only after all the clustering had finished.
                projection = gr.Dropdown(
                    ["none", "pca", "tsne", "pca,tsne"], value="none",
                    label="Projection",
                    info="Scatter plots. t-SNE is O(n^2) and runs per condition",
                )
            with gr.Row():
                categorical_cols = gr.Dropdown(
                    label="Force categorical", multiselect=True,
                    info="Columns to treat as categorical",
                )
                separability_check = gr.Checkbox(label="Separability check", value=False)
                no_standardize = gr.Checkbox(label="Skip StandardScaler", value=False)
            max_composition_plots = gr.Number(
                value=6, label="Max composition plots", precision=0,
                info="Sweep mode emits one per condition x sensitive attribute; 0 = no cap",
            )

        mode = gr.Radio(
            ["Single run", "Full sweep"],
            value="Single run",
            label="Run mode",
            info="Single run audits the configuration above. Full sweep repeats it for "
            "every feature-group combination (REG/SEN/ERR) — thorough, and several times "
            "slower.",
        )
        run_btn = gr.Button("▶ Run", variant="primary", size="lg")

        with gr.Tab("Heatmaps"):
            gallery = gr.Gallery(label="Heatmaps & plots", columns=2, height=520)
        with gr.Tab("Overview table"):
            preview = gr.Dataframe(label="results_summary.csv", wrap=True)
        with gr.Tab("Downloads"):
            files = gr.File(label="All CSV outputs", file_count="multiple")
        with gr.Tab("Log"):
            log = gr.Textbox(label="Run log", lines=18)

        # Populate every column picker from the uploaded CSV header.
        _pickers = [regular, sensitive, continuous_sensitive, proxy, special,
                    categorical_cols, error_col, y_true, y_pred]

        def _fill(f):
            # Reset value too — otherwise a re-upload keeps selections pointing at the
            # previous CSV's columns.
            cols = _cols(f.name) if f else []
            return [gr.update(choices=cols, value=None) for _ in range(len(_pickers))]

        file.change(_fill, inputs=file, outputs=_pickers)
        example_btn.click(_example_dataset, outputs=file)

        # Show only the fields the chosen error type uses. y_true/y_pred stay visible
        # for binary too — the confusion-matrix rate metrics need them.
        def _toggle(t):
            return (
                gr.update(visible=t in ("binary", "regression")),  # error_col
                gr.update(visible=t == "binary"),                  # binary_error_metric
                gr.update(visible=t == "multiclass"),              # mc_option
            )

        error_type.change(
            _toggle, inputs=error_type,
            outputs=[error_col, binary_error_metric, mc_option],
        )

        # Show only the clustering params the chosen algorithm actually uses.
        def _algo_fields(a):
            return (
                gr.update(visible=a == "dbscan"),                          # eps
                gr.update(visible=a in ("dbscan", "hdbscan")),             # min_samples
                gr.update(visible=a in ("kmeans", "bisectingkmeans", "kmedoids", "kprototypes")),  # max_iter
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
                sensitive_gap_test,
                continuous_sensitive,
                proxy,
                special,
                error_label,
                sensitive_labels,
                binary_error_metric,
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
                mode,
                max_composition_plots,
            ],
            outputs=[log, gallery, files, preview],
        )

    with demo.route("Documentation", "/documentation"):
        gr.Markdown(DOC_MD)

    with demo.route("About", "/about"):
        gr.Markdown(ABOUT_MD)

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
    assert c[:4] == [sys.executable, "-m", "c4fairness.main", "--data_path"]
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
    # kprototypes gets --max_iter like the rest of the kmeans family
    c = _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "", "", "per_class",
                   "kprototypes", "gower", 4, 42, "")
    assert "--max_iter" in c
    # single run drops --experiment and the flags that only apply to the sweep
    c = _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "", "", "per_class",
                   "kmeans", "euclidean", 4, 42, "SPECIAL", experiment=False)
    assert "--experiment" not in c and "SPECIAL" not in c
    assert "--max_composition_plots" not in c
    # the sweep keeps them
    c = _build_cmd("d.csv", "/out", "age", "race", "binary", "err", "", "", "per_class",
                   "kmeans", "euclidean", 4, 42, "", max_composition_plots=6)
    assert c[c.index("--max_composition_plots") + 1] == "6"
    # form validation catches the empty pickers before a subprocess is spawned
    assert _validate(None, "", "", "", "binary", "", "", "", "raw")
    assert _validate(object(), "age", "", "", "binary", "err", "", "", "raw")
    assert _validate(object(), "", "", "", "binary", "err", "", "", "raw")
    assert _validate(object(), "age", "race", "", "binary", "err", "", "", "raw") is None
    print("webapp selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
