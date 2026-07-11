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
        "--n_clusters",
        str(int(n_clusters)),
        "--seed",
        str(int(seed)),
        "--output_dir",
        out_dir,
        "--error_type",
        error_type,
    ]
    if regular:
        cmd += ["--regular_cols", regular]
    if sensitive:
        cmd += ["--sensitive_cols", sensitive]
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
        gr.Markdown(
            "# c4f — Clustering for Fairness\n"
            "Upload a CSV, pick your columns, run the experiment. Results "
            "(heatmaps + tables) appear below.\n\n"
            "*Experiment mode needs R ≥ 4.5 + rpy2 on the host for the omnibus "
            "error test.*"
        )
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
            exclude = gr.Textbox(
                label="Exclude groups from conditions", placeholder="e.g. SPECIAL,ERR"
            )

        run_btn = gr.Button("▶ Run experiment", variant="primary", size="lg")

        with gr.Tab("Heatmaps"):
            gallery = gr.Gallery(label="Heatmaps & plots", columns=2, height=520)
        with gr.Tab("Overview table"):
            preview = gr.Dataframe(label="results_summary.csv", wrap=True)
        with gr.Tab("Downloads"):
            files = gr.File(label="All CSV outputs", file_count="multiple")
        with gr.Tab("Log"):
            log = gr.Textbox(label="Run log", lines=18)

        # Populate every column picker from the uploaded CSV header.
        def _fill(f):
            cols = _cols(f.name) if f else []
            return [gr.update(choices=cols) for _ in range(5)]

        file.change(
            _fill, inputs=file, outputs=[regular, sensitive, error_col, y_true, y_pred]
        )

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
