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

# gradio imported lazily inside launch() so _build_cmd stays testable without it.


def _build_cmd(data_path, out_dir, regular, sensitive, error_type, error_col,
               y_true, y_pred, mc_option, algorithm, distance, n_clusters,
               seed, exclude):
    """Assemble the `c4f --experiment ...` argv from form inputs."""
    cmd = [sys.executable, "-m", "c4f.main",
           "--data_path", data_path,
           "--experiment"]
    if exclude:
        cmd.append(exclude)
    cmd += ["--algorithm", algorithm, "--distance", distance,
            "--n_clusters", str(int(n_clusters)), "--seed", str(int(seed)),
            "--output_dir", out_dir, "--error_type", error_type]
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
        cmd += ["--y_true_col", y_true, "--y_pred_col", y_pred,
                "--error_multiclass_option", mc_option]
    return cmd


def _run(file, regular, sensitive, error_type, error_col, y_true, y_pred,
         mc_option, algorithm, distance, n_clusters, seed, exclude):
    if file is None:
        return "Upload a CSV first.", [], []
    out_dir = tempfile.mkdtemp(prefix="c4f_web_")
    cmd = _build_cmd(file.name, out_dir, regular, sensitive, error_type, error_col,
                     y_true, y_pred, mc_option, algorithm, distance, n_clusters,
                     seed, exclude)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    # run_batch_experiment writes into out_dir/<timestamp>_experiment_.../
    runs = sorted(glob.glob(os.path.join(out_dir, "*experiment*")))
    base = runs[-1] if runs else out_dir
    pngs = sorted(glob.glob(os.path.join(base, "*.png")))
    csvs = sorted(glob.glob(os.path.join(base, "*.csv")))
    return log, pngs, csvs


def launch(**kwargs):
    import gradio as gr

    with gr.Blocks(title="c4f — Clustering for Fairness") as demo:
        gr.Markdown("# c4f — Clustering for Fairness\nExperiment mode. "
                    "Needs R + rpy2 on the host for the omnibus error test.")
        file = gr.File(label="Dataset CSV", file_types=[".csv"])
        with gr.Row():
            regular = gr.Textbox(label="Regular cols (comma-sep)")
            sensitive = gr.Textbox(label="Sensitive cols (comma-sep)")
        with gr.Row():
            error_type = gr.Dropdown(["binary", "regression", "multiclass"],
                                     value="binary", label="Error type")
            error_col = gr.Textbox(label="Error col (binary/regression)")
            mc_option = gr.Dropdown(
                ["accuracy", "per_class", "precision", "per_cell",
                 "binary_cells", "onehot", "classwise"],
                value="per_class", label="Multiclass option")
        with gr.Row():
            y_true = gr.Textbox(label="y_true col (regression/multiclass)")
            y_pred = gr.Textbox(label="y_pred col (regression/multiclass)")
        with gr.Row():
            algorithm = gr.Dropdown(
                ["kmeans", "hdbscan", "dbscan", "bisectingkmeans", "kmedoids", "kprototypes"],
                value="kmeans", label="Algorithm")
            distance = gr.Dropdown(["euclidean", "manhattan", "gower"],
                                   value="euclidean", label="Distance")
            n_clusters = gr.Number(value=4, label="n_clusters", precision=0)
            seed = gr.Number(value=42, label="seed", precision=0)
            exclude = gr.Textbox(label="Exclude groups (e.g. SPECIAL,ERR)")
        run_btn = gr.Button("Run experiment", variant="primary")
        log = gr.Textbox(label="Log", lines=12)
        gallery = gr.Gallery(label="Heatmaps & plots", columns=3)
        files = gr.File(label="CSV outputs", file_count="multiple")

        run_btn.click(
            _run,
            inputs=[file, regular, sensitive, error_type, error_col, y_true,
                    y_pred, mc_option, algorithm, distance, n_clusters, seed, exclude],
            outputs=[log, gallery, files],
        )
    demo.launch(**kwargs)


def _selfcheck():
    # binary path
    c = _build_cmd("d.csv", "/out", "age", "race,sex", "binary", "err",
                   "", "", "per_class", "kmeans", "euclidean", 4, 42, "")
    assert c[:4] == [sys.executable, "-m", "c4f.main", "--data_path"]
    assert "--experiment" in c and "--error_col" in c and "err" in c
    assert "--y_true_col" not in c
    # multiclass path adds y_true/y_pred + option, not error_col
    c = _build_cmd("d.csv", "/out", "", "race", "multiclass", "",
                   "yt", "yp", "classwise", "kmeans", "gower", 5, 1, "SPECIAL")
    assert "--y_true_col" in c and "yt" in c and "classwise" in c
    assert "--error_col" not in c
    assert c[c.index("--experiment") + 1] == "SPECIAL"
    print("webapp selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
