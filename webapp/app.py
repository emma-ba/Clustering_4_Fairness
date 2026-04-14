import os, sys, json, uuid, threading, queue, io, base64, traceback, re, math
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify, Response, render_template, stream_with_context

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import cluster
from src.scoring import (
    make_chi2_error_scorer, make_kruskal_error_scorer,
    make_chi2_sensitive_scorer, make_composite_scorer,
)
from src.visualization import reduce_dimensions, plot_clusters, plot_cluster_composition
from src.experiments import (
    create_exp_conditions, run_experiments_generic, make_recap,
    make_chi_tests, recap_quali_metrics, separability_check,
)
from src.preprocessing import encode_categoricals

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

JOBS = {}  # job_id -> dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _QueueWriter(io.TextIOBase):
    """Redirect stdout line-by-line into a queue for SSE streaming."""
    def __init__(self, q):
        self.q = q

    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                self.q.put({'type': 'log', 'text': line})
        return len(s)

    def flush(self):
        pass


def _fig_to_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _df_to_json(df):
    """Serialize DataFrame to list-of-dicts, replacing NaN/inf with None."""
    return json.loads(df.to_json(orient='records'))


# ---------------------------------------------------------------------------
# Analysis workers
# ---------------------------------------------------------------------------

def _run_analysis(job_id, df, params):
    job = JOBS[job_id]
    q = job['queue']
    writer = _QueueWriter(q)

    try:
        with redirect_stdout(writer):
            # --- unpack params ---
            regular_cols   = params.get('regular_cols', [])
            sensitive_cols = params.get('sensitive_cols', [])
            special_cols   = params.get('special_cols', [])
            error_col      = params.get('error_col')
            error_type     = params.get('error_type', 'binary')
            algorithm      = params.get('algorithm', 'hdbscan')
            distance       = params.get('distance', 'euclidean')
            n_clusters     = params.get('n_clusters')
            n_min          = params.get('n_min')
            n_max          = params.get('n_max')
            seed           = params.get('seed', 42)
            min_cs         = params.get('min_cluster_size', 15)
            min_samp       = params.get('min_samples', 5)
            eps            = params.get('eps', 0.5)
            max_iter       = params.get('max_iter', 300)
            min_dp         = params.get('min_datapoints')
            scoring        = params.get('scoring', 'composite')
            experiment     = params.get('experiment', False)
            projection     = params.get('projection', 'tsne')

            # resolve k params
            if n_min or n_max:
                n_clusters = None
                n_min = n_min or 2
                n_max = n_max or 10
            elif n_clusters is None:
                n_clusters = 5

            # scoring function
            scoring_fn = None
            if scoring == 'chi2_error' and error_col:
                fn = make_kruskal_error_scorer if error_type == 'regression' else make_chi2_error_scorer
                scoring_fn = fn(df[error_col].values)
            elif scoring == 'chi2_sensitive' and sensitive_cols:
                scoring_fn = make_chi2_sensitive_scorer(df[sensitive_cols[0]].values)
            elif scoring == 'composite':
                if error_col or sensitive_cols:
                    scoring_fn = make_composite_scorer(
                        error_data=df[error_col].values if error_col else None,
                        sensitive_data=df[sensitive_cols[0]].values if sensitive_cols else None,
                        error_type=error_type,
                    )

            # One-hot encode string/category columns before clustering
            categorical_cols_arg = params.get('categorical_cols', [])
            col_lists = {'regular': regular_cols, 'sensitive': sensitive_cols, 'special': special_cols}
            df, col_lists, categorical_col_names = encode_categoricals(
                df, col_lists, categorical_cols_arg, algorithm
            )
            regular_cols   = col_lists['regular']
            sensitive_cols = col_lists['sensitive']
            special_cols   = col_lists['special']

            if experiment:
                results_data = _run_experiment(
                    df, regular_cols, sensitive_cols, special_cols,
                    error_col, error_type, algorithm, distance,
                    n_clusters, n_min, n_max, seed, min_cs, min_samp,
                    eps, max_iter, min_dp, scoring_fn, projection,
                    categorical_col_names=categorical_col_names,
                )
            else:
                results_data = _run_single(
                    df, regular_cols, sensitive_cols, special_cols,
                    error_col, error_type, algorithm, distance,
                    n_clusters, n_min, n_max, seed, min_cs, min_samp,
                    eps, max_iter, min_dp, scoring_fn, projection,
                    categorical_col_names=categorical_col_names,
                )

        job['status'] = 'done'
        job['results'] = results_data
        q.put({'type': 'done', 'results': results_data})

    except Exception as e:
        tb = traceback.format_exc()
        q.put({'type': 'error', 'text': str(e), 'traceback': tb})
        job['status'] = 'error'


def _run_single(df, regular_cols, sensitive_cols, special_cols,
                error_col, error_type, algorithm, distance,
                n_clusters, n_min, n_max, seed, min_cs, min_samp,
                eps, max_iter, min_dp, scoring_fn, projection,
                categorical_col_names=None):

    feature_set = regular_cols + sensitive_cols + special_cols
    if not feature_set:
        raise ValueError("No feature columns specified. Please assign at least one column as Regular, Sensitive, or Special.")

    print(f"Running single-run analysis")
    print(f"  Algorithm : {algorithm}   Distance : {distance}")
    print(f"  Features  : {', '.join(feature_set)}")
    if error_col:
        print(f"  Error col : {error_col}   Type : {error_type}")
    if n_min:
        print(f"  k range   : {n_min}–{n_max}")
    else:
        print(f"  k         : {n_clusters}")

    cat_names_set = set(categorical_col_names or [])
    cat_features = [i for i, c in enumerate(feature_set) if c in cat_names_set] or None

    result = cluster(
        features=df[feature_set],
        algorithm=algorithm,
        distance=distance,
        n_clusters=n_clusters,
        n_min=n_min, n_max=n_max,
        max_iter=max_iter,
        random_state=seed,
        scoring_fn=scoring_fn,
        min_cluster_size=min_cs,
        min_samples=min_samp,
        eps=eps,
        min_datapoints=min_dp,
        categorical_features=cat_features,
    )

    print(f"  Clusters  : {result.n_clusters}   Noise : {result.n_noise}")
    if result.silhouette is not None:
        print(f"  Silhouette: {result.silhouette:.4f}")

    # Attach cluster labels to original df
    res_df = df.copy()
    if result.mask is not None:
        res_df['clusters'] = -1
        res_df.loc[result.mask, 'clusters'] = result.labels
        masked_idx = res_df.index[result.mask]
    else:
        res_df['clusters'] = result.labels
        masked_idx = res_df.index

    labels = result.labels  # labels only for included rows

    # Recap table
    recap = None
    if error_col:
        recap = make_recap(
            res_df, feature_set,
            sensitive_cols=sensitive_cols,
            error_col=error_col,
            error_type=error_type,
            feature_matrix=result.feature_matrix,
            distance_matrix=result.distance_matrix,
        )
        print(f"  Recap     : {len(recap)} cluster rows")

    images = {}

    # Scatter / projection
    if projection != 'none' and result.n_clusters > 1:
        print(f"  Generating {projection.upper()} scatter plot…")
        X_2d = reduce_dimensions(result.feature_matrix, method=projection)
        fig = plot_clusters(X_2d, labels, title=f"Clusters (n={result.n_clusters})")
        images['scatter'] = _fig_to_b64(fig)
        plt.close(fig)

    # Demographic composition plots
    for attr in sensitive_cols:
        attr_vals = df.loc[masked_idx, attr].values
        fig = plot_cluster_composition(labels, attr_vals, attr)
        images[f'composition_{attr}'] = _fig_to_b64(fig)
        plt.close(fig)

    # Recap heatmap
    recap_image = None
    if recap is not None and len(recap) > 1:
        recap_h = recap.copy().sort_values('diff_vs_rest', ascending=False)
        recap_h['count'] = recap_h['count'] / recap_h['count'].sum()
        recap_h = recap_h.rename(columns={'count': 'size_prop'})
        drop_c = ['c'] + (['n_error'] if 'n_error' in recap_h.columns else [])
        recap_h = recap_h.drop(drop_c, axis=1)
        nc, nr = len(recap_h.columns), len(recap_h)
        fig, ax = plt.subplots(figsize=(max(10, nc * 0.9), max(4, nr * 1.2)))
        sns.heatmap(recap_h, annot=True, fmt='.3g', center=0, cbar=False,
                    cmap=sns.color_palette('vlag', as_cmap=True), robust=True, ax=ax)
        ax.xaxis.tick_top()
        ax.set(xlabel='', ylabel='')
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='y', which='major', length=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
        plt.yticks(rotation='horizontal')
        plt.tight_layout()
        recap_image = _fig_to_b64(fig)
        plt.close(fig)

    print("Done.")
    return {
        'mode': 'single',
        'n_clusters': result.n_clusters,
        'n_noise': result.n_noise,
        'silhouette': result.silhouette,
        'images': images,
        'recap': _df_to_json(recap) if recap is not None else None,
        'recap_image': recap_image,
    }


def _run_experiment(df, regular_cols, sensitive_cols, special_cols,
                    error_col, error_type, algorithm, distance,
                    n_clusters, n_min, n_max, seed, min_cs, min_samp,
                    eps, max_iter, min_dp, scoring_fn, projection,
                    categorical_col_names=None):

    if not error_col:
        raise ValueError("An error column is required in experiment mode.")
    if not sensitive_cols:
        raise ValueError("At least one sensitive column is required in experiment mode.")

    groups = {}
    if regular_cols:   groups['REG']     = regular_cols
    if sensitive_cols: groups['SEN']     = sensitive_cols
    groups['ERR'] = [error_col]
    if special_cols:   groups['SPECIAL'] = special_cols

    exp_condition = create_exp_conditions(groups)
    print(f"Experiment mode: {len(exp_condition)} conditions")
    print(f"  Groups : {list(groups.keys())}")
    print(f"  Algorithm : {algorithm}   Distance : {distance}")
    print()

    results = run_experiments_generic(
        df, exp_condition,
        algorithm=algorithm, distance=distance,
        n_clusters=n_clusters, n_min=n_min, n_max=n_max,
        max_iter=max_iter, seed=seed,
        scoring_fn=scoring_fn,
        sensitive_cols=sensitive_cols,
        error_col=error_col,
        min_cluster_size=min_cs, min_samples=min_samp,
        eps=eps, min_datapoints=min_dp,
        error_type=error_type,
        categorical_col_names=categorical_col_names,
    )

    for i, cname in enumerate(results['cond_name']):
        recap = results['cond_recap'][i]
        nk = len(recap)
        sil = recap['silhouette'].mean() if 'silhouette' in recap.columns and nk > 0 else float('nan')
        sil_s = f"{sil:.3f}" if not math.isnan(sil) else "n/a"
        print(f"  [{i+1}/{len(results['cond_name'])}] {cname.strip()}: {nk} clusters, sil={sil_s}")

    print()
    print("Running statistical tests…")
    chi_res  = make_chi_tests(results, sensitive_cols=sensitive_cols,
                               error_type=error_type, error_col=error_col)
    all_quali = recap_quali_metrics(chi_res, results, exp_condition, sensitive_cols=sensitive_cols)
    actual_sensitive = [c for c in chi_res.columns if c not in ('cond_descr', 'cond_name', 'error')]

    images = {}

    # Chi-squared heatmap
    chi_viz = chi_res[['error'] + actual_sensitive].copy()
    chi_viz.index = chi_res['cond_name'].str.strip()
    fw = max(5, len(chi_viz.columns) + 2)
    fh = max(5, len(chi_viz) * 0.7)
    fig, ax = plt.subplots(figsize=(fw, fh))
    sns.heatmap(chi_viz, annot=True, center=0.05, cbar=False,
                cmap=sns.color_palette('vlag', as_cmap=True), robust=True, ax=ax)
    ax.set_title("Statistical Test p-values")
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='major', length=0)
    ax.tick_params(axis='y', which='major', length=0)
    plt.yticks(rotation=0, ha='right')
    plt.tight_layout()
    images['chi_heatmap'] = _fig_to_b64(fig)
    plt.close(fig)

    # Quality heatmap
    q_cols = ['error'] + actual_sensitive + ['silhouette']
    q_viz = all_quali[q_cols].copy()
    q_viz.index = all_quali['cond_name'].str.strip()
    fig, ax = plt.subplots(figsize=(max(5, len(q_cols) + 2), max(5, len(q_viz) * 0.7)))
    sns.heatmap(q_viz, annot=True, center=0, cbar=False,
                cmap=sns.color_palette('vlag', as_cmap=True), robust=True, ax=ax)
    ax.set(xlabel='', ylabel='')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='major', length=0)
    ax.tick_params(axis='y', which='major', length=0)
    plt.yticks(ha='left')
    plt.tight_layout()
    images['quali_heatmap'] = _fig_to_b64(fig)
    plt.close(fig)

    # Per-condition recap heatmaps
    print("Generating per-condition heatmaps…")
    recap_images = []
    for i, cname in enumerate(results['cond_name']):
        recap = results['cond_recap'][i].copy()
        if len(recap) <= 1:
            continue
        rs = recap.sort_values('diff_vs_rest', ascending=False).copy()
        rs['count'] = rs['count'] / rs['count'].sum()
        rs = rs.rename(columns={'count': 'size_prop'})
        dc = ['c'] + (['n_error'] if 'n_error' in rs.columns else [])
        rs = rs.drop(dc, axis=1)
        nc, nr = len(rs.columns), len(rs)
        fig, ax = plt.subplots(figsize=(max(10, nc * 0.9), max(4, nr * 1.2)))
        sns.heatmap(rs, annot=True, fmt='.3g', center=0, cbar=False,
                    cmap=sns.color_palette('vlag', as_cmap=True), robust=True, ax=ax)
        ax.set_title(re.sub(' +', ' ', cname))
        ax.xaxis.tick_top()
        ax.set(xlabel='', ylabel='')
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='y', which='major', length=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
        plt.yticks(rotation='horizontal')
        plt.tight_layout()
        recap_images.append({'cond_name': cname, 'image': _fig_to_b64(fig)})
        plt.close(fig)

    print(f"Done. {len(recap_images)} heatmaps generated.")
    return {
        'mode': 'experiment',
        'chi_res':  _df_to_json(chi_res),
        'all_quali': _df_to_json(all_quali),
        'images': images,
        'recap_images': recap_images,
        'conditions': [
            {
                'name': results['cond_name'][i],
                'recap': _df_to_json(results['cond_recap'][i]),
            }
            for i in range(len(results['cond_name']))
        ],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400
    try:
        df = pd.read_csv(f)
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            'df': df,
            'queue': queue.Queue(),
            'status': 'pending',
            'results': None,
        }
        columns = [
            {
                'name': c,
                'dtype': str(df[c].dtype),
                'n_unique': int(df[c].nunique()),
                'sample': [str(v) for v in df[c].dropna().head(3).tolist()],
            }
            for c in df.columns
        ]
        preview = json.loads(df.head(5).to_json(orient='records'))
        return jsonify({
            'job_id': job_id,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': columns,
            'preview': preview,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    job_id = data.get('job_id')
    if not job_id or job_id not in JOBS:
        return jsonify({'error': 'Invalid job_id'}), 400
    job = JOBS[job_id]
    # Reset for re-runs
    job['queue'] = queue.Queue()
    job['status'] = 'running'
    job['results'] = None
    t = threading.Thread(
        target=_run_analysis,
        args=(job_id, job['df'], data.get('params', {})),
        daemon=True,
    )
    t.start()
    return jsonify({'status': 'started', 'job_id': job_id})


@app.route('/api/stream/<job_id>')
def stream(job_id):
    if job_id not in JOBS:
        return jsonify({'error': 'Job not found'}), 404
    q = JOBS[job_id]['queue']

    def generate():
        while True:
            try:
                msg = q.get(timeout=60)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg['type'] in ('done', 'error'):
                    break
            except queue.Empty:
                yield 'data: {"type":"heartbeat"}\n\n'

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


if __name__ == '__main__':
    app.run(debug=False, port=5050, threaded=True)
