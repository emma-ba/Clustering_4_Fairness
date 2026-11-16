# Contributing

## Development setup

```bash
pip install -e ".[web]"
pytest
```

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
├── docs/                     # reference docs + example notebooks
│   ├── datasets/             # the COMPAS + student extracts the notebooks read
│   └── images/               # heatmaps used in this README
├── tests/
└── pyproject.toml
```
