"""Scatter and composition plots. Result-table heatmaps live in `result_viz.py`."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from typing import Optional, Literal


def reduce_dimensions(
    X: np.ndarray,
    method: Literal["pca", "tsne", "mds"] = "tsne",
    n_components: int = 2,
    random_state: int = 42,
    precomputed: bool = False,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Reduce a feature matrix (or precomputed distance matrix) to 2D for visualization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features), or a square distance matrix
        (n_samples, n_samples) when precomputed=True.
    method : {"pca", "tsne", "mds"}, default="tsne"
        Pick the one that matches the clustering distance: "pca" for the kmeans
        family (it preserves the Euclidean geometry kmeans optimises), "tsne" for
        the density-based algorithms (it preserves local neighbourhoods), and
        either "tsne" or "mds" with precomputed=True to project a Gower matrix.
    n_components : int, default=2
        Number of output dimensions.
    random_state : int, default=42
        Random seed for reproducibility.
    precomputed : bool, default=False
        If True, X is a square distance matrix. Supported by "tsne" and "mds".

    Returns
    -------
    np.ndarray
        Reduced matrix of shape (n_samples, n_components).
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "tsne":
        # perplexity must be < n_samples; clamp so small cluster sets don't raise.
        perplexity = min(30, max(2, X.shape[0] - 1))
        kw = dict(n_components=n_components, random_state=random_state,
                  perplexity=perplexity)
        if precomputed:
            # Same distance as clustering (e.g. precomputed Gower matrix).
            kw.update(metric="precomputed", init="random")
        else:
            if metric != "euclidean":
                kw.update(metric=metric)   # e.g. manhattan, matching the cluster distance
            if X.shape[1] < 2:
                kw.update(init="random")   # default 'pca' init needs >= 2 features
        reducer = TSNE(**kw)
    elif method == "mds":
        dissimilarity = "precomputed" if precomputed else "euclidean"
        reducer = MDS(
            n_components=n_components,
            random_state=random_state,
            dissimilarity=dissimilarity,
            normalized_stress="auto",
            n_init=4,
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'pca', 'tsne', or 'mds'.")

    return reducer.fit_transform(X)


def plot_clusters(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
    out_path: Optional[str] = None,
    figsize: tuple = (8, 8),
    point_size: int = 10,
    alpha: float = 0.7,
    cmap: str = "tab20",
    show_legend: bool = True,
) -> plt.Figure:
    """
    Plot 2D scatter of clusters.

    Parameters
    ----------
    X_2d : np.ndarray
        2D coordinates of shape (n_samples, 2).
    labels : np.ndarray
        Cluster labels.
    title : str, default="Cluster Visualization"
        Plot title.
    out_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(8, 8)
        Figure size.
    point_size : int, default=10
        Size of scatter points.
    alpha : float, default=0.7
        Point transparency.
    cmap : str, default="tab20"
        Colormap for clusters.
    show_legend : bool, default=True
        Whether to show cluster legend.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        color = "gray" if label == -1 else None
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=point_size,
            alpha=alpha if label != -1 else 0.3,
            c=color,
            label=label_name,
        )

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    if show_legend and len(unique_labels) <= 15:
        ax.legend(markerscale=2, loc="best")

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)

    return fig


def plot_cluster_composition(
    labels: np.ndarray,
    attribute: np.ndarray,
    attribute_name: str,
    attribute_labels: Optional[dict] = None,
    title: Optional[str] = None,
    out_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot stacked bar chart showing demographic composition of each cluster.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels.
    attribute : np.ndarray
        Categorical attribute values (e.g., gender encoded as 0/1).
    attribute_name : str
        Name of the attribute.
    attribute_labels : dict, optional
        Mapping from attribute values to display names.
        Example: {0: "Male", 1: "Female"}
    title : str, optional
        Plot title.
    out_path : str, optional
        Path to save the figure.
    figsize : tuple, default=(10, 6)
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_clusters = sorted(set(labels) - {-1})
    unique_attrs = sorted(set(attribute))

    if attribute_labels is None:
        attribute_labels = {v: str(v) for v in unique_attrs}

    proportions = {attr: [] for attr in unique_attrs}
    for cluster in unique_clusters:
        cluster_mask = labels == cluster
        cluster_size = cluster_mask.sum()
        for attr in unique_attrs:
            count = ((labels == cluster) & (attribute == attr)).sum()
            proportions[attr].append(count / cluster_size if cluster_size > 0 else 0)

    x = np.arange(len(unique_clusters))
    bottom = np.zeros(len(unique_clusters))

    for attr in unique_attrs:
        ax.bar(
            x, proportions[attr], bottom=bottom,
            label=attribute_labels.get(attr, str(attr))
        )
        bottom += np.array(proportions[attr])

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in unique_clusters])
    ax.legend(title=attribute_name)
    ax.set_title(title or f"Cluster Composition by {attribute_name}")

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)

    return fig

