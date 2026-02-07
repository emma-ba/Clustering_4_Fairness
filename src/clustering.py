"""
Clustering module for fairness analysis.

This module provides a flexible clustering function that supports:
- Filtering by prediction outcome (TP, TN, FP, FN)
- Multiple distance metrics (Euclidean, Manhattan, Gower)
- Feature weighting
- Multiple clustering algorithms (HDBSCAN, with placeholders for others)
- Cluster quality evaluation
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal, Union
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from kmodes.kprototypes import KPrototypes                                                                                                                                              

@dataclass
class ClusteringResult:
    """Container for clustering results and evaluation metrics."""

    labels: np.ndarray
    n_clusters: int
    n_noise: int
    silhouette: Optional[float]
    calinski_harabasz: Optional[float]
    cluster_sizes: dict
    feature_matrix: np.ndarray
    mask: Optional[np.ndarray]

def gower_distance(X: np.ndarray,
                   categorical_features: Optional[list[int]] = None,
                   weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Gower distance matrix for mixed-type data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    categorical_features : list[int], optional
        Indices of categorical features. If None, all features are treated as numeric.
    weights : np.ndarray, optional
        Feature weights of shape (n_features,). If None, equal weights are used.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples, n_samples).
    """
    n_samples, n_features = X.shape

    if categorical_features is None:
        categorical_features = []

    if weights is None:
        weights = np.ones(n_features)

    weights = weights / weights.sum()

    numeric_features = [i for i in range(n_features) if i not in categorical_features]

    # Compute ranges for numeric features
    ranges = np.zeros(n_features)
    for i in numeric_features:
        col = X[:, i].astype(float)
        ranges[i] = col.max() - col.min()
        if ranges[i] == 0:
            ranges[i] = 1.0

    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = 0.0
            for k in range(n_features):
                if k in categorical_features:
                    d += weights[k] * (0 if X[i, k] == X[j, k] else 1)
                else:
                    d += weights[k] * abs(float(X[i, k]) - float(X[j, k])) / ranges[k]
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

    return distance_matrix


def cluster(
    features: Union[np.ndarray, pd.DataFrame],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    subset: Optional[Literal["TP", "TN", "FP", "FN", "TP_TN", "FP_FN"]] = None,
    algorithm: Literal["dbscan", "hdbscan", "kmeans", "bisectingkmeans", "kprototypes"] = "hdbscan",
    distance: Literal["euclidean", "manhattan", "gower"] = "euclidean",
    categorical_features: Optional[list[int]] = None,
    feature_weights: Optional[Union[np.ndarray, dict]] = None,
    eps: float = 0.5,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    n_clusters: Optional[int] = None,
    n_min: Optional[int] = None,
    n_max: Optional[int] = None,
    max_iter: int = 300,
    random_state: int = 42,
    standardize: bool = True,
    min_datapoints: Optional[int] = None,
) -> ClusteringResult:
    """
    Perform clustering on features with flexible configuration.

    Parameters
    ----------
    features : np.ndarray or pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
        Can include embeddings, demographic attributes, etc.
    y_true : np.ndarray, optional
        Ground truth labels for computing confusion matrix subsets.
    y_pred : np.ndarray, optional
        Predicted labels for computing confusion matrix subsets.
    subset : {"TP", "TN", "FP", "FN", "TP_TN", "FP_FN"}, optional
        Filter to specific confusion matrix category before clustering.
        TP_TN = correct predictions, FP_FN = errors.
        Requires y_true and y_pred to be provided.
    algorithm : {"hdbscan", "kmeans", "bisecting", "agglomerative"}, default="hdbscan"
        Clustering algorithm to use. "bisecting" uses Bisecting K-Means.
    distance : {"euclidean", "manhattan", "gower"}, default="euclidean"
        Distance metric for clustering.
        Use "gower" for mixed numeric/categorical features.
    categorical_features : list[int], optional
        Indices of categorical features (required for Gower distance).
    feature_weights : np.ndarray or dict, optional
        Weights for each feature. Can be array of shape (n_features,)
        or dict mapping feature names to weights (if features is DataFrame).
    eps : float, default=0.5                                                                                                                                                            
          Maximum distance between samples for neighborhood (DBSCAN only).
    min_cluster_size : int, default=50
        Minimum cluster size (for HDBSCAN).
    min_samples : int, default=10
        Minimum samples in neighborhood (for HDBSCAN).
    n_clusters : int, optional
        Number of clusters (for KMeans, Agglomerative).
    n_min : int, optional
        Minimum number of clusters for range-based search.
    n_max : int, optional
        Maximum number of clusters for range-based search.
    max_iter : int, default=300
        Maximum number of iterations for KMeans/BisectingKMeans.
    random_state : int, default=42
        Random seed for reproducibility.
    standardize : bool, default=True
        Whether to standardize numeric features before clustering.

    Returns
    -------
    ClusteringResult
        Dataclass containing cluster labels, metrics, and metadata.
    """
    # Convert DataFrame to numpy if needed
    if isinstance(features, pd.DataFrame):
        feature_names = features.columns.tolist()
        X = features.values
    else:
        feature_names = None
        X = features.copy()



    # Compute mask for confusion matrix subset
    mask = None
    if subset is not None:
        if y_true is None or y_pred is None:
            raise ValueError(f"subset='{subset}' requires y_true and y_pred")
        if subset == "TP":
            mask = (y_true == 1) & (y_pred == 1)
        elif subset == "TN":
            mask = (y_true == 0) & (y_pred == 0)
        elif subset == "FP":
            mask = (y_true == 0) & (y_pred == 1)
        elif subset == "FN":
            mask = (y_true == 1) & (y_pred == 0)
        elif subset == "TP_TN":
            mask = y_true == y_pred
        elif subset == "FP_FN":
            mask = y_true != y_pred
        else:
            raise ValueError(f"Invalid subset: {subset}")

        X = X[mask]

    # Handle feature weights
    weights = None
    if feature_weights is not None:
        if isinstance(feature_weights, dict) and feature_names is not None:
            weights = np.array([feature_weights.get(name, 1.0) for name in feature_names])
        else:
            weights = np.asarray(feature_weights)

    # Standardize numeric features (skip for kprototypes which handles mixed types internally)
    if standardize and distance != "gower" and algorithm != "kprototypes":
        if categorical_features:
            numeric_mask = [i for i in range(X.shape[1]) if i not in categorical_features]
            scaler = StandardScaler()
            X[:, numeric_mask] = scaler.fit_transform(X[:, numeric_mask].astype(float))
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

    # Apply feature weights (for non-Gower distances)
    if weights is not None and distance != "gower":
        X = X * np.sqrt(weights)

    if algorithm == "dbscan":
        if distance == "gower":
            """ 
            HDBSCAN does not have built-in support for Gower distace. 
            1. We compute the distance matrix manually with our own function written from scratch
            2. We pass to HDBSCAN using metric=="precomputed"
            """
            dist_matrix = gower_distance(X, categorical_features, weights)
            clusterer = DBSCAN(
                eps=eps,                      
                min_samples=min_samples,
                metric="precomputed",
            )
            labels = clusterer.fit_predict(dist_matrix)
        elif distance in ("euclidean", "manhattan"):
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=distance,
            )
            labels = clusterer.fit_predict(X)
    elif algorithm == "hdbscan":
        if distance == "gower":
            """ 
            HDBSCAN does not have built-in support for Gower distace. 
            1. We compute the distance matrix manually with our own function written from scratch
            2. We pass to HDBSCAN using metric=="precomputed"
            """
            dist_matrix = gower_distance(X, categorical_features, weights)
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="precomputed",
            )
            labels = clusterer.fit_predict(dist_matrix)
        elif distance in ("euclidean", "manhattan"):
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=distance,
            )
            labels = clusterer.fit_predict(X)


    elif algorithm == "kmeans":
        if n_clusters is not None:
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=max_iter)
            labels = clusterer.fit_predict(X)
        elif n_min is not None and n_max is not None:
            best_score, best_k, best_labels = -1, n_min, None
            for k in range(n_min, n_max + 1):
                clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=max_iter)
                labels = clusterer.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score, best_k, best_labels = score, k, labels
            print(f"  Best k={best_k} (silhouette={best_score:.3f})")
            labels = best_labels
        else:
            raise ValueError("n_clusters or n_min/n_max required for kmeans")                                                                                                           
# TODO: Look into papers as to how they implemented bisectingkmeans. Pour compenser le pb de kmeans. Peutetre qu'il n'y aurait pas de sens de faire du bisecting, en concordance avec la recherche. Mitzal Radecka - papiers. 
    elif algorithm == "bisectingkmeans":
        if n_clusters is not None:
            clusterer = BisectingKMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)
            labels = clusterer.fit_predict(X)
        elif n_min is not None and n_max is not None:
            best_score, best_k, best_labels = -1, n_min, None
            # TODO: quand on fait des clusters de fairness, on peut avoir d'autres criteres - si cela separe bien les erreurs, et pas seulement le silhouette score. 
            # TODO: Garder cetter fonction qui fait une iteraction, et garder l'option de faire le hyperparamter tuning, et avoir une option e donner le critere. 
            # TODO: Don't forget to include chi square test, ANOVA (pense a commment integrer ca pour choisir le best_k, KPrototypes -> + tard)
            # TODO: Include the rest of the tables from notebook.
            for k in range(n_min, n_max + 1):
                clusterer = BisectingKMeans(n_clusters=k, random_state=random_state, max_iter=max_iter)
                labels = clusterer.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score, best_k, best_labels = score, k, labels
            print(f"  Best k={best_k} (silhouette={best_score:.3f})")
            labels = best_labels
        else:
            raise ValueError("n_clusters or n_min/n_max required for bisectingkmeans")
    elif algorithm == "kprototypes":
        # NOTE: KPrototypes uses its own internal distance metric (Huang's cost function):
        #   - Numeric features: squared Euclidean distance
        #   - Categorical features: simple matching dissimilarity (0 if match, 1 otherwise)
        # Gower distance is not compatible with KPrototypes.
        # For Gower-based mixed-type clustering, use DBSCAN or HDBSCAN with --distance gower.
        # Future: KMedoids with precomputed Gower matrix could provide centroid-based + Gower.
        if categorical_features is None or len(categorical_features) == 0:
            raise ValueError("kprototypes requires categorical_features to be specified")
        if n_clusters is not None:
            clusterer = KPrototypes(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=max_iter)
            labels = clusterer.fit_predict(X, categorical=categorical_features)
        elif n_min is not None and n_max is not None:
            best_score, best_labels = -1, None
            for k in range(n_min, n_max + 1):
                clusterer = KPrototypes(n_clusters=k, random_state=random_state, n_init=10, max_iter=max_iter)
                labels = clusterer.fit_predict(X, categorical=categorical_features)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score, best_labels = score, labels
            print(f"  Best k={k} (silhouette={best_score:.3f})")
            labels = best_labels
        else:
            raise ValueError("n_clusters or n_min/n_max required for kprototypes")
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Filter small clusters to noise if min_datapoints is set
    if min_datapoints is not None:
        for label in set(labels):
            if label == -1:
                continue
            cluster_count = (labels == label).sum()
            if cluster_count < min_datapoints:
                labels = np.where(labels == label, -1, labels)

    # Compute evaluation metrics
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = (labels == -1).sum()

    # Silhouette and Calinski-Harabasz require at least 2 clusters and numeric data
    silhouette = None
    calinski = None
    if n_clusters >= 2:
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > n_clusters:
            # Skip metrics for mixed-type data (kprototypes or gower with categoricals)
            if categorical_features and (algorithm == "kprototypes" or distance == "gower"):
                pass  # Cannot compute silhouette on mixed data
            else:
                silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                calinski = calinski_harabasz_score(X[non_noise_mask], labels[non_noise_mask])

    # Compute cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        cluster_sizes[label] = (labels == label).sum()

    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=silhouette,
        calinski_harabasz=calinski,
        cluster_sizes=cluster_sizes,
        feature_matrix=X,
        mask=mask,
    )
    