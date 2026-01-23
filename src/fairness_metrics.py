"""
Fairness metrics module for evaluating cluster quality and demographic representation.

Computes:
- Cluster purity and separation metrics
- Demographic representation across clusters
- Intersectional fairness metrics
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from .clustering import ClusteringResult


@dataclass
class FairnessMetrics:
    """Container for fairness evaluation metrics."""

    demographic_parity: dict
    representation_ratio: dict
    balance_score: float
    entropy_per_cluster: dict
    overall_entropy: float


def compute_representation_ratio(
    labels: np.ndarray,
    attribute: np.ndarray,
    reference_value: int = 1,
) -> dict:
    """
    Compute representation ratio of a demographic group in each cluster vs overall.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels.
    attribute : np.ndarray
        Binary demographic attribute (e.g., gender: 0=male, 1=female).
    reference_value : int, default=1
        The attribute value to compute ratio for.

    Returns
    -------
    dict
        Mapping from cluster_id to representation ratio.
        Ratio > 1 means over-representation, < 1 means under-representation.
    """
    overall_rate = (attribute == reference_value).mean()

    ratios = {}
    for cluster_id in sorted(set(labels) - {-1}):
        mask = labels == cluster_id
        cluster_rate = (attribute[mask] == reference_value).mean()
        ratios[cluster_id] = cluster_rate / overall_rate if overall_rate > 0 else 0.0

    return ratios


def compute_demographic_parity(
    labels: np.ndarray,
    attribute: np.ndarray,
) -> dict:
    """
    Compute demographic parity: proportion of each demographic in each cluster.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels.
    attribute : np.ndarray
        Categorical demographic attribute.

    Returns
    -------
    dict
        Nested dict: {cluster_id: {attribute_value: proportion}}
    """
    parity = {}
    unique_attrs = sorted(set(attribute))

    for cluster_id in sorted(set(labels) - {-1}):
        mask = labels == cluster_id
        cluster_size = mask.sum()
        parity[cluster_id] = {}
        for attr in unique_attrs:
            count = ((labels == cluster_id) & (attribute == attr)).sum()
            parity[cluster_id][attr] = count / cluster_size if cluster_size > 0 else 0.0

    return parity


def compute_entropy(proportions: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.

    Parameters
    ----------
    proportions : np.ndarray
        Array of proportions (should sum to 1).

    Returns
    -------
    float
        Entropy value (higher = more diverse).
    """
    proportions = proportions[proportions > 0]
    return -np.sum(proportions * np.log2(proportions))


def compute_balance_score(
    labels: np.ndarray,
    attribute: np.ndarray,
) -> float:
    """
    Compute overall balance score across clusters.

    Score of 1.0 means perfect demographic balance across all clusters.
    Lower scores indicate demographic imbalance.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels.
    attribute : np.ndarray
        Categorical demographic attribute.

    Returns
    -------
    float
        Balance score between 0 and 1.
    """
    parity = compute_demographic_parity(labels, attribute)
    unique_attrs = sorted(set(attribute))

    # Compute overall proportions
    overall_props = np.array([(attribute == attr).mean() for attr in unique_attrs])

    # Compute average deviation from overall proportions
    deviations = []
    for cluster_id, props in parity.items():
        cluster_props = np.array([props[attr] for attr in unique_attrs])
        deviation = np.abs(cluster_props - overall_props).mean()
        deviations.append(deviation)

    avg_deviation = np.mean(deviations) if deviations else 0.0
    return 1.0 - avg_deviation


def evaluate_fairness(
    result: ClusteringResult,
    attribute: np.ndarray,
    attribute_name: str = "attribute",
) -> FairnessMetrics:
    """
    Compute comprehensive fairness metrics for clustering result.

    Parameters
    ----------
    result : ClusteringResult
        Output from the cluster() function.
    attribute : np.ndarray
        Demographic attribute to evaluate fairness on.
    attribute_name : str, default="attribute"
        Name of the attribute (for reporting).

    Returns
    -------
    FairnessMetrics
        Dataclass containing all fairness metrics.

    Examples
    --------
    >>> result = cluster(features)
    >>> metrics = evaluate_fairness(result, gender, "gender")
    >>> print(f"Balance score: {metrics.balance_score:.3f}")
    """
    labels = result.labels

    # Demographic parity
    parity = compute_demographic_parity(labels, attribute)

    # Representation ratios
    unique_attrs = sorted(set(attribute))
    representation = {}
    for attr in unique_attrs:
        representation[attr] = compute_representation_ratio(labels, attribute, attr)

    # Balance score
    balance = compute_balance_score(labels, attribute)

    # Entropy per cluster
    entropy_per_cluster = {}
    for cluster_id, props in parity.items():
        prop_array = np.array(list(props.values()))
        entropy_per_cluster[cluster_id] = compute_entropy(prop_array)

    # Overall entropy
    overall_props = np.array([(attribute == attr).mean() for attr in unique_attrs])
    overall_entropy = compute_entropy(overall_props)

    return FairnessMetrics(
        demographic_parity=parity,
        representation_ratio=representation,
        balance_score=balance,
        entropy_per_cluster=entropy_per_cluster,
        overall_entropy=overall_entropy,
    )


def print_fairness_report(
    metrics: FairnessMetrics,
    attribute_name: str = "attribute",
    attribute_labels: Optional[dict] = None,
) -> str:
    """
    Generate a human-readable fairness report.

    Parameters
    ----------
    metrics : FairnessMetrics
        Output from evaluate_fairness().
    attribute_name : str, default="attribute"
        Name of the demographic attribute.
    attribute_labels : dict, optional
        Mapping from attribute values to display names.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append(f"=== Fairness Report: {attribute_name} ===")
    lines.append(f"Overall Balance Score: {metrics.balance_score:.3f}")
    lines.append(f"Overall Entropy: {metrics.overall_entropy:.3f}")
    lines.append("")

    lines.append("Demographic Parity per Cluster:")
    for cluster_id, props in metrics.demographic_parity.items():
        props_str = ", ".join(
            f"{attribute_labels.get(k, k) if attribute_labels else k}: {v:.2%}"
            for k, v in props.items()
        )
        entropy = metrics.entropy_per_cluster[cluster_id]
        lines.append(f"  Cluster {cluster_id}: {props_str} (entropy: {entropy:.3f})")

    lines.append("")
    lines.append("Representation Ratios (vs overall population):")
    for attr, ratios in metrics.representation_ratio.items():
        attr_name = attribute_labels.get(attr, attr) if attribute_labels else attr
        lines.append(f"  {attr_name}:")
        for cluster_id, ratio in ratios.items():
            status = "over" if ratio > 1.1 else "under" if ratio < 0.9 else "balanced"
            lines.append(f"    Cluster {cluster_id}: {ratio:.2f}x ({status})")

    return "\n".join(lines)
