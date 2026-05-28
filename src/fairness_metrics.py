"""
Fairness metrics for clustering analysis.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FairnessMetrics:
    """Container for fairness evaluation metrics."""

    demographic_parity: dict
    representation_ratio: dict
    balance_score: float
    entropy_per_cluster: dict
    overall_entropy: float
