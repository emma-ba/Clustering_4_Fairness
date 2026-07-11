import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from c4f.visualization import reduce_dimensions
from c4f.cli import parse_projection_list


def test_parse_projection_list_multiple():
    assert parse_projection_list("pca,tsne") == ["pca", "tsne"]


def test_parse_projection_list_none_skips_all():
    assert parse_projection_list("none") == []
    assert parse_projection_list("pca,none") == []


def test_parse_projection_list_single():
    assert parse_projection_list("tsne") == ["tsne"]


def test_tsne_precomputed_returns_2d():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    D = squareform(pdist(X))          # Gower-style precomputed distance matrix
    out = reduce_dimensions(D, method="tsne", precomputed=True)
    assert out.shape == (40, 2)


def test_tsne_manhattan_returns_2d():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    out = reduce_dimensions(X, method="tsne", metric="manhattan")
    assert out.shape == (40, 2)


def test_pca_still_works():
    rng = np.random.default_rng(0)
    assert reduce_dimensions(rng.normal(size=(30, 4)), method="pca").shape == (30, 2)


def test_tsne_small_n_does_not_crash_on_perplexity():
    # perplexity must be < n_samples; a tiny cluster set must not raise.
    rng = np.random.default_rng(0)
    out = reduce_dimensions(rng.normal(size=(6, 3)), method="tsne")
    assert out.shape == (6, 2)


def test_tsne_single_feature_does_not_crash_on_pca_init():
    # 1-feature data: default 'pca' init needs >= 2 features -> must fall back to random.
    rng = np.random.default_rng(0)
    out = reduce_dimensions(rng.normal(size=(40, 1)), method="tsne", metric="manhattan")
    assert out.shape == (40, 2)
