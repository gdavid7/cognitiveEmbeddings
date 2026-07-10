"""Tests for the pure-math metrics — the layer the whole evaluation rests on."""

import numpy as np
import pytest

from pipeline_b import metrics


rng = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# RSA                                                                          #
# --------------------------------------------------------------------------- #
def test_rsa_identical_is_one():
    x = rng.standard_normal((30, 8))
    assert metrics.rsa_from_embeddings(x, x.copy()) == pytest.approx(1.0)


def test_rsa_invariant_to_rotation():
    # Euclidean RDMs are invariant to an orthogonal column transform, so RSA=1.
    # (Correlation-distance is NOT — per-row centering doesn't commute with a
    # column rotation — so this uses the euclidean metric deliberately.)
    x = rng.standard_normal((40, 6))
    q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    rho = metrics.rsa(
        metrics.rdm(x, metric="euclidean"),
        metrics.rdm(x @ q, metric="euclidean"),
    )
    assert rho == pytest.approx(1.0, abs=1e-9)


def test_rsa_unrelated_near_zero():
    a = rng.standard_normal((60, 10))
    b = rng.standard_normal((60, 10))
    assert abs(metrics.rsa_from_embeddings(a, b)) < 0.3


def test_rdm_condensed_length():
    x = rng.standard_normal((12, 4))
    assert metrics.rdm(x).shape[0] == 12 * 11 // 2


# --------------------------------------------------------------------------- #
# CKA                                                                          #
# --------------------------------------------------------------------------- #
def test_cka_identical_is_one():
    x = rng.standard_normal((50, 12))
    assert metrics.linear_cka(x, x.copy()) == pytest.approx(1.0, abs=1e-9)


def test_cka_rotation_invariant():
    x = rng.standard_normal((50, 12))
    q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    assert metrics.linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-9)


def test_cka_isotropic_scale_invariant():
    x = rng.standard_normal((50, 12))
    assert metrics.linear_cka(x, 3.7 * x) == pytest.approx(1.0, abs=1e-9)


def test_cka_orthogonal_features_low():
    # Two independent random reps should have modest CKA.
    a = rng.standard_normal((200, 15))
    b = rng.standard_normal((200, 15))
    assert metrics.linear_cka(a, b) < 0.3


# --------------------------------------------------------------------------- #
# E0 linearity                                                                 #
# --------------------------------------------------------------------------- #
def test_linear_map_r2_recovers_affine():
    # "TRIBE adds a rotation" (H0): h_enc is an ORTHOGONAL affine map of the
    # backbone. Then R² ~ 1 (linearly reconstructable) AND CKA ~ 1 (CKA is
    # invariant to orthogonal maps). A *general* linear map would give high R²
    # but lower CKA — which is precisely why the doc checks both.
    x = rng.standard_normal((300, 20))
    q, _ = np.linalg.qr(rng.standard_normal((20, 20)))  # orthogonal
    y = x @ q + 5.0
    res = metrics.linear_map_r2(x, y)
    assert res["r2_uniform"] > 0.99
    assert res["cka"] > 0.99


def test_linear_map_r2_high_but_cka_moderate_for_general_map():
    # General (non-orthogonal) linear map: reconstructable (high R²) but CKA
    # need not be ~1. Documents the R²/CKA dissociation the doc relies on.
    x = rng.standard_normal((300, 20))
    w = rng.standard_normal((20, 16))
    res = metrics.linear_map_r2(x, x @ w + 5.0)
    assert res["r2_uniform"] > 0.99
    assert res["cka"] < 0.99


def test_linear_map_r2_low_for_nonlinear():
    x = rng.standard_normal((300, 8))
    y = np.sin(3 * x) + rng.standard_normal((300, 8)) * 0.1  # strongly nonlinear
    res = metrics.linear_map_r2(x, y)
    assert res["r2_uniform"] < 0.5


# --------------------------------------------------------------------------- #
# Probing                                                                      #
# --------------------------------------------------------------------------- #
def test_linear_probe_separable():
    # Two Gaussian blobs -> linearly separable -> near-perfect probe.
    n = 100
    a = rng.standard_normal((n, 5)) + np.array([4, 0, 0, 0, 0])
    b = rng.standard_normal((n, 5)) - np.array([4, 0, 0, 0, 0])
    X = np.vstack([a, b])
    y = np.array([0] * n + [1] * n)
    assert metrics.linear_probe(X, y)["accuracy"] > 0.95


def test_knn_separable():
    n = 80
    a = rng.standard_normal((n, 5)) + 5
    b = rng.standard_normal((n, 5)) - 5
    X = np.vstack([a, b])
    y = np.array([0] * n + [1] * n)
    assert metrics.knn_accuracy(X, y, k=3)["accuracy"] > 0.95


# --------------------------------------------------------------------------- #
# Noise ceiling                                                                #
# --------------------------------------------------------------------------- #
def test_noise_ceiling_perfect_repeats():
    signal = rng.standard_normal((25, 10))
    repeats = np.stack([signal, signal, signal])  # noiseless -> ceiling ~1
    nc = metrics.noise_ceiling(repeats)
    assert nc["split_half_mean"] == pytest.approx(1.0, abs=1e-6)
    assert nc["mean_vs_one_mean"] == pytest.approx(1.0, abs=1e-6)


def test_noise_ceiling_pure_noise_low():
    repeats = rng.standard_normal((4, 40, 10))  # independent -> ceiling ~0
    nc = metrics.noise_ceiling(repeats)
    assert abs(nc["mean_vs_one_mean"]) < 0.4


def test_normalized_predictivity():
    assert metrics.normalized_predictivity(0.2, 0.4) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Retrieval                                                                    #
# --------------------------------------------------------------------------- #
def test_topk_self_retrieval_excludes_self():
    x = rng.standard_normal((20, 6))
    # correct match is a near-duplicate of each row appended as the gallery tail
    idx = np.arange(20)
    res = metrics.topk_accuracy(x, x, idx, ks=(1,))
    # with self excluded and random data, top-1 self-hit should be near chance
    assert res["top1"] < 0.5


def test_topk_perfect_when_gallery_matches():
    q = rng.standard_normal((15, 6))
    gallery = np.vstack([rng.standard_normal((15, 6)), q])  # true match at 15..29
    correct = np.arange(15, 30)
    res = metrics.topk_accuracy(q, gallery, correct, ks=(1,))
    assert res["top1"] == pytest.approx(1.0)
