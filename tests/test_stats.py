"""Tests for the statistics layer (§5.4)."""

import numpy as np
import pytest

from pipeline_b import stats, metrics


rng = np.random.default_rng(7)


def test_permutation_rsa_significant_when_aligned():
    # Two noisy views of a shared structure -> real RSA -> small p.
    base = rng.standard_normal((40, 6))
    a = base + 0.1 * rng.standard_normal((40, 6))
    b = base + 0.1 * rng.standard_normal((40, 6))
    res = stats.permutation_test_rsa(metrics.rdm(a), metrics.rdm(b), n_permutations=2000, seed=1)
    assert res["observed"] > 0.5
    assert res["p_value"] < 0.01


def test_permutation_rsa_null_when_unrelated():
    a = rng.standard_normal((40, 6))
    b = rng.standard_normal((40, 6))
    res = stats.permutation_test_rsa(metrics.rdm(a), metrics.rdm(b), n_permutations=2000, seed=2)
    assert res["p_value"] > 0.05


def test_permutation_p_never_zero():
    x = rng.standard_normal((20, 4))
    res = stats.permutation_test_rsa(metrics.rdm(x), metrics.rdm(x), n_permutations=500, seed=3)
    assert res["p_value"] == pytest.approx(1 / 501)  # add-one convention


def test_bootstrap_ci_brackets_point():
    base = rng.standard_normal((50, 6))
    a = base + 0.2 * rng.standard_normal((50, 6))
    b = base + 0.2 * rng.standard_normal((50, 6))
    res = stats.bootstrap_rsa_ci(a, b, n_boot=500, seed=4)
    assert res["ci_low"] <= res["point"] <= res["ci_high"]
    assert res["ci_low"] < res["ci_high"]


def test_paired_comparison_detects_better_embedding():
    # a is a clean view of target; b is noise -> a should win the paired test.
    target = rng.standard_normal((60, 6))
    a = target + 0.05 * rng.standard_normal((60, 6))
    b = rng.standard_normal((60, 6))
    res = stats.paired_rsa_comparison(a, b, target, n_boot=500, seed=5)
    assert res["a_better"] is True
    assert res["diff"] > 0
    assert res["p_value"] < 0.05


def test_benjamini_hochberg_basic():
    # one clearly significant, rest null
    p = [0.001, 0.2, 0.4, 0.6, 0.8]
    res = stats.benjamini_hochberg(p, alpha=0.05)
    assert res["reject"][0]
    assert not res["reject"][1:].any()
    assert (res["qvalues"] >= np.array(p)).all()  # q >= p always


def test_benjamini_hochberg_all_null():
    p = [0.5, 0.6, 0.7, 0.9]
    res = stats.benjamini_hochberg(p, alpha=0.05)
    assert not res["reject"].any()


def test_permutation_cka_significant_when_aligned():
    base = rng.standard_normal((50, 8))
    a = base + 0.1 * rng.standard_normal((50, 8))
    res = stats.permutation_test_cka(a, base, n_permutations=1000, seed=6)
    assert res["observed"] > 0.7
    assert res["p_value"] < 0.01
