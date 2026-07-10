"""Tests for pooling, contamination guard, and the E0–E5 drivers."""

import numpy as np
import pytest

from pipeline_b import pooling, experiments, contamination


rng = np.random.default_rng(11)


# --------------------------------------------------------------------------- #
# Pooling                                                                      #
# --------------------------------------------------------------------------- #
def test_mean_pool_shape_and_value():
    segs = [rng.standard_normal((t, 5)) for t in (3, 7, 2)]
    pooled = pooling.mean_pool(segs)
    assert pooled.shape == (3, 5)
    assert pooled[0] == pytest.approx(segs[0].mean(0))


def test_mean_std_pool_doubles_dim():
    segs = [rng.standard_normal((4, 5)) for _ in range(6)]
    assert pooling.mean_std_pool(segs).shape == (6, 10)


def test_zscore_unit_variance():
    x = rng.standard_normal((100, 8)) * 3 + 10
    z = pooling.zscore(x)
    assert z.mean(0) == pytest.approx(np.zeros(8), abs=1e-9)
    assert z.std(0) == pytest.approx(np.ones(8), abs=1e-9)


def test_zscore_handles_dead_dim():
    x = rng.standard_normal((20, 4))
    x[:, 2] = 5.0  # zero variance
    z = pooling.zscore(x)
    assert np.isfinite(z).all()
    assert z[:, 2] == pytest.approx(np.zeros(20))


def test_apply_zscore_reuses_train_stats():
    train = rng.standard_normal((50, 4))
    z, stats_ = pooling.zscore(train, return_stats=True)
    test = rng.standard_normal((10, 4))
    zt = pooling.apply_zscore(test, stats_)
    assert zt.shape == (10, 4)


# --------------------------------------------------------------------------- #
# Contamination guard (§5.1)                                                   #
# --------------------------------------------------------------------------- #
def test_known_studies_blocked():
    for name in ["Algonauts2025", "wen2017", "Lahner2024", "Lebel2023"]:
        assert contamination.is_known_contaminated(name)
        with pytest.raises(contamination.ContaminationError):
            contamination.assert_clean(name, audited_ok=True)


def test_unknown_requires_audit():
    with pytest.raises(contamination.ContaminationError):
        contamination.assert_clean("SomeNovelVideoSet")  # no audited_ok
    contamination.assert_clean("SomeNovelVideoSet", audited_ok=True)  # ok


# --------------------------------------------------------------------------- #
# E0 driver                                                                    #
# --------------------------------------------------------------------------- #
def test_e0_confirms_h0_on_affine():
    # H0-confirmed regime = TRIBE adds a rotation: orthogonal affine map.
    x = rng.standard_normal((300, 20))
    q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    h_enc = x @ q + 2.0
    res = experiments.e0_linearity(x, h_enc)
    assert res["verdict"] == "H0_CONFIRMED"


def test_e0_proceeds_on_nonlinear():
    x = rng.standard_normal((300, 10))
    h_enc = np.tanh(2 * x @ rng.standard_normal((10, 12)))
    res = experiments.e0_linearity(x, h_enc)
    assert res["verdict"] in ("PROCEED", "LARGE_TRANSFORM")


# --------------------------------------------------------------------------- #
# E1 driver                                                                    #
# --------------------------------------------------------------------------- #
def test_e1_candidate_beats_control():
    from pipeline_b import metrics

    target = rng.standard_normal((40, 6))
    h_enc = target + 0.05 * rng.standard_normal((40, 6))       # close to human
    concat = rng.standard_normal((40, 6))                      # unrelated
    human_rdm = metrics.rdm(target)
    res = experiments.e1_behavioral_rsa(
        {"h_enc": h_enc, "concat_backbone": concat}, human_rdm,
        n_permutations=500, n_boot=300, seed=1,
    )
    assert res["per_embedding"]["h_enc"]["rsa"] > res["per_embedding"]["concat_backbone"]["rsa"]
    assert res["candidate_beats_control"]["a_better"] is True


# --------------------------------------------------------------------------- #
# E3 dissociation                                                              #
# --------------------------------------------------------------------------- #
def test_e3_detects_brain_filter_signature():
    n = 120
    coarse = rng.integers(0, 2, n)          # coarse: 2 classes
    fine = rng.integers(0, 8, n)            # fine: 8 classes
    # h_enc encodes coarse cleanly but scrambles fine; clip encodes both.
    h_enc = np.column_stack([coarse * 6.0, rng.standard_normal((n, 4))])
    clip = np.column_stack([coarse * 6.0, fine * 6.0, rng.standard_normal((n, 2))])
    res = experiments.e3_semantic_probing(
        {"h_enc": h_enc, "clip": clip}, coarse, fine
    )
    sig = res["brain_filter_signature"]
    assert sig["coarse_win_vs_clip"] is True
    assert sig["fine_loss_vs_clip"] is True
    assert sig["signature_present"] is True


# --------------------------------------------------------------------------- #
# E4 ladder + E5 retrieval + FDR                                              #
# --------------------------------------------------------------------------- #
def test_e4_attributes_gains():
    embs = {n: rng.standard_normal((30, 5)) for n in experiments.LADDER}
    res = experiments.e4_ablation(embs, scorer=lambda e: float(e.mean()))
    assert set(res["scores"]) == set(experiments.LADDER)
    assert "temporal_cross_modal_attention" in res["attributed_gains"]


def test_e5_retrieval_runs():
    embs = {"h_enc": rng.standard_normal((20, 6))}
    res = experiments.e5_retrieval(embs, {}, np.arange(20), ks=(1, 5))
    assert "top1" in res["per_embedding"]["h_enc"]


def test_apply_fdr_grid():
    res = experiments.apply_fdr({"E1/h_enc": 0.001, "E1/backbone": 0.4, "E2/h_lr": 0.03})
    assert res["E1/h_enc"]["reject"]
    assert all("q" in v for v in res.values())
