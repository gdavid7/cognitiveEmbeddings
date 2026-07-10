"""Experiments E0–E5 (§5.3), operating on already-extracted embedding matrices.

Pure-math: every function takes ``(n_stimuli, dim)`` arrays and returns a result
dict that bundles the numbers WITH the doc's interpretation, so a caller can't
read a bare ρ and forget what it means. The model/extraction layer produces the
matrices; this layer decides what they say.

Preregister E0–E3 predictions before viewing any results (§5.4). See
``prereg_template.md``.
"""

from __future__ import annotations

import numpy as np

from . import metrics, stats, pooling

Array = np.ndarray


# --------------------------------------------------------------------------- #
# E0 — kill H0 (the linearity test). RUN FIRST. (§5.3)                         #
# --------------------------------------------------------------------------- #
def e0_linearity(concat_backbone: Array, h_enc: Array, n_splits: int = 5) -> dict:
    """Fit ``concat_backbone -> h_enc`` linearly; report held-out R² and CKA.

    Verdict thresholds are the doc's (§5.3 table):
      R²>0.95 & CKA>0.95 -> H0 confirmed: TRIBE only rotates. Pipeline B is a
                            negative result — write it up and stop.
      R²≈0.6–0.9         -> real nonlinear/temporal structure. Proceed.
      R²<0.5             -> large transform; proceed but re-check extraction.
    """
    res = metrics.linear_map_r2(concat_backbone, h_enc, n_splits=n_splits)
    r2, cka = res["r2_uniform"], res["cka"]
    if r2 > 0.95 and cka > 0.95:
        verdict = "H0_CONFIRMED"
        meaning = ("TRIBE adds ~a rotation. Pipeline B is dead; write up as a "
                   "clean negative result and stop (§6).")
    elif r2 < 0.5:
        verdict = "LARGE_TRANSFORM"
        meaning = ("Large transformation. Proceed, but verify extraction isn't "
                   "broken before trusting it (§5.3).")
    else:
        verdict = "PROCEED"
        meaning = "TRIBE adds real nonlinear/temporal structure. Proceed."
    return {"experiment": "E0", "r2": r2,
            "r2_variance_weighted": res["r2_variance_weighted"],
            "cka": cka, "verdict": verdict, "meaning": meaning, "detail": res}


# --------------------------------------------------------------------------- #
# E1 — behavioral alignment (the payoff). (§5.3)                              #
# --------------------------------------------------------------------------- #
def e1_behavioral_rsa(
    embeddings: dict,
    human_rdm_condensed: Array,
    candidate: str = "h_enc",
    control: str = "concat_backbone",
    metric: str = "correlation",
    n_permutations: int = 10000,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """RSA of each embedding vs the human-judgment RDM; test candidate>control.

    Prediction: ``h_enc`` > ``concat_backbone`` — brain supervision should keep
    the categorical structure humans use and discard perceptual detail they
    ignore. ``human_rdm_condensed`` is the behavioral RDM (e.g. from THINGS
    odd-one-out), in condensed (upper-triangle) form, aligned to the stimulus
    order of the embeddings.
    """
    per_embed = {}
    for name, emb in embeddings.items():
        rdm = metrics.rdm(emb, metric=metric)
        perm = stats.permutation_test_rsa(
            rdm, human_rdm_condensed, n_permutations=n_permutations, seed=seed
        )
        per_embed[name] = {"rsa": perm["observed"], "p_value": perm["p_value"],
                           "null_mean": perm["null_mean"]}

    paired = None
    if candidate in embeddings and control in embeddings:
        paired = stats.paired_rsa_comparison(
            embeddings[candidate], embeddings[control],
            _square_target(human_rdm_condensed), n_boot=n_boot, metric=metric, seed=seed,
        )
    return {"experiment": "E1", "per_embedding": per_embed,
            "candidate": candidate, "control": control,
            "candidate_beats_control": paired}


def _square_target(human_rdm_condensed: Array) -> Array:
    """paired_rsa_comparison needs an embedding-like target, but the human RDM
    is given condensed. Recover a metric embedding whose pairwise correlation-
    distance reproduces the RDM via classical MDS, so the paired bootstrap can
    resample stimuli consistently."""
    from scipy.spatial.distance import squareform

    d = squareform(np.asarray(human_rdm_condensed, dtype=np.float64), checks=False)
    n = d.shape[0]
    d2 = d ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    vals = np.clip(vals[::-1], 0, None)
    vecs = vecs[:, ::-1]
    return vecs * np.sqrt(vals)


# --------------------------------------------------------------------------- #
# E2 — brain predictivity (a DIAGNOSTIC, not a result). (§5.3)                #
# --------------------------------------------------------------------------- #
def e2_brain_predictivity(
    embeddings: dict, fmri: Array, noise_ceiling_vec: Array, n_splits: int = 5
) -> dict:
    """Ridge ``embedding -> fMRI``, Pearson r normalized by the noise ceiling.

    Prediction: ``h_lr`` ≥ ``h_enc`` > ``concat_backbone``. But TRIBE was
    TRAINED for this, so a win proves little — its value is diagnostic: if
    ``h_enc`` does NOT beat the backbones here, extraction is broken (§5.3).
    Every number is reported as a fraction of the noise ceiling (§2.5).
    """
    ceiling_mean = float(np.nanmean(noise_ceiling_vec))
    per_embed = {}
    for name, emb in embeddings.items():
        pred = metrics.ridge_predictivity(emb, fmri, n_splits=n_splits)
        per_embed[name] = {
            "pearson_r": pred["pearson_r"],
            "normalized": pred["pearson_r"] / max(ceiling_mean, 1e-8),
        }
    enc = per_embed.get("h_enc", {}).get("pearson_r")
    bb = per_embed.get("concat_backbone", {}).get("pearson_r")
    diagnostic_ok = (enc is None or bb is None) or (enc > bb)
    return {"experiment": "E2", "per_embedding": per_embed,
            "noise_ceiling_mean": ceiling_mean,
            "extraction_sane": bool(diagnostic_ok),
            "note": ("If h_enc does not beat concat_backbone here, extraction "
                     "is likely broken — treat E2 as a sanity check, not a win.")}


# --------------------------------------------------------------------------- #
# E3 — semantic probing ("what got destroyed?"). (§5.3)                        #
# --------------------------------------------------------------------------- #
def e3_semantic_probing(
    embeddings: dict, coarse_labels: Array, fine_labels: Array,
    clip_name: str = "clip", candidate: str = "h_enc", n_splits: int = 5,
) -> dict:
    """Linear probe + k-NN for coarse and fine categories; look for the
    coarse-win / fine-loss dissociation vs CLIP.

    Prediction (the interesting one): ``h_enc`` UNDERperforms CLIP on fine
    categories while MATCHING/BEATING it on coarse ones. That dissociation is
    the signature of a genuinely brain-shaped representation — a lossy filter
    that keeps what ventral temporal cortex distinguishes and drops the rest.
    """
    per_embed = {}
    for name, emb in embeddings.items():
        per_embed[name] = {
            "coarse_linear": metrics.linear_probe(emb, coarse_labels, n_splits=n_splits)["accuracy"],
            "coarse_knn": metrics.knn_accuracy(emb, coarse_labels, n_splits=n_splits)["accuracy"],
            "fine_linear": metrics.linear_probe(emb, fine_labels, n_splits=n_splits)["accuracy"],
            "fine_knn": metrics.knn_accuracy(emb, fine_labels, n_splits=n_splits)["accuracy"],
        }
    dissociation = None
    if candidate in per_embed and clip_name in per_embed:
        c, cl = per_embed[candidate], per_embed[clip_name]
        coarse_win = c["coarse_linear"] >= cl["coarse_linear"]
        fine_loss = c["fine_linear"] < cl["fine_linear"]
        dissociation = {
            "coarse_win_vs_clip": bool(coarse_win),
            "fine_loss_vs_clip": bool(fine_loss),
            "signature_present": bool(coarse_win and fine_loss),
            "coarse_gap": c["coarse_linear"] - cl["coarse_linear"],
            "fine_gap": c["fine_linear"] - cl["fine_linear"],
        }
    return {"experiment": "E3", "per_embedding": per_embed,
            "brain_filter_signature": dissociation}


# --------------------------------------------------------------------------- #
# E4 — stage-wise ablation ladder. (§5.3)                                      #
# --------------------------------------------------------------------------- #
LADDER = ["h_raw", "concat_backbone", "h_proj", "h_enc", "h_lr"]


def e4_ablation(embeddings: dict, scorer, ladder=LADDER) -> dict:
    """Run one scoring function across the full ladder and attribute each gain.

    ``scorer`` maps an embedding array -> float (e.g. behavioral RSA, or a probe
    accuracy). Attributes deltas to the stage that produced them:
      concat_backbone - h_raw          : concatenation
      h_proj          - concat_backbone: per-modality projection
      h_enc           - h_proj         : temporal / cross-modal attention
      h_lr            - h_enc          : brain-objective compression
    """
    scores = {name: float(scorer(embeddings[name])) for name in ladder if name in embeddings}
    present = [n for n in ladder if n in scores]
    deltas = {}
    labels = {
        ("concat_backbone", "h_raw"): "concatenation",
        ("h_proj", "concat_backbone"): "per_modality_projection",
        ("h_enc", "h_proj"): "temporal_cross_modal_attention",
        ("h_lr", "h_enc"): "brain_objective_compression",
    }
    for i in range(1, len(present)):
        lo, hi = present[i - 1], present[i]
        key = labels.get((hi, lo), f"{lo}->{hi}")
        deltas[key] = scores[hi] - scores[lo]
    return {"experiment": "E4", "scores": scores, "attributed_gains": deltas}


# --------------------------------------------------------------------------- #
# E5 — retrieval (smoke test). (§5.3)                                          #
# --------------------------------------------------------------------------- #
def e5_retrieval(embeddings: dict, query: dict, correct_idx: Array, ks=(1, 5)) -> dict:
    """Top-k retrieval per embedding (§5.3). The model's home turf; a cheap
    smoke test that the embeddings carry stimulus identity at all."""
    out = {}
    for name, gallery in embeddings.items():
        q = query.get(name, gallery)
        out[name] = metrics.topk_accuracy(q, gallery, correct_idx, ks=ks)
    return {"experiment": "E5", "per_embedding": out}


# --------------------------------------------------------------------------- #
# Grid-wide FDR (§5.4)                                                         #
# --------------------------------------------------------------------------- #
def apply_fdr(pvalue_dict: dict, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg across every p-value in the tap × experiment grid.

    ``pvalue_dict`` maps a label -> p. Returns per-label q-values and reject
    flags, so significance survives multiple comparisons (§5.4).
    """
    labels = list(pvalue_dict.keys())
    pvals = [pvalue_dict[k] for k in labels]
    res = stats.benjamini_hochberg(pvals, alpha=alpha)
    return {lab: {"p": pvals[i], "q": float(res["qvalues"][i]),
                  "reject": bool(res["reject"][i])} for i, lab in enumerate(labels)}
