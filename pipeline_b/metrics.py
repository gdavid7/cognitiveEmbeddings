"""Representation-comparison metrics for Pipeline B.

Everything here operates on plain 2-D ``numpy`` arrays of shape
``(n_stimuli, dim)`` — one row per stimulus. No torch, no model. This is the
layer the whole evaluation (E0–E5) is built on, so it is the layer that is
unit-tested.

Design-doc anchors:
  §2.2 RSA          -> ``rdm`` + ``rsa``
  §2.3 CKA          -> ``linear_cka``
  §2.4 linear probe -> ``linear_probe``, ``knn_accuracy``
  §2.5 noise ceiling-> ``noise_ceiling``, ``normalized_predictivity``
  E0   linearity    -> ``linear_map_r2``
  E5   retrieval    -> ``topk_accuracy``
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

Array = np.ndarray


def _as_2d(x: Array) -> Array:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D (n_stimuli, dim) array, got shape {x.shape}")
    return x


# --------------------------------------------------------------------------- #
# RSA (§2.2)                                                                   #
# --------------------------------------------------------------------------- #
def rdm(x: Array, metric: str = "correlation") -> Array:
    """Representational Dissimilarity Matrix, returned in *condensed* form.

    Returns the upper triangle (``scipy`` pdist ordering), length
    ``n(n-1)/2``. Condensed form is what ``rsa`` consumes and what the
    bootstrap/permutation machinery in ``stats`` expects, so we never
    materialize the square matrix unless asked.

    ``metric='correlation'`` -> dissimilarity = 1 - Pearson r between stimuli,
    the standard choice for RSA. Any ``scipy.spatial.distance.pdist`` metric
    works (``'cosine'``, ``'euclidean'``, ...).
    """
    x = _as_2d(x)
    if x.shape[0] < 2:
        raise ValueError("need at least 2 stimuli to build an RDM")
    return pdist(x, metric=metric)


def square_rdm(condensed: Array) -> Array:
    """Condensed RDM -> square symmetric matrix with zero diagonal."""
    from scipy.spatial.distance import squareform

    return squareform(condensed, checks=False)


def rsa(rdm_a: Array, rdm_b: Array) -> float:
    """Spearman correlation between two condensed RDMs (§2.2).

    A single number: how similarly the two systems rank stimulus pairs by
    dissimilarity. Higher = the systems organize the world the same way.
    """
    rdm_a = np.asarray(rdm_a, dtype=np.float64).ravel()
    rdm_b = np.asarray(rdm_b, dtype=np.float64).ravel()
    if rdm_a.shape != rdm_b.shape:
        raise ValueError(f"RDM shapes differ: {rdm_a.shape} vs {rdm_b.shape}")
    rho, _ = spearmanr(rdm_a, rdm_b)
    return float(rho)


def rsa_from_embeddings(a: Array, b: Array, metric: str = "correlation") -> float:
    """Convenience: build both RDMs from embedding matrices, then RSA."""
    return rsa(rdm(a, metric=metric), rdm(b, metric=metric))


# --------------------------------------------------------------------------- #
# CKA (§2.3)                                                                   #
# --------------------------------------------------------------------------- #
def _center_columns(x: Array) -> Array:
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(a: Array, b: Array) -> float:
    """Linear Centered Kernel Alignment between two representations (§2.3).

    Invariant to rotation and isotropic scaling but NOT to arbitrary invertible
    linear maps. So ``linear_cka(concat_backbone, h_enc) ~ 1`` is strong
    evidence for H0 (TRIBE only rotates/reweights its inputs).

    Uses the feature-space (Gram-of-features) form, which is exact and cheaper
    than the n×n kernel form when dim < n:
        CKA = ||A^T B||_F^2 / (||A^T A||_F * ||B^T B||_F)
    on column-centered A, B.
    """
    a = _center_columns(_as_2d(a))
    b = _center_columns(_as_2d(b))
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"row counts differ: {a.shape[0]} vs {b.shape[0]}")
    # ||A^T B||_F^2 == sum of squared cross-covariances
    cross = a.T @ b
    hsic_ab = float(np.sum(cross ** 2))
    hsic_aa = float(np.linalg.norm(a.T @ a, ord="fro"))
    hsic_bb = float(np.linalg.norm(b.T @ b, ord="fro"))
    denom = hsic_aa * hsic_bb
    if denom == 0.0:
        return 0.0
    return hsic_ab / denom


# --------------------------------------------------------------------------- #
# E0 — the linearity test (§5.3, kill H0)                                      #
# --------------------------------------------------------------------------- #
def linear_map_r2(
    source: Array,
    target: Array,
    n_splits: int = 5,
    alpha: float = 1.0,
    random_state: int = 0,
) -> dict:
    """Cross-validated R² of the best linear map ``source -> target`` (E0).

    Fits ridge regression on train folds and scores held-out folds. This is the
    quantitative form of H0: if a linear map reconstructs ``h_enc`` from
    ``concat_backbone`` at high R², TRIBE added (approximately) only an affine
    transform.

    Returns per-fold and mean R², under two averaging conventions:
      * ``uniform``  — plain mean over output dims (what the doc's thresholds
        implicitly mean; sensitive to dead dims).
      * ``variance_weighted`` — weights dims by their variance (robust).
    Report ``uniform``; keep ``variance_weighted`` as a sanity check.
    """
    source = _as_2d(source)
    target = _as_2d(target)
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same number of rows")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_uniform, r2_varw = [], []
    for train_idx, test_idx in kf.split(source):
        xs = StandardScaler().fit(source[train_idx])
        xtr, xte = xs.transform(source[train_idx]), xs.transform(source[test_idx])
        model = Ridge(alpha=alpha)
        model.fit(xtr, target[train_idx])
        pred = model.predict(xte)
        r2_uniform.append(r2_score(target[test_idx], pred, multioutput="uniform_average"))
        r2_varw.append(r2_score(target[test_idx], pred, multioutput="variance_weighted"))
    return {
        "r2_uniform": float(np.mean(r2_uniform)),
        "r2_variance_weighted": float(np.mean(r2_varw)),
        "r2_uniform_folds": [float(v) for v in r2_uniform],
        "cka": linear_cka(source, target),
    }


# --------------------------------------------------------------------------- #
# Linear probing / k-NN (§2.4, E3)                                            #
# --------------------------------------------------------------------------- #
def linear_probe(
    embeddings: Array,
    labels: Array,
    n_splits: int = 5,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 0,
) -> dict:
    """Cross-validated accuracy of a linear classifier on frozen embeddings.

    Linearity is the point (§2.4): it asks whether the label is *explicitly*
    represented — laid out along directions — not merely recoverable in
    principle. Embeddings are standardized inside each fold.
    """
    embeddings = _as_2d(embeddings)
    labels = np.asarray(labels)
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels and embeddings row counts differ")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for tr, te in kf.split(embeddings):
        scaler = StandardScaler().fit(embeddings[tr])
        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(scaler.transform(embeddings[tr]), labels[tr])
        accs.append(clf.score(scaler.transform(embeddings[te]), labels[te]))
    return {"accuracy": float(np.mean(accs)), "folds": [float(a) for a in accs]}


def knn_accuracy(
    embeddings: Array,
    labels: Array,
    k: int = 5,
    n_splits: int = 5,
    metric: str = "cosine",
    random_state: int = 0,
) -> dict:
    """Cross-validated k-NN accuracy — a non-parametric read on local structure."""
    embeddings = _as_2d(embeddings)
    labels = np.asarray(labels)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for tr, te in kf.split(embeddings):
        scaler = StandardScaler().fit(embeddings[tr])
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
        clf.fit(scaler.transform(embeddings[tr]), labels[tr])
        accs.append(clf.score(scaler.transform(embeddings[te]), labels[te]))
    return {"accuracy": float(np.mean(accs)), "folds": [float(a) for a in accs]}


# --------------------------------------------------------------------------- #
# Brain predictivity + noise ceiling (§2.5, E2)                               #
# --------------------------------------------------------------------------- #
def ridge_predictivity(
    embeddings: Array,
    fmri: Array,
    n_splits: int = 5,
    alpha: float = 1000.0,
    random_state: int = 0,
) -> dict:
    """Cross-validated Pearson r from ``embeddings -> fmri`` (E2, raw).

    Returns the mean-over-vertices Pearson r on held-out folds. This is the
    RAW number; it MUST be divided by the noise ceiling before interpretation
    (§2.5) — see ``normalized_predictivity``.
    """
    embeddings = _as_2d(embeddings)
    fmri = _as_2d(fmri)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_r = []
    for tr, te in kf.split(embeddings):
        scaler = StandardScaler().fit(embeddings[tr])
        model = Ridge(alpha=alpha)
        model.fit(scaler.transform(embeddings[tr]), fmri[tr])
        pred = model.predict(scaler.transform(embeddings[te]))
        fold_r.append(_columnwise_pearson(fmri[te], pred).mean())
    return {"pearson_r": float(np.mean(fold_r)), "folds": [float(r) for r in fold_r]}


def _columnwise_pearson(y_true: Array, y_pred: Array) -> Array:
    """Per-column (per-vertex) Pearson r, safe against zero-variance columns."""
    yt = y_true - y_true.mean(0, keepdims=True)
    yp = y_pred - y_pred.mean(0, keepdims=True)
    num = (yt * yp).sum(0)
    den = np.sqrt((yt ** 2).sum(0) * (yp ** 2).sum(0))
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def noise_ceiling(repeats: Array) -> dict:
    """Estimate the noise ceiling from repeated measurements (§2.5).

    ``repeats`` has shape ``(n_repeats, n_stimuli, n_vertices)`` — the SAME
    stimuli measured multiple times. The ceiling is the correlation the data
    has with itself; no model can beat it.

    Two estimators, both reported:
      * ``split_half``  — mean pairwise Pearson r between single repeats,
        Spearman-Brown corrected to the full number of repeats.
      * ``mean_vs_one`` — r between each single repeat and the mean of the
        others (leave-one-out); a common encoding-model convention.
    """
    repeats = np.asarray(repeats, dtype=np.float64)
    if repeats.ndim != 3:
        raise ValueError("repeats must be (n_repeats, n_stimuli, n_vertices)")
    n_rep = repeats.shape[0]
    if n_rep < 2:
        raise ValueError("need >= 2 repeats to estimate a noise ceiling")

    # split-half: average r over all repeat pairs, per vertex, then Spearman-Brown
    pair_r = []
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            pair_r.append(_columnwise_pearson(repeats[i], repeats[j]))
    r1 = np.mean(pair_r, axis=0)  # per-vertex single-repeat reliability
    sb = (n_rep * r1) / (1.0 + (n_rep - 1.0) * r1)  # Spearman-Brown to n repeats

    # leave-one-out: each repeat vs mean of the rest
    loo_r = []
    for i in range(n_rep):
        others = np.delete(repeats, i, axis=0).mean(axis=0)
        loo_r.append(_columnwise_pearson(repeats[i], others))
    mean_vs_one = np.mean(loo_r, axis=0)

    return {
        "split_half": sb,               # per-vertex, corrected to n repeats
        "mean_vs_one": mean_vs_one,     # per-vertex
        "split_half_mean": float(np.nanmean(sb)),
        "mean_vs_one_mean": float(np.nanmean(mean_vs_one)),
    }


def normalized_predictivity(pearson_r, ceiling, eps: float = 1e-8):
    """Predictivity as a fraction of the noise ceiling (§2.5).

    Accepts scalars or per-vertex arrays. Values > 1 (model appears to beat the
    ceiling, from estimator noise) are clipped to 1 for the summary but the raw
    ratio is returned for auditing.
    """
    r = np.asarray(pearson_r, dtype=np.float64)
    c = np.asarray(ceiling, dtype=np.float64)
    ratio = r / np.maximum(c, eps)
    return ratio


# --------------------------------------------------------------------------- #
# Retrieval (§5.3, E5)                                                         #
# --------------------------------------------------------------------------- #
def topk_accuracy(query: Array, gallery: Array, correct_idx: Array, ks=(1, 5)) -> dict:
    """Top-k retrieval accuracy (E5).

    For each ``query`` row, rank ``gallery`` rows by cosine similarity; a hit is
    when ``correct_idx[i]`` appears in the top-k. If ``query is gallery`` (same
    array), self-matches are excluded.
    """
    query = _as_2d(query)
    gallery = _as_2d(gallery)
    correct_idx = np.asarray(correct_idx)

    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    gn = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    sim = qn @ gn.T  # (n_query, n_gallery)

    same = query.shape == gallery.shape and np.shares_memory(query, gallery)
    if same:
        np.fill_diagonal(sim, -np.inf)

    order = np.argsort(-sim, axis=1)  # descending
    out = {}
    for k in ks:
        topk = order[:, :k]
        hits = (topk == correct_idx[:, None]).any(axis=1)
        out[f"top{k}"] = float(hits.mean())
    return out
