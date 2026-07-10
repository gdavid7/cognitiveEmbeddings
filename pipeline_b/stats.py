"""Statistics for RSA/CKA comparisons (§5.4).

The failure mode this module defends against: an RSA over *n* stimuli produces
*n(n-1)/2* correlated cells, so naive parametric significance is meaningless and
easy to manufacture. The doc mandates:

  * permutation tests over *stimulus labels* (not RDM cells) for RSA/CKA;
  * bootstrap over *stimuli* (not cells) for confidence intervals;
  * paired tests when comparing embeddings on the same stimuli;
  * FDR correction across the tap × experiment grid.

All randomness flows through an explicit ``numpy.random.Generator`` seed so
every reported number is reproducible.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .metrics import rsa, linear_cka

Array = np.ndarray


def _n_from_condensed(length: int) -> int:
    n = int((1 + np.sqrt(1 + 8 * length)) / 2)
    if n * (n - 1) // 2 != length:
        raise ValueError(f"{length} is not a valid condensed-RDM length")
    return n


# --------------------------------------------------------------------------- #
# Permutation test for RSA (§5.4)                                             #
# --------------------------------------------------------------------------- #
def permutation_test_rsa(
    rdm_a: Array,
    rdm_b: Array,
    n_permutations: int = 10000,
    alternative: str = "greater",
    seed: int = 0,
) -> dict:
    """Exact-ish permutation p-value for an observed RSA between two RDMs.

    The null: no stimulus-to-stimulus correspondence between the two systems.
    We realize it by permuting the stimulus labels of system A — i.e. shuffling
    rows *and* columns of A's square RDM together — and recomputing RSA. This
    respects the dependency structure of the RDM (whole stimuli move, not
    individual cells), which cell-shuffling would destroy.

    ``alternative``: 'greater' (default; RSA is directional), 'two-sided', or
    'less'. p uses the (1 + #null-as-extreme) / (1 + n_permutations)
    add-one convention so p is never exactly 0.
    """
    rdm_a = np.asarray(rdm_a, dtype=np.float64).ravel()
    rdm_b = np.asarray(rdm_b, dtype=np.float64).ravel()
    if rdm_a.shape != rdm_b.shape:
        raise ValueError("RDMs must be the same length")
    n = _n_from_condensed(rdm_a.size)

    observed = rsa(rdm_a, rdm_b)
    sq_a = squareform(rdm_a, checks=False)
    iu = np.triu_indices(n, k=1)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        permuted = sq_a[np.ix_(perm, perm)][iu]
        null[i] = rsa(permuted, rdm_b)

    p = _p_value(observed, null, alternative)
    return {
        "observed": observed,
        "p_value": p,
        "n_permutations": n_permutations,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
    }


def permutation_test_cka(
    a: Array, b: Array, n_permutations: int = 10000, alternative: str = "greater", seed: int = 0
) -> dict:
    """Permutation p-value for linear CKA between two embedding matrices.

    Same logic as RSA but on raw representations: permute the row order
    (stimulus labels) of A, recompute CKA.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError("row counts differ")
    n = a.shape[0]
    observed = linear_cka(a, b)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        null[i] = linear_cka(a[rng.permutation(n)], b)
    return {
        "observed": observed,
        "p_value": _p_value(observed, null, alternative),
        "n_permutations": n_permutations,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
    }


def _p_value(observed: float, null: Array, alternative: str) -> float:
    n = null.size
    if alternative == "greater":
        count = int(np.sum(null >= observed))
    elif alternative == "less":
        count = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        count = int(np.sum(np.abs(null - null.mean()) >= abs(observed - null.mean())))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return (1 + count) / (1 + n)


# --------------------------------------------------------------------------- #
# Bootstrap over stimuli (§5.4)                                               #
# --------------------------------------------------------------------------- #
def bootstrap_rsa_ci(
    emb_a: Array,
    emb_b: Array,
    n_boot: int = 2000,
    ci: float = 95.0,
    metric: str = "correlation",
    seed: int = 0,
) -> dict:
    """Bootstrap CI for RSA, resampling STIMULI (rows), not RDM cells.

    Each iteration draws ``n`` stimuli with replacement, subsets both embedding
    matrices, rebuilds both RDMs, and computes RSA. RDMs must be rebuilt on the
    resampled set (not sub-indexed from a precomputed RDM) because resampling
    changes the population of pairs. Requires embeddings, not RDMs.
    """
    emb_a = np.asarray(emb_a, dtype=np.float64)
    emb_b = np.asarray(emb_b, dtype=np.float64)
    n = emb_a.shape[0]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = rsa(pdist(emb_a[idx], metric=metric), pdist(emb_b[idx], metric=metric))
    lo, hi = np.nanpercentile(vals, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    point = rsa(pdist(emb_a, metric=metric), pdist(emb_b, metric=metric))
    return {
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "boot_mean": float(np.nanmean(vals)),
        "ci_level": ci,
    }


def paired_rsa_comparison(
    emb_a: Array,
    emb_b: Array,
    target: Array,
    n_boot: int = 2000,
    metric: str = "correlation",
    seed: int = 0,
) -> dict:
    """Paired test: is RSA(A, target) > RSA(B, target)? (e.g. h_enc vs backbone)

    Bootstraps over stimuli. Because A and B are compared on the *same*
    resampled stimuli each iteration, the difference is paired and the
    stimulus-sampling noise largely cancels. Returns the observed difference,
    its bootstrap CI, and a two-sided bootstrap p (fraction of resamples whose
    difference has the opposite sign of the observed, doubled).
    """
    emb_a = np.asarray(emb_a, dtype=np.float64)
    emb_b = np.asarray(emb_b, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = emb_a.shape[0]
    rng = np.random.default_rng(seed)

    def _diff(idx):
        rt = pdist(target[idx], metric=metric)
        return rsa(pdist(emb_a[idx], metric=metric), rt) - rsa(pdist(emb_b[idx], metric=metric), rt)

    observed = _diff(np.arange(n))
    diffs = np.array([_diff(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled difference flips sign
    if observed >= 0:
        p = 2 * (1 + np.sum(diffs <= 0)) / (1 + n_boot)
    else:
        p = 2 * (1 + np.sum(diffs >= 0)) / (1 + n_boot)
    return {
        "diff": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "a_better": bool(observed > 0),
    }


# --------------------------------------------------------------------------- #
# Multiple comparisons (§5.4)                                                 #
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR control across the tap × experiment grid.

    Returns per-hypothesis rejection flags and BH-adjusted q-values (monotone,
    clipped to 1), in the original input order.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    # adjusted q in ranked order, enforced monotone from the top
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q = np.empty(m, dtype=np.float64)
    q[order] = q_ranked
    return {"reject": q <= alpha, "qvalues": q}
