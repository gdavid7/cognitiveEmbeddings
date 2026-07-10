"""Pool per-TR embeddings to per-clip vectors, and standardize (§4.4).

TRIBE taps produce one vector per TR (1 Hz). Clip-level similarity needs one
vector per clip. The doc prescribes:
  * primary: mean-pool over time;
  * secondary: [mean ‖ std] to test whether temporal variability carries
    signal that mean-pooling destroys — cheap, report both;
  * z-score per dimension across the corpus before ANY distance, so a handful
    of high-variance dims don't dominate every cosine.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def mean_pool(segments) -> Array:
    """List of ``(T_i, D)`` per-clip TR sequences -> ``(n_clips, D)`` means."""
    return np.stack([np.asarray(s, dtype=np.float64).mean(axis=0) for s in segments])


def mean_std_pool(segments) -> Array:
    """``[mean ‖ std]`` per clip -> ``(n_clips, 2D)`` (§4.4 secondary).

    ``std`` uses ddof=0 and is 0 for single-TR clips, which is correct: no
    temporal variability to report.
    """
    rows = []
    for s in segments:
        s = np.asarray(s, dtype=np.float64)
        rows.append(np.concatenate([s.mean(axis=0), s.std(axis=0)]))
    return np.stack(rows)


def zscore(x: Array, eps: float = 1e-8, return_stats: bool = False):
    """Per-dimension z-score across the corpus (§4.4).

    Must be applied before distance computation. Zero-variance dims are left at
    zero rather than amplified. If ``return_stats``, also returns ``(mean, std)``
    so the same transform can be reapplied to a held-out set.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    safe = np.where(std < eps, 1.0, std)
    z = (x - mean) / safe
    if return_stats:
        return z, (mean, std)
    return z


def apply_zscore(x: Array, stats, eps: float = 1e-8) -> Array:
    """Reapply a stored ``(mean, std)`` z-score to new rows (e.g. test split)."""
    mean, std = stats
    safe = np.where(std < eps, 1.0, std)
    return (np.asarray(x, dtype=np.float64) - mean) / safe
