"""Baseline / control embeddings to extract alongside the TRIBE taps (§4.3).

MODEL-DEPENDENT for ``concat_backbone`` (needs the batch's raw backbone
features) and ``clip`` (needs a CLIP/SigLIP model). ``random_proj`` and
``shuffled_tribe`` are pure numpy and importable anywhere.

| name            | what it is                                    | why it's here                    |
|-----------------|-----------------------------------------------|----------------------------------|
| concat_backbone | TRIBE's own input features, pooled to TR grid | THE critical control (H0)        |
| clip            | CLIP/SigLIP on sampled frames, mean-pooled    | standard semantic reference      |
| random_proj     | random Gaussian projection of concat_backbone | dimensionality control           |
| shuffled_tribe  | h_enc with row order permuted                 | RSA/CKA null distribution        |
"""

from __future__ import annotations

import numpy as np

from .extraction import to_tr_grid, _segment_keep_mask


def random_projection(concat_backbone: np.ndarray, out_dim: int = 1152, seed: int = 0) -> np.ndarray:
    """Random Gaussian projection to ``out_dim`` (§4.3).

    Controls for dimensionality: any metric on which this scores well is a
    metric that measures nothing. Columns scaled by 1/sqrt(out_dim) (JL).
    """
    x = np.asarray(concat_backbone, dtype=np.float64)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((x.shape[1], out_dim)) / np.sqrt(out_dim)
    return x @ proj


def shuffled(embedding: np.ndarray, seed: int = 0) -> np.ndarray:
    """Row-permuted copy of an embedding — the RSA/CKA null (§4.3)."""
    x = np.asarray(embedding)
    rng = np.random.default_rng(seed)
    return x[rng.permutation(x.shape[0])].copy()


def extract_concat_backbone(xp, events):
    """TRIBE's INPUT features, pooled to the TR grid and masked like predict().

    This is the H0 control (§4.3): TRIBE's raw multimodal input, concatenated
    across modalities. We tap ``model.aggregate_features`` — the same tensor
    the encoder receives, before any transformer fusion — so it is exactly
    comparable, row-for-row, with ``h_proj``/``h_enc``.
    """
    import torch
    from einops import rearrange

    model = xp._model
    n_out_t = model.n_output_timesteps
    loader = xp.data.get_loaders(events=events, split_to_build="all")["all"]

    captured = {}

    def hook(module, inputs, output):
        captured["x"] = output.detach()

    # aggregate_features is a method, not a module; wrap via a pre-hook on the
    # encoder input instead — that tensor IS aggregate_features' output
    # (pre temporal_smoothing differences aside). We reuse the encoder pre-hook.
    handle = model.encoder.register_forward_pre_hook(
        lambda m, i: captured.__setitem__("x", i[0].detach())
    )

    rows, segs = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(model.device)
            batch_segments, keep = _segment_keep_mask(batch, xp)
            _ = model(batch)
            h = to_tr_grid(captured["x"], n_out_t)
            h = rearrange(h, "b t d -> (b t) d").cpu().numpy()
            assert h.shape[0] == len(keep), "concat_backbone alignment broken"
            rows.append(h[keep])
            segs.extend([s for i, s in enumerate(batch_segments) if keep[i]])
    handle.remove()
    return np.concatenate(rows), segs
