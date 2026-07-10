"""Tap internal activations from a live TRIBE v2 checkpoint (§3, §4.2).

MODEL-DEPENDENT: needs ``torch``, ``einops``, and an installed ``tribev2``.
Kept out of ``pipeline_b/__init__`` so the analysis layer imports without them.

Tap points (§3):
  h_proj : input to the transformer encoder (post per-modality MLP + concat,
           pre temporal fusion) — isolates fusion-by-concatenation.
  h_enc  : output of the 8-layer transformer — the PRIMARY candidate; has
           temporal context + cross-modal attention.
  h_lr   : output of low_rank_head (2048) — most shaped by the brain objective
           (may be absent from the released checkpoint; guarded).

Every correctness hazard from §4.1 is enforced here, because none of them
throw — they silently produce plausible, wrong numbers:
  1. temporal misalignment  -> ``to_tr_grid`` (adaptive_avg_pool1d) before masking
  2. the ``keep`` mask       -> rebuilt to match predict(), applied after pooling
  3. (b t) ordering          -> asserted, not trusted
  4. modality dropout        -> asserted model is in eval()
  5. average_subjects        -> subject_id is not an axis of variation (caller's concern)
  6. hemodynamic offset      -> in targets, not features; nothing to correct here
  7. low_rank_head existence -> hasattr-guarded before hooking
"""

from __future__ import annotations

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn.functional as F  # noqa: F401
        from einops import rearrange  # noqa: F401
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "extraction requires torch + einops (and an installed tribev2). "
            "Install with: pip install '.[extract]'"
        ) from e


class TribeTaps:
    """Capture internal activations from ``FmriEncoderModel`` via forward hooks."""

    def __init__(self, model):
        _require_torch()
        self.acts = {}
        self.handles = []
        # Hazard #4: dropout is gated on self.training; from_pretrained calls
        # .eval(). If it fires during extraction, embeddings are stochastic.
        assert not model.training, "model must be in eval() or dropout will fire"

        # h_proj: input to the transformer encoder (post-projection, post-pos-embed)
        self.handles.append(
            model.encoder.register_forward_pre_hook(self._save_input("h_proj"))
        )
        # h_enc: output of the transformer encoder
        self.handles.append(
            model.encoder.register_forward_hook(self._save_output("h_enc"))
        )
        # h_lr: the brain-relevant bottleneck (hazard #7: may not exist)
        if hasattr(model, "low_rank_head"):
            self.handles.append(
                model.low_rank_head.register_forward_hook(self._save_output("h_lr"))
            )

    def _save_output(self, name):
        def hook(module, inputs, output):
            self.acts[name] = output.detach()  # (B, T, D)

        return hook

    def _save_input(self, name):
        def hook(module, inputs):
            self.acts[name] = inputs[0].detach()  # (B, T, D)

        return hook

    def clear(self):
        self.acts = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def to_tr_grid(h, n_output_timesteps):
    """``(B, T_feat, D) -> (B, T_TR, D)``. Mirrors ``model.pooler`` exactly.

    Hazard #1: features run at 2 Hz, fMRI at 1 Hz; the downsample lives in
    pooler, AFTER every tap point. So hooked tensors have ~2× the timesteps of
    predict()'s output. Apply the identical AdaptiveAvgPool1d before anything
    else, and only then the keep mask.
    """
    import torch.nn.functional as F

    h = h.transpose(1, 2)  # (B, D, T_feat)
    h = F.adaptive_avg_pool1d(h.float(), n_output_timesteps)  # (B, D, T_TR)
    return h.transpose(1, 2)  # (B, T_TR, D)


def _segment_keep_mask(batch, xp):
    """Rebuild the exact per-TR segment list + keep mask that predict() builds.

    Hazard #2: predict() drops event-free segments when remove_empty_segments
    is set. Hooked activations must be masked identically or every index shifts.
    """
    batch_segments = []
    for segment in batch.segments:
        for t in np.arange(0, segment.duration - 1e-2, xp.data.TR):
            batch_segments.append(segment.copy(offset=t, duration=xp.data.TR))
    if xp.remove_empty_segments:
        keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
    else:
        keep = np.ones(len(batch_segments), dtype=bool)
    return batch_segments, keep


def extract_embeddings(xp, events, taps_wanted=("h_proj", "h_enc", "h_lr")):
    """Return ``({tap: (n_segments, D)}, segments)`` aligned 1:1 with predict().

    ``xp`` is a TRIBE experiment handle (exposes ``_model``, ``data``,
    ``remove_empty_segments``). Runs the model purely to populate hooks; the
    predictions themselves are discarded.
    """
    _require_torch()
    import torch
    from einops import rearrange

    model = xp._model
    n_out_t = model.n_output_timesteps
    taps = TribeTaps(model)
    loader = xp.data.get_loaders(events=events, split_to_build="all")["all"]

    out = {k: [] for k in taps_wanted}
    all_segments = []

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(model.device)
            taps.clear()

            batch_segments, keep = _segment_keep_mask(batch, xp)

            _ = model(batch)  # forward pass purely to populate hooks

            for name in taps_wanted:
                if name not in taps.acts:
                    continue
                h = to_tr_grid(taps.acts[name], n_out_t)  # (B, T_TR, D)
                # Hazard #3: predict() flattens as (b t) d; match it, then assert.
                h = rearrange(h, "b t d -> (b t) d").cpu().numpy()
                assert h.shape[0] == len(keep), (
                    f"{name}: {h.shape[0]} rows vs {len(keep)} segments — "
                    "alignment broken (see §4.1 hazards 1-3)"
                )
                out[name].append(h[keep])

            all_segments.extend([s for i, s in enumerate(batch_segments) if keep[i]])

    taps.remove()
    return {k: np.concatenate(v) for k, v in out.items() if v}, all_segments
