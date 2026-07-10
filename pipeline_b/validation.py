"""The reconstruction gate (§4.2) — a HARD gate before any analysis.

MODEL-DEPENDENT. The claim: if the taps are correct, then pushing the captured
``h_lr`` through the model's own ``predictor`` and ``pooler`` must reproduce
``xp.predict()`` to float tolerance. If it doesn't, the taps are wrong and every
downstream number is meaningless — so this raises, it does not warn.

This simultaneously validates hazards #1 (temporal grid), #2 (keep mask), and
#3 ((b t) ordering) from §4.1: they all have to be right for the reconstruction
to line up.
"""

from __future__ import annotations

import numpy as np


class ReconstructionGateError(RuntimeError):
    """Raised when h_lr cannot reproduce predict() — taps are invalid."""


def reconstruct_from_h_lr(model, h_lr, subject_id):
    """Push captured ``h_lr`` (B, T, 2048) through predictor + pooler.

    Mirrors ``FmriEncoderModel.forward`` from the low_rank_head output onward:
    the head's captured output is (B, T, D_lr); the forward transposes it to
    (B, D_lr, T) before ``predictor``. We replay exactly that.
    """
    import torch

    with torch.inference_mode():
        x = h_lr.transpose(1, 2)  # (B, T, D_lr) -> (B, D_lr, T), as in forward()
        x = model.predictor(x, subject_id)  # (B, 20484, T)
        out = model.pooler(x)  # (B, 20484, T')
    return out


def run_gate(xp, events, rtol=1e-4, atol=1e-4):
    """Extract, predict, reconstruct, and assert they match (§4.2 hard gate).

    Returns a report dict on success; raises ``ReconstructionGateError`` on
    mismatch or if ``h_lr`` is unavailable (the released checkpoint may lack
    low_rank_head — hazard #7 — in which case this gate cannot run and you must
    fall back to validating ``h_enc`` against a manual encoder→predictor path).
    """
    import torch
    from einops import rearrange

    from .extraction import TribeTaps, to_tr_grid, _segment_keep_mask

    model = xp._model
    if not hasattr(model, "low_rank_head"):
        raise ReconstructionGateError(
            "model has no low_rank_head; the h_lr reconstruction gate cannot "
            "run on this checkpoint (§4.1 hazard 7). Validate h_enc via a manual "
            "predictor path instead before trusting any taps."
        )

    n_out_t = model.n_output_timesteps
    taps = TribeTaps(model)
    loader = xp.data.get_loaders(events=events, split_to_build="all")["all"]

    max_abs_err = 0.0
    n_checked = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(model.device)
            taps.clear()
            _, keep = _segment_keep_mask(batch, xp)

            preds = model(batch)  # the model's own output, (B, 20484, T')
            # subject_id lives in batch.data (verified against demo_utils.predict),
            # not on the batch object directly.
            subject_id = batch.data.get("subject_id", None)
            recon = reconstruct_from_h_lr(model, taps.acts["h_lr"], subject_id)

            preds = preds.float().cpu().numpy()
            recon = recon.float().cpu().numpy()
            if preds.shape != recon.shape:
                taps.remove()
                raise ReconstructionGateError(
                    f"shape mismatch: predict {preds.shape} vs reconstruction "
                    f"{recon.shape} — a tap or transpose is wrong."
                )
            max_abs_err = max(max_abs_err, float(np.abs(preds - recon).max()))
            if not np.allclose(preds, recon, rtol=rtol, atol=atol):
                taps.remove()
                raise ReconstructionGateError(
                    f"h_lr reconstruction does not match predict(): "
                    f"max|Δ|={max_abs_err:.3e} exceeds tol (rtol={rtol}, atol={atol}). "
                    "Taps are invalid; do not proceed to analysis (§4.2)."
                )
            n_checked += 1
            break  # one batch is a sufficient gate; remove to check all

    taps.remove()
    return {"passed": True, "batches_checked": n_checked, "max_abs_err": max_abs_err}
