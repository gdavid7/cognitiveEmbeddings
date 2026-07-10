# Design Doc — Pipeline B: Brain-Aligned Embeddings from TRIBE v2

**Author:** David Gershony
**Status:** Proposed
**Scope:** Extract internal representations from `facebookresearch/tribev2` and evaluate whether they constitute a meaningfully *brain-aligned* embedding space, distinct from the frozen backbones they're built from.

---

## 1. The research question, stated precisely

TRIBE v2 predicts human fMRI responses to media. It does this by taking four frozen foundation models (Llama-3.2-3B, DINOv2-large, V-JEPA2-ViTg, Wav2Vec2-BERT), fusing their features with a transformer, and projecting to 20,484 cortical vertices.

Pipeline B proposes to throw away the brain map and keep the **transformer's internal hidden state** as an embedding.

The interesting question is not "does this produce vectors?" (trivially yes). It is:

> **Does brain supervision reshape the representation into something that isn't already present in the frozen backbones?**

This matters because TRIBE's *inputs* are already excellent embeddings. V-JEPA2 and DINOv2 are strong on their own. Everything TRIBE adds is (a) **multimodal fusion**, (b) **temporal context** via the transformer, and (c) **a training objective that says "keep what the cortex cares about, discard what it doesn't."**

So there is a sharp, falsifiable null hypothesis, and the whole design hangs off it:

> **H₀ (the boring outcome):** The TRIBE hidden state is approximately an *affine function* of the concatenated backbone features. Fusion adds reweighting, not new structure. Any apparent gain is the backbones' gain, laundered.

If H₀ holds, Pipeline B is not worth building — you should concatenate DINOv2 + V-JEPA2 + Llama features and skip TRIBE entirely. **The entire experimental design below exists to kill or confirm H₀.** Everything else is secondary.

---

## 2. Background fundamentals

Assume nothing. Here is every concept the design depends on.

### 2.1 What "an embedding" is, and what makes one *good*

An embedding is a map from a complicated object (a video clip) to a vector in ℝᵈ, such that **geometric closeness in ℝᵈ means semantic closeness in the world**. The vector itself is meaningless; only the *relative distances* matter.

This is the key mental shift for this project: **we never evaluate a single embedding vector. We only ever evaluate the structure of distances among many of them.** A method that scrambles all the vectors but preserves all pairwise distances is, for our purposes, the identical embedding.

### 2.2 Representational Similarity Analysis (RSA)

The workhorse of this project. Borrowed from neuroscience precisely because it lets you compare two systems with *totally different* internal formats — a 1152-d transformer state and a 20,484-d brain map, or a brain and a human's button presses.

The trick: don't compare the systems, compare their **distance matrices**.

1. Take *n* stimuli. Embed all of them in System A → get an *n×n* matrix of pairwise distances. This is the **RDM** (Representational Dissimilarity Matrix).
2. Do the same for System B → another *n×n* RDM.
3. Correlate the two RDMs (Spearman, over the upper triangle).

**Analogy:** Suppose I want to know if you and I have the same taste in music, but you rate songs 1–10 and I rate them as colors. Comparing "7" to "teal" is nonsense. But we can each say *"song A is more similar to song B than to song C."* If our *rankings of similarity* agree across all triples, we share a taste structure — even though our formats never touch. RSA is that, formalized.

Result is a single number: Spearman ρ between RDMs. Higher = the two systems organize the world the same way.

### 2.3 CKA (Centered Kernel Alignment)

A second, complementary similarity measure between representations. RSA on ranks is robust but throws away magnitude; **linear CKA** asks specifically: *how much of representation A is recoverable from representation B by a linear map?*

This is exactly the tool for testing H₀. Linear CKA is invariant to rotation and isotropic scaling but **not** to arbitrary invertible linear maps, so `CKA(backbone_concat, tribe_hidden) ≈ 1.0` is strong evidence for the boring outcome.

### 2.4 Linear probing

To ask "is information X present in embedding E?", freeze E and train **only a linear classifier** on top to predict X.

The linearity is the whole point. Any information is *technically* present in a lossless embedding — a sufficiently deep MLP could dig it out. Asking that the information be *linearly decodable* is asking that it be **explicitly represented**, i.e. laid out along directions in the space rather than tangled up. "Good embedding" ≈ "the interesting variables are linear directions."

### 2.5 The noise ceiling

Real fMRI is noisy. If you scan the same person watching the same movie twice, the two brain maps won't match — correlation might only be r ≈ 0.4.

So a model that predicts brain data at r = 0.4 is not "40% good." It is **perfect**, given the data. The noise ceiling is the correlation the *data has with itself*, and it is the true upper bound for any model.

**Every brain-predictivity number in this project must be reported as a fraction of the noise ceiling**, or it is uninterpretable. This is the single most common way brain-encoding results get oversold.

### 2.6 Why we need behavioral data, not just brain data

TRIBE was *trained* to predict fMRI. Evaluating it on "can it predict fMRI" is close to circular — it will win, and we'll have learned nothing.

The load-bearing evaluation is therefore against **human behavioral similarity judgments**: independent ground truth about cognitive similarity, never seen by the model, collected from people making explicit "which of these is the odd one out?" choices. If brain supervision buys anything real, it should transfer *here*.

---

## 3. Where to tap the model

From `model.py`, `FmriEncoderModel.forward()`:

```python
x = self.aggregate_features(batch)          # (B, T, 1152)  ← per-modality MLP projections, concatenated
if hasattr(self, "temporal_smoothing"): ...
x = self.transformer_forward(x, subject_id) # (B, T, 1152)  ← TAP: h_enc
x = x.transpose(1, 2)                       # (B, 1152, T)
if self.config.low_rank_head is not None:
    x = self.low_rank_head(x.transpose(1,2)).transpose(1,2)  # ← TAP: h_lr, (B, T, 2048)
x = self.predictor(x, subject_id)           # (B, 20484, T)
out = self.pooler(x)                        # (B, 20484, T')  ← T' = duration_trs
```

Four tap points, chosen to isolate *what each stage contributes*:

| Tap | Where | Dim | What it isolates |
|---|---|---|---|
| `h_raw` | `batch.data[modality]` | ~varies | **Baseline.** Frozen backbone features, no TRIBE at all. |
| `h_proj` | pre-hook on `encoder` | 1152 | After per-modality MLP + concat, **before** temporal fusion. Isolates fusion-by-concatenation. |
| `h_enc` | forward hook on `encoder` | 1152 | **Primary candidate.** After 8-layer transformer. Has temporal context + cross-modal attention. |
| `h_lr` | forward hook on `low_rank_head` | 2048 | The brain-relevant bottleneck. Most compressed toward cortex. |

The `h_enc` vs `h_proj` contrast measures **what the transformer adds**. The `h_enc` vs `h_raw` contrast measures **what TRIBE adds in total**. These are the two numbers the project exists to produce.

> **Note on `h_lr`:** its dim (2048) *exceeds* `h_enc` (1152), so it is not a bottleneck in the dimensional sense — it is an up-projection. It is a bottleneck only in the *information* sense: it's the last representation before the per-subject linear readout, so it has been maximally shaped by the brain objective. Worth including; don't assume it's the winner.

---

## 4. Implementation

### 4.1 Correctness hazards (read before writing code)

These will not throw errors. They will quietly produce plausible, wrong numbers.

1. **Temporal misalignment (the big one).**
   Features run at 2 Hz (`data.frequency = 2`); fMRI output runs at 1 Hz (`TR = 1/neuro.frequency = 1`). The downsample happens in `self.pooler`, which sits *after* every tap point. So hooked tensors have `T ≈ 2 × duration_trs` timesteps, while `predict()` returns `duration_trs` per segment.
   **Fix:** apply the same `AdaptiveAvgPool1d(n_output_timesteps)` to hooked activations before doing anything else. Then, and only then, apply the `keep` mask.

2. **The `keep` mask.** `predict()` drops segments with no events (`remove_empty_segments=True`). Hooked activations must be masked identically or every index shifts.

3. **Ordering.** `predict()` does `rearrange(y_pred, "b d t -> (b t) d")`. Hooks give `(B, T, D)`, so the matching op is `rearrange(h, "b t d -> (b t) d")`. Same `(b t)` ordering — verify with an assert, don't trust it.

4. **Modality dropout is training-only.** `modality_dropout=0.3` and `temporal_dropout` are gated on `self.training`. `from_pretrained` calls `.eval()`. Confirm — if dropout fires during extraction, embeddings are stochastic garbage.

5. **`average_subjects=True`.** Set by `from_pretrained`. The embedding is of the *average* cortex. Fine, but it means `subject_id` is not a meaningful axis of variation; don't analyze it.

6. **The 5s hemodynamic offset is in the *targets*, not the features** (`neuro_extractor.offset = 5`). So `h_enc` at time *t* corresponds to the *stimulus* at time *t*, and predicts brain activity at *t+5s*. For stimulus-similarity work this is what you want. Don't "correct" for it.

7. **Verify `low_rank_head` exists in the released checkpoint.** `defaults.py` has `low_rank_head: 2048`, but the shipped `config.yaml` on HF is authoritative. `hasattr(model, "low_rank_head")` before hooking.

### 4.2 Extraction code

```python
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from tribev2 import TribeModel


class TribeTaps:
    """Capture internal activations from FmriEncoderModel via forward hooks."""

    def __init__(self, model):
        self.acts = {}
        self.handles = []
        assert not model.training, "model must be in eval() or dropout will fire"

        # h_proj: input to the transformer encoder (post-projection, post-pos-embed)
        self.handles.append(
            model.encoder.register_forward_pre_hook(self._save_input("h_proj"))
        )
        # h_enc: output of the transformer encoder
        self.handles.append(
            model.encoder.register_forward_hook(self._save_output("h_enc"))
        )
        # h_lr: the brain-relevant bottleneck (may not exist)
        if hasattr(model, "low_rank_head"):
            self.handles.append(
                model.low_rank_head.register_forward_hook(self._save_output("h_lr"))
            )

    def _save_output(self, name):
        def hook(module, inputs, output):
            self.acts[name] = output.detach()   # (B, T, D)
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


def to_tr_grid(h, n_output_timesteps):
    """(B, T_feat, D) -> (B, T_TR, D). Mirrors model.pooler exactly."""
    h = h.transpose(1, 2)                                   # (B, D, T_feat)
    h = F.adaptive_avg_pool1d(h.float(), n_output_timesteps)  # (B, D, T_TR)
    return h.transpose(1, 2)                                # (B, T_TR, D)


@torch.inference_mode()
def extract_embeddings(xp, events, taps_wanted=("h_proj", "h_enc", "h_lr")):
    """Returns {tap: (n_segments, D)} aligned 1:1 with xp.predict()'s segments."""
    model = xp._model
    n_out_t = model.n_output_timesteps
    taps = TribeTaps(model)
    loader = xp.data.get_loaders(events=events, split_to_build="all")["all"]

    out = {k: [] for k in taps_wanted}
    all_segments = []

    for batch in loader:
        batch = batch.to(model.device)
        taps.clear()

        # Rebuild the exact segment list predict() would build
        batch_segments = []
        for segment in batch.segments:
            for t in np.arange(0, segment.duration - 1e-2, xp.data.TR):
                batch_segments.append(segment.copy(offset=t, duration=xp.data.TR))
        if xp.remove_empty_segments:
            keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
        else:
            keep = np.ones(len(batch_segments), dtype=bool)

        _ = model(batch)  # forward pass purely to populate hooks

        for name in taps_wanted:
            if name not in taps.acts:
                continue
            h = to_tr_grid(taps.acts[name], n_out_t)      # (B, T_TR, D)
            h = rearrange(h, "b t d -> (b t) d").cpu().numpy()
            assert h.shape[0] == len(keep), \
                f"{name}: {h.shape[0]} rows vs {len(keep)} segments — alignment broken"
            out[name].append(h[keep])

        all_segments.extend([s for i, s in enumerate(batch_segments) if keep[i]])

    taps.remove()
    return {k: np.concatenate(v) for k, v in out.items() if v}, all_segments
```

**Validation gate (do this first, before any analysis):**

Run `extract_embeddings` and `xp.predict()` on the same clip. Then reconstruct the brain map from `h_lr` by hand:

```python
# h_lr -> predictor -> pooler should reproduce preds exactly
```
If it doesn't match `preds` to within float tolerance, the taps are wrong and every downstream number is meaningless. **This is a hard gate.** No analysis until it passes.

### 4.3 Baselines to extract alongside

| Name | Definition | Why |
|---|---|---|
| `concat_backbone` | `batch.data[modality]` for all modalities, layer-aggregated, concatenated, pooled to TR grid | **The critical control.** This is TRIBE's input. |
| `clip` | CLIP/SigLIP on sampled frames, mean-pooled | Standard semantic embedding reference point. |
| `random_proj` | Random Gaussian projection of `concat_backbone` to 1152-d | Controls for dimensionality. Any metric where this scores well is a metric that measures nothing. |
| `shuffled_tribe` | `h_enc` with row order permuted | Null distribution for RSA/CKA. |

### 4.4 Pooling to a per-clip vector

Segment embeddings are per-TR (1 Hz). For clip-level similarity:
- **Primary:** mean-pool over time.
- **Secondary:** also try `[mean ‖ std]` concatenation, to test whether temporal *variability* carries signal that mean-pooling destroys. Cheap to add; report both.

Standardize (z-score per dimension across the corpus) before any distance computation — otherwise a few high-variance dimensions dominate every cosine.

---

## 5. Evaluation protocol

### 5.1 Contamination: the threat to validity

TRIBE v2 trained on **500+ hours from 700+ subjects**, spanning images, podcasts, videos, and text. The repo's `studies/` directory names four: `Algonauts2025`, `Wen2017`, `Lahner2024`, `Lebel2023`.

**Therefore: those four datasets cannot be used for evaluation.** Any result on them is train-set performance.

Mitigations, in priority order:
1. **Use stimuli the model provably never saw.** Novel video clips, or behavioral datasets (below) that aren't fMRI at all.
2. Audit the v2 paper/model card for the full training corpus list before touching *any* public dataset.
3. When in doubt, treat a dataset as contaminated. The cost of a false positive here is a wasted project.

### 5.2 Ground truth for "cognitive similarity"

The strongest available: **human odd-one-out triplet judgments** (e.g. the THINGS behavioral dataset — people see three items, pick the odd one). From many triplets you derive an empirical human similarity space. This is a direct behavioral readout of cognitive similarity, and it is *not* fMRI, so TRIBE was never trained on it.

**Complication:** THINGS is images, and TRIBE has **no native still-image path** (`get_events_dataframe` accepts only `.txt`/audio/video). You must synthesize a short static video per image. This is out-of-distribution and is itself a confound — a static clip gives V-JEPA2 no motion to chew on.

**Mitigation:** treat "static-video" as an explicit experimental condition. Run a sanity check first: does `h_enc` on static clips even produce sensible brain maps (strong occipital/ventral response, weak motion-area response)? Use `summarize_by_roi()` from `utils.py`. If the ROI profile is incoherent, the image path is invalid and the behavioral experiment must move to **video** ground truth instead.

A video fallback: collect (or reuse) human similarity judgments on short video clips. Smaller, but in-distribution and confound-free. **I'd recommend budgeting for this from the start rather than treating it as a fallback** — it's the only clean route.

### 5.3 The experiments

Each states a prediction. Each can fail.

---

**E0 — Kill H₀ (the linearity test). *Run this before anything else.***

Fit a linear map `W: concat_backbone → h_enc` on a train split; measure R² on held-out data. Also compute `linear_CKA(concat_backbone, h_enc)`.

| Outcome | Meaning |
|---|---|
| R² > 0.95, CKA > 0.95 | **H₀ confirmed.** TRIBE adds a rotation. Pipeline B is dead; write it up as a negative result and stop. |
| R² ≈ 0.6–0.9 | TRIBE adds real nonlinear/temporal structure. Proceed. |
| R² < 0.5 | Large transformation. Proceed, but check you haven't broken the extraction. |

This experiment is cheap and decides whether the rest of the project is worth running. **It is the single highest-value thing in this doc.** Do not skip it out of enthusiasm.

---

**E1 — Behavioral alignment (the payoff experiment).**

RSA between each embedding's RDM and the human-judgment RDM.

*Prediction:* `h_enc` > `concat_backbone`. Brain supervision should discard perceptual detail humans ignore and preserve the categorical structure humans use. If TRIBE has learned anything about cognition, it shows up here.

*If it fails:* TRIBE's representation is brain-aligned but not *behavior*-aligned — an interesting result in itself, and worth reporting honestly.

---

**E2 — Brain predictivity on held-out data.**

Ridge regression from each embedding → real fMRI on an uncontaminated dataset. Report Pearson r **normalized by the noise ceiling**.

*Prediction:* `h_lr` ≥ `h_enc` > `concat_backbone`.

*Caveat, stated loudly:* TRIBE was trained for exactly this. A win is expected and proves little. Its value is as a **sanity check** — if `h_enc` does *not* beat the backbones at brain prediction, the extraction is broken. Treat E2 as a diagnostic, not a result.

---

**E3 — Semantic probing (the "what got destroyed?" experiment).**

Linear probe + k-NN for category labels (object class, scene type, action). Compare against CLIP.

*Prediction, and this is the interesting one:* `h_enc` **underperforms CLIP** on fine-grained categories while **matching or beating it** on coarse superordinate categories (animate/inanimate, face/scene/object).

Why predict a *loss*? Because that's the hypothesis. The brain objective is a lossy filter: it should preserve what cortex distinguishes (coarse category, driven by the well-known face/scene/body/object selectivity of ventral temporal cortex) and discard what cortex does not explicitly code. **A clean win on coarse + clean loss on fine is the signature of a genuinely brain-shaped representation**, and is a much more compelling result than "TRIBE beats CLIP at everything" (which would suggest a confound).

---

**E4 — Stage-wise ablation.**

Full ladder: `h_raw → concat_backbone → h_proj → h_enc → h_lr → brain_map`, all metrics, on every experiment.

Isolates: does the gain come from **concatenation** (`h_proj` vs `concat_backbone`), from **temporal/cross-modal attention** (`h_enc` vs `h_proj`), or from **brain-objective compression** (`h_lr` vs `h_enc`)?

Additional single-modality runs (video-only, audio-only, text-only, via zeroing `batch.data`) test whether fusion is doing real integrative work or whether one modality dominates. TRIBE's own papers report multimodality mattering most in high-level associative cortex — that should be visible here.

---

**E5 — Retrieval.**

Top-1/top-5 accuracy retrieving the correct stimulus from its embedding. The repo already uses `TopkAcc` as a training metric, so this is the model's home turf. Cheap; run it as a smoke test.

### 5.4 Statistics

Non-negotiable, because RSA on *n* stimuli gives you *n(n−1)/2* correlated cells and it is trivially easy to manufacture significance.

- **Permutation tests** for all RSA/CKA comparisons: shuffle stimulus labels ≥10,000×, build the null, report exact *p*.
- **Bootstrap over stimuli** (not over RDM cells — the cells are not independent) for confidence intervals.
- **Paired tests** when comparing embeddings on the same stimuli.
- **Correct for multiple comparisons** across the tap × experiment grid (FDR).
- **Preregister E0–E3 predictions** before looking at results. The failure mode here — a large model, many tap points, many metrics — is garden-of-forking-paths overfitting. Writing the predictions down first is the cheapest possible defense.

---

## 6. Success criteria

The project **succeeds** — meaning it produces a publishable, true claim — under any of:

- **E0 shows R² < 0.9** *and* **E1 shows `h_enc` beats `concat_backbone` on behavioral RSA**, permutation *p* < 0.01. → *TRIBE embeddings are meaningfully brain-aligned.* Strong positive result.
- **E3 shows the predicted coarse-win/fine-loss dissociation.** → *Brain supervision acts as a semantic filter.* Interesting even standalone.
- **E0 shows R² > 0.95.** → *Brain-encoding models add nothing beyond their backbones as embeddings.* A clean, useful **negative result**, and cheap to obtain.

The project **fails** only if extraction is never validated (§4.2 gate) or contamination is never controlled (§5.1). Both are avoidable by construction. Note that there is no experimental outcome that constitutes failure — this is by design, and is why E0 comes first.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Training-set contamination | **Critical** | Audit corpus; use behavioral ground truth; assume contamination by default |
| Silent temporal misalignment | **Critical** | §4.2 reconstruction gate before any analysis |
| Still images are out-of-distribution | High | ROI sanity check; budget for video-based ground truth from day one |
| H₀ is true | Medium | E0 first — costs ~1 day, saves ~2 months |
| Forking paths / p-hacking | Medium | Preregister; permutation tests; FDR |
| CC-BY-NC-4.0 license | Low (research) | Research use only; no product path without licensing |
| Compute (ViT-giant + Llama-3B feature extraction) | Low | Feature cache is built in (`cache_folder`); extraction is one-time per corpus |

---

## 8. Milestones

| # | Deliverable | Gate |
|---|---|---|
| 1 | `TribeTaps` + `extract_embeddings`; reconstruction gate passes | Hard gate — no progress without it |
| 2 | Baselines extracted; corpus contamination audit written down | — |
| 3 | **E0 (linearity test)** | **Go/no-go for the whole project** |
| 4 | Preregistration of E1–E3 predictions | Before any results are viewed |
| 5 | E2 + E5 (diagnostics) | Confirms extraction is sane |
| 6 | E1 + E3 (the real experiments) | — |
| 7 | E4 ablation ladder; write-up | — |

Milestone 3 is the decision point. Everything before it is ~1 week of work; everything after it is the actual project.

---

## 9. Open design questions

1. **Pool over time, or model it?** Mean-pooling to one vector per clip discards the temporal structure the transformer was built to capture. Worth a variant where similarity is computed as DTW/alignment distance over the TR-level sequences.
2. **Is `h_lr` really the most brain-shaped tap?** It's up-projected (1152→2048), so "bottleneck" is a misnomer. It might just be a rotated `h_enc`. Test with `CKA(h_enc, h_lr)` — if ≈1, drop it and save a tap.
3. **Should the ROI structure be exploited?** `summarize_by_roi()` gives interpretable, low-dim signatures. An ROI-space embedding (~360-d, HCP parcels) sits between `h_lr` and the raw brain map, is far more interpretable, and might be the actual sweet spot for downstream use. Consider adding as a fifth tap.

---

## Appendix — Key constants

```
mesh              = fsaverage5     → 10,242 vertices/hemi → n_outputs = 20,484
TR                = 1.0 s          (neuro frequency = 1 Hz)
feature frequency = 2 Hz           → T_feat ≈ 2 × T_TR    ← alignment hazard
hidden            = 1152
low_rank_head     = 2048
encoder depth     = 8
n_output_timesteps = duration_trs
hemodynamic offset = 5 s (applied to targets, not features)
modality_dropout  = 0.3 (train only); average_subjects = True at inference
license           = CC-BY-NC-4.0
```
