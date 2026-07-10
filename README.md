# Pipeline B — Brain-Aligned Embeddings from TRIBE v2

Implementation of [`tribev2_pipelineB_design.md`](tribev2_pipelineB_design.md).
Extracts the TRIBE v2 fusion transformer's internal hidden state as an
embedding and evaluates, against a falsifiable null (H₀), whether brain
supervision reshapes the representation into something not already present in
the frozen backbones.

## The one question this answers

> **H₀:** The TRIBE hidden state is ~an affine function of the concatenated
> backbone features — fusion adds reweighting, not new structure.

If H₀ holds, don't build Pipeline B; concatenate DINOv2 + V-JEPA2 + Llama and
skip TRIBE. **E0 decides this in ~a day** and is the highest-value step in the
project (design §1, §5.3).

## Layout

Two layers, deliberately separated so the analysis never depends on torch.

| Module | Layer | Needs |
|---|---|---|
| `metrics.py` | pure-math | numpy/scipy/sklearn |
| `stats.py` | pure-math | permutation / bootstrap / FDR (§5.4) |
| `pooling.py` | pure-math | per-TR → per-clip pooling + z-score (§4.4) |
| `experiments.py` | pure-math | E0–E5 drivers with verdicts (§5.3) |
| `contamination.py` | pure-math | training-set guard (§5.1) |
| `extraction.py` | model-dep | `TribeTaps`, `extract_embeddings` (§4.2) |
| `baselines.py` | model-dep | `concat_backbone` + controls (§4.3) |
| `validation.py` | model-dep | reconstruction **hard gate** (§4.2) |

`import pipeline_b` pulls in only the pure-math layer. Import
`pipeline_b.extraction` / `.baselines` / `.validation` explicitly when you have
torch + a `tribev2` checkout.

## Install & test

```bash
pip install -e .          # pure-math layer
pip install -e '.[extract]'   # + torch/einops for live tapping
pytest                    # 40 tests, no torch required
```

## Workflow (maps to design §8 milestones)

### 1–2. Extract (model-dependent)

`extract` needs a live TRIBE experiment handle, which is environment-specific.
The harness:

```python
import numpy as np
from pipeline_b.validation import run_gate
from pipeline_b.extraction import extract_embeddings
from pipeline_b.baselines import extract_concat_backbone, random_projection, shuffled
from pipeline_b.contamination import assert_clean
from pipeline_b import pooling

# xp = <TRIBE experiment>; model built via from_pretrained (eval, average_subjects=True)
assert_clean("YourEvalDataset", audited_ok=True)   # §5.1 — blocks the 4 training studies
run_gate(xp, events)                                # §4.2 HARD GATE — no analysis until this passes

taps, segs = extract_embeddings(xp, events)         # {h_proj, h_enc, h_lr}
cb, _      = extract_concat_backbone(xp, events)    # the H0 control

# per-TR sequences → per-clip vectors, grouped by clip, then standardized (§4.4)
# (group `taps[name]`/`cb` rows by their segment's clip id, mean-pool, z-score)
emb = {name: pooling.zscore(pooling.mean_pool(seqs_by_clip[name])) for name in taps}
emb["concat_backbone"] = pooling.zscore(pooling.mean_pool(cb_by_clip))
emb["random_proj"]     = random_projection(emb["concat_backbone"])
emb["shuffled_tribe"]  = shuffled(emb["h_enc"])
np.savez("embeddings.npz", **emb)
```

The **reconstruction gate is not optional**: if `h_lr → predictor → pooler`
doesn't reproduce `xp.predict()` to float tolerance, the taps are wrong and
every downstream number is meaningless (§4.2). It jointly validates the three
silent hazards — temporal grid, `keep` mask, `(b t)` ordering.

### 3. E0 — kill H₀ (go/no-go)

```bash
python -m pipeline_b.runner analyze --embeddings embeddings.npz
```

`R²>0.95 & CKA>0.95` → **STOP**, write the negative result. Otherwise proceed.
Note both are checked: a general linear map gives high R² but lower CKA, so
CKA is what distinguishes "just a rotation" from "a real reweighting."

### 4. Preregister

Fill in [`prereg_template.md`](prereg_template.md) **before** viewing E1–E3
results (§5.4). This is the cheapest defense against garden-of-forking-paths
overfitting across the tap × experiment grid.

### 5–7. Run experiments

```bash
python -m pipeline_b.runner analyze \
    --embeddings embeddings.npz \
    --ground-truth ground_truth.npz \
    --out report.json
```

`ground_truth.npz` may contain: `human_rdm` (condensed behavioral RDM → E1),
`fmri` + `noise_ceiling` (→ E2), `coarse_labels` + `fine_labels` (→ E3),
`correct_idx` (→ E5). Each present key triggers its experiment.

## What each experiment claims (§5.3)

- **E0** linearity — kills or confirms H₀. *Run first.*
- **E1** behavioral RSA — the payoff: does `h_enc` beat `concat_backbone`
  against human odd-one-out judgments? Uncontaminated ground truth.
- **E2** brain predictivity, noise-ceiling normalized — a **diagnostic**, not a
  result (TRIBE was trained for this). If `h_enc` doesn't beat the backbones
  here, extraction is broken.
- **E3** semantic probing — looks for the coarse-win / fine-loss dissociation
  vs CLIP: the signature of a brain-shaped lossy filter.
- **E4** stage-wise ablation ladder — attributes gains to concatenation vs
  attention vs brain-objective compression.
- **E5** retrieval — cheap smoke test.

All RSA/CKA comparisons carry permutation p-values and stimulus-bootstrap CIs;
apply `experiments.apply_fdr` across the full grid (§5.4).

## Contamination & noise ceiling — the two things that void results

- Never evaluate on Algonauts2025 / Wen2017 / Lahner2024 / Lebel2023 — TRIBE
  trained on them. `contamination.assert_clean` hard-blocks them and forces an
  explicit audit acknowledgement for anything else (§5.1).
- Every brain-predictivity number is reported as a fraction of the noise
  ceiling, or it is uninterpretable (§2.5). Provide repeated measurements to
  `metrics.noise_ceiling`.

## Status vs the design doc

Fully implemented and tested: the entire pure-math evaluation layer (metrics,
stats, pooling, E0–E5, contamination). Implemented to TRIBE's API but
unrunnable here (no torch / no checkpoint): extraction, baselines, and the
reconstruction gate — wire these to your `tribev2` install per the harness
above. Open design questions (§9: DTW over TR sequences, dropping `h_lr` if
`CKA(h_enc,h_lr)≈1`, an ROI-space fifth tap) are left as noted extension points.
