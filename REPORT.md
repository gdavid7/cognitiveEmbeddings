# Pipeline B — Findings Report

**Brain-aligned embeddings from TRIBE v2**
Author: David Gershony · Repo: `github.com/gdavid7/cognitiveEmbeddings`
*Revised after internal review (see §Revisions) — the headline magnitude and mechanism changed.*

---

## TL;DR

We tapped TRIBE v2's internal representation and asked whether brain supervision reshapes it into something more human-like than its ingredient models. The direction survives scrutiny, but the honest effect is **modest, transformer-carried, and not a fusion result**:

- **The hidden state is not a trivial re-mix of the backbones** (E0) — H₀ rejected on video, though the static-image regime where the payoff experiments run is closer to linear (R² 0.80 vs 0.68).
- **It is more human-aligned than its own inputs and its best backbone** (E1) — but the effect is **~0.05–0.08 in RSA**, not the 0.19 a naive raw-concatenation comparison implied. About half of that naive gap was **dead-modality dilution** (a constant audio tone), and most of the rest is free dimensionality reduction.
- **The one robust, significant brain-supervision signal is the transformer step** (Δ = +0.059, p = 0.0007). The learned per-modality projection adds ~nothing over a random projection of the de-diluted features.
- **It is not a stronger category classifier than CLIP** (E3), and the "loses more on fine than coarse" story is **suggestive, not significant** (CIs overlap).

**Revised headline claim:** *Brain supervision — specifically the transformer stage — makes TRIBE's (mostly-visual, single-effective-backbone) representation modestly but significantly more aligned with human similarity geometry than its own pre-transformer input; the effect over its best backbone is real but marginal (p ≈ 0.04), and this configuration does not test multimodal fusion.*

---

## The question and setup

TRIBE v2 fuses Llama-3.2-3B, DINOv2, V-JEPA2, and Wav2Vec2-BERT with a transformer to predict fMRI. Pipeline B keeps the transformer hidden state as an embedding and tests **H₀: the hidden state is ~an affine function of the concatenated backbone features.**

**Configuration caveat (load-bearing):** the payoff experiments (E1/E3/E4) run on THINGS still images shown as *static video*, with the audio track a constant tone and the text/Llama modality zeroed. So effectively the model runs on **the visual backbones only** (DINOv2 + a near-inert V-JEPA2), audio dead, text off. This is **not** a multimodal-fusion configuration.

---

## Results

### E0 — Kill H₀ (linearity)
Real video corpus (Big Buck Bunny + Elephant's Dream, 1,251 rows). Reconstruction gate **passed, max|Δ| = 0.0** (extraction verified). `concat_backbone → h_enc`: CKA 0.39, R² 0.68 → **PROCEED**. `CKA(h_enc, h_lr)` = 0.97 → `h_lr` dropped.
*Spot-check (added in review):* in the **static-image** regime the same map is R² 0.80 / CKA 0.54 — still rejects H₀ (<0.95) but is meaningfully more linear than the video regime. The payoff experiments run where H₀ is *less* strongly rejected.

### E1 — Behavioral alignment (RSA vs human SPoSE), decomposed
235 concepts, correlation-distance RSA, stimulus-bootstrap paired tests:

| Representation | RSA vs human |
|---|---|
| `concat_raw` (raw backbones **with dead audio**) | 0.128 |
| `random_proj` (random 1152-d, zero learning) | 0.239 ± 0.005 |
| **`video_only`** (raw visual backbones, tone sliced out) | 0.276 |
| `h_proj` (learned projection) | 0.272 |
| **`h_enc`** (transformer output) | **0.330** |

Paired (all significant unless noted):
- `h_enc` vs **best backbone** (`video_only`): Δ = **+0.054** [+0.003, +0.103], **p = 0.039** *(marginal)*
- `h_enc` vs `random_proj`: Δ = +0.078 [+0.020, +0.132], p = 0.012
- `h_enc` vs `h_proj` (**transformer step**): Δ = +0.059 [+0.025, +0.092], **p = 0.0007**
- `h_proj` vs `random_proj` (learned projection): **not significant** (Δ ≈ +0.03, p ≈ 0.09)

**Reading:** `concat_raw`'s low score was ~half **dead-audio dilution** (1,917 of 4,864 dims were a constant tone); de-diluting to `video_only` nearly doubles it. A *random* projection recovers most of the rest. So the naive "h_enc beats raw backbones by 0.19" overstates the learned effect ~3×. `h_enc` still beats the best backbone and a random baseline — but modestly.

### E3 — Semantic probing vs CLIP (balanced accuracy, 5-fold, with CIs)

| Rep | Coarse (animacy) | Fine (category) |
|---|---|---|
| CLIP | 0.913 [0.890, 0.937] | 0.835 [0.739, 0.930] |
| `concat_raw` | 0.850 | 0.749 [0.695, 0.803] |
| `h_proj` | 0.825 | 0.654 [0.577, 0.730] |
| `h_enc` | 0.810 [0.772, 0.847] | 0.614 [0.538, 0.689] |

CLIP is the stronger decoder everywhere → preregistered signature **ABSENT**. `h_enc − CLIP`: coarse −0.10 [−0.15, −0.06], fine −0.22 [−0.32, −0.13]. The two CIs **overlap**, so "loses ~2× more on fine than coarse" is **suggestive, not significant**. The transformer's effect on fine decodability (−0.04 [−0.10, +0.02]) includes zero.

### E4 — Stage-wise ladder (revised attribution)
Human-alignment rises `concat_raw` 0.13 → `video_only`/`h_proj` ~0.27 → `h_enc` 0.33. But once dead-audio dilution is removed, the stages attribute as:
- **dead-audio de-dilution + dimensionality:** the large `concat_raw → video_only` jump (not a learned brain effect)
- **learned per-modality projection:** ≈ 0 over de-diluted video (not significant)
- **transformer:** +0.06, **p = 0.0007** — *the only robust brain-supervision stage*

Category decodability falls along the ladder, more for fine, but (per E3 CIs) not significantly so.

---

## The good
- Extraction **provably correct** (gate at max|Δ| = 0.0) before any analysis.
- Two silent bugs caught by design tripwires: a cache collision (constant RDM) and a label misalignment (CLIP-at-chance).
- The reviewable controls were actually run: `random_proj` and best-backbone baselines are **now formally paired-tested**, and the direction (h_enc > backbones > random, transformer significant) **survives**.
- Reported the E3 prereg failure and the magnitude correction straight.

## The bad / what shrank under scrutiny
- **Magnitude:** the human-alignment effect is ~0.05–0.08 RSA, not the 0.19 first reported. Half of the naive gap was dead-audio dilution; most of the remainder is dimensionality.
- **Mechanism:** it is the **transformer**, not the "69% projection" the first draft claimed. The learned projection does not significantly beat random.
- **Best-backbone margin is marginal:** `h_enc` beats `video_only` at only p = 0.039 — would likely not survive strict multiple-comparison correction.
- **Not fusion:** audio dead, text off, V-JEPA2 near-inert on static frames → effectively a single visual backbone. "Fusion" is unsupported here.
- **E3 graded claim:** suggestive only (overlapping CIs).
- **OOD regime:** static images; and H₀ is less strongly rejected there (R² 0.80).

## Done vs. not done
**Done:** extraction + gate; contamination control; preregistration; E0, E1, E3, E4; the review controls (random_proj, video-only, static-linearity, E3 CIs); tested analysis package. All committed.
**Not done:** E2 (needs uncontaminated fMRI); a proper modality-integration test (needs naturalistic audiovisual stimuli — vacuous here); the full four-modality run (text/Llama enabled) and true DINOv2-vs-V-JEPA2 separation; scale-up (more concepts / multiple exemplars for instance-level fine probing); the formal write-up.

## Bottom line
The result **holds in direction but not in magnitude or mechanism as first claimed.** What is defensible: TRIBE's **transformer stage** produces a representation significantly more human-aligned than its pre-transformer input (Δ ≈ 0.06, p < 0.001), and modestly more than its best backbone (Δ ≈ 0.05, p ≈ 0.04). The learned projection and the raw-concatenation comparison were red herrings (dimensionality + dead-modality dilution). This is a real but small brain-supervision effect on a mostly-visual representation — not a fusion result, and not the ~0.19 headline. The cleanest next steps are the full four-modality run on naturalistic video, and E2 on uncontaminated fMRI.

---

## Revisions (after internal review)
An internal review flagged that the `random_proj` control — never formally paired-tested in the first draft — undercut the mechanism story, that a per-modality (DINOv2-alone) baseline was missing, that "fusion" was unsupported by this configuration, and that the E3 graded claim lacked CIs. All four were correct. This report reflects the added controls (`analyze_rebuttal.py`, `focused_rebuttal.py`; results in `results/rebuttal_result.txt`), which lowered the effect size ~3×, reassigned the mechanism from projection to transformer, dropped the fusion claim, and demoted the E3 graded result to suggestive. The core direction survived formal testing; the overclaims did not.

*Reproducibility: `analyze_e{1,3,4}.py`, `analyze_rebuttal.py`, `focused_rebuttal.py`, `pipeline_b/`; extraction in `modal_tribe.py`; results in `results/`; predictions in `preregistration_E1_E3.md`. Compute: Modal A10G, ~$16.*
