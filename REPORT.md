# Pipeline B — Findings Report

**Brain-aligned embeddings from TRIBE v2**
Author: David Gershony · Status: core experimental arc complete · Repo: `github.com/gdavid7/cognitiveEmbeddings`

---

## TL;DR

We tapped the internal representation of TRIBE v2 (a brain-encoding model built on four frozen foundation models) and asked whether brain supervision reshapes it into something more human-like than its raw ingredients. Across four preregistered experiments the answer is a qualified **yes, with a clear mechanism**:

- **The fusion is not cosmetic** (E0): TRIBE's hidden state is not just a rotation of its backbone features.
- **It is more human-aligned** (E1): its similarity geometry matches human odd-one-out judgments substantially better than the raw backbones (ρ = 0.36 vs 0.13, p < 0.001).
- **But it is not a better category classifier** (E3): CLIP linearly decodes object categories better at every level.
- **These reconcile into one trade-off** (E4): going deeper into TRIBE trades linear category separability — especially fine-grained — for alignment with human similarity structure, and this happens gradually, stage by stage.

**Headline claim:** *Brain supervision makes TRIBE's embedding geometry more human-like by spending fine-grained category separability, and the trade is visible along the network's internal ladder.*

---

## The question

TRIBE v2 fuses Llama-3.2-3B, DINOv2, V-JEPA2, and Wav2Vec2-BERT with a transformer and predicts fMRI. Pipeline B keeps the **transformer's hidden state** as an embedding and tests a falsifiable null:

> **H₀:** the hidden state is ~an affine function of the concatenated backbone features — fusion adds reweighting, not new structure.

If H₀ held, the project would be dead (just concatenate the backbones). Everything hinged on killing or confirming it first.

---

## What was built

- A tested pure-math evaluation package (`pipeline_b/`): RSA, linear CKA, linear-map R², probing, noise ceiling, retrieval, and the statistics (stimulus-permutation tests, stimulus bootstrap, FDR). **40 unit tests, all passing.**
- Model-dependent extraction verified against the **real** `facebook/tribev2` checkpoint via a hard reconstruction gate.
- A Modal harness (cloud A10G GPUs) that stands up the full TRIBE stack, extracts taps, and runs feature extraction — since the model doesn't fit on a laptop.
- Analysis scripts `analyze_e1.py` / `analyze_e3.py` / `analyze_e4.py`, and a committed preregistration.

---

## Results by experiment

### E0 — Kill H₀ (the linearity test)
*Real corpus: Big Buck Bunny + Elephant's Dream, 1,251 TR rows.*

| Test | Result |
|---|---|
| Reconstruction gate (extraction correctness) | **passed, max\|Δ\| = 0.0** |
| `concat_backbone → h_enc` linear map | CKA 0.39, cross-val R² 0.68 |
| `CKA(h_enc, h_lr)` | 0.97 |

**Verdict: PROCEED.** Both CKA and R² are far below the H₀ thresholds (>0.95), so TRIBE's fusion is *not* an affine re-mix of its inputs. Bonus: `h_lr` is ~a rotation of `h_enc` (CKA 0.97), so it was dropped as a redundant tap.

### E1 — Behavioral alignment (the payoff)
*150 THINGS concepts as static clips; RSA vs human SPoSE odd-one-out similarity.*

| Representation | RSA vs human |
|---|---|
| **`h_enc` (TRIBE output)** | **0.247** |
| `h_proj` / `h_agg` (projected input) | 0.212 |
| `concat_raw` (raw backbones) | 0.092 |
| `random_proj` (control) | 0.180 |
| `shuffled` (null) | −0.003 (p = 0.62 ✓) |

Preregistered **P1** (paired `h_enc` > `concat_backbone`): Δρ = **+0.188** [95% CI +0.124, +0.250], permutation **p = 0.0004 → PASS**. **P2** (beats nulls) **PASS**. On the larger 235-concept set (E4) the gap is even cleaner: 0.36 vs 0.13.

### E3 — Semantic probing (what got destroyed?)
*235 concepts (150 + 85 animate-balancing); balanced-accuracy linear probe.*

| Representation | Coarse (animate/inanimate) | Fine (category) |
|---|---|---|
| **CLIP** | **0.91** | **0.84** |
| `concat_raw` | 0.85 | 0.75 |
| `h_proj` | 0.83 | 0.65 |
| **`h_enc`** | 0.81 | 0.61 |

Preregistered dissociation (h_enc vs CLIP): coarse gap −0.10 (**P5 FAIL**), fine gap −0.22 (**P4 PASS**) → **signature ABSENT** as strictly defined. CLIP is the stronger category decoder everywhere. **But** h_enc's deficit is ~2× larger on fine than coarse — graded support that brain supervision preferentially sheds fine detail.

### E4 — Stage-wise ablation ladder
*235 concepts, both metrics up the ladder.*

| Stage | RSA↑human | Coarse | Fine |
|---|---|---|---|
| `concat_raw` | 0.133 | 0.850 | 0.749 |
| `h_agg` (after projection) | 0.287 | 0.825 | 0.654 |
| `h_proj` (after combiner) | 0.287 | 0.825 | 0.654 |
| `h_enc` (after transformer) | 0.355 | 0.810 | 0.614 |

Human-alignment gain `concat_raw → h_enc` = **+0.222**, of which **69% is the learned per-modality projection** and **31% is the transformer** (the combiner + positional embedding contributes exactly 0 — it's affine). Meanwhile category decodability *falls* along the same ladder, more for fine than coarse. **The alignment gain and the decodability loss are two ends of one trade-off, shown mechanistically.**

---

## The good

- **A clean, preregistered positive result** (E0 PROCEED + E1 pass) that meets the design's top success criterion.
- **A coherent mechanistic story** — E4 unifies the apparent E1/E3 tension into a single stage-by-stage trade-off. That's more compelling than "TRIBE wins at everything," which would smell like a confound.
- **Extraction is provably correct** — the reconstruction gate matched the model's own output to 0.0 before any analysis ran.
- **Honest controls worked** — the shuffled null sits at p = 0.62; a random projection is beaten by the learned one; contamination is avoided (THINGS is behavioral, never in TRIBE's fMRI training).
- **Fully reproducible and cheap** — public data, scripted analyses, ~$16 of a $30 compute budget.

## The bad / limitations

- **Out-of-distribution stimuli.** TRIBE has no still-image path, so THINGS images were shown as frozen "static video." V-JEPA2 sees no motion; the ROI sanity check (§5.2) showed ventral-visual-dominated maps (face → fusiform), so it's usable — but it's a confound, and absolute RSA values (~0.25–0.36) are modest.
- **Partial model.** The text/Llama modality was zeroed to avoid a fragile transcription dependency and Llama's gated weights. Results describe the **video+audio configuration**, not the full four-modality model. (Adding text can only add transformation, so it wouldn't overturn E0/E1.)
- **The preregistered E3 signature did not appear.** CLIP beats h_enc on coarse categories too; only the *graded* version of the hypothesis survived. Reported as a partial result, not spun.
- **Dimensionality confounds one comparison.** `concat_raw` (4,864-d) has more linear-probe capacity than `h_enc` (1,152-d), so "raw beats h_enc on probing" is partly capacity, not information.
- **"Fine" is basic-level, not instance-level.** One image per concept means true fine-grained (dog-breed-level) discrimination couldn't be tested.
- **Modest n and a single behavioral ground truth** (SPoSE-derived). No fMRI generalization test yet.

## Bugs caught (validity hygiene)

Two silent errors were caught *because* the pipeline has built-in tripwires, not by luck:

- **Cache collision** — reusing one temp filename made TRIBE's feature cache return the first image's features for all 150. Caught immediately by a **constant RDM** (std = 0), not a plausible-but-wrong number. Fixed with unique paths; verified on a 3-image run before re-spending.
- **Label misalignment** — `im.mat`'s image order diverges from the concept-name list at index 246, scrambling index-based labels. Caught because **even CLIP fell to chance** on animacy (impossible if aligned). Fixed by labeling per concept name via WordNet.

Both are worth stating: the results survived because wrong answers looked obviously wrong.

## Done vs. not done

**Done**
- Extraction + reconstruction gate (verified on the real checkpoint)
- Contamination control and preregistration
- E0 (linearity / kill H₀)
- E1 (behavioral RSA — the payoff)
- E3 (semantic probing vs CLIP)
- E4 (stage-wise ablation ladder)
- Tested analysis package, all committed to the repo

**Not done**
- **E2 (brain predictivity)** — needs sourcing an uncontaminated fMRI dataset; it's a diagnostic, not a headline.
- **Modality-integration ablation** — our tone-only audio makes it vacuous here; needs naturalistic audiovisual stimuli.
- **Full four-modality run** (text/Llama enabled).
- **Scale-up** — more concepts and multiple exemplars per concept (would strengthen E3 and enable instance-level fine probing).
- **The formal write-up / paper.**

---

## Bottom line

The two load-bearing experiments landed positive and preregistered, the "failure" (E3) is the kind that makes the story credible, and E4 explains the mechanism. The defensible claim is specific: **brain supervision reshapes TRIBE's embedding toward human similarity geometry by trading away fine-grained category separability — measurably, stage by stage.** The main asterisks are the out-of-distribution image path and the video+audio-only configuration; the natural next steps are a naturalistic-video replication and the full four-modality run.

*Reproducibility: analysis scripts `analyze_e{0,1,3,4}` and `pipeline_b/`; extraction in `modal_tribe.py`; raw result JSON/TXT under `results/`; predictions locked in `preregistration_E1_E3.md`. Compute: Modal A10G, ~$16 total.*
