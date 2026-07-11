# Preregistration — Pipeline B, Experiments E1–E3

**Committed:** 2026-07-11 (before any E1–E3 results have been computed or viewed).
**Author:** David Gershony
**Status:** Locked. Milestone 4 of `tribev2_pipelineB_design.md`.

This document locks predictions and analysis decisions *before* results exist,
per design §5.4. The failure mode being defended against: one large model, four
tap points, many metrics → garden-of-forking-paths. Writing predictions down
first is the cheapest defense. Any deviation from this document in the final
write-up will be reported as a deviation, with reason.

---

## 0. Context: E0 result (already obtained — this is why we proceed)

E0 ran first (Milestone 3) and is recorded here for provenance; E1–E3 were not
computed at commit time.

- **Corpus:** Big Buck Bunny + Elephant's Dream, 1,251 TR rows (> hidden dim 1,152, so the linear map is well-posed). Video+audio configuration (text/Llama modality zero-filled).
- **Reconstruction gate (§4.2):** PASSED, max|Δ| = 0.0.
- **H₀ test — raw `concat_backbone → h_enc`:** CKA = 0.39, cross-validated ridge R² = 0.68.
- **Internal proxies:** `h_agg → h_enc` CKA 0.54 / R² 0.46; `h_proj → h_enc` CKA 0.54 / R² 0.45.
- **`CKA(h_enc, h_lr)` = 0.97** → `h_lr` is ~a rotation of `h_enc`.
- **Verdict: PROCEED.** All CKA/R² are far below the H₀-confirmed thresholds (both > 0.95). Brain-supervised fusion is not merely an affine function of the backbones.

**Consequence for what follows:** `h_lr` is **dropped** as a redundant tap
(Open Question #2 resolved). The analyzed taps are: `concat_backbone`, `h_agg`,
`h_proj`, `h_enc`, plus the `clip` reference and `random_proj` / `shuffled` nulls.

---

## 1. Datasets & contamination

- **E1 ground truth:** human odd-one-out similarity (THINGS behavioral). It is
  NOT fMRI, so TRIBE was never trained on it (§5.2). Image path viability is
  gated by the §5.2 ROI sanity check (run separately, before E1). If static
  images yield incoherent ROI profiles, E1 pivots to **video** similarity
  ground truth.
- **E2 ground truth:** an fMRI dataset confirmed absent from the TRIBE v2
  training corpus. **Never** Algonauts2025 / Wen2017 / Lahner2024 / Lebel2023.
  Gated by `contamination.assert_clean(name, audited_ok=True)`.
- **E3 labels:** object class (fine) + superordinate category (coarse) for the
  E1 stimulus set.

---

## 2. Locked predictions

### E1 — behavioral alignment (the payoff)
- **P1 (primary):** `RSA(h_enc, human) > RSA(concat_backbone, human)`, paired,
  permutation *p* < 0.01. Directional (one-sided): brain supervision should
  preserve the categorical structure humans use and discard perceptual detail
  they ignore.
- **P2:** `h_enc` also > `random_proj` and > `shuffled` nulls (sanity).
- **Registered fallback:** if P1 fails, report "brain-aligned but not
  behavior-aligned" honestly — an interesting result, not a hidden one.

### E2 — brain predictivity (diagnostic, NOT a headline)
- **P3:** ordering `h_lr* ≥ h_enc > concat_backbone` on noise-ceiling-normalized
  Pearson r. (*`h_lr` dropped per E0; if reinstated for E2 only, this ordering
  is the prediction.*)
- **Interpretation lock:** E2 is a sanity check. TRIBE was trained for it, so a
  win proves little. If `h_enc` does NOT beat `concat_backbone` here, we treat
  it as evidence extraction is broken, not as a null result.

### E3 — semantic probing (the interesting dissociation)
- **P4:** `h_enc` **underperforms** `clip` on FINE categories (linear probe).
- **P5:** `h_enc` **matches or beats** `clip` on COARSE superordinate categories
  (animate/inanimate, face/scene/object).
- The conjunction P4 ∧ P5 is the registered "brain-shaped lossy filter"
  signature. A clean win everywhere would instead suggest a confound.

---

## 3. Analysis decisions (locked)

- **Pooling:** mean over time (primary); `[mean ‖ std]` (secondary, reported alongside).
- **Distance for RDMs:** correlation distance (primary).
- **Standardization:** per-dimension z-score across the corpus before any distance.
- **Permutation tests:** ≥ 10,000 stimulus-label shuffles for every RSA/CKA comparison.
- **Bootstrap:** over **stimuli** (not RDM cells), ≥ 2,000 resamples, 95% CI.
- **Paired tests** when comparing embeddings on the same stimuli.
- **Multiple comparisons:** Benjamini-Hochberg FDR across the full tap × experiment grid, α = 0.05.
- **Probes:** linear (logistic) primary + k-NN secondary, 5-fold CV, standardized within fold.
- **Seeds:** all randomness seeded and reported.

---

## 4. Success criteria (design §6)

The project produces a publishable true claim under any of:
- **E0 R² < 0.9 AND E1 P1 holds (p < 0.01)** → TRIBE embeddings are meaningfully brain-aligned (strong positive). *E0 R² = 0.68 already satisfies the first half.*
- **E3 P4 ∧ P5 dissociation holds** → brain supervision acts as a semantic filter (interesting standalone).
- **E0 R² > 0.95** → clean negative result (backbones suffice). *Ruled out by E0.*

No experimental outcome constitutes failure; only never validating extraction
(gate — passed) or never controlling contamination (guarded) would.
