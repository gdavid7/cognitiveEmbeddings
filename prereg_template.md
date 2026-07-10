# Preregistration — Pipeline B, Experiments E1–E3

Fill this in **before** looking at any E1–E3 result (design §5.4, Milestone 4).
Commit it to git with a timestamp. The failure mode — one large model, four
tap points, many metrics — is garden-of-forking-paths overfitting; writing the
predictions down first is the cheapest defense.

- **Date committed:**
- **Embeddings file (sha):**
- **Evaluation dataset(s):** ____  — confirmed uncontaminated via `assert_clean(..., audited_ok=True)`? ☐
- **Ground-truth source (E1):** ____ (e.g. THINGS odd-one-out; static-video vs native-video condition: ____)

## E0 result (record before continuing — it gates everything)

- R² (uniform): ____   CKA: ____   → verdict: ☐ H0_CONFIRMED (stop) ☐ PROCEED ☐ LARGE_TRANSFORM

## Predictions (lock these now)

### E1 — behavioral alignment
- [ ] `h_enc` RSA > `concat_backbone` RSA against human judgments, paired, permutation *p* < 0.01.
- Directional? ☐ yes (one-sided). Primary distance metric: ☐ correlation ☐ cosine ☐ euclidean.
- Fallback if it fails: report "brain-aligned but not behavior-aligned" honestly.

### E2 — brain predictivity (diagnostic only)
- [ ] `h_lr` ≥ `h_enc` > `concat_backbone`, Pearson r normalized by noise ceiling.
- This is a sanity check; a win proves little (TRIBE was trained for it).

### E3 — semantic probing (the interesting dissociation)
- [ ] `h_enc` **underperforms** CLIP on fine-grained categories.
- [ ] `h_enc` **matches or beats** CLIP on coarse superordinate categories.
- Coarse label set: ____   Fine label set: ____   Probe: linear (primary) + k-NN.

## Analysis decisions (lock these too)

- Pooling: ☐ mean (primary) ☐ mean‖std (secondary, report both).
- Permutations: ____ (≥10,000).  Bootstrap resamples: ____.  Bootstrap unit: **stimuli**.
- Multiple comparisons: Benjamini-Hochberg FDR across the full tap × experiment grid, α = ____.
- Taps analyzed: ☐ h_proj ☐ h_enc ☐ h_lr ☐ concat_backbone ☐ clip ☐ random_proj ☐ shuffled_tribe.

## Success criteria (design §6) — which would you claim?

- ☐ E0 R² < 0.9 **and** E1 `h_enc` > `concat_backbone` (p < 0.01) → strong positive.
- ☐ E3 coarse-win / fine-loss dissociation → brain supervision is a semantic filter.
- ☐ E0 R² > 0.95 → clean negative result (backbones are enough).
