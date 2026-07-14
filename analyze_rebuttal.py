"""Rebuttal analyses — stress-test the E1/E4 mechanism claim.

All local, on already-extracted 235-concept embeddings:
  [1] random-projection decomposition + paired test h_enc vs random_proj
  [2] per-modality baseline (slice constant-tone audio out -> video-only RSA)
  [3] E0-style linearity spot-check in the static-image regime
  [4] E3 probe balanced-accuracy CIs (is the fine>coarse deficit real?)
"""
import numpy as np
from scipy.spatial.distance import pdist
from analyze_e1 import human_rdm
from analyze_e3 import _merge, _lexname_by_word, _probe, ANIMATE, FINE_MIN
from pipeline_b import metrics, stats, pooling, baselines

TSV = ("/private/tmp/claude-501/-Users-davidgershony-cognitiveEmbeddings/"
       "d191a05b-78e8-4828-bfff-0dc5304ea555/scratchpad/things/items1854names.tsv")
SPOSE = "pipeline_b_data/spose_embedding_49d_sorted.txt"
IDS = "pipeline_b_data/unique_id.txt"

emb, idx, words = _merge("/tmp/e1_embeddings.npz", "/tmp/e1_extra.npz")
spose = np.loadtxt(SPOSE)
name2row = {n: i for i, n in enumerate(open(IDS).read().split())}
H = spose[[name2row[str(w)] for w in words]]
hr = human_rdm(H)  # cosine dissimilarity


def rsa_of(x):
    return metrics.rsa(metrics.rdm(pooling.zscore(x)), hr)


print(f"=== REBUTTAL ANALYSES ({len(idx)} concepts) ===\n")

# ------------------------------------------------------------------ [1]
print("[1] Random-projection decomposition (RSA vs human)")
r_concat = rsa_of(emb["concat_raw"])
r_proj = rsa_of(emb["h_proj"])
r_enc = rsa_of(emb["h_enc"])
K = 25
rp = np.array([rsa_of(baselines.random_projection(emb["concat_raw"], 1152, seed=s))
               for s in range(K)])
print(f"  concat_raw           {r_concat:.3f}")
print(f"  random_proj (K={K})   {rp.mean():.3f} ± {rp.std():.3f}  [{rp.min():.3f}, {rp.max():.3f}]")
print(f"  h_proj (learned)     {r_proj:.3f}")
print(f"  h_enc                {r_enc:.3f}")
tot = r_enc - r_concat
dim = rp.mean() - r_concat
learn = r_proj - rp.mean()
trans = r_enc - r_proj
print(f"\n  gain h_enc-concat_raw = {tot:+.3f}")
print(f"    dimensionality (random-concat):   {dim:+.3f} ({100*dim/tot:.0f}%)")
print(f"    learned projection (h_proj-random):{learn:+.3f} ({100*learn/tot:.0f}%)")
print(f"    transformer (h_enc-h_proj):        {trans:+.3f} ({100*trans/tot:.0f}%)")
print(f"  brain-supervision gain over random baseline (h_enc-random) = {r_enc-rp.mean():+.3f}")

# paired tests (stimulus bootstrap), cosine throughout (as in E1)
rp0 = baselines.random_projection(emb["concat_raw"], 1152, seed=0)
for a, b, name in [(emb["h_enc"], rp0, "h_enc vs random_proj"),
                   (emb["h_enc"], emb["h_proj"], "h_enc vs h_proj (transformer step)"),
                   (emb["h_proj"], rp0, "h_proj vs random_proj (learned projection)")]:
    r = stats.paired_rsa_comparison(pooling.zscore(a), pooling.zscore(b), H,
                                    n_boot=5000, metric="cosine", seed=0)
    print(f"  paired {name:38s} Δ={r['diff']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] p={r['p_value']:.4f}")

# ------------------------------------------------------------------ [2]
print("\n[2] Per-modality baseline (audio = constant tone, sliced by zero variance)")
cr = emb["concat_raw"]
col_std = cr.std(0)
audio = col_std < 1e-6
print(f"  constant(audio) dims: {int(audio.sum())} | varying(video) dims: {int((~audio).sum())}")
if (~audio).sum() > 0:
    video = cr[:, ~audio]
    r_video = rsa_of(video)
    print(f"  video-only RSA (DINOv2+V-JEPA2): {r_video:.3f}")
    print(f"  concat_raw (with dead audio):    {r_concat:.3f}")
    print(f"  h_enc:                           {r_enc:.3f}")
    rv = stats.paired_rsa_comparison(pooling.zscore(emb["h_enc"]), pooling.zscore(video), H,
                                     n_boot=5000, metric="cosine", seed=0)
    print(f"  paired h_enc vs video-only: Δ={rv['diff']:+.3f} [{rv['ci_low']:+.3f},{rv['ci_high']:+.3f}] p={rv['p_value']:.4f}")

# ------------------------------------------------------------------ [3]
print("\n[3] E0-style linearity spot-check, STATIC-image regime")
for src, nm in [(emb["concat_raw"], "concat_raw"), (cr[:, ~audio] if (~audio).sum() else cr, "video_only")]:
    res = metrics.linear_map_r2(src, emb["h_enc"])
    print(f"  {nm:11s} -> h_enc:  R²={res['r2_uniform']:.3f}  CKA={res['cka']:.3f}")
print("  (E0 on video was R²=0.68, CKA=0.39 -> PROCEED; compare regimes)")

# ------------------------------------------------------------------ [4]
print("\n[4] E3 probe balanced-accuracy CIs (5-fold; is fine-deficit > coarse-deficit real?)")
c = np.load("/tmp/clip_all.npz", allow_pickle=True)
cmap = {int(i): r for i, r in zip(c["idx"], c["clip"])}
emb["clip"] = np.stack([cmap[i] for i in idx])
lex_by_word = _lexname_by_word(TSV)
fine = np.array([lex_by_word.get(str(w), "MISSING") for w in words])
animate = np.array([1 if f in ANIMATE else 0 for f in fine])
from collections import Counter
keep = {k for k, n in Counter(fine).items() if n >= FINE_MIN and k != "MISSING"}
fmask = np.array([f in keep for f in fine])


def ci(a):
    a = np.asarray(a); return a.mean(), a.mean() - 1.96*a.std(ddof=1)/len(a)**.5, a.mean()+1.96*a.std(ddof=1)/len(a)**.5


folds = {}
for r in ["h_enc", "h_proj", "clip", "concat_raw"]:
    cl, _ = _probe(emb[r], animate)
    fn, _ = _probe(emb[r][fmask], fine[fmask])
    folds[r] = (cl, fn)
    (m1, l1, h1), (m2, l2, h2) = ci(cl), ci(fn)
    print(f"  {r:11s} coarse {m1:.3f}[{l1:.3f},{h1:.3f}]  fine {m2:.3f}[{l2:.3f},{h2:.3f}]")
# paired fold deltas: h_enc-CLIP on coarse vs fine, and is fine-deficit larger?
cg = folds["h_enc"][0] - folds["clip"][0]
fg = folds["h_enc"][1] - folds["clip"][1]
print(f"\n  h_enc-CLIP coarse gap {cg.mean():+.3f} [{ci(cg)[1]:+.3f},{ci(cg)[2]:+.3f}]")
print(f"  h_enc-CLIP fine   gap {fg.mean():+.3f} [{ci(fg)[1]:+.3f},{ci(fg)[2]:+.3f}]")
diff = fg.mean() - cg.mean()
print(f"  fine-deficit minus coarse-deficit = {diff:+.3f}  (negative => loses more on fine)")
# h_enc vs h_proj (transformer effect on decodability), fine
tp = folds["h_enc"][1] - folds["h_proj"][1]
print(f"  transformer effect on fine decode (h_enc-h_proj) = {tp.mean():+.3f} [{ci(tp)[1]:+.3f},{ci(tp)[2]:+.3f}]")
