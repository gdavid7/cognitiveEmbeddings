"""Metric-consistent recompute of the two decisive controls (correlation
distance throughout, for both embeddings and the human SPoSE RDM, so point RSAs
and paired bootstrap Deltas are directly comparable)."""
import numpy as np
from scipy.spatial.distance import pdist
from analyze_e3 import _merge
from pipeline_b import metrics, stats, pooling, baselines

SPOSE = "pipeline_b_data/spose_embedding_49d_sorted.txt"
IDS = "pipeline_b_data/unique_id.txt"
M = "correlation"  # ONE metric everywhere

emb, idx, words = _merge("/tmp/e1_embeddings.npz", "/tmp/e1_extra.npz")
spose = np.loadtxt(SPOSE)
n2r = {n: i for i, n in enumerate(open(IDS).read().split())}
H = spose[[n2r[str(w)] for w in words]]
hr = pdist(H, metric=M)


def rsa_of(x):
    return metrics.rsa(pdist(pooling.zscore(x), metric=M), hr)


def paired(a, b, nb=3000):
    return stats.paired_rsa_comparison(pooling.zscore(a), pooling.zscore(b), H,
                                       n_boot=nb, metric=M, seed=0)


# video-only = concat_raw minus constant (tone) audio columns
cr = emb["concat_raw"]
audio = cr.std(0) < 1e-6
video = cr[:, ~audio]
rp0 = baselines.random_projection(cr, 1152, seed=0)
K = 25
rp_rsa = np.array([rsa_of(baselines.random_projection(cr, 1152, seed=s)) for s in range(K)])

print(f"=== metric-consistent ({M}) recompute, {len(idx)} concepts ===\n")
print("point RSA vs human:")
for nm, x in [("concat_raw", cr), ("video_only", video), ("random_proj~", None),
              ("h_proj", emb["h_proj"]), ("h_enc", emb["h_enc"])]:
    if nm == "random_proj~":
        print(f"  {nm:12s} {rp_rsa.mean():.3f} ± {rp_rsa.std():.3f}")
    else:
        print(f"  {nm:12s} {rsa_of(x):.3f}")

print("\npaired (stimulus bootstrap):")
for a, b, nm in [(emb["h_enc"], video, "h_enc vs video_only (best backbone)"),
                 (emb["h_enc"], rp0, "h_enc vs random_proj"),
                 (emb["h_enc"], emb["h_proj"], "h_enc vs h_proj (transformer)"),
                 (video, rp0, "video_only vs random_proj")]:
    r = paired(a, b)
    print(f"  {nm:38s} Δ={r['diff']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] p={r['p_value']:.4f}")
