"""E1 analysis — behavioral RSA of TRIBE embeddings vs human odd-one-out.

Runs LOCALLY (pure-math, no GPU) on:
  * e1_embeddings.npz  — per-image TRIBE taps, from run_e1_extract (Modal Volume)
  * spose_embedding_49d_sorted.txt — human similarity space (THINGS, OSF)
  * unique_id.txt      — concept names, row-aligned with SPoSE

Human RDM: SPoSE is a validated model of the odd-one-out data; human similarity
between two concepts is the dot product of their (non-negative) 49-d vectors, so
the human dissimilarity is 1 - normalized dot product. We restrict SPoSE to the
extracted concepts (matched by name), in the SAME order as the embeddings.

Executes the preregistered E1 test (preregistration_E1_E3.md):
  P1: RSA(h_enc, human) > RSA(concat_backbone, human), paired, permutation p<0.01.
  P2: h_enc also beats random_proj and shuffled nulls.
"""

import sys
import numpy as np

from pipeline_b import metrics, stats, pooling, baselines


def load(npz_path, spose_path, ids_path):
    d = np.load(npz_path, allow_pickle=True)
    words = [str(w) for w in d["words"]]
    emb = {k: d[k] for k in d.files if k not in ("words", "idx")}

    spose = np.loadtxt(spose_path)
    ids = open(ids_path).read().split()
    name_to_row = {n: i for i, n in enumerate(ids)}
    rows = [name_to_row[w] for w in words]  # align SPoSE to embedding order
    human = spose[rows]
    return words, emb, human


def human_rdm(human_vecs, metric="cosine"):
    """SPoSE dot-product similarity -> dissimilarity. Cosine (normalized dot)
    is the natural dissimilarity for non-negative SPoSE vectors."""
    from scipy.spatial.distance import pdist
    return pdist(human_vecs, metric=metric)


def main(npz="e1_embeddings.npz",
         spose="pipeline_b_data/spose_embedding_49d_sorted.txt",
         ids="pipeline_b_data/unique_id.txt"):
    words, emb, human = load(npz, spose, ids)
    n = len(words)
    print(f"E1: {n} concepts | taps: {list(emb)}")

    # z-score each tap across the corpus before distances (§4.4)
    embz = {k: pooling.zscore(v) for k, v in emb.items()}
    # add nulls
    if "concat_raw" in embz:
        embz["random_proj"] = pooling.zscore(baselines.random_projection(emb["concat_raw"], out_dim=1152))
    embz["shuffled_h_enc"] = baselines.shuffled(embz["h_enc"])

    hr = human_rdm(human)

    # RSA of every tap vs human, with permutation p
    print("\n-- RSA vs human (SPoSE) --")
    rsa_rows = {}
    for k, v in embz.items():
        res = stats.permutation_test_rsa(metrics.rdm(v), hr, n_permutations=10000, seed=0)
        rsa_rows[k] = res
        print(f"  {k:16s} rho={res['observed']:+.4f}  p={res['p_value']:.4f}")

    # Preregistered paired test: h_enc > concat_backbone
    name_backbone = "concat_raw"
    print(f"\n-- P1 paired: h_enc vs {name_backbone} --")
    # paired_rsa_comparison needs an embedding-like target; use SPoSE vectors directly
    paired = stats.paired_rsa_comparison(embz["h_enc"], embz[name_backbone], human,
                                         n_boot=5000, metric="cosine", seed=0)
    print(f"  Δrho(h_enc - {name_backbone}) = {paired['diff']:+.4f} "
          f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]  p={paired['p_value']:.4f} "
          f"| h_enc better: {paired['a_better']}")

    # FDR across the tap grid
    fdr = stats.benjamini_hochberg([rsa_rows[k]["p_value"] for k in rsa_rows])
    print("\n-- verdict --")
    p1 = paired["a_better"] and paired["p_value"] < 0.01
    p2 = (rsa_rows["h_enc"]["observed"] > rsa_rows.get("random_proj", {"observed": 1})["observed"]
          and rsa_rows["h_enc"]["observed"] > rsa_rows["shuffled_h_enc"]["observed"])
    print(f"  P1 (h_enc > {name_backbone}, p<0.01): {'PASS' if p1 else 'FAIL'}")
    print(f"  P2 (h_enc > nulls):                    {'PASS' if p2 else 'FAIL'}")
    return {"rsa": {k: (rsa_rows[k]["observed"], rsa_rows[k]["p_value"]) for k in rsa_rows},
            "paired": paired, "P1": p1, "P2": p2}


if __name__ == "__main__":
    import json
    out = main(*sys.argv[1:])
    print("\n" + json.dumps(out, default=float, indent=2))
