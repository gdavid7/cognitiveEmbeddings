"""E4 analysis — stage-wise ablation ladder.

Runs the two load-bearing metrics (E1 behavioral RSA vs human, E3 category
probing) across the ladder and attributes each gain to the stage that produced
it. No new extraction: reuses the 235-concept embeddings.

Ladder (our tap points -> design stages):
  concat_raw  raw backbone features (design's concat_backbone)
    | per-modality projection (learned MLP projectors)
  h_agg       aggregate_features output (combiner input)
    | combiner + positional embedding
  h_proj      transformer-encoder input
    | transformer (temporal + cross-modal attention)
  h_enc       transformer-encoder output  <- the primary embedding

(h_lr dropped per E0: CKA(h_enc,h_lr)=0.97, ~a rotation.)

Interpretation: which stage buys the human-alignment (E1) and how category
decodability (E3) changes along the way.
"""

import numpy as np

from analyze_e1 import load, human_rdm
from analyze_e3 import _merge, _lexname_by_word, _probe, ANIMATE, FINE_MIN
from pipeline_b import metrics, pooling, stats

LADDER = ["concat_raw", "h_agg", "h_proj", "h_enc"]
STAGE = {
    ("concat_raw", "h_agg"): "per-modality projection",
    ("h_agg", "h_proj"): "combiner + pos-embed",
    ("h_proj", "h_enc"): "transformer (temporal + cross-modal attn)",
}
TSV = ("/private/tmp/claude-501/-Users-davidgershony-cognitiveEmbeddings/"
       "d191a05b-78e8-4828-bfff-0dc5304ea555/scratchpad/things/items1854names.tsv")
SPOSE = "pipeline_b_data/spose_embedding_49d_sorted.txt"
IDS = "pipeline_b_data/unique_id.txt"


def main():
    emb, idx, words = _merge("/tmp/e1_embeddings.npz", "/tmp/e1_extra.npz")

    # human RDM (SPoSE, matched by concept name)
    spose = np.loadtxt(SPOSE)
    name2row = {n: i for i, n in enumerate(open(IDS).read().split())}
    H = spose[[name2row[str(w)] for w in words]]
    hr = human_rdm(H)

    # labels for probing
    lex_by_word = _lexname_by_word(TSV)
    fine = np.array([lex_by_word.get(str(w), "MISSING") for w in words])
    animate = np.array([1 if f in ANIMATE else 0 for f in fine])
    from collections import Counter
    keep = {c for c, n in Counter(fine).items() if n >= FINE_MIN and c != "MISSING"}
    fmask = np.array([f in keep for f in fine])

    # metrics per ladder tap
    rows = {}
    for tap in LADDER:
        v = pooling.zscore(emb[tap])
        rsa = metrics.rsa(metrics.rdm(v), hr)
        cl, _ = _probe(emb[tap], animate)
        fn, _ = _probe(emb[tap][fmask], fine[fmask])
        rows[tap] = dict(rsa=rsa, coarse=cl.mean(), fine=fn.mean())

    print(f"E4 ladder attribution ({len(idx)} concepts)\n")
    print(f"{'stage':38s} {'RSA↑human':>10s} {'coarse↑':>9s} {'fine↑':>8s}")
    print("-" * 70)
    for tap in LADDER:
        r = rows[tap]
        print(f"{tap:38s} {r['rsa']:>10.3f} {r['coarse']:>9.3f} {r['fine']:>8.3f}")
    print("\nper-stage delta (attribution):")
    print(f"{'stage':42s} {'ΔRSA':>8s} {'Δcoarse':>9s} {'Δfine':>8s}")
    print("-" * 70)
    for a, b in zip(LADDER, LADDER[1:]):
        dr = rows[b]["rsa"] - rows[a]["rsa"]
        dc = rows[b]["coarse"] - rows[a]["coarse"]
        df = rows[b]["fine"] - rows[a]["fine"]
        print(f"{STAGE[(a, b)]:42s} {dr:>+8.3f} {dc:>+9.3f} {df:>+8.3f}")

    total_rsa = rows["h_enc"]["rsa"] - rows["concat_raw"]["rsa"]
    proj = rows["h_agg"]["rsa"] - rows["concat_raw"]["rsa"]
    trans = rows["h_enc"]["rsa"] - rows["h_proj"]["rsa"]
    print(f"\nHuman-alignment (RSA) gain concat_raw->h_enc = {total_rsa:+.3f}")
    if total_rsa != 0:
        print(f"  projection stage: {proj:+.3f} ({100*proj/total_rsa:.0f}% of gain)")
        print(f"  transformer stage: {trans:+.3f} ({100*trans/total_rsa:.0f}% of gain)")
    return rows


if __name__ == "__main__":
    import json
    rows = main()
    print("\n" + json.dumps({k: {m: float(x) for m, x in v.items()} for k, v in rows.items()}, indent=2))
