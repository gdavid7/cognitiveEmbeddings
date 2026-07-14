"""E3 analysis — semantic probing: what does brain supervision destroy?

Merges the 150 + 85-balancing TRIBE embeddings, aligns CLIP, and probes coarse
(animate/inanimate) vs fine (WordNet lexname category) with linear + k-NN
classifiers, balanced accuracy, 5-fold CV.

Preregistered prediction (preregistration_E1_E3.md):
  P4: h_enc UNDERperforms CLIP on FINE categories.
  P5: h_enc MATCHES/BEATS CLIP on COARSE categories.
  P4 ∧ P5 = the brain-shaped lossy-filter signature.

Balanced accuracy is used throughout so class imbalance can't flatter a
majority-class guesser, and the h_enc-vs-CLIP comparison is on identical folds.
"""

import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

ANIMATE = {"noun.animal", "noun.plant", "noun.body", "noun.person"}
FINE_MIN = 15  # keep lexname classes with >= this many members


def _merge(npz_a, npz_b):
    a = np.load(npz_a, allow_pickle=True)
    b = np.load(npz_b, allow_pickle=True)
    taps = [k for k in a.files if k not in ("words", "idx")]
    emb = {k: np.vstack([a[k], b[k]]) for k in taps}
    idx = np.concatenate([a["idx"], b["idx"]]).astype(int)
    words = np.concatenate([a["words"], b["words"]])
    return emb, idx, words


def _probe(X, y, seed=0):
    """5-fold balanced-accuracy for logistic + k-NN, per fold (for pairing)."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    lin, knn = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
        lin.append(balanced_accuracy_score(y[te], lr.predict(Xte)))
        kn = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(Xtr, y[tr])
        knn.append(balanced_accuracy_score(y[te], kn.predict(Xte)))
    return np.array(lin), np.array(knn)


def _lexname_by_word(tsv_path):
    """Map concept name -> WordNet lexname. Keyed by NAME (uniqueID column),
    NOT index: im.mat image order diverges from unique_id.txt, so index-based
    labels are misaligned. The saved `words` are the ground-truth concept."""
    from nltk.corpus import wordnet as wn
    uid2wnid = {}
    with open(tsv_path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 6:
                uid2wnid[p[5]] = p[2]
    out = {}
    for uid, wnid in uid2wnid.items():
        if wnid[:1] == "n":
            try:
                out[uid] = wn.synset_from_pos_and_offset("n", int(wnid[1:])).lexname()
            except Exception:
                out[uid] = "MISSING"
        else:
            out[uid] = "MISSING"
    return out


def main(e1="e1_embeddings.npz", extra="e1_extra.npz", clip="clip_all.npz",
         tsv="/private/tmp/claude-501/-Users-davidgershony-cognitiveEmbeddings/"
             "d191a05b-78e8-4828-bfff-0dc5304ea555/scratchpad/things/items1854names.tsv"):
    emb, idx, words = _merge(e1, extra)
    # align CLIP by idx
    c = np.load(clip, allow_pickle=True)
    cmap = {int(i): r for i, r in zip(c["idx"], c["clip"])}
    emb["clip"] = np.stack([cmap[i] for i in idx])

    lex_by_word = _lexname_by_word(tsv)
    fine = np.array([lex_by_word.get(str(w), "MISSING") for w in words])
    animate = np.array([1 if fine[k] in ANIMATE else 0 for k in range(len(fine))])

    # restrict fine to well-populated classes
    from collections import Counter
    keep = {c for c, n in Counter(fine).items() if n >= FINE_MIN and c != "MISSING"}
    fmask = np.array([f in keep for f in fine])
    print(f"E3: {len(idx)} concepts | animate {int(animate.sum())}/{len(animate)} | "
          f"fine classes {sorted(keep)} on {int(fmask.sum())} concepts\n")

    reps = ["h_enc", "concat_raw", "h_proj", "h_agg", "clip"]
    res = {}
    for r in reps:
        cl_lin, cl_knn = _probe(emb[r], animate)
        fn_lin, fn_knn = _probe(emb[r][fmask], fine[fmask])
        res[r] = dict(coarse_lin=cl_lin, coarse_knn=cl_knn, fine_lin=fn_lin, fine_knn=fn_knn)
        print(f"  {r:11s} coarse(animacy) lin={cl_lin.mean():.3f} knn={cl_knn.mean():.3f} | "
              f"fine(category) lin={fn_lin.mean():.3f} knn={fn_knn.mean():.3f}")

    # dissociation: h_enc vs CLIP, paired across folds
    def gap(metric):
        d = res["h_enc"][metric] - res["clip"][metric]
        return d.mean(), d
    cg, cd = gap("coarse_lin")
    fg, fd = gap("fine_lin")
    coarse_win = cg >= 0
    fine_loss = fg < 0
    print("\n-- dissociation (h_enc - CLIP, linear, balanced acc) --")
    print(f"  coarse gap = {cg:+.3f}  (h_enc {'>=' if coarse_win else '<'} CLIP)")
    print(f"  fine gap   = {fg:+.3f}  (h_enc {'<' if fine_loss else '>='} CLIP)")
    print(f"\n  P5 coarse-win : {'PASS' if coarse_win else 'FAIL'}")
    print(f"  P4 fine-loss  : {'PASS' if fine_loss else 'FAIL'}")
    print(f"  SIGNATURE (P4 and P5): {'PRESENT' if (coarse_win and fine_loss) else 'ABSENT'}")
    return {"coarse_gap": float(cg), "fine_gap": float(fg),
            "signature": bool(coarse_win and fine_loss),
            "acc": {r: {k: float(v.mean()) for k, v in res[r].items()} for r in reps}}


if __name__ == "__main__":
    import json
    out = main(*sys.argv[1:])
    print("\n" + json.dumps(out, indent=2))
