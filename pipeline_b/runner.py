"""CLI wiring the milestones (§8) into runnable steps.

Two entry paths:
  * ``extract``  — model-dependent (needs torch + tribev2). Runs the §4.2
    reconstruction gate, then taps + baselines, then saves an .npz.
  * ``analyze``  — pure-math. Loads a saved .npz of pooled embeddings plus
    ground-truth arrays and runs E0–E5 with statistics + FDR.

Usage:
  python -m pipeline_b.runner extract  --out embeddings.npz   [tribe args...]
  python -m pipeline_b.runner analyze  --embeddings embeddings.npz --ground-truth gt.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import experiments, pooling


def _pool_and_standardize(raw: dict, segments_key="_segments") -> dict:
    """Given a dict of per-clip TR-sequence lists, mean-pool then z-score."""
    out = {}
    for name, seqs in raw.items():
        if name.startswith("_"):
            continue
        out[name] = pooling.zscore(pooling.mean_pool(seqs))
    return out


def cmd_extract(args):
    """Milestones 1–2: reconstruction gate, then extract taps + baselines."""
    # Imported lazily: this branch is the only one that needs torch/tribev2.
    from .validation import run_gate
    from .extraction import extract_embeddings
    from .baselines import extract_concat_backbone
    from .contamination import assert_clean

    raise SystemExit(
        "extract requires a live TRIBE v2 experiment handle (xp) and an events "
        "frame, which are environment-specific. Wire this to your tribev2 setup:\n"
        "  1. build xp (from_pretrained, .eval(), average_subjects=True)\n"
        "  2. run_gate(xp, events)                 # §4.2 HARD GATE\n"
        "  3. assert_clean(dataset_name, audited_ok=...)   # §5.1\n"
        "  4. taps, segs = extract_embeddings(xp, events)\n"
        "  5. cb, _       = extract_concat_backbone(xp, events)\n"
        "  6. pool + zscore, then np.savez(args.out, **matrices)\n"
        "See README 'Extraction' for the copy-paste harness."
    )


def cmd_analyze(args):
    """Milestones 3–7: E0–E5 on saved, pooled, standardized embeddings."""
    emb = dict(np.load(args.embeddings, allow_pickle=True))
    gt = dict(np.load(args.ground_truth, allow_pickle=True)) if args.ground_truth else {}

    report = {}
    if "concat_backbone" in emb and "h_enc" in emb:
        report["E0"] = experiments.e0_linearity(emb["concat_backbone"], emb["h_enc"])
        print(f"[E0] R²={report['E0']['r2']:.3f} CKA={report['E0']['cka']:.3f} "
              f"-> {report['E0']['verdict']}: {report['E0']['meaning']}")
        if report["E0"]["verdict"] == "H0_CONFIRMED":
            print("[E0] Go/no-go: STOP — negative result (§6).")

    if "human_rdm" in gt:
        report["E1"] = experiments.e1_behavioral_rsa(emb, gt["human_rdm"])
    if "fmri" in gt and "noise_ceiling" in gt:
        report["E2"] = experiments.e2_brain_predictivity(emb, gt["fmri"], gt["noise_ceiling"])
    if "coarse_labels" in gt and "fine_labels" in gt:
        report["E3"] = experiments.e3_semantic_probing(emb, gt["coarse_labels"], gt["fine_labels"])
    if "correct_idx" in gt:
        report["E5"] = experiments.e5_retrieval(emb, {}, gt["correct_idx"])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)
        print(f"wrote {args.out}")
    return report


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main(argv=None):
    p = argparse.ArgumentParser(prog="pipeline_b.runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="tap a live TRIBE v2 model (needs torch)")
    pe.add_argument("--out", required=True)
    pe.set_defaults(func=cmd_extract)

    pa = sub.add_parser("analyze", help="run E0–E5 on saved embeddings")
    pa.add_argument("--embeddings", required=True, help=".npz of pooled+zscored matrices")
    pa.add_argument("--ground-truth", default=None, help=".npz of human_rdm/fmri/labels/...")
    pa.add_argument("--out", default=None, help="write JSON report here")
    pa.set_defaults(func=cmd_analyze)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
