"""Modal app: stand up TRIBE v2, run the §4.2 reconstruction gate, and a
preliminary E0 (linearity / H0) on a live checkpoint.

Run:  modal run modal_tribe.py

Design-doc milestones 1 & 3. Deliberately scoped and cheap:
  * bypasses the whisperx transcription path (audio_only=True) -> no uvx, and
    the text/Llama modality is absent (zero-filled), sidestepping Llama's HF
    gating. Audio (Wav2Vec2-BERT) + video (DINOv2, V-JEPA2) backbones are public.
  * self-contained: reconstructs from h_enc through the real forward tail
    (model.py lines 170-175), so the gate holds with or without low_rank_head.
"""

import modal

app = modal.App("tribe-pipelineb")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "git+https://github.com/facebookresearch/tribev2",
        "scikit-learn>=1.3",
    )
    # Separate layer so the big tribev2 layer stays cached: ROI atlas needs
    # nibabel; §5.2 stimuli come from scikit-image's bundled sample photos.
    .pip_install("nibabel", "scikit-image")
    .env({"MNE_DATASETS_SAMPLE_PATH": "/mne", "MNE_DATA": "/mne"})
)
mne_vol = modal.Volume.from_name("tribe-mne", create_if_missing=True)

# A Volume caches the HF checkpoint + backbone weights across runs so we only
# download the ~10GB of model assets once.
cache_vol = modal.Volume.from_name("tribe-cache", create_if_missing=True)
# Persist HF-downloaded backbone weights across runs so they download only once.
hf_vol = modal.Volume.from_name("tribe-hf", create_if_missing=True)
CACHE = "/cache"
HF_CACHE = "/root/.cache/huggingface"


def _make_sample_video(path: str, duration_s: int = 90, fps: int = 4, size: int = 224):
    """A synthetic clip: moving noise frames + a sine tone. Enough to exercise
    the video+audio backbones and produce ~duration_s TR rows. Content is
    irrelevant for the gate and for a linearity read on internal structure."""
    import numpy as np
    import soundfile as sf
    from moviepy import ImageSequenceClip, AudioFileClip

    sr = 16000
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sf.write("/tmp/tone.wav", 0.2 * np.sin(2 * np.pi * 220 * t), sr)

    rng = np.random.default_rng(0)
    n = duration_s * fps
    # smoothly drifting noise so consecutive frames are correlated (gives the
    # temporal transformer something non-trivial to fuse)
    base = rng.random((size, size, 3))
    frames = []
    for i in range(n):
        base = 0.9 * base + 0.1 * rng.random((size, size, 3))
        frames.append((base * 255).astype("uint8"))
    clip = ImageSequenceClip(frames, fps=fps).with_audio(AudioFileClip("/tmp/tone.wav"))
    clip.write_videofile(path, fps=fps, audio_codec="aac", logger=None)


def _linear_cka(a, b):
    import numpy as np

    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    cross = a.T @ b
    num = float(np.sum(cross ** 2))
    den = float(np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro"))
    return num / den if den else 0.0


def _cv_ridge_r2(src, tgt, alpha=1000.0, n_splits=5):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=min(n_splits, len(src)), shuffle=True, random_state=0)
    scores = []
    for tr, te in kf.split(src):
        sc = StandardScaler().fit(src[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(src[tr]), tgt[tr])
        scores.append(r2_score(tgt[te], m.predict(sc.transform(src[te])),
                               multioutput="uniform_average"))
    return float(np.mean(scores))


@app.function(image=image, gpu="A10G", timeout=3600,
              volumes={CACHE: cache_vol})
def run_gate_and_e0():
    import numpy as np
    import pandas as pd
    import torch
    from einops import rearrange
    from tribev2 import TribeModel
    from tribev2.demo_utils import get_audio_and_text_events

    torch.manual_seed(0)
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------ #
    # 1. sample video + events (no transcription)                        #
    # ------------------------------------------------------------------ #
    vid = "/tmp/sample.mp4"
    _make_sample_video(vid)
    event = {"type": "Video", "filepath": vid, "start": 0,
             "timeline": "default", "subject": "default"}
    events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
    print("events rows:", len(events))

    # ------------------------------------------------------------------ #
    # 2. load model                                                      #
    # ------------------------------------------------------------------ #
    xp = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
    model = xp._model
    model.eval()
    has_lr = hasattr(model, "low_rank_head")
    print("low_rank_head present:", has_lr,
          "| n_output_timesteps:", model.n_output_timesteps,
          "| training:", model.training)

    # ------------------------------------------------------------------ #
    # 3. hooks                                                            #
    # ------------------------------------------------------------------ #
    acts = {}
    handles = [
        model.encoder.register_forward_pre_hook(
            lambda m, i: acts.__setitem__("h_proj", i[0].detach())),
        model.encoder.register_forward_hook(
            lambda m, i, o: acts.__setitem__("h_enc", o.detach())),
    ]
    if hasattr(model, "combiner"):
        handles.append(model.combiner.register_forward_pre_hook(
            lambda m, i: acts.__setitem__("h_agg", i[0].detach())))
    if has_lr:
        handles.append(model.low_rank_head.register_forward_hook(
            lambda m, i, o: acts.__setitem__("h_lr", o.detach())))

    def to_tr(h, n):  # mirror pooler
        h = h.transpose(1, 2)
        h = torch.nn.functional.adaptive_avg_pool1d(h.float(), n)
        return h.transpose(1, 2)

    def recon_from_h_enc(h_enc, subj):  # replicate forward() lines 170-175
        x = h_enc.transpose(1, 2)              # B,H,T
        if has_lr:
            x = model.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
        x = model.predictor(x, subj)           # B,O,T
        return model.pooler(x)                 # B,O,T'

    # ------------------------------------------------------------------ #
    # 4. one pass = gate + capture (replicates demo_utils.predict loop)  #
    # ------------------------------------------------------------------ #
    n_out_t = model.n_output_timesteps
    loader = xp.data.get_loaders(events=events, split_to_build="all")["all"]
    caps = {k: [] for k in ("h_agg", "h_proj", "h_enc", "h_lr")}
    max_gate_err = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(model.device)
            acts.clear()
            segs = []
            for segment in batch.segments:
                for t in np.arange(0, segment.duration - 1e-2, xp.data.TR):
                    segs.append(segment.copy(offset=t, duration=xp.data.TR))
            keep = (np.array([len(s.ns_events) > 0 for s in segs])
                    if xp.remove_empty_segments else np.ones(len(segs), bool))

            preds = model(batch)                       # B,O,T'
            subj = batch.data.get("subject_id", None)
            recon = recon_from_h_enc(acts["h_enc"], subj)

            p = rearrange(preds.float().cpu().numpy(), "b d t -> (b t) d")[keep]
            r = rearrange(recon.float().cpu().numpy(), "b d t -> (b t) d")[keep]
            max_gate_err = max(max_gate_err, float(np.abs(p - r).max()))

            for k in caps:
                if k in acts:
                    h = to_tr(acts[k], n_out_t)
                    h = rearrange(h, "b t d -> (b t) d").cpu().numpy()
                    assert h.shape[0] == len(keep), f"{k} alignment broke"
                    caps[k].append(h[keep])
    for h in handles:
        h.remove()

    emb = {k: np.concatenate(v) for k, v in caps.items() if v}
    shapes = {k: list(v.shape) for k, v in emb.items()}
    print("captured:", shapes, "| gate max|Δ|:", max_gate_err)

    gate_passed = max_gate_err < 1e-3

    # ------------------------------------------------------------------ #
    # 5. preliminary E0 (H0 / linearity)                                 #
    # ------------------------------------------------------------------ #
    e0 = {}
    if gate_passed and "h_enc" in emb:
        for src in ("h_agg", "h_proj"):
            if src in emb:
                e0[f"{src}->h_enc"] = {
                    "cka": _linear_cka(emb[src], emb["h_enc"]),
                    "ridge_r2": _cv_ridge_r2(emb[src], emb["h_enc"]),
                }
        if has_lr and "h_lr" in emb:
            e0["cka(h_enc,h_lr)"] = _linear_cka(emb["h_enc"], emb["h_lr"])

    result = {"gate_passed": gate_passed, "gate_max_abs_err": max_gate_err,
              "has_low_rank_head": has_lr, "shapes": shapes,
              "n_rows": int(next(iter(emb.values())).shape[0]) if emb else 0,
              "E0_preliminary": e0}

    # Persist to the Volume so a detached run's result survives client disconnect.
    import json
    with open(f"{CACHE}/result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    cache_vol.commit()
    print("\n===== RESULT =====")
    print(json.dumps(result, indent=2, default=float))
    return result


def _e0_verdict(r2, cka):
    if r2 > 0.95 and cka > 0.95:
        return "H0_CONFIRMED", "TRIBE adds ~a rotation; Pipeline B is a negative result (stop)."
    if r2 < 0.5:
        return "LARGE_TRANSFORM", "Large transformation; proceed but re-check extraction."
    return "PROCEED", "TRIBE adds real nonlinear/temporal structure; proceed."


# Real natural-video corpus (Google public test bucket; direct downloads, real
# motion/scenes/audio). ~1250s total -> ~1250 TR rows > hidden dim (1152), so the
# concat->h_enc linear map is well-posed. E0 is an internal-structure test, so
# contamination is irrelevant here (no ground truth involved).
CORPUS = [
    ("bbb", "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"),
    ("ed", "https://archive.org/download/ElephantsDream/ed_1024_512kb.mp4"),
]


def _download(url, path):
    """Fetch with a browser UA (default urllib UA gets 403'd) + follow redirects."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


@app.function(image=image, gpu="A10G", timeout=18000,
              volumes={CACHE: cache_vol, HF_CACHE: hf_vol})
def run_e0_corpus():
    import json
    import numpy as np
    import pandas as pd
    import torch
    from einops import rearrange
    from tribev2 import TribeModel
    from tribev2.demo_utils import get_audio_and_text_events

    torch.manual_seed(0)
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))

    xp = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
    model = xp._model
    model.eval()
    has_lr = hasattr(model, "low_rank_head")
    n_out_t = model.n_output_timesteps

    # ------------------------------------------------------------------ #
    # hooks: encoder in/out, combiner in (h_agg), per-modality raw feats  #
    # ------------------------------------------------------------------ #
    acts = {}
    raw = {}
    handles = [
        model.encoder.register_forward_pre_hook(
            lambda m, i: acts.__setitem__("h_proj", i[0].detach())),
        model.encoder.register_forward_hook(
            lambda m, i, o: acts.__setitem__("h_enc", o.detach())),
    ]
    if hasattr(model, "combiner"):
        handles.append(model.combiner.register_forward_pre_hook(
            lambda m, i: acts.__setitem__("h_agg", i[0].detach())))
    if has_lr:
        handles.append(model.low_rank_head.register_forward_hook(
            lambda m, i, o: acts.__setitem__("h_lr", o.detach())))
    # raw backbone features = input to each per-modality projector (post
    # layer-aggregation, pre-projection) == design's concat_backbone building block
    for name, proj in model.projectors.items():
        handles.append(proj.register_forward_pre_hook(
            (lambda nm: (lambda m, i: raw.__setitem__(nm, i[0].detach())))(name)))

    def to_tr(h, n):
        h = h.transpose(1, 2)
        h = torch.nn.functional.adaptive_avg_pool1d(h.float(), n)
        return h.transpose(1, 2)

    def recon_from_h_enc(h_enc, subj):
        x = h_enc.transpose(1, 2)
        if has_lr:
            x = model.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
        x = model.predictor(x, subj)
        return model.pooler(x)

    caps = {k: [] for k in ("h_agg", "h_proj", "h_enc", "h_lr", "concat_raw")}
    max_gate_err = 0.0
    modality_order = sorted(model.projectors.keys())
    print("modalities:", modality_order, "| low_rank_head:", has_lr)

    for clip_name, url in CORPUS:
        path = f"/tmp/{clip_name}.mp4"
        print(f"downloading {clip_name} ...", flush=True)
        _download(url, path)
        event = {"type": "Video", "filepath": path, "start": 0,
                 "timeline": "default", "subject": "default"}
        events = get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)
        loader = xp.data.get_loaders(events=events, split_to_build="all")["all"]
        print(f"[{clip_name}] extracting features ...", flush=True)

        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(model.device)
                acts.clear(); raw.clear()
                segs = []
                for segment in batch.segments:
                    for t in np.arange(0, segment.duration - 1e-2, xp.data.TR):
                        segs.append(segment.copy(offset=t, duration=xp.data.TR))
                keep = (np.array([len(s.ns_events) > 0 for s in segs])
                        if xp.remove_empty_segments else np.ones(len(segs), bool))

                preds = model(batch)
                subj = batch.data.get("subject_id", None)
                recon = recon_from_h_enc(acts["h_enc"], subj)
                p = rearrange(preds.float().cpu().numpy(), "b d t -> (b t) d")[keep]
                r = rearrange(recon.float().cpu().numpy(), "b d t -> (b t) d")[keep]
                max_gate_err = max(max_gate_err, float(np.abs(p - r).max()))

                for k in ("h_agg", "h_proj", "h_enc", "h_lr"):
                    if k in acts:
                        h = to_tr(acts[k], n_out_t)
                        h = rearrange(h, "b t d -> (b t) d").cpu().numpy()
                        assert h.shape[0] == len(keep)
                        caps[k].append(h[keep])
                # raw concat across the modalities that actually fired
                present = [m for m in modality_order if m in raw]
                if present:
                    parts = [rearrange(to_tr(raw[m], n_out_t), "b t d -> (b t) d").cpu().numpy()
                             for m in present]
                    concat = np.concatenate(parts, axis=1)
                    caps["concat_raw"].append(concat[keep])
        cache_vol.commit()

    for h in handles:
        h.remove()

    emb = {k: np.concatenate(v) for k, v in caps.items() if v}
    shapes = {k: list(v.shape) for k, v in emb.items()}
    print("captured:", shapes, "| gate max|Δ|:", max_gate_err, flush=True)
    np.savez(f"{CACHE}/e0_embeddings.npz", **emb)

    # ------------------------------------------------------------------ #
    # E0 proper: CKA + CV-ridge R² for each control -> h_enc             #
    # ------------------------------------------------------------------ #
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    def cka(a, b):
        a = a - a.mean(0, keepdims=True); b = b - b.mean(0, keepdims=True)
        num = float(np.sum((a.T @ b) ** 2))
        den = float(np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro"))
        return num / den if den else 0.0

    def cv_ridge_r2(src, tgt, alpha=1000.0):
        kf = KFold(5, shuffle=True, random_state=0); s = []
        for tr, te in kf.split(src):
            sc = StandardScaler().fit(src[tr])
            m = Ridge(alpha=alpha).fit(sc.transform(src[tr]), tgt[tr])
            s.append(r2_score(tgt[te], m.predict(sc.transform(src[te])),
                              multioutput="uniform_average"))
        return float(np.mean(s))

    tgt = emb["h_enc"]
    e0 = {}
    for src in ("concat_raw", "h_agg", "h_proj"):
        if src in emb:
            r2 = cv_ridge_r2(emb[src], tgt)
            c = cka(emb[src], tgt)
            verdict, meaning = _e0_verdict(r2, c)
            e0[f"{src}->h_enc"] = {"cka": c, "ridge_r2": r2,
                                   "verdict": verdict, "meaning": meaning}
    if has_lr and "h_lr" in emb:
        e0["cka(h_enc,h_lr)"] = cka(emb["h_enc"], emb["h_lr"])

    result = {"n_rows": int(tgt.shape[0]), "shapes": shapes,
              "gate_passed": bool(max_gate_err < 1e-3),
              "gate_max_abs_err": max_gate_err,
              "corpus": [c for c, _ in CORPUS], "E0": e0}
    with open(f"{CACHE}/e0_result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    cache_vol.commit()
    print("\n===== E0 RESULT =====")
    print(json.dumps(result, indent=2, default=float))
    return result


# --------------------------------------------------------------------------- #
# §5.2 image-path sanity check: do static-image clips yield coherent brain     #
# maps (strong occipital/ventral, weak motion)? Gates whether E1 can use image #
# ground truth (THINGS) or must pivot to video similarity judgments.           #
# --------------------------------------------------------------------------- #
ROI_GROUPS = {
    "early_visual": ["V1", "V2", "V3", "V4"],
    "ventral_category": ["FFC", "PIT", "VVC", "VMV*", "PHA*", "TF"],
    "motion": ["MT", "MST", "FST", "V4t", "V3A", "V6", "V7"],
    "auditory": ["A1", "A4", "A5", "LBelt", "MBelt", "PBelt"],
}


def _static_video(img, path, duration=6, fps=4):
    """One still image -> a static clip with a faint tone (so the audio path has
    input). This is the out-of-distribution 'image as video' hack the design
    flags: V-JEPA2 sees no motion."""
    import numpy as np
    import soundfile as sf
    from moviepy import ImageClip, AudioFileClip

    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sf.write("/tmp/tone.wav", 0.05 * np.sin(2 * np.pi * 220 * t), sr)
    clip = ImageClip(np.asarray(img), duration=duration).with_fps(fps)
    clip = clip.with_audio(AudioFileClip("/tmp/tone.wav"))
    clip.write_videofile(path, fps=fps, audio_codec="aac", logger=None)


@app.function(image=image, gpu="A10G", timeout=3600,
              volumes={CACHE: cache_vol, HF_CACHE: hf_vol, "/mne": mne_vol})
def run_image_roi_check():
    import json
    import numpy as np
    import pandas as pd
    import torch
    from skimage import data as skdata
    from tribev2 import TribeModel
    from tribev2.demo_utils import get_audio_and_text_events
    from tribev2.utils import summarize_by_roi, get_hcp_roi_indices, get_hcp_labels

    xp = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
    model = xp._model
    model.eval()

    # tribev2's get_topk_rois does np.array(dict_keys) -> 0-d array (bug), so
    # build the label list ourselves; summarize_by_roi returns values in this
    # same key order.
    roi_labels = list(get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both").keys())

    def top_rois(brain_1d, k=12):
        vals = summarize_by_roi(brain_1d)
        order = np.argsort(vals)[::-1][:k]
        return [roi_labels[i] for i in order]

    stimuli = {"face_astronaut": skdata.astronaut(),
               "object_coffee": skdata.coffee(),
               "object_rocket": skdata.rocket()}

    # Precompute ROI-group vertex indices once.
    group_idx = {}
    for g, rois in ROI_GROUPS.items():
        idxs = []
        for r in rois:
            try:
                idxs.append(get_hcp_roi_indices(r))
            except ValueError:
                pass
        group_idx[g] = np.concatenate(idxs) if idxs else np.array([], dtype=int)

    out = {}
    for name, img in stimuli.items():
        path = f"/tmp/{name}.mp4"
        _static_video(img, path)
        events = get_audio_and_text_events(
            pd.DataFrame([{"type": "Video", "filepath": path, "start": 0,
                           "timeline": "default", "subject": "default"}]),
            audio_only=True)
        preds, segs = xp.predict(events, verbose=False)   # (n_kept, 20484)
        brain = preds.mean(axis=0)                         # 1D over vertices
        # z-score across vertices so group means are comparable across stimuli
        bz = (brain - brain.mean()) / (brain.std() + 1e-8)
        group_means = {g: float(bz[idx].mean()) if len(idx) else None
                       for g, idx in group_idx.items()}
        out[name] = {
            "top_rois": top_rois(brain, k=12),
            "roi_group_z": group_means,
            "visual_minus_motion": (group_means["early_visual"] - group_means["motion"]),
            "ventral_minus_motion": (group_means["ventral_category"] - group_means["motion"]),
        }
        print(name, "top:", out[name]["top_rois"][:6],
              "| vis-mot:", round(out[name]["visual_minus_motion"], 2), flush=True)

    # Coherent image path => visual & ventral consistently exceed motion.
    coherent = all(v["visual_minus_motion"] > 0 and v["ventral_minus_motion"] > 0
                   for v in out.values())
    result = {"coherent_image_path": bool(coherent), "per_stimulus": out,
              "interpretation": (
                  "Image path VIABLE for E1 (static clips drive visual/ventral > "
                  "motion)." if coherent else
                  "Image path INCOHERENT — pivot E1 to video similarity ground truth (§5.2).")}
    with open(f"{CACHE}/roi_check_result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    cache_vol.commit()
    print("\n===== ROI CHECK =====")
    print(json.dumps(result, indent=2, default=float))
    return result


@app.local_entrypoint()
def main():
    import json
    res = run_gate_and_e0.remote()
    print("\n===== RESULT =====")
    print(json.dumps(res, indent=2))


@app.local_entrypoint()
def e0():
    import json
    res = run_e0_corpus.remote()
    print("\n===== E0 RESULT =====")
    print(json.dumps(res, indent=2))
