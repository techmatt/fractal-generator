"""Pre-label preprocess: score a head batch with the deployed wallpaper head (v2)
and stamp the prediction, then re-sort images.jsonl for descending-quality labeling.

Runs AFTER a head batch finishes, BEFORE labeling. Purpose is labeling speed; the
anchoring bias is accepted (see prompts/prompt_prelabel_score_sort.md).

What it does (inference + stamp + sort ONLY — no re-render, split, merge, retrain):
  * Runs head-v2 on every crop through the EXACT deploy transform
    (classifier.data.Transform(train=False): 1280x720 -> 384x224 bicubic stretch +
    normalize), the deterministic mirror of present.rs's JPG path.
  * CORN conditional sigmoids -> marginals (cumprod). Continuous quality readout =
    EXPECTED TIER = 1 + Σ_k marg[:,k]  (monotone across all four tiers 1..4).
  * Stamps a NEW top-level `head_v2` block on each row {pred, p_ge2, p_ge3, p_ge4,
    score, ckpt} plus a flat `head_v2_pred` scalar (the sort key). Existing blocks
    (render / provenance / label) and all existing stamps are left untouched.
  * Rewrites images.jsonl in DESCENDING head_v2_pred order (best first). The UI
    (wallpaper_label.html) honors file order when rows carry head_v2_pred.

    uv run python tools/wallpaper/prelabel_score_v2.py \
        --batch 2026-07-09_wallpaper_headbatch_dramatic_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from classifier.inference import load_scorer  # noqa: E402
CKPT = REPO / "data/wallpaper_head/v2/model_best.pt"
BATCHES = REPO / "data/wallpaper_corpus/batches"


@torch.no_grad()
def score_batch(scorer, crop_paths, batch_size: int = 64):
    """Returns (cond, marg, ssum) aligned to crop_paths.

    cond[:,k] = σ(logit_k) = CORN CONDITIONAL P(rank>k | rank>=k)
    marg      = cumprod(cond) = marginal P(rank>=k+1) = P(tier>=k+2)  (k=0->P>=2 ...)
    ssum      = Σ σ(logit_k)  (raw CORN score, kept for reference)
    """
    from PIL import Image

    dev = scorer.device
    conds, ssums = [], []
    buf = []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(dev)
        with torch.autocast(device_type=dev.split(":")[0], enabled=(dev != "cpu")):
            logits = scorer.model(x)
        logits = logits.float()
        conds.append(torch.sigmoid(logits).cpu().numpy())
        from classifier.model import score_from_logits
        ssums.append(score_from_logits(logits, "ordinal").cpu().numpy())
        buf.clear()

    for p in crop_paths:
        with Image.open(p) as im:
            im.load()
            buf.append(scorer.transform(im.convert("RGB")))
        if len(buf) == batch_size:
            flush()
    flush()

    cond = np.concatenate(conds, axis=0).astype(np.float64)
    ssum = np.concatenate(ssums, axis=0).astype(np.float64)
    marg = np.cumprod(cond, axis=1)
    return cond, marg, ssum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="batch dir name under data/wallpaper_corpus/batches/")
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()

    batch_dir = BATCHES / args.batch
    images_path = batch_dir / "images.jsonl"
    crops_dir = batch_dir / "crops"
    rows = [json.loads(l) for l in images_path.read_text().splitlines() if l.strip()]
    print(f"[load] {len(rows)} rows from {images_path}")

    scorer = load_scorer(args.ckpt)
    print(f"[model] v2 loaded on {scorer.device} · target={scorer.target} "
          f"· K={scorer.config.get('num_classes')} · geometry={scorer.config['geometry']}")

    ids = [r["image_id"] for r in rows]
    paths = [crops_dir / f"{iid}.jpg" for iid in ids]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"[err] {len(missing)} crops missing, e.g. {missing[0]}")

    cond, marg, ssum = score_batch(scorer, paths)
    # expected tier E[tier] = 1 + Σ_k P(rank>k) = 1 + Σ marginals ; in [1, K]
    pred = 1.0 + marg.sum(axis=1)
    ckpt_rel = str(Path(args.ckpt).relative_to(REPO)).replace("\\", "/")

    for i, r in enumerate(rows):
        r["head_v2"] = {
            "pred": float(pred[i]),          # expected tier, monotone across all 4 tiers
            "p_ge2": float(marg[i, 0]),
            "p_ge3": float(marg[i, 1]),
            "p_ge4": float(marg[i, 2]),
            "score": float(ssum[i]),         # raw Σσ(logit_k), reference only
            "ckpt": ckpt_rel,
        }
        r["head_v2_pred"] = float(pred[i])   # flat sort key the UI honors

    # descending predicted quality (best first); tie-break image_id for stability
    order = sorted(range(len(rows)), key=lambda i: (-pred[i], rows[i]["image_id"]))
    rows_sorted = [rows[i] for i in order]

    with images_path.open("w", encoding="utf-8") as f:
        for r in rows_sorted:
            f.write(json.dumps(r) + "\n")
    print(f"[write] re-sorted {len(rows_sorted)} rows -> {images_path} (descending head_v2_pred)")

    # report + spot-check
    p = pred
    qs = np.quantile(p, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"[pred] E[tier] min/q25/med/q75/max = {np.round(qs, 3).tolist()}")

    def bucket(r):
        pv = r.get("provenance", {})
        return pv.get("curation_bucket") or ("bad_inject" if pv.get("bad_rank") is not None else "?")

    print("\n[top 8 predicted]")
    for r in rows_sorted[:8]:
        pv = r.get("provenance", {})
        print(f"  {r['head_v2_pred']:.3f}  {r['image_id']:<12} {bucket(r):<10} "
              f"{r['render'].get('fractal_type',''):<12} {r['render'].get('palette','')}")
    print("\n[bottom 8 predicted]")
    for r in rows_sorted[-8:]:
        pv = r.get("provenance", {})
        print(f"  {r['head_v2_pred']:.3f}  {r['image_id']:<12} {bucket(r):<10} "
              f"{r['render'].get('fractal_type',''):<12} {r['render'].get('palette','')}")

    # sanity: where do bad_inject rows land vs topk?
    buckets = np.array([bucket(r) for r in rows_sorted], dtype=object)
    n = len(rows_sorted)
    for b in ("topk", "bad_inject"):
        idx = np.where(buckets == b)[0]
        if len(idx):
            print(f"\n[{b}] n={len(idx)} · rank quantiles (0=top) "
                  f"q25/med/q75 = {np.round(np.quantile(idx, [.25,.5,.75])/n, 3).tolist()} of {n}")


if __name__ == "__main__":
    main()
