"""Integrate render-mode pilot + scale label sets -> one mining-head dataset.

Combines render_mode_scale_v1 (1000) with the reusable pilot roster labels
(399: pilot minus exp_smoothing[too_close_to_smooth flag] minus trap_circle
[dropped mode] minus direct_trap_screen [pre-rolloff, superseded]). Applies the
scale split map (location->side) to the pilot rasters so they inherit the same
eval/train split. Emits train.jsonl / eval.jsonl in the wallpaper-head harness
row format + a summary.json.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # tools/render_mode_pilot/ -> repo root
BATCHES = ROOT / "data" / "render_mode_corpus" / "batches"
PILOT_DIR = BATCHES / "2026-07-10_render_mode_pilot_v1"
SCALE_DIR = BATCHES / "2026-07-11_render_mode_scale_v1"
PILOT_LABELS = ROOT / "labels" / "render_mode_pilot_v1.json"
SCALE_LABELS = ROOT / "labels" / "render_mode_scale_v1.json"
SPLIT_MAP = ROOT / "data" / "render_mode_corpus" / "rms_split_map.json"
OUT_DIR = ROOT / "data" / "render_mode_corpus" / "dataset_v1"

DROP_MODES = {"trap_circle", "direct_trap_screen"}  # pilot-only drops
EVAL_FRAC = 0.4  # matches split map; used only for deterministic orphan fallback


def load_batch(batch_dir, labels_path):
    labels = json.loads(labels_path.read_text())
    rows = []
    for line in (batch_dir / "images.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r["image_id"]
        if iid not in labels:
            raise ValueError(f"{batch_dir.name}: {iid} unlabeled")
        r["_label"] = labels[iid]
        rows.append(r)
    extra = set(labels) - {r["image_id"] for r in rows}
    if extra:
        raise ValueError(f"{batch_dir.name}: {len(extra)} labels w/o row")
    return rows


def orphan_side(loc: str) -> str:
    h = int(hashlib.md5(loc.encode()).hexdigest(), 16) % 1000
    return "eval" if h < EVAL_FRAC * 1000 else "train"


def main():
    smap = json.load(open(SPLIT_MAP))["location_side"]
    pilot = load_batch(PILOT_DIR, PILOT_LABELS)
    scale = load_batch(SCALE_DIR, SCALE_LABELS)

    orphans = []

    def side_of(loc):
        if loc in smap:
            return smap[loc], False
        s = orphan_side(loc)
        orphans.append((loc, s))
        return s, True

    dataset = []  # unified rows

    # --- scale: all 1000 in; split from map (cross-check against stamped side) ---
    stamp_mismatch = []
    for r in scale:
        loc = r["provenance"]["location_key"]
        side, orph = side_of(loc)
        stamped = r["provenance"].get("split_side")
        if stamped is not None and stamped != side:
            stamp_mismatch.append((r["image_id"], stamped, side))
        dataset.append(_row(r, "scale", side, PILOT=False))

    # --- pilot reusables: drop flagged (exp_smoothing) + dropped modes ---
    pilot_drop = Counter()
    for r in pilot:
        mode = r["render"]["render_mode"]
        if r.get("too_close_to_smooth"):
            pilot_drop["too_close_to_smooth(exp_smoothing)"] += 1
            continue
        if mode in DROP_MODES:
            pilot_drop[f"mode:{mode}"] += 1
            continue
        loc = r["provenance"]["location_key"]
        side, orph = side_of(loc)
        dataset.append(_row(r, "pilot", side, PILOT=True))

    # --- location-disjointness across the whole union ---
    loc_sides = defaultdict(set)
    for d in dataset:
        loc_sides[d["location_key"]].add(d["split"])
    spanning = {l: s for l, s in loc_sides.items() if len(s) > 1}

    # --- verify crops exist ---
    missing = [d["image_id"] for d in dataset if not (ROOT / d["crop"]).exists()]

    # --- write ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = [d for d in dataset if d["split"] == "train"]
    ev = [d for d in dataset if d["split"] == "eval"]
    with open(OUT_DIR / "train.jsonl", "w") as fh:
        for d in train:
            fh.write(json.dumps(d) + "\n")
    with open(OUT_DIR / "eval.jsonl", "w") as fh:
        for d in ev:
            fh.write(json.dumps(d) + "\n")

    # --- summary tables ---
    def tier_hist(rows):
        c = Counter(d["label"] for d in rows)
        return {t: c.get(t, 0) for t in (1, 2, 3)}

    def per_mode(rows):
        m = defaultdict(Counter)
        for d in rows:
            m[d["mode"]][d["label"]] += 1
        return {mode: {t: c.get(t, 0) for t in (1, 2, 3)} | {"total": sum(c.values())}
                for mode, c in sorted(m.items())}

    summary = {
        "combined_count": len(dataset),
        "by_batch": dict(Counter(d["batch"] for d in dataset)),
        "pilot_dropped": dict(pilot_drop),
        "pilot_reusable": sum(1 for d in dataset if d["batch"] == "pilot"),
        "orphans_assigned": [{"location_key": l, "side": s} for l, s in orphans],
        "scale_stamp_vs_map_mismatch": stamp_mismatch,
        "split": {
            "train": {"n": len(train), "loc": len({d["location_key"] for d in train}),
                      "tier": tier_hist(train)},
            "eval": {"n": len(ev), "loc": len({d["location_key"] for d in ev}),
                     "tier": tier_hist(ev)},
        },
        "location_disjoint": len(spanning) == 0,
        "spanning_locations": list(spanning.items())[:10],
        "eval_q3_count": tier_hist(ev)[3],
        "overall_tier": tier_hist(dataset),
        "per_mode_overall": per_mode(dataset),
        "per_mode_train": per_mode(train),
        "per_mode_eval": per_mode(ev),
        "crops_missing": missing,
        "paths": {"train": str((OUT_DIR / "train.jsonl").relative_to(ROOT)),
                  "eval": str((OUT_DIR / "eval.jsonl").relative_to(ROOT))},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _row(r, batch, side, PILOT):
    iid = r["image_id"]
    rd = r["render"]
    prov = r["provenance"]
    batch_dir = "2026-07-10_render_mode_pilot_v1" if PILOT else "2026-07-11_render_mode_scale_v1"
    crop = f"data/render_mode_corpus/batches/{batch_dir}/crops/{iid}.jpg"
    return {
        "image_id": iid,
        "crop": crop,
        "label": int(r["_label"]),
        "split": side,
        "batch": batch,
        "mode": rd["render_mode"],
        "mode_kind": prov.get("mode_kind"),
        "mode_params": prov.get("mode_params", {}),
        "family": prov.get("family"),
        "location_key": prov["location_key"],
        "render": rd,
        "color_params": prov.get("color_params", {}),
    }


if __name__ == "__main__":
    main()
