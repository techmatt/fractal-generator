#!/usr/bin/env python
"""q4 stage-1 — build a fresh, G-AIMED next-to-label batch.

The 2026-07-23 window batch is fully labeled (211 filter-survivors, 0 left). The
next labeling pass is NOT its leftovers — it is a fresh candidate sweep scored by
the linear goodness field G (q4_stage1_linear_fit, tier T2_cells), so Matt labels
where the model is *uncertain* (teaches the boundary) plus a confident-accept slug
(audits precision) plus a uniform-random control (audits confident-and-wrong).

Candidate source = BOTH:
  * dense-grid harvest over the 30 existing fields (out/q4_stage1/fields/) — cheap,
    no new renders; deduped against the already-labeled windows so the batch is new.
  * a handful of NEW minibrots (fresh Newton nuclei at unused ∂M anchors) — diversity.

Pipeline: train G on the current 228 accept/reject labels -> harvest v2-survivor
windows (dense sweep, per-scale NMS) with G,p -> select slugs (uncertain / top_g /
control) with per-minibrot caps -> render crops -> a NEW registered store batch
(schema-parity with q4_window_reader) -> label in tools/viz/q4_window_label.html.

Run:  uv run python -m tools.studies.q4_stage1_g_batch build
      uv run python -m tools.studies.q4_stage1_g_batch serve   # launch the labeler
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mpmath as mp  # noqa: E402
from tools.corpus import q4_window_reader as qr  # noqa: E402
from tools.sourcing import deep_center_finder as dcf  # noqa: E402
from tools.studies import q4_stage1_labelset as LS  # noqa: E402
from tools.studies import q4_stage1_linear_fit as LF  # noqa: E402

NEW_BATCH_ID = "2026-07-23_q4_g_aimed"
NEW_STORE = ROOT / "data" / "q4_window_corpus" / "batches" / NEW_BATCH_ID
NEW_CROPS = NEW_STORE / "crops"
GEN_OUT = ROOT / "out" / "q4_stage1" / "g_batch"

TIER, C = "T2_cells", 2.0        # the fit's chosen tier + C
SCALES = LS.SCALES               # [0.06, 0.09, 0.14]
HARVEST_STRIDE_FRAC = 0.22       # coarse-ish (NMS dedups); bounds featurize cost
NMS_IOU = 0.25                   # dedup among harvested candidates
DEDUP_EXISTING_IOU = 0.30        # drop candidates overlapping an already-labeled window
N_NEW_MB = 6
SEED = 0
TARGET = dict(uncertain=64, top_g=24, control=24)
PER_MB_CAP = 8                   # per-minibrot cap within each slug (diversity)
WORKERS = 4                      # hard cap (project rule)

# NEW ∂M anchors, distinct from LS.ANCHORS -> genuinely different minibrots.
NEW_ANCHORS = [
    (-1.76850, 0.00042, "west_p3"),
    (0.43700, 0.34100, "ne_far"),
    (-0.62500, 0.42500, "nw_mid"),
    (0.37900, 0.20000, "ne2"),
    (-1.25400, 0.38000, "west_up"),
    (-0.10100, 0.95630, "north_seahorse"),
    (0.25400, 0.00030, "elephant_axis"),
    (-0.79350, 0.16500, "seahorse_up"),
]
PERIODS = list(range(4, 80))
DEDUP_DPS = 22


# --------------------------------------------------------------------------- #
# Model.                                                                       #
# --------------------------------------------------------------------------- #
def train_model():
    rows = LF.build_dataset()
    _, _, sc, clf = LF.surviving_weights(rows, TIER, C)
    keys = LF.FEATURES[TIER]
    acc_idx = list(clf.classes_).index(1)
    return sc, clf, keys, acc_idx


# --------------------------------------------------------------------------- #
# New minibrots (fresh Newton nuclei; dedup vs the existing 30).               #
# --------------------------------------------------------------------------- #
def gen_new_minibrots():
    existing = LS.load_minibrots()
    seen = {(round(float(m["cx"]), 6), round(float(m["cy"]), 6)) for m in existing}
    mp.mp.dps = 60
    tol = mp.mpf(10) ** (-(mp.mp.dps - 6))
    found = {}
    for ar, ai, aname in NEW_ANCHORS:
        seed = mp.mpc(ar, ai)
        for p in PERIODS:
            r = dcf.newton_nucleus(seed, p)
            if not r.converged:
                continue
            if not LS._minimal_period(r.c, p, tol):
                continue
            size = dcf.nucleus_size_estimate(r.c, p)
            sabs = float(abs(size)) if size != 0 else 0.0
            if not (LS.SIZE_LO <= sabs <= LS.SIZE_HI):
                continue
            key = (mp.nstr(r.c.real, DEDUP_DPS), mp.nstr(r.c.imag, DEDUP_DPS))
            if key in found:
                continue
            dc = dcf.make_deep_center(r)
            rc = (round(float(dc.cx), 6), round(float(dc.cy), 6))
            if any(abs(rc[0] - e[0]) < 1e-5 and abs(rc[1] - e[1]) < 1e-5 for e in seen):
                continue
            found[key] = dict(anchor=aname, period=p, cx=dc.cx, cy=dc.cy,
                              fw=dc.fw_suggest, maxiter=dc.render_maxiter, size=sabs)
            seen.add(rc)
            break                          # one fresh minibrot per anchor
    recs = list(found.values())
    recs.sort(key=lambda d: d["period"])
    recs = recs[:N_NEW_MB]
    for i, d in enumerate(recs):
        d["id"] = f"nb{i:02d}_p{d['period']:02d}"
    GEN_OUT.mkdir(parents=True, exist_ok=True)
    (GEN_OUT / "new_minibrots.json").write_text(json.dumps(recs, indent=2))
    print(f"new minibrots: {len(recs)}  periods {[d['period'] for d in recs]}")
    # dump fields (into the shared fields dir so load_field_values finds them)
    LS.FIELDS.mkdir(parents=True, exist_ok=True)
    for d in recs:
        b = LS.FIELDS / f"{d['id']}.bin"
        if b.exists() and b.with_suffix(".json").exists():
            continue
        LS.dump_field(d, b)
        print(f"  field {d['id']} fw={d['fw']} maxiter={d['maxiter']}")
    return recs


# --------------------------------------------------------------------------- #
# Harvest (one minibrot -> deduped G-scored candidates). Top-level = picklable. #
# --------------------------------------------------------------------------- #
def harvest_one(args):
    mb_id, sc, clf, keys, acc_idx, existing_boxes = args
    field, fw, fh = LS.load_field_values(mb_id)
    out = []
    for s in SCALES:
        Wp = max(8, int(round(s * fw)))
        Hp = max(8, int(round(Wp * 9 / 16)))
        if Hp >= fh or Wp >= fw:
            continue
        st = max(4, int(round(HARVEST_STRIDE_FRAC * Wp)))
        local = []
        for y in range(0, fh - Hp + 1, st):
            for x in range(0, fw - Wp + 1, st):
                f = LF.featurize(field[y:y + Hp, x:x + Wp])
                if f is None or LF._v2_drop(f):
                    continue
                u, v, uw, vh = x / fw, y / fh, Wp / fw, Hp / fh
                if any(LS._iou((u, v, uw, vh), eb) > DEDUP_EXISTING_IOU
                       for eb in existing_boxes):
                    continue
                local.append((u, v, uw, vh, [f[k] for k in keys]))
        if not local:
            continue
        Xg = np.array([c[4] for c in local])
        g = clf.decision_function(sc.transform(Xg))
        p = clf.predict_proba(sc.transform(Xg))[:, acc_idx]
        order = np.argsort(g)[::-1]
        kept = []
        for i in order:
            u, v, uw, vh, _ = local[i]
            box = (u, v, uw, vh)
            if all(LS._iou(box, k["box"]) <= NMS_IOU for k in kept):
                kept.append(dict(minibrot_id=mb_id, scale=s, box=box,
                                 G=float(g[i]), p=float(p[i])))
        out.extend(kept)
    return out


def harvest_all(mb_ids, model, existing_boxes_by_mb):
    sc, clf, keys, acc_idx = model
    tasks = [(m, sc, clf, keys, acc_idx, existing_boxes_by_mb.get(m, []))
             for m in mb_ids]
    cands = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(harvest_one, tasks)):
            cands.extend(res)
            print(f"  harvested {mb_ids[i]}: {len(res)} candidates "
                  f"(running total {len(cands)})", flush=True)
    return cands


# --------------------------------------------------------------------------- #
# Select the slugs.                                                            #
# --------------------------------------------------------------------------- #
def _take(pool, key, n, reverse, used, per_mb):
    picks = []
    seen_mb = defaultdict(int)
    for c in sorted(pool, key=key, reverse=reverse):
        cid = id(c)
        if cid in used:
            continue
        if seen_mb[c["minibrot_id"]] >= per_mb:
            continue
        seen_mb[c["minibrot_id"]] += 1
        used.add(cid)
        picks.append(c)
        if len(picks) >= n:
            break
    return picks


def select(cands):
    for c in cands:
        c["margin"] = abs(c["p"] - 0.5)
    rng = np.random.default_rng(SEED)
    used = set()
    uncertain = _take(cands, lambda c: c["margin"], TARGET["uncertain"], False,
                      used, PER_MB_CAP)
    top_g = _take(cands, lambda c: c["G"], TARGET["top_g"], True, used, PER_MB_CAP)
    rest = [c for c in cands if id(c) not in used]
    rng.shuffle(rest)
    control = []
    seen_mb = defaultdict(int)
    for c in rest:
        if seen_mb[c["minibrot_id"]] >= PER_MB_CAP:
            continue
        seen_mb[c["minibrot_id"]] += 1
        control.append(c)
        if len(control) >= TARGET["control"]:
            break
    for slug, grp in (("uncertain", uncertain), ("top_g", top_g), ("control", control)):
        for c in grp:
            c["slug"] = slug
    sel = uncertain + top_g + control
    print(f"selected: {len(sel)}  uncertain={len(uncertain)} top_g={len(top_g)} "
          f"control={len(control)}  over {len({c['minibrot_id'] for c in sel})} minibrots")
    return sel


# --------------------------------------------------------------------------- #
# Render crops + write the store batch.                                        #
# --------------------------------------------------------------------------- #
def write_batch(sel, mb_by_id):
    from PIL import Image

    NEW_CROPS.mkdir(parents=True, exist_ok=True)
    LS.FRAMES.mkdir(parents=True, exist_ok=True)
    by_mb = defaultdict(list)
    for c in sel:
        by_mb[c["minibrot_id"]].append(c)

    rows = []
    for mbid, cs in by_mb.items():
        frame = LS.FRAMES / f"{mbid}.png"
        if not frame.exists():
            LS.render_full_frame(mb_by_id[mbid], frame)      # new minibrots only
        full = Image.open(frame).convert("RGB")
        fw_px, fh_px = full.size
        field, ffw, ffh = LS.load_field_values(mbid)
        mb = mb_by_id[mbid]
        for c in cs:
            u, v, w, h = c["box"]
            wk = f"{mbid}|{c['scale']}|{round(u,5)}|{round(v,5)}"
            wid = f"{mbid}_s{int(c['scale']*1000):03d}_{hashlib.sha1(wk.encode()).hexdigest()[:8]}"
            x0, y0 = int(round(u * fw_px)), int(round(v * fh_px))
            x1, y1 = int(round((u + w) * fw_px)), int(round((v + h) * fh_px))
            full.crop((x0, y0, x1, y1)).save(NEW_CROPS / f"{wid}.jpg", quality=90)
            # stored features = the compute_metrics 10 (UI caption + schema parity)
            m = LS.compute_metrics(LF.crop_field(field, ffw, ffh,
                                                 dict(u=u, v=v, w=w, h=h)))
            rows.append(dict(
                window_id=wid, minibrot_id=mbid, period=mb["period"],
                render=dict(cx=mb["cx"], cy=mb["cy"], fw=mb["fw"],
                            maxiter=mb["maxiter"], family="mandelbrot",
                            width=LS.W, height=LS.H, aspect=LS.ASPECT, palette=LS.PALETTE),
                window=dict(u=round(u, 5), v=round(v, 5), w=round(w, 5), h=round(h, 5)),
                scale=c["scale"], band=None,
                score_composite=round(c["G"], 5),       # G is the ranking score now
                g_score=round(c["G"], 5), p_accept=round(c["p"], 4), slug=c["slug"],
                features={k: round(float(m[k]), 5) for k in LS.FEATURE_KEYS},
                label=dict(klass=None)))
        print(f"  {mbid}: {len(cs)} crops", flush=True)

    # deterministic order: minibrot, scale, position
    rows.sort(key=lambda r: (r["minibrot_id"], r["scale"],
                             r["window"]["u"], r["window"]["v"]))
    NEW_STORE.mkdir(parents=True, exist_ok=True)
    with (NEW_STORE / "windows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = dict(
        batch_id=NEW_BATCH_ID, created="2026-07-23",
        generator="tools/studies/q4_stage1_g_batch.py",
        purpose="G-aimed next-to-label pass: linear goodness field (T2_cells) ranks "
                "fresh v2-survivor windows. Slugs: uncertain (boundary-teaching) / "
                "top_g (precision audit) / control (uniform, confident-and-wrong audit).",
        label_classes=["accept", "reject", "filter_leak"],
        model=dict(tier=TIER, C=C, source="q4_stage1_linear_fit"),
        n_windows=len(rows), n_minibrots=len(by_mb),
        slugs={s: sum(1 for r in rows if r["slug"] == s)
               for s in ("uncertain", "top_g", "control")},
        candidate_source="dense-grid harvest over existing 30 fields (deduped vs the "
                         "labeled batch) + fresh Newton minibrots",
        render=dict(width=LS.W, height=LS.H, aspect=LS.ASPECT, palette=LS.PALETTE),
        prefilter="v2 ceilings applied at construction (interior>=0.10 | flat>=0.88 | "
                  "speckle_ratio>=0.30) -> all rows are survivors; no auto_filter_v2 needed.",
        feature_keys=LS.FEATURE_KEYS,
        separate_store="q4 window store; canonical reader tools/corpus/q4_window_reader.py.",
    )
    (NEW_STORE / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {len(rows)} windows -> {NEW_STORE.relative_to(ROOT)}")
    return rows


def register_batch():
    """Append NEW_BATCH_ID to q4_window_reader.REGISTERED_BATCHES (idempotent)."""
    p = ROOT / "tools" / "corpus" / "q4_window_reader.py"
    txt = p.read_text()
    if NEW_BATCH_ID in txt:
        print(f"already registered: {NEW_BATCH_ID}")
        return
    needle = '    "2026-07-23_q4_stage1_windows",\n'
    if needle not in txt:
        print("WARN: could not find REGISTERED_BATCHES anchor; register manually")
        return
    txt = txt.replace(needle, needle + f'    "{NEW_BATCH_ID}",\n')
    p.write_text(txt)
    print(f"registered {NEW_BATCH_ID} in q4_window_reader.REGISTERED_BATCHES")


# --------------------------------------------------------------------------- #
def stage_build():
    GEN_OUT.mkdir(parents=True, exist_ok=True)
    model = train_model()
    print(f"G model: {TIER} C={C}")

    # existing labeled windows -> boxes to dedup against (per minibrot)
    existing_boxes = defaultdict(list)
    mb_by_id = {m["id"]: m for m in LS.load_minibrots()}
    for row, _ in qr.iter_windows("2026-07-23_q4_stage1_windows"):
        w = row["window"]
        existing_boxes[row["minibrot_id"]].append((w["u"], w["v"], w["w"], w["h"]))

    new_mbs = gen_new_minibrots()
    for m in new_mbs:
        mb_by_id[m["id"]] = m
    mb_ids = sorted(mb_by_id.keys())

    print(f"harvesting {len(mb_ids)} minibrots (30 existing + {len(new_mbs)} new)...")
    cands = harvest_all(mb_ids, model, existing_boxes)
    print(f"total candidates: {len(cands)}")
    (GEN_OUT / "candidates.json").write_text(json.dumps(
        [dict(minibrot_id=c["minibrot_id"], scale=c["scale"], box=list(c["box"]),
              G=round(c["G"], 4), p=round(c["p"], 4)) for c in cands]))

    sel = select(cands)
    write_batch(sel, mb_by_id)
    register_batch()
    print(f"\nNEXT: uv run python -m tools.studies.q4_stage1_g_batch serve")


def stage_serve(port=8017):
    import http.server, socketserver, functools
    url = (f"http://localhost:{port}/tools/viz/q4_window_label.html"
           f"?batch={NEW_BATCH_ID}")
    print(f"serving repo root at :{port}\n  LABELER -> {url}\n(Ctrl-C to stop)")
    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    with socketserver.TCPServer(("", port), Handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "serve"])
    ap.add_argument("--port", type=int, default=8017)
    args = ap.parse_args()
    if args.stage == "build":
        stage_build()
    elif args.stage == "serve":
        stage_serve(args.port)
