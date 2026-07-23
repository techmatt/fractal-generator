"""Build the native multibrot (mb3/mb4/mb5) t_good-calibration LABEL batch.

Goal (see prompts/native_multibrot_labels.md): native mb3/mb4/mb5 have never been
human-measured; their t_good sits at an uncalibrated 0.50. This produces a stratified
~300-item (~100/family) LABEL-INPUT batch spanning stored p_good bands — including
sub-threshold and rejected candidates — rendered at the *scored presentation*
(640x360 ss2, twilight_shifted, the reframe search fidelity the stored p_good refers
to). Analysis / threshold changes come LATER, after labeling.

Score axis (uniform, current-decoded v7):
  * mb3, mb5 — the v7 campaign ledgers already span the full p_good range natively:
      admissions  (outcome_ledger, all 4 lanes)      score = p_good      (>= t_good)
      canon-rejects (campaign2 harvest_log, admitted=False) score = canon_pgood
  * mb4 — v7 campaigns are admissions-heavy (7 sub-0.5 rows), so the sub-cut bands are
      filled from the gather/multibrot4 harvest RE-SCORED under v7
      (tools/corpus/rescore_gather_mb4_v7.py). Those fills are tagged
      source=gather_v6_rescored_v7 / lineage=gather — LABEL-BATCH material ONLY; they
      are never ledger admissions and never enter any generation/pool path.

Banding: 5 fixed p_good bands split at the t_good=0.5 operating point; target 20/band
(=100/family). Thin bands are filled from adjacent bands and the realized counts are
reported (no silent rebalance). Selection within a band is deterministic (sort by
(fw,cx,cy), evenly-spaced pick) — reproducible, spread across scale/location.

Deliverable: the registered batch dir (images.jsonl + batch.json + crops), a whole-batch
round-trip byte-identity gate, and a one-page manifest. The SIDECAR_LABELS registry
entry is DEFERRED until labels exist (registering an empty sidecar would make
label_store.assert_sidecars_joined raise for the whole corpus) — the exact one-line diff
is printed in the manifest.

Run: uv run python tools/corpus/build_native_multibrot_band.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import corpus_common as cc  # noqa: E402
from active_ckpt import BIN, PALETTE, JPG_Q, auto_maxiter  # noqa: E402

BATCH_ID = "2026-07-22_native_multibrot_band_v1"
GEN_VER = "native_multibrot_band_v1"
FAMS = ("multibrot3", "multibrot4", "multibrot5")
T_GOOD = 0.50  # the uncalibrated native-multibrot operating point being measured
BANDS = [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 1.0001)]
BAND_LABELS = ["0.00-0.20", "0.20-0.35", "0.35-0.50", "0.50-0.65", "0.65-1.00"]
PER_BAND = 20  # -> 100/family target
WORKERS = 4

BATCH_DIR = ROOT / "data" / "label_corpus" / "batches" / BATCH_ID
CROPS = BATCH_DIR / "crops"
SIDECAR = ROOT / "labels" / f"{GEN_VER}.json"
MB4_CACHE = BATCH_DIR / "mb4_gather_v7_rescore.jsonl"
MANIFEST = ROOT / "docs" / "findings" / "native_multibrot_band_v1_manifest.md"

OUTCOME_LANES = ["campaign1/breadth", "campaign1/dive", "campaign2/breadth", "campaign2/dive"]
HARVEST_LANES = ["campaign2/breadth", "campaign2/dive"]  # campaign1 harvest has NO coords


def band_of(p):
    for i, (a, b) in enumerate(BANDS):
        if a <= p < b:
            return i
    return None


def gather_guard_verdict():
    """gather_id -> v6 guard_verdict string (the cache stored only guard_pass bool)."""
    m = {}
    for line in open(ROOT / "data/discovery/gather/multibrot4/outcome_ledger.jsonl", encoding="utf-8"):
        r = json.loads(line)
        m[r.get("id")] = r.get("guard_verdict")
    return m


def load_pool(fam):
    """Distinct-by-coord candidate pool for `fam`. Each cand:
    {cx,cy,fw, score(v7 p_good), p_notbad, source, lineage, ledger_id, k3,
     decoded_class, guard_verdict, admitted, src_ledger}."""
    pool = {}  # coordkey -> cand

    def key(cx, cy, fw):
        return (round(cx, 10), round(cy, 10), round(fw, 12))

    # 1) admissions (all 4 lanes), native only. score = p_good (v7).
    for lane in OUTCOME_LANES:
        p = ROOT / "data/discovery" / lane / "outcome_ledger.jsonl"
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("family") != fam or r.get("julia_c_re") is not None:
                continue
            k = key(r["outcome_cx"], r["outcome_cy"], r["outcome_fw"])
            pool.setdefault(k, {
                "cx": r["outcome_cx"], "cy": r["outcome_cy"], "fw": r["outcome_fw"],
                "score": r.get("p_good"), "p_notbad": r.get("p_notbad"),
                "source": "campaign_v7", "lineage": "campaign", "ledger_id": r.get("id"),
                "k3": r.get("k3"), "decoded_class": r.get("decoded_class"),
                "guard_verdict": r.get("guard_verdict"), "admitted": True,
                "src_ledger": f"{lane}/outcome_ledger.jsonl",
            })

    # 2) campaign2 harvest canon-rejects (admitted=False), score = canon_pgood (v7).
    for lane in HARVEST_LANES:
        p = ROOT / "data/discovery" / lane / "harvest_log.jsonl"
        for line in open(p, encoding="utf-8"):
            h = json.loads(line)
            if h.get("partition") != fam or h.get("julia_c_re") is not None or "cx" not in h:
                continue
            if h.get("admitted"):  # admitted rows carry SEED coords (dupes of §1) — skip
                continue
            cp = h.get("canon_pgood")
            if cp is None:
                continue
            k = key(h["cx"], h["cy"], h["fw"])
            if k in pool:
                continue
            pool[k] = {
                "cx": h["cx"], "cy": h["cy"], "fw": h["fw"],
                "score": cp, "p_notbad": h.get("canon_nb"),
                "source": "campaign_v7", "lineage": "campaign",
                "ledger_id": f"c2{lane.split('/')[1][0]}_n{h['node_id']}",
                "k3": None, "decoded_class": h.get("canon_decoded"),
                "guard_verdict": None, "admitted": False,
                "src_ledger": f"{lane}/harvest_log.jsonl",
            }

    # 3) mb4 only: gather harvest RE-SCORED under v7 (sub-cut fill). score = v7_p_good.
    if fam == "multibrot4":
        gv = gather_guard_verdict()
        for line in open(MB4_CACHE, encoding="utf-8"):
            d = json.loads(line)
            k = key(d["cx"], d["cy"], d["fw"])
            if k in pool:  # campaign coord wins
                continue
            pool[k] = {
                "cx": d["cx"], "cy": d["cy"], "fw": d["fw"],
                "score": d["v7_p_good"], "p_notbad": d["v7_p_notbad"],
                "source": "gather_v6_rescored_v7", "lineage": "gather",
                "ledger_id": d.get("gather_id"),
                "k3": d.get("v6_k3"), "decoded_class": d.get("v6_decoded_class"),
                "guard_verdict": gv.get(d.get("gather_id")), "admitted": False,
                "src_ledger": "gather/multibrot4/outcome_ledger.jsonl (v7-rescored)",
            }

    return [c for c in pool.values() if c["score"] is not None]


def stratified_pick(cands):
    """Deterministic ~PER_BAND/band pick, filling thin bands from adjacent. Returns
    (selected, realized) where realized[band] = requested/picked bookkeeping."""
    by_band = {i: [] for i in range(len(BANDS))}
    for c in cands:
        b = band_of(c["score"])
        if b is not None:
            by_band[b].append(c)
    for b in by_band:
        by_band[b].sort(key=lambda c: (c["fw"], c["cx"], c["cy"]))

    selected = []
    picked_ids = set()

    def take_even(rows, n):
        n = min(n, len(rows))
        if n <= 0:
            return []
        if n == len(rows):
            return list(rows)
        step = len(rows) / n
        return [rows[int(i * step)] for i in range(n)]

    realized = {}
    deficits = {}
    for b in range(len(BANDS)):
        got = take_even(by_band[b], PER_BAND)
        for c in got:
            picked_ids.add(id(c))
        selected += got
        realized[b] = {"requested": PER_BAND, "available": len(by_band[b]), "picked": len(got)}
        deficits[b] = PER_BAND - len(got)

    # fill deficits from adjacent bands (nearest first), never re-picking.
    total_deficit = sum(deficits.values())
    filled_from = {b: 0 for b in range(len(BANDS))}
    for b in range(len(BANDS)):
        need = deficits[b]
        if need <= 0:
            continue
        order = sorted(range(len(BANDS)), key=lambda o: abs(o - b))
        for o in order:
            if need <= 0:
                break
            if o == b:
                continue
            extra = [c for c in by_band[o] if id(c) not in picked_ids]
            take = extra[:need]
            for c in take:
                picked_ids.add(id(c))
                selected.append(c)
                filled_from[o] += 1
            need -= len(take)
        deficits[b] = need

    return selected, realized, filled_from, deficits


def sanitize(s):
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(s))


def camptag(src_ledger):
    """Short source tag so image_id is globally unique — ledger_ids restart per lane
    (campaign1/breadth and campaign2/breadth both emit `st_..._breadth_000324`)."""
    if src_ledger.startswith("gather"):
        return "gth"
    c = "c1" if src_ledger.startswith("campaign1") else "c2"
    lane = "b" if "/breadth/" in src_ledger else "d"
    return c + lane


def render_crop(fractal_type, cx_s, cy_s, fw_s, maxiter, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN), "render-one", "--cx", cx_s, "--cy", cy_s, "--fw", fw_s,
        "--width", "640", "--height", "360", "--supersample", "2",
        "--maxiter", str(maxiter), "--palette", PALETTE, "--jpg-quality", str(JPG_Q),
        "--out", str(out), "--family", fractal_type,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and out.exists()
    return ok, ("" if ok else r.stderr[-300:])


def build_rows(fam, selected):
    rows = []
    for c in selected:
        cx_s, cy_s, fw_s = cc.hp_str(c["cx"]), cc.hp_str(c["cy"]), cc.hp_str(c["fw"])
        maxiter = auto_maxiter(float(c["fw"]))
        stratum = BAND_LABELS[band_of(c["score"])]
        image_id = f"nmb{fam[-1]}_{camptag(c['src_ledger'])}_{sanitize(c['ledger_id'])}"
        render = cc.render_block(
            cx=cx_s, cy=cy_s, fw=fw_s, maxiter=maxiter, palette=PALETTE,
            composition="center", width=640, height=360, ss=2,
            filter="lanczos3", interior_mode="black",
        )
        # native multibrot: family lives in the render block (jm-band precedent); no c.
        render["fractal_type"] = fam
        render["c_re"] = None
        render["c_im"] = None
        prov = cc.provenance_block(
            GEN_VER, BATCH_ID, family=fam,
            p_good=c["score"], p_notbad=c["p_notbad"], t_good=T_GOOD,
            stratum=stratum, scorer_version="v7", ledger_id=c["ledger_id"],
            source=c["source"], lineage=c["lineage"],
            k3=c["k3"], decoded_class=c["decoded_class"], guard_verdict=c["guard_verdict"],
        )
        rows.append((image_id, cc.make_row(image_id, render, prov, cc.label_block())))
    return rows


def main():
    assert MB4_CACHE.exists(), (
        f"mb4 v7 rescore cache missing: {MB4_CACHE}\n"
        f"run: uv run python tools/corpus/rescore_gather_mb4_v7.py")
    CROPS.mkdir(parents=True, exist_ok=True)

    all_rows = []
    report = {}
    seen_ids = set()
    for fam in FAMS:
        pool = load_pool(fam)
        selected, realized, filled_from, deficits = stratified_pick(pool)
        rows = build_rows(fam, selected)
        for iid, _ in rows:
            if iid in seen_ids:
                raise SystemExit(f"duplicate image_id {iid}")
            seen_ids.add(iid)
        all_rows += rows
        # source / band breakdown
        band_src = {bl: {"campaign_v7": 0, "gather_v6_rescored_v7": 0} for bl in BAND_LABELS}
        for c in selected:
            band_src[BAND_LABELS[band_of(c["score"])]][c["source"]] += 1
        report[fam] = {
            "pool": len(pool), "selected": len(selected), "realized": realized,
            "filled_from": filled_from, "deficits": deficits, "band_src": band_src,
        }
        print(f"{fam}: pool={len(pool)} selected={len(selected)}", flush=True)

    # --- render crops (parallel, capped) ---
    print(f"\nrendering {len(all_rows)} crops @640x360 ss2 {PALETTE} ...", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for iid, row in all_rows:
            r = row["render"]
            out = CROPS / f"{iid}.jpg"
            futs[ex.submit(render_crop, r["fractal_type"], r["cx"], r["cy"], r["fw"],
                           r["maxiter"], out)] = iid
        n = 0
        for fut in cf.as_completed(futs):
            ok, err = fut.result()
            n += 1
            if not ok:
                raise SystemExit(f"crop render failed [{futs[fut]}]: {err}")
            if n % 50 == 0 or n == len(all_rows):
                print(f"  {n}/{len(all_rows)} ({time.time()-t0:.1f}s)", flush=True)

    # --- whole-batch round-trip byte-identity gate ---
    print("\nround-trip byte-identity gate (rebuild every crop from its stamp) ...", flush=True)
    tmp = BATCH_DIR / "_gate_tmp"
    tmp.mkdir(exist_ok=True)
    tg = time.time()
    mismatches = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for iid, row in all_rows:
            r = row["render"]
            out = tmp / f"{iid}.jpg"
            futs[ex.submit(render_crop, r["fractal_type"], r["cx"], r["cy"], r["fw"],
                           r["maxiter"], out)] = (iid, out)
        for fut in cf.as_completed(futs):
            ok, err = fut.result()
            iid, out = futs[fut]
            if not ok:
                raise SystemExit(f"gate re-render failed [{iid}]: {err}")
            orig = (CROPS / f"{iid}.jpg").read_bytes()
            if orig != out.read_bytes():
                mismatches.append(iid)
    for f in tmp.glob("*.jpg"):
        f.unlink()
    tmp.rmdir()
    if mismatches:
        raise SystemExit(f"ROUND-TRIP GATE FAILED: {len(mismatches)} byte-mismatch: "
                         f"{mismatches[:5]}")
    print(f"  gate PASS: {len(all_rows)}/{len(all_rows)} byte-identical ({time.time()-tg:.1f}s)",
          flush=True)

    # --- write images.jsonl ---
    cc.write_jsonl([row for _, row in all_rows], str(BATCH_DIR / "images.jsonl"))

    # --- batch.json ---
    per_family = {f: report[f]["selected"] for f in FAMS}
    batch = {
        "created": "2026-07-22",
        "labeler": None,
        "generator_version": GEN_VER,
        "source_run": ("v7 campaign ledgers (campaign1/2 breadth+dive) admissions + "
                       "campaign2 harvest canon-rejects; mb4 sub-cut filled from "
                       "gather/multibrot4 harvest RE-SCORED under v7"),
        "purpose": ("Calibrate native multibrot mb3/mb4/mb5 t_good (uncalibrated 0.50). "
                    "Labels-input ONLY; threshold analysis comes later."),
        "schema_extension": ("render block adds fractal_type (multibrot3/4/5) + c_re/c_im "
                             "(null for native) so the crop rebuilds as the right family "
                             "(jm-band precedent)."),
        "score_axis": {
            "scorer_version": "v7", "t_good_measured": T_GOOD,
            "bands": BAND_LABELS, "per_band_target": PER_BAND,
            "band_score": ("p_good for admissions; canon_pgood for campaign2 harvest "
                           "rejects; v7-rescored p_good for gather/multibrot4 fills"),
        },
        "band": {
            "families": list(FAMS),
            "per_family_selected": per_family,
            "n_rows": sum(per_family.values()),
            "realized": {f: report[f]["realized"] for f in FAMS},
            "filled_from_adjacent": {f: report[f]["filled_from"] for f in FAMS},
            "band_source_breakdown": {f: report[f]["band_src"] for f in FAMS},
        },
        "mb4_note": ("gather_v6_rescored_v7 rows are LABEL-BATCH material only "
                     "(source=gather_v6_rescored_v7, scorer=v7). They are NOT ledger "
                     "admissions and must never enter a generation/pool path."),
        "render_defaults": {
            "width": 640, "height": 360, "ss": 2, "maxiter": "auto_maxiter(fw)",
            "filter": "lanczos3", "composition": "center", "interior_mode": "black",
            "palette": PALETTE, "jpg_quality": JPG_Q,
        },
        "render_recipe": {
            "path": "render-one --family <fractal_type> --palette twilight_shifted",
            "colormaps": "data/palettes/clean_colormaps.json (render-one default)",
            "note": "byte-identical to the reframe/prescreen scored presentation.",
        },
        "sidecar_registration_deferred": {
            "reason": ("registering an empty sidecar in label_store.SIDECAR_LABELS makes "
                       "assert_sidecars_joined raise for the whole corpus (joined[bid]==0). "
                       "Add the line below AFTER labels are exported."),
            "add_to_SIDECAR_LABELS": f'"{BATCH_ID}": "{GEN_VER}.json",',
            "sidecar_path": f"labels/{GEN_VER}.json",
        },
    }
    (BATCH_DIR / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")

    # --- empty sidecar placeholder (labels land here; NOT yet registered) ---
    if not SIDECAR.exists():
        SIDECAR.parent.mkdir(parents=True, exist_ok=True)
        SIDECAR.write_text("{}\n", encoding="utf-8")

    write_manifest(report, per_family)
    print(f"\nDONE. batch={BATCH_ID}  rows={sum(per_family.values())}")
    print(f"  images.jsonl + batch.json -> {BATCH_DIR.relative_to(ROOT)}")
    print(f"  manifest -> {MANIFEST.relative_to(ROOT)}")


def write_manifest(report, per_family):
    L = []
    L.append(f"# Native multibrot band batch — `{BATCH_ID}`\n")
    L.append(f"**Generator version:** `{GEN_VER}`  ·  **Total rows:** "
             f"{sum(per_family.values())}  ·  **Render:** 640×360 ss2 `{PALETTE}` "
             f"(the scored presentation)  ·  **Score axis:** v7 p_good, t_good measured "
             f"= {T_GOOD}\n")
    L.append("Labels-input only. Native mb3/mb4/mb5 have never been human-measured; this "
             "batch calibrates their t_good (currently an uncalibrated 0.50). Threshold "
             "analysis comes after labeling.\n")

    L.append("## Per-family × per-band counts (requested vs realized)\n")
    L.append(f"Target {PER_BAND}/band → {PER_BAND*len(BANDS)}/family. Bands are v7 p_good; "
             "the split at 0.50 is the measured operating point.\n")
    for fam in FAMS:
        r = report[fam]
        L.append(f"### {fam}  (pool {r['pool']}, selected {r['selected']})\n")
        L.append("| band (p_good) | available | picked | source: campaign_v7 / gather_v6→v7 |")
        L.append("|---|---|---|---|")
        for i, bl in enumerate(BAND_LABELS):
            av = r["realized"][i]["available"]
            pk = r["realized"][i]["picked"]
            bs = r["band_src"][bl]
            fill = r["filled_from"].get(i, 0)
            note = f" (+{fill} filled into other bands)" if fill else ""
            L.append(f"| {bl} | {av} | {pk} | {bs['campaign_v7']} / "
                     f"{bs['gather_v6_rescored_v7']} |{note}")
        # deficits
        defs = {BAND_LABELS[b]: d for b, d in r["deficits"].items() if d > 0}
        if defs:
            L.append(f"\n*Unfilled band deficits (data-limited): {defs}*")
        L.append("")

    L.append("## Source ledgers\n")
    L.append("- **Admissions** (score = `p_good`): `outcome_ledger.jsonl` in "
             "campaign1/breadth, campaign1/dive, campaign2/breadth, campaign2/dive "
             "(native `multibrot{3,4,5}`, `julia_c` null).")
    L.append("- **Canon-rejects / sub-cut** (score = `canon_pgood`): "
             "`harvest_log.jsonl` in campaign2/breadth, campaign2/dive "
             "(`admitted=False`; campaign1 harvest carries no coordinates → excluded).")
    L.append("- **mb4 sub-cut fill** (score = v7-rescored `p_good`): "
             "`gather/multibrot4/outcome_ledger.jsonl`, every distinct candidate "
             "re-rendered at 640×360 ss2 and re-scored under v7 "
             "(`tools/corpus/rescore_gather_mb4_v7.py`; cache "
             "`batches/{}/mb4_gather_v7_rescore.jsonl`). v7-decoded native mb4 has only "
             "~7 sub-0.5 rows, so its below-threshold bands come entirely from this "
             "current-decoded re-score.".format(BATCH_ID))
    L.append("\nEach row records its exact `provenance.src`/`lineage`/`ledger_id`/"
             "`scorer_version` (per-item source is recoverable from `images.jsonl`).\n")

    L.append("## Provenance / pool-safety\n")
    L.append("`gather_v6_rescored_v7` rows are LABEL-BATCH material only "
             "(`source=gather_v6_rescored_v7`, `scorer=v7`, `lineage=gather`). They are "
             "**not** ledger admissions and were **not** written to any discovery ledger "
             "or generation/pool path.\n")

    L.append("## Registration\n")
    L.append(f"- Batch registered in the store: `data/label_corpus/batches/{BATCH_ID}/` "
             "(`images.jsonl` + `batch.json` + `crops/`). Never-delete.")
    L.append(f"- Empty label sidecar created: `labels/{GEN_VER}.json` (`{{}}`).")
    L.append("- **Sidecar registry entry is DEFERRED** — adding an empty sidecar to "
             "`label_store.SIDECAR_LABELS` now makes `assert_sidecars_joined` raise for "
             "the whole corpus (the batch joins 0 rows until labeled). After labels are "
             "exported, add this one line to `tools/corpus/label_store.py`:")
    L.append(f"\n```python\n    \"{BATCH_ID}\": \"{GEN_VER}.json\",\n```\n")

    L.append("## Labeling\n")
    L.append("- View: `tools/viz/corpus_label.html` (blind — no scores shown; family "
             "may be visible). `image_id` collides across batches, so calibration reads "
             "must key on `render.cx/cy/fw` + `fractal_type`, all stamped per row.")
    L.append("- `label.score ∈ {null,1,2,3}` (bad/okay/good). Export → the sidecar above.\n")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
