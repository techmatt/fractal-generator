#!/usr/bin/env python
r"""Phoenix Phase B — the stratified human-label batch (corpus rules in full).

Turns a grid run's `all_outcomes.jsonl` (every scored walk: q3 AND sub-threshold AND guarded
reject) into a registered label_corpus batch of ~500 items, stratified TWO ways at once
(prompt phoenix_phase_b.md):
  * across SEEDS — many seeds x few items each, so human labels support their own
    between/within-seed split (round-robin over seed_idx within each band).
  * across p_good BANDS including sub-threshold rungs and rejects — without below-cut items
    the v7-calibration read is selection-biased. Bands:
        HIGH   guard-pass, decoded q3, p_good >= HIGH_CUT
        Q3     guard-pass, decoded q3, t_good <= p_good < HIGH_CUT
        SUB    guard-pass, decoded 1/2 (sub-threshold real locations)
        REJECT guard-fail (degenerate) — rendered at deploy fidelity as a true reject
    (SUB may be thin — phoenix outcomes tend to be bimodal q3-vs-guarded; realized band
    populations are reported, never faked.)

**The render block stamps the COMPLETE parameter identity** — fractal_type AND (c,p,z_-1) —
because any absent axis silently round-trips to the Ushiki default and Guard B alone can't
catch it (the baserate_v1 lesson with three axes, docs/findings/invariant_audit.md). So before
rendering, every row is asserted to round-trip through `location.from_render_block` with its
c/p/z_-1 present and equal to the source seed; then Guard B (`verify_render_path.check_batch`)
confirms byte-reproducibility.

Crops are the canonical corpus recipe (`render_corpus_crop`: render-one --palette --colormaps,
1280x720 ss4 lanczos3 q90). Batch registered per CORPUS_SCHEMA (images.jsonl + batch.json with
the render_recipe stamp + an empty scores.json); admission + reject contact sheets emitted.

  uv run python tools/phoenix/phoenix_label_batch.py --run data/discovery/phoenix_grid/grid --target 500
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import corpus_common as cc                    # noqa: E402
import location as loc_mod                     # noqa: E402
import verify_render_path as vrp               # noqa: E402
from active_ckpt import auto_maxiter           # noqa: E402

GENERATOR_VERSION = "phoenix_grid"
SCORE3 = ROOT / "data" / "palettes" / "score3_colormaps.json"
BIN = ROOT / "target" / "release" / "fractal-generator.exe"

# canonical label-crop render spec (matches the gather_v6 batch — deploy wallpaper fidelity)
W, H, SS, FILTER, INTERIOR, COMPOSITION, JPGQ = 1280, 720, 4, "lanczos3", "black", "center", 90
HIGH_CUT = 0.30      # split the q3 band so the label set spans the p_good range, not just its floor
WORKERS = 4          # project cap

# Ushiki defaults — the values a DROPPED phoenix axis silently round-trips to (the hazard we guard).
USHIKI = {"c_re": 0.5667, "c_im": 0.0, "p_re": -0.5, "p_im": 0.0, "zm1_re": 0.0, "zm1_im": 0.0}


def band_of(o: dict, t_good: float) -> str:
    if not o.get("guard_pass"):
        return "REJECT"
    if o.get("decoded_class") == 3:
        return "HIGH" if float(o["p_good"]) >= HIGH_CUT else "Q3"
    return "SUB"


def dedup_key(o: dict) -> tuple:
    """Identity-aware coarse-viewport key so two walks landing on the same phoenix place (same
    (c,p,z_-1) and ~same viewport) don't both enter the batch. Distinct (c,p,z_-1) never collide."""
    fw = float(o["outcome_fw"])
    q = max(fw, 1e-12)
    return (round(o["phoenix_c_re"], 6), round(o["phoenix_c_im"], 6),
            round(o["phoenix_p_re"], 6), round(o["phoenix_p_im"], 6),
            round(o["phoenix_zm1_re"], 6), round(o["phoenix_zm1_im"], 6),
            round(float(o["outcome_cx"]) / q, 1), round(float(o["outcome_cy"]) / q, 1),
            round(math.log10(q), 1))


def palette_for(image_id: str, names: list[str]) -> str:
    h = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16)
    return names[h % len(names)]


def allocate(bands: dict[str, list], target: int) -> dict[str, int]:
    """Even target split across the 4 bands, redistributing any band's shortfall to bands with
    surplus (so a thin SUB band doesn't cost total count). Deterministic."""
    order = ["HIGH", "Q3", "SUB", "REJECT"]
    quota = {b: target // len(order) for b in order}
    for b in order[:target % len(order)]:
        quota[b] += 1
    # clamp to availability, pool the deficit, redistribute round-robin to bands with headroom
    alloc = {b: min(quota[b], len(bands.get(b, []))) for b in order}
    deficit = target - sum(alloc.values())
    while deficit > 0 and any(len(bands.get(b, [])) > alloc[b] for b in order):
        for b in order:
            if deficit <= 0:
                break
            if len(bands.get(b, [])) > alloc[b]:
                alloc[b] += 1
                deficit -= 1
    return alloc


def stratify_across_seeds(items: list[dict], n: int) -> list[dict]:
    """Round-robin across seed_idx (max seed spread), best-p_good first within a seed."""
    by_seed: dict[int, list] = defaultdict(list)
    for o in items:
        by_seed[int(o["seed_idx"])].append(o)
    for s in by_seed.values():
        s.sort(key=lambda r: (-float(r.get("p_good", 0.0)), r["id"]))
    seeds = sorted(by_seed)
    out, i = [], 0
    while len(out) < n and any(by_seed[s] for s in seeds):
        s = seeds[i % len(seeds)]
        if by_seed[s]:
            out.append(by_seed[s].pop(0))
        i += 1
    return out


def build_render_block(o: dict) -> dict:
    """Version-invariant render block + the FULL phoenix identity stamped at top level."""
    fw = float(o["outcome_fw"])
    r = cc.render_block(cx=o["outcome_cx"], cy=o["outcome_cy"], fw=fw,
                        maxiter=int(auto_maxiter(fw)), palette="__set_by_caller__",
                        composition=COMPOSITION, width=W, height=H, ss=SS,
                        filter=FILTER, interior_mode=INTERIOR)
    r["fractal_type"] = "phoenix"
    r["c_re"] = cc.hp_str(o["phoenix_c_re"]); r["c_im"] = cc.hp_str(o["phoenix_c_im"])
    r["p_re"] = cc.hp_str(o["phoenix_p_re"]); r["p_im"] = cc.hp_str(o["phoenix_p_im"])
    r["zm1_re"] = cc.hp_str(o["phoenix_zm1_re"]); r["zm1_im"] = cc.hp_str(o["phoenix_zm1_im"])
    return r


def assert_identity_stamped(render: dict, o: dict):
    """The baserate_v1 guard, three axes: the render block MUST round-trip through
    from_render_block with all of (c,p,z_-1) present and equal to the source seed — else a
    dropped axis silently renders the Ushiki default and Guard B can't tell (both crops match)."""
    loc = loc_mod.from_render_block(render)
    assert loc.family == "phoenix", f"fractal_type lost: {loc.family}"
    fp = dict(loc.family_params)
    for k in ("p_re", "p_im", "zm1_re", "zm1_im"):
        assert fp.get(k) is not None, f"{k} absent from render block (would round-trip to Ushiki)"
    assert loc.c_re is not None and loc.c_im is not None, "c absent from render block"
    # exact round-trip to the source values (the true anti-silent-default check)
    got = {"c_re": float(loc.c_re), "c_im": float(loc.c_im),
           "p_re": float(fp["p_re"]), "p_im": float(fp["p_im"]),
           "zm1_re": float(fp["zm1_re"]), "zm1_im": float(fp["zm1_im"])}
    src = {"c_re": float(o["phoenix_c_re"]), "c_im": float(o["phoenix_c_im"]),
           "p_re": float(o["phoenix_p_re"]), "p_im": float(o["phoenix_p_im"]),
           "zm1_re": float(o["phoenix_zm1_re"]), "zm1_im": float(o["phoenix_zm1_im"])}
    for k in got:
        assert abs(got[k] - src[k]) < 1e-9, f"{k} stamp {got[k]} != source {src[k]}"


def make_row(o: dict, band: str, batch_id: str, names: list[str]) -> dict:
    iid = o["id"]
    render = build_render_block(o)
    render["palette"] = palette_for(iid, names)
    assert_identity_stamped(render, o)
    prov = cc.provenance_block(
        GENERATOR_VERSION, batch_id, family="phoenix", branch=o.get("branch"),
        theta=o.get("theta"), offset=o.get("offset"),
        z_class=("nonreal" if abs(float(o["phoenix_zm1_im"])) > 1e-9
                 else ("zero" if abs(float(o["phoenix_zm1_re"])) < 1e-12 else "real")),
        seed_index=int(o["seed_idx"]), stratum=band, selection_role=band,
        k3=o.get("k3"), filter_score=o.get("k3"), decoded_class=o.get("decoded_class"),
        p_good=o.get("p_good"), p_notbad=o.get("p_notbad"), t_good=o.get("t_good"),
        descend_mode="phoenix_grid", lineage="phoenix_grid", scorer_version=o.get("scorer_version", "v7"),
        ledger_id=iid)
    return cc.make_row(iid, render, prov, cc.label_block())


def render_crop(row: dict, crops: Path):
    out = crops / f"{row['image_id']}.jpg"
    cc.render_corpus_crop(row["render"], out, palette_source=SCORE3, bin_path=BIN,
                          jpg_quality=JPGQ, cwd=str(ROOT))
    return row["image_id"]


def contact_sheet(rows: list[dict], crops: Path, out_png: Path, title: str, ncol: int = 8):
    from PIL import Image, ImageDraw
    TW, TH, PAD, LBL, GUT = 200, 112, 4, 14, 34
    items = [r for r in rows if (crops / f"{r['image_id']}.jpg").exists()]
    if not items:
        return None
    nrow = (len(items) + ncol - 1) // ncol
    cw, ch = TW + 2 * PAD, TH + LBL + 2 * PAD
    sheet = Image.new("RGB", (ncol * cw, GUT + nrow * ch), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((8, 10), title, fill=(235, 235, 235))
    for k, r in enumerate(items):
        rr, cc_ = divmod(k, ncol)
        x, y = cc_ * cw + PAD, GUT + rr * ch + PAD
        try:
            im = Image.open(crops / f"{r['image_id']}.jpg").convert("RGB").resize((TW, TH))
            sheet.paste(im, (x, y))
        except Exception:
            pass
        pg = r["provenance"].get("p_good")
        lab = f"{r['provenance']['stratum']} pg={pg:.2f}" if pg is not None else r["provenance"]["stratum"]
        d.rectangle([x, y + TH, x + TW, y + TH + LBL], fill=(28, 28, 32))
        d.text((x + 2, y + TH + 1), lab[:30], fill=(210, 210, 218))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="grid run dir (all_outcomes.jsonl, summary.json)")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--batch-id", default=None, help="default: 2026-07-21_phoenix_grid")
    ap.add_argument("--date", default="2026-07-21")
    ap.add_argument("--guard-k", type=int, default=8, help="Guard-B sample size")
    ap.add_argument("--no-render", action="store_true", help="write rows only (skip crop render + Guard B)")
    args = ap.parse_args(argv)

    run = Path(args.run)
    outcomes = [json.loads(l) for l in open(run / "all_outcomes.jsonl", encoding="utf-8") if l.strip()]
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    t_good = summ["config"]["t_good"]
    batch_id = args.batch_id or f"{args.date}_phoenix_grid"
    batch_dir = Path(cc.batch_dir(batch_id))
    crops = batch_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    names = [p["name"] for p in json.loads(SCORE3.read_text(encoding="utf-8"))]

    # dedup then band
    seen, uniq = set(), []
    for o in outcomes:
        k = dedup_key(o)
        if k in seen:
            continue
        seen.add(k); uniq.append(o)
    bands: dict[str, list] = defaultdict(list)
    for o in uniq:
        bands[band_of(o, t_good)].append(o)
    avail = {b: len(bands.get(b, [])) for b in ("HIGH", "Q3", "SUB", "REJECT")}
    alloc = allocate(bands, args.target)
    print(f"batch {batch_id}: {len(outcomes)} outcomes -> {len(uniq)} unique places")
    print(f"  band availability {avail}  |  allocation {alloc}")

    selected = []
    for b in ("HIGH", "Q3", "SUB", "REJECT"):
        selected += [(o, b) for o in stratify_across_seeds(bands.get(b, []), alloc[b])]
    rows = [make_row(o, b, batch_id, names) for o, b in selected]
    cc.write_jsonl(rows, str(batch_dir / "images.jsonl"))
    (batch_dir / "scores.json").write_text("{}\n", encoding="utf-8")

    n_seeds = len({r["provenance"]["seed_index"] for r in rows})
    band_real = Counter(r["provenance"]["stratum"] for r in rows)
    batch_json = {
        "created": args.date, "labeler": None, "generator_version": GENERATOR_VERSION,
        "source_run": str(run),
        "schema_extension": "render block adds fractal_type + (c_re/c_im, p_re/p_im, zm1_re/zm1_im) "
                            "— the FULL phoenix (c,p,z_-1) identity; provenance adds theta/offset/z_class",
        "sampling_metaparameters": {
            "grid": {"n_seeds": summ["config"]["n_seeds"], "k": summ["config"]["k"],
                     "walks_per_descent": summ["config"]["walks_per_descent"],
                     "depth": summ["config"]["depth"], "t_good": t_good,
                     "scorer_version": summ["config"]["scorer_version"]},
            "stratification": {"bands": ["HIGH", "Q3", "SUB", "REJECT"], "high_cut": HIGH_CUT,
                               "across_seeds": "round-robin over seed_idx within band",
                               "availability": avail, "allocation": alloc,
                               "realized": dict(band_real), "n_seeds_in_batch": n_seeds},
            "palette_pool": "data/palettes/score3_colormaps.json (seeded per image_id)",
            "dedup": "identity-aware coarse-viewport (distinct (c,p,z_-1) never collide)",
        },
        "present_gates": None,
        "render_defaults": {"width": W, "height": H, "ss": SS, "filter": FILTER,
                            "interior_mode": INTERIOR, "maxiter": "auto_maxiter(fw)",
                            "composition": COMPOSITION, "jpg_quality": JPGQ},
        "render_recipe": cc.render_recipe_stamp(SCORE3, jpg_quality=JPGQ),
    }
    (batch_dir / "batch.json").write_text(json.dumps(batch_json, indent=2), encoding="utf-8")
    print(f"  wrote {len(rows)} rows ({n_seeds} seeds) realized bands {dict(band_real)}")

    if args.no_render:
        print("  --no-render: skipped crops + Guard B")
        return 0

    # render crops (<=4 workers)
    print(f"  rendering {len(rows)} crops @ {W}x{H} ss{SS}...")
    fails = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(render_crop, r, crops): r["image_id"] for r in rows}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:
                fails.append((futs[fut], str(e)[:160]))
            if i % 50 == 0:
                print(f"    {i}/{len(rows)}")
    if fails:
        print(f"  WARN {len(fails)} crop renders failed; first: {fails[0]}")

    # Guard B — byte-reproducibility of the render blocks (identity round-trip already asserted)
    res = vrp.check_batch(batch_dir, k=min(args.guard_k, len(rows)), verbose=True)
    print(f"  Guard B: {'PASS' if res['ok'] else 'FAIL'} worst mean|d|={res['worst']:.3f}")

    # contact sheets (admissions + rejects)
    adm_rows = [r for r in rows if r["provenance"]["stratum"] in ("HIGH", "Q3")]
    rej_rows = [r for r in rows if r["provenance"]["stratum"] in ("SUB", "REJECT")]
    sd = ROOT / "out" / "phoenix_grid" / "label_sheets"
    a = contact_sheet(adm_rows, crops, sd / f"{batch_id}_admissions.png",
                      f"{batch_id} — admissions HIGH+Q3 ({len(adm_rows)})")
    r = contact_sheet(rej_rows, crops, sd / f"{batch_id}_rejects.png",
                      f"{batch_id} — sub-threshold + rejects ({len(rej_rows)})")
    print(f"  sheets -> {a}\n           {r}")
    print(f"  batch  -> {batch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
