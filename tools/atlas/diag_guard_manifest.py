#!/usr/bin/env python
"""Pin the guard thresholds and emit the drop manifest (report-only).

Thin sibling of `diag_outcome_guards.py`. The guard MEASURES are already decided
(that script rendered the fields and located the gaps); this step PINS both
thresholds and emits the exact **drop manifest** — the ID set each gate removes —
so the guard-implementation step has a control to reproduce. It reads the already
produced diagnostic table; **no re-rendering.**

  Gates (pinned):
    interior gate:  interior_frac >= 0.25   -> drop   (confirmed black-gate)
    flat gate:      field_std     <  6      -> drop   (center of the 5-7 empty band)
    hf_energy: carried in the table for reference only, NOT a gate here.

Applied over the 81 DISTINCT outcomes (near-dups are not reward-pool members).

  Scope — HARD boundary: report + manifest + marked sheet only. Wires no guard,
  touches no ledger, refits no theta_hat. Outputs are ephemeral (`out/`).

  uv run python tools/atlas/diag_guard_manifest.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools" / "atlas_probe", ROOT / "tools" / "reframe"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from production_seeder import build_contact_sheet, NCOL_DUP  # noqa: E402  reuse tiling

OUT = ROOT / "out" / "atlas" / "diag_outcome_guards"
TABLE = OUT / "table.csv"
TILE_DIR = OUT / "tiles"
MARK_DIR = OUT / "marked_tiles"        # ephemeral: union tiles with a drop X drawn on
MANIFEST = OUT / "drop_manifest.json"

# Pinned thresholds.
INTERIOR_THR = 0.25    # interior_frac >= INTERIOR_THR -> drop
FIELD_STD_THR = 6.0    # field_std < FIELD_STD_THR -> drop

# Sparse-but-structured dendrites that MUST survive the flat gate (floor sanity).
SURVIVE_IDS = ("000123", "000060")


def load_distinct():
    rows = []
    with open(TABLE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["distinct"] = r["distinct"].strip().lower() == "true"
            for k in ("k3", "interior_frac", "field_std", "hf_energy", "grad_mag"):
                r[k] = float(r[k])
            rows.append(r)
    distinct = [r for r in rows if r["distinct"]]
    return rows, distinct


def short(oid):
    return oid[-6:]


def mark_tile(src: Path, dst: Path, label: str):
    """Copy a tile with a red X + border drawn over it (drop marker)."""
    from PIL import Image, ImageDraw
    im = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    red = (235, 45, 45)
    for t in range(4):
        d.line([(t, 0), (w, h - t)], fill=red, width=3)
        d.line([(0, h - t), (w - t, 0)], fill=red, width=3)
    for t in range(4):
        d.rectangle([t, t, w - 1 - t, h - 1 - t], outline=red)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)


def main():
    rows, distinct = load_distinct()
    n = len(distinct)
    print(f"=== diag_guard_manifest: {len(rows)} outcomes, {n} distinct ===")
    print(f"gates (pinned): interior_frac >= {INTERIOR_THR}  |  field_std < {FIELD_STD_THR}")
    assert n == 81, f"expected 81 distinct, got {n}"

    by_id = {r["id"]: r for r in distinct}
    interior_hit = {r["id"] for r in distinct if r["interior_frac"] >= INTERIOR_THR}
    flat_hit = {r["id"] for r in distinct if r["field_std"] < FIELD_STD_THR}

    # ---- floor sanity: the sparse-but-structured dendrites must survive ---- #
    print("\n--- flat-floor sanity (these dendrites MUST survive field_std<6) ---")
    bad_floor = False
    for tag in SURVIVE_IDS:
        hit = [r for r in distinct if short(r["id"]) == tag]
        if not hit:
            print(f"  {tag}: NOT FOUND among distinct (cannot confirm) -- STOP")
            bad_floor = True
            continue
        r = hit[0]
        dropped = r["id"] in flat_hit
        state = "DROPPED (floor mis-set!)" if dropped else "survives"
        print(f"  {tag}: field_std={r['field_std']:.2f}  hf={r['hf_energy']:.1f}  "
              f"if={r['interior_frac']:.3f}  k3={r['k3']:.3f}  -> {state}")
        if dropped:
            bad_floor = True
    if bad_floor:
        print("\nFLOOR MIS-SET: a required-survivor dendrite was dropped. Stopping before manifest.")
        sys.exit(1)
    print("  floor OK: both dendrites retained.")

    union = interior_hit | flat_hit
    both = interior_hit & flat_hit
    interior_only = interior_hit - flat_hit
    flat_only = flat_hit - interior_hit
    retained = [r for r in distinct if r["id"] not in union]

    def ids_desc(idset):
        return [r["id"] for r in sorted(distinct, key=lambda x: -x["k3"]) if r["id"] in idset]

    # ---- 1. per-gate drop lists ---- #
    print("\n--- per-gate drop lists (k3-descending) ---")
    print(f"interior gate (interior_frac >= {INTERIOR_THR}):  {len(interior_hit)} dropped")
    for oid in ids_desc(interior_hit):
        r = by_id[oid]
        print(f"    {short(oid)}  if={r['interior_frac']:.3f}  fs={r['field_std']:7.2f}  "
              f"k3={r['k3']:.3f}  {r['mix_source']}")
    print(f"flat gate (field_std < {FIELD_STD_THR}):  {len(flat_hit)} dropped")
    for oid in ids_desc(flat_hit):
        r = by_id[oid]
        print(f"    {short(oid)}  fs={r['field_std']:6.3f}  if={r['interior_frac']:.3f}  "
              f"hf={r['hf_energy']:.2f}  k3={r['k3']:.3f}  {r['mix_source']}")

    # ---- 2. buckets ---- #
    print("\n--- buckets (union partition) ---")
    print(f"interior-only: {len(interior_only)}   flat-only: {len(flat_only)}   both: {len(both)}")
    print(f"  interior-only ids: {[short(i) for i in ids_desc(interior_only)]}")
    print(f"  flat-only     ids: {[short(i) for i in ids_desc(flat_only)]}")
    print(f"  both          ids: {[short(i) for i in ids_desc(both)]}")
    print("  flat-only interior_frac (must be ~0.000 -> interior gate waves them through,"
          " so field_std is load-bearing):")
    for oid in ids_desc(flat_only):
        r = by_id[oid]
        print(f"    {short(oid)}  interior_frac={r['interior_frac']:.6f}  field_std={r['field_std']:.3f}"
              f"  k3={r['k3']:.3f}")

    # ---- 3. union / retained counts ---- #
    print("\n--- union / retained ---")
    print(f"  |union dropped| = {len(union)} / {n}   |retained| = {len(retained)}")

    # ---- 4. mix-source breakdown of the union ---- #
    print("\n--- mix-source breakdown ---")
    src_union = Counter(by_id[i]["mix_source"] for i in union)
    src_all = Counter(r["mix_source"] for r in distinct)
    for k in sorted(src_all):
        print(f"  {k:8s}: dropped {src_union.get(k,0):2d} / {src_all[k]:2d} distinct")
    print("  (expect exploit stays clean ~6/52; explore/native carry the junk.)")

    # ---- 5. reward-baseline preview (no refit) ---- #
    import statistics as st
    k3_drop = [by_id[i]["k3"] for i in union]
    k3_keep = [r["k3"] for r in retained]
    k3_all = [r["k3"] for r in distinct]
    print("\n--- reward-baseline preview (magnitude the theta_hat recompute will act on) ---")
    print(f"  mean k3  dropped  = {st.mean(k3_drop):.4f}   (n={len(k3_drop)})")
    print(f"  mean k3  retained = {st.mean(k3_keep):.4f}   (n={len(k3_keep)})")
    print(f"  mean k3  overall  = {st.mean(k3_all):.4f}   (n={len(k3_all)})")
    print(f"  retained - overall = {st.mean(k3_keep) - st.mean(k3_all):+.4f}  "
          f"(reward pool lifts by this once the junk is gated)")

    # ---- 6. drop_manifest.json (authoritative artifact) ---- #
    manifest = {
        "interior_gate": ids_desc(interior_hit),
        "flat_gate": ids_desc(flat_hit),
        "both": ids_desc(both),
        "union": ids_desc(union),
        "thresholds": {"interior": INTERIOR_THR, "field_std": FIELD_STD_THR},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  drop_manifest -> {MANIFEST}")
    print(f"    interior_gate={len(manifest['interior_gate'])} flat_gate={len(manifest['flat_gate'])} "
          f"both={len(manifest['both'])} union={len(manifest['union'])}")

    # ---- 7. marked contact sheet (union tiles get a drop X; k3-descending) ---- #
    def _lab(r, mark):
        return (f"{short(r['id'])} k{r['k3']:.2f} i{r['interior_frac']*100:.0f}% "
                f"fs{r['field_std']:.0f}{mark}")

    dtiles = []
    for r in sorted(distinct, key=lambda x: -x["k3"]):
        oid = r["id"]
        if oid in union:
            gates = ("I" if oid in interior_hit else "") + ("F" if oid in flat_hit else "")
            mtile = MARK_DIR / f"{oid}.jpg"
            mark_tile(TILE_DIR / f"{oid}.jpg", mtile, gates)
            dtiles.append((mtile, _lab(r, f" DROP-{gates}")))
        else:
            dtiles.append((TILE_DIR / f"{oid}.jpg", _lab(r, "")))
    sheet = OUT / "contact_sheet_marked.png"
    build_contact_sheet(
        dtiles, [], sheet,
        f"guard drop manifest — {len(union)}/{n} dropped (red X = dropped; "
        f"DROP-I interior_frac>={INTERIOR_THR}, DROP-F field_std<{FIELD_STD_THR}); k3-desc")
    print(f"  marked sheet  -> {sheet}")
    print("\nDONE (report-only; ledger + theta_hat untouched).")


if __name__ == "__main__":
    main()
