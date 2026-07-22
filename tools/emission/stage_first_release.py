#!/usr/bin/env python
r"""stage_first_release.py — assemble the emission-driver SNAPSHOT for the first real
release by UNIONING the two committed intake passes WITHOUT re-embedding.

Honors prompts/first_release.md: "Intake is already done; start from the current library,
do not re-intake." The current-decoded library is the union of the two committed intake
passes:

  * library_intake_2 (out/emission/library_intake_2) — 819 admitted / 745 clusters, adds
    phoenix; ids + cluster tags are kept VERBATIM so the measure's phoenix#196..236 override
    stays valid (campaign1 has no phoenix).
  * campaign1        (out/emission/campaign1)        — 568 admitted / 523 clusters.

The two passes clustered SEPARATELY and their run-scoped ids COLLIDE: 60 `st_<fam>_<arm>_<seq>`
ids are reused across campaigns for DIFFERENT locations (verified: same id, different coords).
A naive union by id would silently collapse 60 distinct wallpapers. So campaign1 ids are
DISAMBIGUATED with a `c1__` prefix — in id-prefixed ledger COPIES (so the driver's own
load_admitted re-derives the prefixed ids) AND in this snapshot. campaign1's `<fam>#<k>`
cluster indices are offset past library_intake_2's per-family count so the two tag namespaces
never collide. library_intake_2 stays the unprefixed, measure-native space.

Result: 1387 globally-distinct locations, each with a reused morph-CLIP embedding + cached
640x360 smooth field (no re-render, no re-embed). The driver then runs with
`--out out/first_release` and `--ledger <the 6 ledgers>` (2 prefixed campaign1 copies + 4
library_intake_2 originals) and REUSES this snapshot (descriptor.load_embs + the cached
intake.json), skipping intake entirely.

Outputs (out/first_release/):
  ledgers/c1__breadth.jsonl, c1__dive.jsonl   id-prefixed campaign1 ledger copies
  intake.json                                  {cluster_tags, fields, n_admitted}
  morph_embs.npz                               descriptor._save_embs format (ids, emb)
  stage_report.json                            reconcile + validation + stale-rejection proof

  uv run python tools/emission/stage_first_release.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "wallpaper"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import corpus_common as cc                                # noqa: E402
from tools.emission import descriptor as D                # noqa: E402
from tools.wallpaper import library_annotate as la        # noqa: E402
from tools.wallpaper import library_store as store        # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

C1_DIR = ROOT / "out" / "emission" / "campaign1"
I2_DIR = ROOT / "out" / "emission" / "library_intake_2"
OUT = ROOT / "out" / "first_release"

C1_LEDGERS = [
    ("c1_breadth", ROOT / "data" / "discovery" / "campaign1" / "breadth" / "outcome_ledger.jsonl"),
    ("c1_dive",    ROOT / "data" / "discovery" / "campaign1" / "dive"    / "outcome_ledger.jsonl"),
]
I2_LEDGERS = [
    ("c2_breadth",      ROOT / "data" / "discovery" / "campaign2" / "breadth" / "outcome_ledger.jsonl"),
    ("c2_dive",         ROOT / "data" / "discovery" / "campaign2" / "dive"    / "outcome_ledger.jsonl"),
    ("phoenix_grid",    ROOT / "data" / "discovery" / "phoenix_grid" / "grid" / "outcome_ledger_v7_t45.jsonl"),
    ("classic_phoenix", ROOT / "data" / "discovery" / "classic_phoenix" / "outcome_ledger.jsonl"),
]
C1_PREFIX = "c1__"
# The 41 classic-phoenix clusters the measure up-weights; MUST survive verbatim.
MEASURE_PHOENIX = [f"phoenix#{k}" for k in range(196, 237)]


def _admitted_rows(ledgers) -> dict:
    """id -> row over `ledgers`, cross-ledger id-dedup first-wins — the same union the
    intake pass used, so its cluster_tags keys line up with these ids."""
    out = {}
    for _tag, path in ledgers:
        for r in D.load_admitted(path):
            out.setdefault(r["id"], r)      # first ledger wins (matches driver dedup)
    return out


def _field_paths(row) -> tuple[Path, Path]:
    loc = D.location_of(row)
    stem = store.field_stem(loc, "smooth", la.W, la.H, la.SS)
    return (stem + ".bin"), (stem + ".json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ledgers").mkdir(parents=True, exist_ok=True)

    c1_tags = json.loads((C1_DIR / "intake.json").read_text(encoding="utf-8"))["cluster_tags"]
    i2_tags = json.loads((I2_DIR / "intake.json").read_text(encoding="utf-8"))["cluster_tags"]
    print(f"[stage] campaign1 tags={len(c1_tags)}  library_intake_2 tags={len(i2_tags)}", flush=True)

    # per-family offset so campaign1 <fam>#<k> lands past every intake2 <fam> index.
    off: dict = {}
    for tag in i2_tags.values():
        fam, k = tag.rsplit("#", 1)
        off[fam] = max(off.get(fam, 0), int(k) + 1)

    c1_rows = _admitted_rows(C1_LEDGERS)
    i2_rows = _admitted_rows(I2_LEDGERS)
    assert set(c1_rows) == set(c1_tags), \
        f"campaign1 admitted rows ({len(c1_rows)}) != its cluster_tags ({len(c1_tags)})"
    assert set(i2_rows) == set(i2_tags), \
        f"library_intake_2 admitted rows ({len(i2_rows)}) != its cluster_tags ({len(i2_tags)})"

    merged_tags: dict = {}
    embs: dict = {}
    fields: dict = {}
    missing_field, missing_emb = [], []

    def _ingest(rid_out, orig_id, tag_out, row, pass_dir):
        binf, jsonf = _field_paths(row)
        binp = pass_dir / "fields" / binf
        embp = pass_dir / "embs" / f"{orig_id}.npy"
        if not binp.exists():
            missing_field.append(rid_out)
        if not embp.exists():
            missing_emb.append(rid_out)
            return
        merged_tags[rid_out] = tag_out
        fields[rid_out] = [str(binp), str(pass_dir / "fields" / jsonf)]
        embs[rid_out] = np.load(embp).astype(np.float32)

    # library_intake_2 — verbatim ids + tags (measure-native, phoenix here).
    for rid, tag in i2_tags.items():
        _ingest(rid, rid, tag, i2_rows[rid], I2_DIR)

    # campaign1 — prefixed ids, family-offset cluster indices.
    for rid, tag in c1_tags.items():
        fam, k = tag.rsplit("#", 1)
        _ingest(C1_PREFIX + rid, rid, f"{fam}#{int(k) + off[fam]}", c1_rows[rid], C1_DIR)

    if missing_field or missing_emb:
        raise SystemExit(f"[stage] missing artifacts — fields:{len(missing_field)} "
                         f"embs:{len(missing_emb)} (e.g. field {missing_field[:2]} "
                         f"emb {missing_emb[:2]})")

    # id-prefixed campaign1 ledger COPIES (ALL rows; only the id string changes).
    for tag, path in C1_LEDGERS:
        dst = OUT / "ledgers" / f"{C1_PREFIX}{path.parent.name}.jsonl"
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["id"] = C1_PREFIX + r["id"]
            lines.append(json.dumps(r))
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[stage] wrote {dst.relative_to(ROOT)} ({len(lines)} rows)", flush=True)

    # ---- validation --------------------------------------------------------- #
    n = len(merged_tags)
    expected = len(i2_tags) + len(c1_tags)
    assert n == expected, f"merged {n} != expected {expected}"
    # namespace collision check: every campaign1 tag distinct from every intake2 tag.
    i2_vals = set(i2_tags.values())
    c1_new = {merged_tags[C1_PREFIX + r] for r in c1_tags}
    collide = i2_vals & c1_new
    assert not collide, f"campaign1/intake2 tag collision: {list(collide)[:5]}"
    # phoenix measure address preserved verbatim.
    have_phx = set(merged_tags.values())
    missing_phx = [t for t in MEASURE_PHOENIX if t not in have_phx]
    assert not missing_phx, f"measure phoenix clusters missing: {missing_phx[:5]}"
    # embedding dim consistent.
    dims = {e.shape[0] for e in embs.values()}
    assert len(dims) == 1, f"inconsistent emb dims: {dims}"

    # driver-parity reconcile: load the 6 ledgers exactly as the driver will and confirm
    # the admitted id set equals the snapshot id set (no deferral, no collision).
    six = [(C1_PREFIX + "breadth", OUT / "ledgers" / "c1__breadth.jsonl"),
           (C1_PREFIX + "dive",    OUT / "ledgers" / "c1__dive.jsonl")] + I2_LEDGERS
    driver_ids = set(_admitted_rows(six))
    only_snap = set(merged_tags) - driver_ids
    only_drv = driver_ids - set(merged_tags)
    assert not only_snap and not only_drv, \
        f"driver/snapshot id mismatch: snap-only {list(only_snap)[:3]} drv-only {list(only_drv)[:3]}"

    # ---- stale-rejection proof (acceptance) --------------------------------- #
    # A real v6 (non-current) row from a legacy ledger: prove the admission predicate
    # rejects it. All 1387 library rows are already current (0 stale in the 6 ledgers).
    stale = {"checked": False}
    legacy = ROOT / "data" / "discovery" / "outcome_ledger.jsonl"
    if legacy.exists():
        for line in legacy.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not cc.is_current_decoded(r):
                admitted_here = {x["id"] for x in D.load_admitted(legacy)}
                stale = {"checked": True, "ledger": str(legacy.relative_to(ROOT)),
                         "id": r["id"], "scorer_version": r.get("scorer_version"),
                         "is_current_decoded": cc.is_current_decoded(r),
                         "in_load_admitted": r["id"] in admitted_here}
                assert not stale["is_current_decoded"] and not stale["in_load_admitted"], \
                    "stale row leaked past load_admitted"
                break

    # ---- write snapshot ----------------------------------------------------- #
    D._save_embs(embs, OUT / "morph_embs.npz")
    (OUT / "intake.json").write_text(json.dumps(
        {"cluster_tags": merged_tags, "fields": fields, "n_admitted": n},
        indent=0), encoding="utf-8")

    fam_counts = Counter(t.rsplit("#", 1)[0] for t in merged_tags.values())
    n_clusters = len(set(merged_tags.values()))
    report = {
        "n_admitted": n,
        "n_distinct_clusters": n_clusters,
        "from_library_intake_2": len(i2_tags),
        "from_campaign1_prefixed": len(c1_tags),
        "emb_dim": dims.pop(),
        "per_type_clusters": {f: len({t for t in merged_tags.values() if t.rsplit('#', 1)[0] == f})
                              for f in sorted(fam_counts)},
        "per_type_locations": dict(Counter(
            (i2_rows[r]["family"] if r in i2_tags else c1_rows[r[len(C1_PREFIX):]]["family"])
            for r in merged_tags)),
        "measure_phoenix_preserved": True,
        "stale_rejection_proof": stale,
    }
    (OUT / "stage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[stage] snapshot: {n} locations, {n_clusters} distinct clusters "
          f"({len(i2_tags)} intake2 + {len(c1_tags)} campaign1-prefixed)", flush=True)
    print(f"[stage] per-type clusters: {report['per_type_clusters']}", flush=True)
    if stale.get("checked"):
        print(f"[stage] stale-rejection PROVEN: {stale['id']} ({stale['scorer_version']}) "
              f"is_current={stale['is_current_decoded']} in_admitted={stale['in_load_admitted']}",
              flush=True)
    print(f"[stage] wrote {(OUT / 'intake.json').relative_to(ROOT)}, "
          f"{(OUT / 'morph_embs.npz').relative_to(ROOT)}, stage_report.json", flush=True)


if __name__ == "__main__":
    main()
