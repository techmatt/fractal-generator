"""Build the v4 augmentation-cache render plan + cache manifest.

The augmentation scheme (identical 42-slot multiset for every location, blind to
class — see `prompts/v4_augmentation_precompute.md`):

  axes        : palette(6) x scale(3) x shift(2) x AA(2)
  aliased ss1 : full 6 x 3 x 2 = 36 grid   (cheap, genuinely aliased)
  ss4         : 6 palettes x scale 1.0 x center = 6 (deploy-quality)
  total       : 42 renders / location  (152124 over 3622 locations)

`fw_slot = scale * fw`. The `shifted` center is offset by `0.4 * fw_slot` in a
per-(palette,scale)-combo deterministic direction: the 18 palette x scale combos
get angles 2*pi*k/18, the SAME schedule for every location, so shift directions
are diverse yet class-balanced. The ss4 set pairs 1:1 with its aliased twins at
(palette, scale 1.0, center).

Outputs (resolution-independent — render resolution is a `v4-render-batch` flag):
  data/v4/plan.jsonl            : one render/row `{cx,cy,fw,palette,ss,filter,out}`
  data/v4/cache_manifest.jsonl  : one row/render, full provenance + render axes

  uv run python tools/v4/build_plan.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# aug_cache is relocated out of the tree: the manifest "path" stays repo-relative
# (portable + parity-stable), but the plan "out" the Rust batch writes to must be
# the resolved real location so a rebuild never re-materializes the cache in-tree.
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
from artifacts import resolve as resolve_artifact  # noqa: E402

MANIFEST = ROOT / "data" / "v4" / "manifest.jsonl"
ROSTER = ROOT / "data" / "v4" / "aug_roster.json"
CACHE_DIR = ROOT / "data" / "v4" / "aug_cache"
PLAN_OUT = ROOT / "data" / "v4" / "plan.jsonl"
CACHE_MANIFEST_OUT = ROOT / "data" / "v4" / "cache_manifest.jsonl"

SCALES = [0.7, 1.0, 1.3]
SHIFT_FRAC = 0.4          # |offset| = SHIFT_FRAC * fw_slot  (< 0.5*fw_slot bound)
FRACTAL_TYPE = "mandelbrot"


def scale_tok(s: float) -> str:
    # 0.7 -> "0.7", 1.0 -> "1.0", 1.3 -> "1.3" (stable, dot-free issues avoided)
    return f"{s:.1f}"


def fmt_f64(x: float) -> str:
    """Full-precision round-trippable f64 string (matches what Rust to_f64 reads)."""
    return repr(float(x))


def main() -> None:
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    palettes = [r["name"] for r in roster]
    fam_of = {r["name"]: r["palette_family"] for r in roster}
    assert len(palettes) == 6, f"expected 6 palettes, got {len(palettes)}"

    # Fixed shift-angle schedule over the 18 (palette, scale) combos. combo index
    # k = pi*3 + si; angle = 2*pi*k/18. Identical for every location.
    n_combo = len(palettes) * len(SCALES)  # 18
    angle_of = {}
    for pi, pal in enumerate(palettes):
        for si, sc in enumerate(SCALES):
            k = pi * len(SCALES) + si
            angle_of[(pal, sc)] = 2.0 * math.pi * k / n_combo

    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]

    plan_f = PLAN_OUT.open("w", encoding="utf-8")
    cm_f = CACHE_MANIFEST_OUT.open("w", encoding="utf-8")
    n_plan = 0

    for loc_id, r in enumerate(rows):
        cx0, cy0 = float(r["cx"]), float(r["cy"])
        fw0 = float(r["fw"])
        loc_dir = CACHE_DIR / str(loc_id)
        # carry-through provenance for the cache manifest
        base = dict(label=r["label"], split=r["split"], group_id=r["group_id"],
                    source=r["source"], biased=r["biased"])

        def emit(pal: str, sc: float, shift_id: str, ss: int):
            nonlocal n_plan
            fw_slot = sc * fw0
            if shift_id == "shifted":
                ang = angle_of[(pal, sc)]
                mag = SHIFT_FRAC * fw_slot
                dx, dy = mag * math.cos(ang), mag * math.sin(ang)
                cx, cy = cx0 + dx, cy0 + dy
            else:
                dx = dy = 0.0
                cx, cy = cx0, cy0
            aa = "aliased" if ss == 1 else "antialiased"
            aa_tok = f"ss{ss}"
            filt = "box" if ss == 1 else "lanczos3"
            fname = f"{pal}__s{scale_tok(sc)}__sh{shift_id}__{aa_tok}.jpg"
            # repo-relative (forward-slash) so the cache + manifest stay portable
            # across machines/model versions; the Rust batch runs from repo root.
            out = (loc_dir / fname).relative_to(ROOT).as_posix()
            # plan row (Rust render executor) — cx/cy/fw are f64-exact strings.
            # "out" is the RESOLVED absolute path (relocated out of the tree).
            plan_f.write(json.dumps({
                "cx": fmt_f64(cx), "cy": fmt_f64(cy), "fw": fmt_f64(fw_slot),
                "palette": pal, "ss": ss, "filter": filt,
                "out": resolve_artifact(out).as_posix(),
            }) + "\n")
            # cache manifest row (training-side provenance)
            cm_f.write(json.dumps({
                "location_id": loc_id, **base,
                "palette": pal, "palette_family": fam_of[pal],
                "scale": sc, "shift_id": shift_id,
                "shift_dx": dx, "shift_dy": dy,
                "aa_level": aa, "fractal_type": FRACTAL_TYPE,
                "path": out,
            }) + "\n")
            n_plan += 1

        # aliased ss1: full palette x scale x shift = 36
        for pal in palettes:
            for sc in SCALES:
                for shift_id in ("center", "shifted"):
                    emit(pal, sc, shift_id, ss=1)
        # antialiased ss4: 6 palettes x scale 1.0 x center = 6
        for pal in palettes:
            emit(pal, 1.0, "center", ss=4)

    plan_f.close()
    cm_f.close()
    print(f"locations           : {len(rows)}")
    print(f"renders/location    : 42  (36 aliased ss1 + 6 ss4)")
    print(f"plan rows           : {n_plan}")
    print(f"  expected          : {len(rows) * 42}")
    assert n_plan == len(rows) * 42
    print(f"wrote {PLAN_OUT}")
    print(f"wrote {CACHE_MANIFEST_OUT}")


if __name__ == "__main__":
    main()
