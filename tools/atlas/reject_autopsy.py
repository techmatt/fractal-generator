#!/usr/bin/env python
r"""Reject autopsy — fate-stratified visual samples of a steered run's admissions AND rejects.

The julia dup bug lived in the rejects, where no human ever looked: a metric that was
aggregate-plausible but semantically wrong, caught only by eyeballing an *absence*. This
module makes the rejects visible. It buckets a run's outcomes by FATE and renders a small
random sample of each into one fate-labeled contact sheet, at the exact reframe search
fidelity (640x360 ss2, deploy palette) the classifier scored — so a wrong-but-plausible
reject shows up as a picture, not just a count.

Fates (each renderable now that harvest_log carries cx/cy/fw + julia c):
  * admit_q3        — ledger: distinct q3 (the kept population, for contrast)
  * coord_dup       — ledger: q3 but near an existing cloud place (distinct == False)
  * guard_fail      — ledger: guard sentinel (guard_pass == False)
  * precanon_dup    — harvest_log: skipped before the confirmation render (item-4 filter)
  * canon_not_q3    — harvest_log: canonical decode != q3 (never reached reframe)

Standing use: `julia_fix_readout.py` calls `render_autopsy_sheet(...)` by default, so every
readout drops a visual sample beside the numbers. Older runs whose harvest_log predates the
coord logging silently skip the two harvest-only fates (a note is emitted).

Standalone:
  uv run python tools/atlas/reject_autopsy.py --run data/discovery/campaign1/breadth --n 10
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

import prescreen
import production_seeder as ps
from steered_frontier import render_family_of

WORKERS = 4                       # project hard cap on parallel workers
FATES = ["admit_q3", "coord_dup", "guard_fail", "precanon_dup", "canon_not_q3"]
FATE_COLOR = {
    "admit_q3": (60, 200, 90), "coord_dup": (235, 180, 40), "guard_fail": (220, 70, 70),
    "precanon_dup": (150, 120, 230), "canon_not_q3": (110, 170, 230),
}


def _load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def _seed_c(re, im):
    return None if re is None else (str(re), str(im))


def collect_fates(run: Path) -> dict:
    """Bucket a run's outcomes into renderable {fate: [item]} where each item carries
    (partition, render_family, c, cx, cy, fw, caption). Ledger fates always have coords;
    harvest fates need cx/cy/fw (present only in coord-logged runs)."""
    rows = _load_jsonl(run / "outcome_ledger.jsonl")
    harvest = _load_jsonl(run / "harvest_log.jsonl")
    out: dict[str, list] = {f: [] for f in FATES}

    for r in rows:
        part = r.get("family", "mandelbrot")
        base = dict(partition=part, c=_seed_c(r.get("julia_c_re"), r.get("julia_c_im")),
                    cx=r["outcome_cx"], cy=r["outcome_cy"], fw=r["outcome_fw"])
        gp = r.get("guard_pass", True)
        dc = r.get("decoded_class")
        if not gp:
            base["caption"] = f"{part} d{r.get('reached_depth','?')} guard-fail"
            out["guard_fail"].append(base)
        elif dc == 3 and r.get("distinct"):
            base["caption"] = f"{part} d{r.get('reached_depth','?')} pg{_f(r.get('canon_pgood'))}"
            out["admit_q3"].append(base)
        elif dc == 3 and r.get("distinct") is False:
            base["caption"] = f"{part} d{r.get('reached_depth','?')} dup~{str(r.get('dup_of'))[-6:]}"
            out["coord_dup"].append(base)

    # harvest-only fates — need coords (coord-logged runs only)
    coordless = 0
    for h in harvest:
        if h.get("cx") is None:
            coordless += 1
            continue
        part = h["partition"]
        base = dict(partition=part, render_family=render_family_of(part),
                    c=_seed_c(h.get("julia_c_re"), h.get("julia_c_im")),
                    cx=h["cx"], cy=h["cy"], fw=h["fw"])
        cd = h.get("canon_decoded")
        if h.get("precanon_dup") is not None:
            base["caption"] = f"{part} d{h.get('depth','?')} precanon~{str(h['precanon_dup'])[-6:]}"
            out["precanon_dup"].append(base)
        elif cd == 3 and not h.get("admitted"):
            # canonical-q3 but rejected at the pre-reframe near-dup (a coord-dup caught before
            # the 12-render reframe — never gets a ledger row, only this log line).
            base["caption"] = f"{part} d{h.get('depth','?')} q3-dup pg{_f(h.get('canon_pgood'))}"
            out["coord_dup"].append(base)
        elif cd is not None and cd != 3 and not h.get("admitted"):
            base["caption"] = f"{part} d{h.get('depth','?')} canon={cd} pg{_f(h.get('canon_pgood'))}"
            out["canon_not_q3"].append(base)
    out["_coordless_harvest"] = coordless
    return out


def _f(x):
    return "?" if x is None else f"{float(x):.2f}"


def _render_family_of(part: str) -> str:
    return "mandelbrot" if part == "mandelbrot" else render_family_of(part)


def render_autopsy_sheet(run: Path, out_png: Path, n_per_fate: int = 10,
                         seed: int = 0, workers: int = WORKERS) -> dict:
    """Render up to `n_per_fate` random tiles per fate and lay them out as one labeled sheet.
    Returns {fate: n_rendered, ...} + the sheet path. ~= len(FATES)*n_per_fate tiles."""
    buckets = collect_fates(run)
    rng = random.Random(seed)
    tmp = out_png.parent / (out_png.stem + "_tiles")
    tmp.mkdir(parents=True, exist_ok=True)

    jobs = []       # (fate, idx, item, tile_path)
    for fate in FATES:
        items = buckets[fate]
        pick = rng.sample(items, min(n_per_fate, len(items)))
        for i, it in enumerate(pick):
            jobs.append((fate, i, it, tmp / f"{fate}_{i}.jpg"))

    def _do(job):
        fate, i, it, tile = job
        ok, err = prescreen._render(it["cx"], it["cy"], it["fw"], tile,
                                    family=_render_family_of(it["partition"]), c=it["c"])
        return job, ok

    rendered = {f: 0 for f in FATES}
    done = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for job, ok in ex.map(_do, jobs):
            fate, i, it, tile = job
            if ok:
                rendered[fate] += 1
                done[(fate, i)] = (it, tile)

    _layout(run, out_png, buckets, done, n_per_fate)
    return {"sheet": str(out_png), "rendered": rendered,
            "counts": {f: len(buckets[f]) for f in FATES},
            "coordless_harvest": buckets["_coordless_harvest"]}


# thumbnail + grid geometry
TW, TH = 240, 135
PAD, CAPH, LBLW = 5, 16, 150
COLS = 10


def _fate_rows(shown: int) -> int:
    return (shown + COLS - 1) // COLS if shown else 0


def _layout(run: Path, out_png: Path, buckets: dict, done: dict, n_per_fate: int):
    cell_w, cell_h = TW + 2 * PAD, TH + CAPH + 2 * PAD
    shown = {f: min(n_per_fate, len(buckets[f])) for f in FATES}
    W = LBLW + COLS * cell_w
    H = 40 + sum(24 + _fate_rows(shown[f]) * cell_h for f in FATES)
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), f"reject autopsy — {run.name}   "
           f"(fidelity 640x360 ss2, deploy palette; green=admit for contrast)",
           fill=(235, 235, 235))
    y = 40
    for fate in FATES:
        col = FATE_COLOR[fate]
        d.rectangle([10, y, 22, y + 12], fill=col)
        d.text((28, y), f"{fate}   (n={len(buckets[fate])}, showing {shown[fate]})", fill=col)
        y += 20
        for i in range(shown[fate]):
            r_, c_ = divmod(i, COLS)
            x = LBLW + c_ * cell_w + PAD
            yy = y + r_ * cell_h + PAD
            entry = done.get((fate, i))
            if entry is None:
                d.rectangle([x, yy, x + TW, yy + TH], outline=(90, 40, 40))
                d.text((x + 4, yy + TH // 2), "render fail", fill=(200, 120, 120))
                continue
            it, tile = entry
            try:
                im = Image.open(tile).convert("RGB").resize((TW, TH))
                sheet.paste(im, (x, yy))
            except Exception:
                pass
            d.rectangle([x - 1, yy - 1, x + TW, yy + TH], outline=col)
            d.rectangle([x, yy + TH, x + TW, yy + TH + CAPH], fill=(28, 28, 32))
            d.text((x + 3, yy + TH + 2), it["caption"][:46], fill=(210, 210, 215))
        y += _fate_rows(shown[fate]) * cell_h + 4
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a steered run dir (ledger + harvest_log)")
    ap.add_argument("--out", default=None, help="sheet PNG (default: <run>/reject_autopsy.png)")
    ap.add_argument("--n", type=int, default=10, help="tiles per fate (~50 total across 5 fates)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    out = Path(args.out) if args.out else run / "reject_autopsy.png"
    res = render_autopsy_sheet(run, out, n_per_fate=args.n, seed=args.seed)
    print(json.dumps(res, indent=2))
    if res["coordless_harvest"]:
        print(f"note: {res['coordless_harvest']} harvest rows lack coords (pre-coord-logging run) "
              f"-> precanon_dup/canon_not_q3 fates undersampled for this run.")


if __name__ == "__main__":
    main()
