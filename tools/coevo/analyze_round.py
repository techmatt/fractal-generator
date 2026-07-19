#!/usr/bin/env python
r"""Co-evolution round — filter + stratify by v6, render review thumbnails, build
per-band contact sheets + report.md.

Consumes the round's guard-OFF gather ledger (out/coevo_round/<ts>/gather/mandelbrot/
outcome_ledger.jsonl) written by coevo_round.py. Steps 2-5 of the prompt:

  2. Confirm the v6 stamp on every fresh row. Drop model-free degenerate-guard fails
     (guard_verdict != "pass"). Over guard-pass locations, stratify by the live v6
     deg-2 p_good (t_good=0.24) into three bands: clear-reject / near-boundary
     (bracketing 0.24) / clear-pass. Report band counts + the overall v6-q3 pass-rate
     on the new-default guard-pass population.
  3. Sample up to N per band (pad thin bands). Render each via Recipe 1
     (corpus_common.render_corpus_crop = render-one --palette) with the ablation's
     fixed neutral palette (twilight_shifted) at browsable res — single palette,
     geometry triage. NOT the 76-palette corpus-crop; NOT full-res.
  4. Per-band contact sheets; each thumbnail tagged with its v6 numbers (p_good,
     p_notbad, k3, decoded_class), ordered within band by p_good.
  5. report.md — band counts, new-default v6-pass rate, sheet pointers, framing.

No verdict (the eye pass is Matt's). No relabel/retrain/corpus-crop/full-res.

  uv run python -u tools/coevo/analyze_round.py --round-ts <ts> --n-per-band 40
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))   # corpus_common, location

from corpus_common import render_block, render_corpus_crop, is_current_decoded  # noqa: E402

# --- render recipe (Recipe 1 = render-one --palette) at browsable/triage res -----
PALETTE = "twilight_shifted"                          # the ablation's fixed neutral palette
PALETTE_SOURCE = ROOT / "data" / "palettes" / "clean_colormaps.json"
CROP_W, CROP_H, CROP_SS, CROP_FILTER = 640, 360, 2, "lanczos3"   # browsable, NOT full-res
CROP_MAXITER = 8000                                   # matches the gather corpus recipe
JPGQ = 90
WORKERS = 4                                           # project cap

# --- v6 deg-2 operating point + band thresholds on p_good (bracketing 0.24) -------
T_GOOD = 0.24
BAND_LO = 0.15      # clear-reject   : p_good <  0.15
BAND_HI = 0.35      # clear-pass     : p_good >= 0.35 ; near-boundary : [0.15, 0.35)
BANDS = ["clear_reject", "near_boundary", "clear_pass"]
BAND_COLOR = {"clear_reject": (230, 90, 80), "near_boundary": (240, 200, 60),
              "clear_pass": (80, 210, 110)}


def band_of(p_good: float) -> str:
    if p_good < BAND_LO:
        return "clear_reject"
    if p_good < BAND_HI:
        return "near_boundary"
    return "clear_pass"


def load_rows(round_dir: Path) -> list[dict]:
    led = round_dir / "gather" / "mandelbrot" / "outcome_ledger.jsonl"
    if not led.exists():
        raise SystemExit(f"no ledger at {led}")
    rows = []
    for line in open(led, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def sample_band(rows: list[dict], n: int) -> list[dict]:
    """Up to n rows, sorted by p_good. If the band is larger than n, take an even
    stride across the p_good-sorted band so the sheet spans the whole band (keeps the
    boundary legible); thin bands pass through whole (padded = take all)."""
    s = sorted(rows, key=lambda r: r["p_good"])
    if len(s) <= n:
        return s
    idx = [round(i * (len(s) - 1) / (n - 1)) for i in range(n)]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(s[i])
    return out


def render_one(row: dict, crops_dir: Path) -> tuple[str, bool]:
    oid = row["id"]
    out = crops_dir / f"{oid}.jpg"
    if out.exists():
        return oid, True
    block = render_block(cx=row["outcome_cx"], cy=row["outcome_cy"], fw=row["outcome_fw"],
                         maxiter=CROP_MAXITER, palette=PALETTE, composition="center",
                         width=CROP_W, height=CROP_H, ss=CROP_SS, filter=CROP_FILTER,
                         interior_mode="black")
    block["fractal_type"] = "mandelbrot"
    try:
        render_corpus_crop(block, out, palette_source=str(PALETTE_SOURCE),
                           jpg_quality=JPGQ, cwd=str(ROOT))
    except Exception as e:
        sys.stderr.write(f"[render {oid}] FAILED: {e}\n")
        return oid, False
    return oid, True


def build_sheet(band: str, rows: list[dict], crops_dir: Path, out_png: Path, subtitle: str):
    """Per-band contact sheet, tiles ordered by p_good, tagged with v6 numbers."""
    TW, TH, PAD, LBL, HDR = 360, 203, 8, 30, 46
    NCOL = 5
    rows = sorted(rows, key=lambda r: r["p_good"])
    n = max(1, len(rows))
    nrow = (n + NCOL - 1) // NCOL
    cw, ch = TW + 2 * PAD, TH + LBL + 2 * PAD
    W, H = NCOL * cw, HDR + nrow * ch
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    col = BAND_COLOR[band]
    d.rectangle([0, 0, W, HDR], fill=(30, 30, 34))
    d.text((10, 8), f"[{band}]  {subtitle}", fill=col)
    d.text((10, 26), "ordered by p_good ->  (tag: pg=p_good  nb=p_notbad  k3  cls=decoded_class)",
           fill=(180, 180, 188))
    for k, r in enumerate(rows):
        rr, cc = divmod(k, NCOL)
        x, y = cc * cw + PAD, HDR + rr * ch + PAD
        tp = crops_dir / f"{r['id']}.jpg"
        if tp.exists():
            im = Image.open(tp).convert("RGB").resize((TW, TH))
            sheet.paste(im, (x, y))
        else:
            d.rectangle([x, y, x + TW, y + TH], fill=(50, 30, 30))
            d.text((x + 6, y + 6), "render failed", fill=(230, 160, 160))
        for t in range(2):
            d.rectangle([x - 1 - t, y - 1 - t, x + TW + t, y + TH + t], outline=col)
        d.rectangle([x, y + TH, x + TW, y + TH + LBL], fill=(26, 26, 30))
        tag = (f"pg={r['p_good']:.3f} nb={r['p_notbad']:.3f} "
               f"k3={r['k3']:.2f} cls{r['decoded_class']}")
        d.text((x + 4, y + TH + 3), tag, fill=(215, 215, 222))
        d.text((x + 4, y + TH + 16), r["id"][-10:], fill=(150, 150, 160))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round-ts", required=True, help="round dir name under out/coevo_round/")
    ap.add_argument("--n-per-band", type=int, default=40)
    args = ap.parse_args()

    round_dir = ROOT / "out" / "coevo_round" / args.round_ts
    if not round_dir.exists():
        raise SystemExit(f"no round dir {round_dir}")
    crops_dir = round_dir / "crops"
    sheets_dir = round_dir / "sheets"
    crops_dir.mkdir(exist_ok=True)
    sheets_dir.mkdir(exist_ok=True)

    rows = load_rows(round_dir)
    n_total = len(rows)

    # --- step 2: confirm the current-model stamp on every fresh row -----------
    from corpus_common import active_scorer_version
    cur = active_scorer_version()
    n_cur = sum(1 for r in rows if is_current_decoded(r))
    stamp_clean = (n_cur == n_total)
    print(f"current stamp: {n_cur}/{n_total} rows scorer_version=='{cur}'  "
          f"({'CLEAN' if stamp_clean else 'MIXED — investigate'})")

    # --- drop degenerate-guard fails (not the co-evolution question) ----------
    verdict_counts = {}
    for r in rows:
        v = r.get("guard_verdict", "?")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    guard_pass = [r for r in rows if r.get("guard_verdict") == "pass"]
    n_gp = len(guard_pass)
    print(f"guard verdict split: " + "  ".join(f"{k}={v}" for k, v in sorted(verdict_counts.items())))
    print(f"guard-pass population (kept): {n_gp}/{n_total}")

    # --- stratify guard-pass by p_good ---------------------------------------
    by_band = {b: [] for b in BANDS}
    for r in guard_pass:
        by_band[band_of(r["p_good"])].append(r)
    n_q3 = sum(1 for r in guard_pass if r.get("decoded_class") == 3)
    q3_rate = n_q3 / n_gp if n_gp else 0.0
    print(f"\nband counts over guard-pass (p_good thresholds {BAND_LO}/{BAND_HI}, t_good={T_GOOD}):")
    for b in BANDS:
        print(f"  {b:14s}: {len(by_band[b])}")
    print(f"overall v6-q3 pass-rate on guard-pass pop: {n_q3}/{n_gp} = {q3_rate:.1%}")

    # --- step 3: sample + render ---------------------------------------------
    sampled = {b: sample_band(by_band[b], args.n_per_band) for b in BANDS}
    to_render = [r for b in BANDS for r in sampled[b]]
    print(f"\nrendering {len(to_render)} crops (Recipe 1, palette={PALETTE}, "
          f"{CROP_W}x{CROP_H} ss{CROP_SS} maxiter{CROP_MAXITER}, {WORKERS} workers)...")
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _oid, good in ex.map(lambda r: render_one(r, crops_dir), to_render):
            ok += int(good)
    print(f"  rendered {ok}/{len(to_render)} ok")

    # --- step 4: per-band sheets ---------------------------------------------
    sheet_paths = {}
    for b in BANDS:
        sp = sheets_dir / f"band_{b}.png"
        sub = (f"{len(sampled[b])} shown of {len(by_band[b])} guard-pass  "
               f"(v6 deg-2, t_good={T_GOOD})")
        build_sheet(b, sampled[b], crops_dir, sp, sub)
        sheet_paths[b] = sp
        print(f"  sheet [{b}] -> {sp}")

    # --- persist the stratification (traceable for a later relabel round) -----
    strat = {
        "round_ts": args.round_ts, "n_total": n_total, "n_current_stamped": n_cur,
        "current_version": cur,
        "stamp_clean": stamp_clean, "guard_verdict_counts": verdict_counts,
        "guard_pass": n_gp, "t_good": T_GOOD, "band_lo": BAND_LO, "band_hi": BAND_HI,
        "band_counts": {b: len(by_band[b]) for b in BANDS},
        "band_sampled": {b: len(sampled[b]) for b in BANDS},
        "v6_q3_pass": n_q3, "v6_q3_pass_rate": q3_rate,
        "sampled_ids": {b: [r["id"] for r in sampled[b]] for b in BANDS},
    }
    (round_dir / "stratification.json").write_text(json.dumps(strat, indent=2), encoding="utf-8")

    # --- step 5: report.md ----------------------------------------------------
    write_report(round_dir, strat, sheet_paths, verdict_counts)
    print(f"\nreport -> {round_dir / 'report.md'}")


def write_report(round_dir, s, sheet_paths, verdict_counts):
    bc = s["band_counts"]
    bs = s["band_sampled"]
    vc = "  ".join(f"`{k}`={v}" for k, v in sorted(verdict_counts.items()))
    md = f"""# Co-evolution round — v6-gap diagnosis (new random-survivor default)

Round `{s['round_ts']}` · Mandelbrot c-plane deg-2 · guard-OFF gather on the shipped
random-survivor selection default (no `--selection` flag) · scored by v6.

## current-model stamp
{s['n_current_stamped']}/{s['n_total']} fresh rows carry `scorer_version=="{s.get('current_version','?')}"` — \
**{'CLEAN' if s['stamp_clean'] else 'MIXED (investigate before consuming as current verdicts)'}**. \
Unlike the head-v3 slice, this round's ledger is a clean current-model decode.

## Population
- Total gather outcomes (guard-OFF): **{s['n_total']}**
- Degenerate-guard verdict split: {vc}
- **Guard-pass (kept)**: **{s['guard_pass']}** — the model-free degenerate-guard fails are
  dropped (genuinely bad, not under-rated; not the co-evolution question).

## v6 stratification of the guard-pass population
p_good bands bracket the deg-2 operating point `t_good={s['t_good']}` \
(thresholds {s['band_lo']} / {s['band_hi']}):

| band | count | sampled to sheet |
|---|---:|---:|
| clear_reject (p_good < {s['band_lo']}) | {bc['clear_reject']} | {bs['clear_reject']} |
| near_boundary ({s['band_lo']} ≤ p_good < {s['band_hi']}) | {bc['near_boundary']} | {bs['near_boundary']} |
| clear_pass (p_good ≥ {s['band_hi']}) | {bc['clear_pass']} | {bs['clear_pass']} |

**Overall v6-q3 pass-rate on the new-default guard-pass population: \
{s['v6_q3_pass']}/{s['guard_pass']} = {s['v6_q3_pass_rate']:.1%}** \
(q3 = decoded_class==3 = p_notbad≥0.5 ∧ p_good≥{s['t_good']}).

## Review sheets
Each thumbnail is Recipe 1 (`render-one --palette`) at the ablation's fixed neutral
palette (`twilight_shifted`), browsable res — single-palette geometry triage, ordered
within band by p_good so the boundary is legible. Tags: `pg`=p_good `nb`=p_notbad `k3`
`cls`=decoded_class.

- clear_reject : `{sheet_paths['clear_reject'].relative_to(round_dir)}`
- near_boundary: `{sheet_paths['near_boundary'].relative_to(round_dir)}`
- clear_pass   : `{sheet_paths['clear_pass'].relative_to(round_dir)}`

## Framing (the eye pass is Matt's — no verdict here)
The **v6-reject band (clear_reject + the reject side of near_boundary) is the
hypothesis-bearing population**: if it is full of locations worth passing, then
good-but-rejected = v6 is under-rating the shifted distribution the random-survivor
default now surfaces → a relabel need. The **clear_pass band is the agreement
control** — watch for the opposite error there (bad-but-passed). What sizes the next
step: if the reject / near-boundary sheets are full of keepers, the gap is real and the
next prompt seeds a relabel round from exactly these locations (corpus-crop → label UI
→ merge → v6 retrain). If not, v6 is keeping up with the new default and no urgent
relabel — the null dies cheap here.

_Stratification detail + sampled ids: `stratification.json`. No relabel/retrain/
corpus-crop/full-res performed._
"""
    (round_dir / "report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
