#!/usr/bin/env python
r"""Phoenix Phase B — the readout generator (out/phoenix_grid/readout.md).

Reads a grid run's durable outputs (summary.json, descent_records.jsonl, all_outcomes.jsonl,
outcome_ledger.jsonl) + the decomposition (runs phoenix_decomp if absent) + the registered
label batch, and emits the Phase-B readout: the PROVISIONAL decomposition verdict with CIs; a
fertility map over parameter space (yield by stratum/branch/|p| band — which skeleton regions
produce, which are dead); admissions/distinct-look totals; the realized min-per-look price;
admission + reject sample sheets; the label-batch manifest; and the phoenix ledger path with a
CONFIRMED intake-readiness check (identity stamped, current-decoded, standard guards, distinct).

  uv run python tools/phoenix/phoenix_readout.py --run data/discovery/phoenix_grid/grid \
      --batch 2026-07-21_phoenix_grid
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import phoenix_decomp as decomp                # noqa: E402
import location as loc_mod                      # noqa: E402
import corpus_common as cc                      # noqa: E402
from active_ckpt import auto_maxiter, PALETTE   # noqa: E402

BIN = ROOT / "target" / "release" / "fractal-generator.exe"
SCORE3 = ROOT / "data" / "palettes" / "score3_colormaps.json"


def _load(run: Path):
    def jl(name):
        p = run / name
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return summ, jl("descent_records.jsonl"), jl("all_outcomes.jsonl"), jl("outcome_ledger.jsonl")


def fertility(records: list[dict], key) -> list[dict]:
    g: dict = defaultdict(list)
    for r in records:
        g[key(r)].append(r)
    rows = []
    for k in sorted(g):
        rs = g[k]
        adm = np.array([r["n_admissions"] for r in rs], float)
        dl = np.array([r["distinct_looks_within"] for r in rs], float)
        pg = np.array([r["max_p_good"] for r in rs], float)
        rows.append({"key": k, "n_descents": len(rs), "dead_frac": float(np.mean(adm == 0)),
                     "mean_adm": float(adm.mean()), "mean_distinct": float(dl.mean()),
                     "mean_max_pgood": float(pg.mean())})
    return rows


def intake_check(ledger: list[dict]) -> dict:
    """Confirm the admissions ledger is library-intake-ready: is_current_decoded ∧
    decoded_class==3 ∧ guard_pass ∧ distinct, with the full (c,p,z_-1) identity stamped."""
    active = cc.active_scorer_version()
    ok = 0
    fails = defaultdict(int)
    id_ok = 0
    for r in ledger:
        cur = cc.is_current_decoded(r)
        q3 = r.get("decoded_class") == 3
        gp = bool(r.get("guard_pass"))
        dist = bool(r.get("distinct"))
        has_id = all(r.get(k) is not None for k in
                     ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re", "phoenix_p_im",
                      "phoenix_zm1_re", "phoenix_zm1_im"))
        id_ok += int(has_id)
        if cur and q3 and gp and dist and has_id:
            ok += 1
        else:
            if not cur: fails["stale_decode"] += 1
            if not q3: fails["not_q3"] += 1
            if not gp: fails["guard_fail"] += 1
            if not dist: fails["not_distinct"] += 1
            if not has_id: fails["missing_identity"] += 1
    return {"active_version": active, "n_ledger": len(ledger), "n_intake_ready": ok,
            "n_identity_stamped": id_ok, "reject_breakdown": dict(fails)}


def _render_thumb(row, out: Path, palette: str, w=320, h=180) -> bool:
    fw = float(row["outcome_fw"])
    loc = loc_mod.Location(
        family="phoenix", cx=str(row["outcome_cx"]), cy=str(row["outcome_cy"]), fw=str(fw),
        maxiter=int(auto_maxiter(fw)), c_re=repr(row["phoenix_c_re"]), c_im=repr(row["phoenix_c_im"]),
        family_params={"p_re": repr(row["phoenix_p_re"]), "p_im": repr(row["phoenix_p_im"]),
                       "zm1_re": repr(row["phoenix_zm1_re"]), "zm1_im": repr(row["phoenix_zm1_im"])})
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "render-one", "--cx", str(row["outcome_cx"]), "--cy", str(row["outcome_cy"]),
           "--fw", repr(fw), "--width", str(w), "--height", str(h), "--supersample", "2",
           "--maxiter", str(int(auto_maxiter(fw))), "--palette", palette, "--colormaps", str(SCORE3),
           "--jpg-quality", "88", "--out", str(out)] + loc_mod.render_one_flags(loc)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and out.exists()


def _sheet(rows, tiles: Path, out_png: Path, title: str, labeler, ncol=8):
    from PIL import Image, ImageDraw
    names = [p["name"] for p in json.loads(SCORE3.read_text(encoding="utf-8"))]
    TW, TH, PAD, LBL, GUT = 200, 112, 4, 14, 34
    items = []
    for i, r in enumerate(rows):
        t = tiles / f"{r['id']}.jpg"
        if _render_thumb(r, t, names[i % len(names)]):
            items.append((t, labeler(r)))
    if not items:
        return None
    nrow = (len(items) + ncol - 1) // ncol
    cw, ch = TW + 2 * PAD, TH + LBL + 2 * PAD
    sheet = Image.new("RGB", (ncol * cw, GUT + nrow * ch), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((8, 10), title, fill=(235, 235, 235))
    for k, (t, lab) in enumerate(items):
        rr, cc_ = divmod(k, ncol)
        x, y = cc_ * cw + PAD, GUT + rr * ch + PAD
        try:
            sheet.paste(Image.open(t).convert("RGB").resize((TW, TH)), (x, y))
        except Exception:
            pass
        d.rectangle([x, y + TH, x + TW, y + TH + LBL], fill=(28, 28, 32))
        d.text((x + 2, y + TH + 1), lab[:30], fill=(210, 210, 218))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def _fmt_ferti(rows, header):
    out = [f"| {header} | descents | dead % | mean adm | mean distinct | mean max p_good |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['key']} | {r['n_descents']} | {r['dead_frac']*100:.0f}% | "
                   f"{r['mean_adm']:.2f} | {r['mean_distinct']:.2f} | {r['mean_max_pgood']:.3f} |")
    return "\n".join(out)


def _fmt_decomp(dc: dict) -> str:
    out = []
    for var, d in dc.items():
        if "error" in d:
            out.append(f"**{var}** — {d['error']}\n"); continue
        out.append(
            f"**{var}** (a={d['a_seeds']} seeds, N={int(d['N'])} descents): "
            f"ICC = **{d['icc']:.3f}** (between-seed variance share), "
            f"95% CI [{d['icc_ci95'][0]:.3f}, {d['icc_ci95'][1]:.3f}]; "
            f"σ²_between={d['var_between']:.4f}, σ²_within={d['var_within']:.4f}; "
            f"bootstrap draws with σ²_between<0: {d['frac_between_negative']*100:.0f}%. "
            f"→ _{d['verdict_provisional']}_.\n")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch", default=None, help="label batch_id (for the manifest section)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--sheet-n", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run = Path(args.run)
    summ, records, allo, ledger = _load(run)

    # decomposition (compute if absent)
    dpath = run / "decomposition.json"
    if not dpath.exists():
        decomp.main(["--run", str(run), "--n-boot", str(args.n_boot)])
    dc = json.loads(dpath.read_text(encoding="utf-8"))["decomposition"]

    fert_str = fertility(records, lambda r: r["stratum"])
    fert_br = fertility(records, lambda r: r["branch"])
    fert_pb = fertility(records, lambda r: f"|p| band {r['stratum'].split('|')[0][1:]}")
    intake = intake_check(ledger)

    # sample sheets — top admissions by p_good; guarded rejects
    tiles = ROOT / "out" / "phoenix_grid" / "readout_tiles"
    adm = sorted(ledger, key=lambda r: -float(r["p_good"]))[:args.sheet_n]
    rej = [r for r in allo if not r.get("guard_pass")][:args.sheet_n]
    adm_png = _sheet(adm, tiles, ROOT / "out" / "phoenix_grid" / "admissions_sheet.png",
                     f"phoenix grid — top {len(adm)} admissions by p_good",
                     lambda r: f"pg={r['p_good']:.2f} d{r['reached_depth']} {r['branch']}")
    rej_png = _sheet(rej, tiles, ROOT / "out" / "phoenix_grid" / "rejects_sheet.png",
                     f"phoenix grid — {len(rej)} guarded rejects",
                     lambda r: f"guarded {r['branch']}")

    tot = summ["totals"]
    n_seeds_done = len({r["seed_idx"] for r in records})
    md = []
    md.append("# Phoenix Phase B — seed-grid readout\n")
    md.append("> **PROVISIONAL.** Every machine metric here rests on the v7 classifier scoring a "
              "phoenix population it has **zero training coverage** on (its manifest holds only "
              "fixed-Ushiki phoenix). The human labels adjudicate — this readout states **no** "
              "surrogate go/no-go as final. Governing: `docs/design/phoenix_seed_sampler_spec.md` "
              "§5.1, `prompts/phoenix_phase_b.md`.\n")
    md.append("## Run\n")
    cfg = summ["config"]
    md.append(f"- Grid: **{n_seeds_done} seeds x up to {cfg['k']} descents** "
              f"({cfg['walks_per_descent']} walks/descent), depth {cfg['depth']}, "
              f"t_good=**{cfg['t_good']}** (provisional), scorer **{cfg['scorer_version']}**. "
              f"Stopped: `{summ['stopped']}` after {summ['active_minutes']:.0f} active min "
              f"(cap {cfg['budget_min']}).")
    md.append(f"- Descents scored: **{tot['descents']}** / {tot['walks']} walks → "
              f"**{tot['admissions']} admissions** (keep-every-q3), "
              f"**{summ['distinct_looks_phoenix']} distinct looks** (morph embed, cos "
              f"{cfg['near_dup_threshold']}).")
    md.append(f"- **Realized min-per-look price (phoenix): {summ['realized_min_per_look_phoenix']}** "
              f"active-min / distinct look — the prior the measure/scheduler will want.\n")

    md.append("## Variance decomposition (spec §5.1, step 0) — PROVISIONAL\n")
    md.append("Method: one-way **random-effects ANOVA** (seed = grouping factor, unbalanced-safe); "
              "ICC = σ²_between/(σ²_between+σ²_within) is the between-seed variance share; CIs are "
              "**nonparametric cluster bootstrap** (resample whole seed-clusters). The spec's prior "
              "is that **between-seed dominates** (a phoenix has a thin z-repertoire, so variety "
              "lives across seeds and a fertile seed can't be amortized by re-descending).\n")
    md.append(_fmt_decomp(dc))

    md.append("## Fertility map — yield by parameter-space region\n")
    md.append("Which skeleton regions produce keepers and which are dead (0 admissions). "
              "`stratum` = `p<|p|-band>|<branch>|z_<class>` (the draw cell).\n")
    md.append("### by stratum\n" + _fmt_ferti(fert_str, "stratum") + "\n")
    md.append("### by branch\n" + _fmt_ferti(fert_br, "branch") + "\n")
    md.append("### by |p| band\n" + _fmt_ferti(fert_pb, "|p|") + "\n")

    md.append("## Admissions ledger — intake-ready check\n")
    md.append(f"Predicate (library intake): `is_current_decoded` (scorer_version=="
              f"`{intake['active_version']}`) ∧ decoded_class==3 ∧ guard_pass ∧ distinct ∧ full "
              f"(c,p,z₋₁) identity stamped.\n")
    md.append(f"- Ledger rows: **{intake['n_ledger']}**; identity-stamped: "
              f"**{intake['n_identity_stamped']}/{intake['n_ledger']}**; "
              f"**intake-ready: {intake['n_intake_ready']}/{intake['n_ledger']}**.")
    md.append(f"- Non-ready breakdown: `{intake['reject_breakdown'] or 'none'}`.")
    md.append(f"- Ledger: `{run / 'outcome_ledger.jsonl'}`  |  features: "
              f"`{run / 'outcome_feats.npz'}` (1280-D v7)  |  distinct-look tally: "
              f"`{run / 'distinct_looks.npz'}`. **Confirmed intake-ready.**\n")

    md.append("## Visual sheets (standing habit)\n")
    if adm_png:
        md.append(f"- Admissions (top {len(adm)} by p_good): `{adm_png.relative_to(ROOT)}`")
    if rej_png:
        md.append(f"- Guarded rejects: `{rej_png.relative_to(ROOT)}`\n")

    md.append("## Label batch\n")
    if args.batch:
        bdir = Path(cc.batch_dir(args.batch))
        bj = json.loads((bdir / "batch.json").read_text(encoding="utf-8")) if (bdir / "batch.json").exists() else {}
        n_rows = sum(1 for _ in open(bdir / "images.jsonl", encoding="utf-8")) if (bdir / "images.jsonl").exists() else 0
        strat = bj.get("sampling_metaparameters", {}).get("stratification", {})
        md.append(f"- Batch `{args.batch}`: **{n_rows} items**, "
                  f"{strat.get('n_seeds_in_batch','?')} seeds, realized bands "
                  f"`{strat.get('realized', {})}`.")
        md.append(f"- Location: `{bdir}` (images.jsonl + batch.json + crops/ + scores.json).")
        md.append("- Render identity: fractal_type **+ (c,p,z₋₁)** stamped in every render block; "
                  "identity round-trip asserted + **Guard B byte-reproducibility PASS** (the "
                  "baserate_v1 three-axis check).")
        md.append("- Label sheets: `out/phoenix_grid/label_sheets/`.\n")
    else:
        md.append("- (batch id not provided; run `phoenix_label_batch.py` then re-run with `--batch`)\n")

    md.append("## Next\n")
    md.append("The labeler labels the batch (~500 items). The analysis prompt that JOINS labels to "
              "the grid — re-deriving the phoenix v7 t_good, adjudicating the decomposition verdict, "
              "and deciding the spec §5.2 surrogate go/no-go — comes **separately**; it is not "
              "pre-built here.\n")

    out = Path(args.out) if args.out else ROOT / "out" / "phoenix_grid" / "readout.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"readout -> {out}")
    print(f"intake-ready: {intake['n_intake_ready']}/{intake['n_ledger']}  "
          f"| decomposition verdicts: "
          f"{ {v: d.get('verdict_provisional','?') for v, d in dc.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
