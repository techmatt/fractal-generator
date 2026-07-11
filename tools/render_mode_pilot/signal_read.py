"""Flag too-close-to-smooth + per-mode signal read (flag-and-signal-read.md).

Step 1: persist `too_close_to_smooth` on each images.jsonl row (dE76 < 8 AND
        1-SSIM < 0.12, from smooth_pass.py distances.json).
Step 2: per-mode signal read on render_mode_pilot_v1 labels joined to provenance,
        EXCLUDING flagged rasters.

    uv run python tools/render_mode_pilot/signal_read.py            # report only
    uv run python tools/render_mode_pilot/signal_read.py --write    # + persist flag
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "data/render_mode_corpus/batches/2026-07-10_render_mode_pilot_v1"
IMAGES = BATCH / "images.jsonl"
DIST = REPO / "out/render_mode_pilot/smooth_pass/distances.json"
LABELS = REPO / "labels/render_mode_pilot_v1.json"

DE_CUT, OMS_CUT = 8.0, 0.12

for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding="utf-8")
    except Exception: pass


def load():
    rows = [json.loads(l) for l in IMAGES.read_text().splitlines() if l.strip()]
    dist = {d["image_id"]: d for d in json.load(open(DIST))}
    labels = json.load(open(LABELS))
    for r in rows:
        d = dist[r["image_id"]]
        r["_de76"] = d["de76"]; r["_oms"] = d["one_minus_ssim"]
        r["_score"] = labels.get(r["image_id"])
        r["_flag"] = (d["de76"] < DE_CUT) and (d["one_minus_ssim"] < OMS_CUT)
    return rows


def qtriple(g):
    c = Counter(r["_score"] for r in g)
    return c.get(1, 0), c.get(2, 0), c.get(3, 0)


def rate_line(name, g, width=34):
    q1, q2, q3 = qtriple(g)
    n = q1 + q2 + q3
    if n == 0:
        return f"{name:<{width}} n= 0   ---"
    q3r = q3 / n; q23r = (q2 + q3) / n
    return (f"{name:<{width}} n={n:3d}  q1={q1:2d} q2={q2:2d} q3={q3:2d}  "
            f"q3={q3r:5.1%}  q2+q3={q23r:5.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = load()

    # ---- Step 1: flag ------------------------------------------------------
    flagged = [r for r in rows if r["_flag"]]
    print(f"=== STEP 1  too_close_to_smooth  (dE76 < {DE_CUT} AND 1-SSIM < {OMS_CUT}) ===")
    print(f"flagged: {len(flagged)} / {len(rows)}")
    bymode = Counter(r["render"]["render_mode"] for r in flagged)
    for m, c in bymode.most_common():
        print(f"   {m:30s} {c}")
    # ranges among flagged, for sanity
    if flagged:
        des = [r["_de76"] for r in flagged]; oms = [r["_oms"] for r in flagged]
        print(f"   flagged dE76 range [{min(des):.2f}, {max(des):.2f}]  "
              f"1-SSIM range [{min(oms):.4f}, {max(oms):.4f}]")

    if args.write:
        lines = []
        for l in IMAGES.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            d = None
            r["too_close_to_smooth"] = next(
                x["_flag"] for x in rows if x["image_id"] == r["image_id"])
            lines.append(json.dumps(r))
        IMAGES.write_text("\n".join(lines) + "\n")
        print(f"[write] persisted too_close_to_smooth on {len(lines)} rows -> {IMAGES.relative_to(REPO)}")

    # ---- Step 2: signal read (exclude flagged) -----------------------------
    kept = [r for r in rows if not r["_flag"]]
    print(f"\n=== STEP 2  per-mode signal read  (excluding {len(flagged)} flagged; n={len(kept)}) ===")

    # 2a per mode, sorted by q3-rate
    modes = defaultdict(list)
    for r in kept:
        modes[r["render"]["render_mode"]].append(r)
    def q3rate(g):
        q1, q2, q3 = qtriple(g); n = q1 + q2 + q3
        return q3 / n if n else -1
    print("\n-- per mode (sorted by q3-rate) --")
    for m in sorted(modes, key=lambda m: -q3rate(modes[m])):
        print(rate_line(m, modes[m]))

    # 2b transfer_dropped cross-tab (Rust-path modes = modes where drop can happen)
    print("\n-- transfer_dropped cross-tab (Rust-path modes only) --")
    droppable = {m for m, g in modes.items()
                 if any(r["provenance"]["transfer_dropped"] for r in g)}
    allrust = [r for r in kept if r["render"]["render_mode"] in droppable]
    for label, sub in (("DROPPED (grad lost)", [r for r in allrust if r["provenance"]["transfer_dropped"]]),
                       ("KEPT (grad)",          [r for r in allrust if not r["provenance"]["transfer_dropped"]])):
        print(rate_line(label, sub, width=22))
    print("   per-mode:")
    for m in sorted(droppable):
        g = modes[m]
        dr = [r for r in g if r["provenance"]["transfer_dropped"]]
        kp = [r for r in g if not r["provenance"]["transfer_dropped"]]
        print(f"     {m:32s}  drop {rate_line('',dr,0).strip()}")
        print(f"     {'':32s}  keep {rate_line('',kp,0).strip()}")

    # 2c direct-trap param grid: q3-rate by opacity x threshold
    print("\n-- direct-trap param grid  (q3-rate by opacity x threshold; pooled over 4 direct modes) --")
    direct = [r for r in kept if r["provenance"]["mode_kind"] == "direct"]
    cell = defaultdict(list)
    ops, ths = set(), set()
    for r in direct:
        mp = r["provenance"]["mode_params"]
        o = mp.get("direct_opacity"); t = mp.get("direct_threshold")
        cell[(o, t)].append(r); ops.add(o); ths.add(t)
    ops, ths = sorted(ops), sorted(ths)
    hdr = "opacity\\thresh " + "".join(f"{t:>12}" for t in ths)
    print(hdr)
    for o in ops:
        parts = [f"{o:<14}"]
        for t in ths:
            g = cell.get((o, t), [])
            q1, q2, q3 = qtriple(g); n = q1 + q2 + q3
            parts.append(f"{(q3/n if n else 0):5.0%}({n:2d})".rjust(12) if n else "  --".rjust(12))
        print("".join(parts))
    # also per direct mode
    print("   per direct mode:")
    for m in sorted(m for m in modes if any(r["provenance"]["mode_kind"]=="direct" for r in modes[m])):
        print("     " + rate_line(m, modes[m], 30))

    # 2d family x mode coarse (mark thin cells n<3)
    print("\n-- family x mode  (q3-rate; thin cells n<3 marked *) --")
    fams = sorted({r["provenance"]["family"] for r in kept})
    modenames = sorted(modes)
    print(f"{'family':22s}" + "".join(f"{i:>6}" for i in range(len(modenames))))
    for i, m in enumerate(modenames):
        print(f"  [{i:2d}] {m}")
    for f in fams:
        parts = [f"{f:22s}"]
        for m in modenames:
            g = [r for r in modes[m] if r["provenance"]["family"] == f]
            q1, q2, q3 = qtriple(g); n = q1 + q2 + q3
            if n == 0:
                parts.append("   -  ")
            else:
                mark = "*" if n < 3 else " "
                parts.append(f"{int(round(100*q3/n)):3d}{mark} ")
        print("".join(parts))

    # 2e overall
    q1, q2, q3 = qtriple(kept); n = q1 + q2 + q3
    print(f"\n-- overall (post-exclusion) --")
    print(f"n={n}  q1={q1} ({q1/n:.1%})  q2={q2} ({q2/n:.1%})  q3={q3} ({q3/n:.1%})")

    # side: exp_smoothing full breakdown
    exp = [r for r in rows if r["render"]["render_mode"] == "exp_smoothing"]
    print(f"\n-- SIDE: exp_smoothing (all {len(exp)}, incl flagged) --")
    print("   " + rate_line("exp_smoothing", exp, 14))
    print(f"   flagged {sum(r['_flag'] for r in exp)} / {len(exp)}")


if __name__ == "__main__":
    main()
