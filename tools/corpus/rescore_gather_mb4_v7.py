"""Re-score the native multibrot4 GATHER harvest under the CURRENT (v7) scorer.

Why: the native mb4 t_good is uncalibrated (0.50) and the current-decoded (v7)
campaign ledgers hold almost no SUB-threshold mb4 (7 rows below p_good 0.5) — the
campaigns ran mb4 admissions-heavy. The v6-decoded `gather/multibrot4` harvest DOES
have abundant native sub-threshold mb4, but its stored k3/decoded_class are v6, not
current. This renders every distinct gather-mb4 candidate at the *scored presentation*
(640x360 ss2, twilight_shifted — the reframe search fidelity that campaign canon_pgood
also refers to) and scores it with the live v7 checkpoint, yielding a current-decoded
p_good directly comparable to the campaign ledgers.

Output cache (resumable, append-only): the batch-scoped jsonl below. These rows are
LABEL-BATCH material only (source=gather_v6, scorer=v7) — they are never ledger
admissions and must not enter any generation/pool path.

Run: uv run python tools/corpus/rescore_gather_mb4_v7.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

from active_ckpt import BIN, PALETTE, JPG_Q, auto_maxiter, ACTIVE_VERSION  # noqa: E402
from score_lib import Scorer  # noqa: E402
from active_ckpt import ACTIVE_CKPT  # noqa: E402

GATHER = ROOT / "data" / "discovery" / "gather" / "multibrot4" / "outcome_ledger.jsonl"
BATCH_ID = "2026-07-22_native_multibrot_band_v1"
CACHE = ROOT / "data" / "label_corpus" / "batches" / BATCH_ID / "mb4_gather_v7_rescore.jsonl"
TILES = ROOT / "out" / "native_multibrot_band" / "mb4_rescore_tiles"
WORKERS = 4  # project cap


def distinct_gather() -> list[dict]:
    seen: dict = {}
    for line in open(GATHER, encoding="utf-8"):
        r = json.loads(line)
        k = (round(r["outcome_cx"], 10), round(r["outcome_cy"], 10), round(r["outcome_fw"], 12))
        if k not in seen or (r.get("k3") or -1) > (seen[k].get("k3") or -1):
            seen[k] = r
    return list(seen.values())


def render(cx, cy, fw, out: Path) -> tuple[bool, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN), "render-one", "--cx", repr(float(cx)), "--cy", repr(float(cy)),
        "--fw", repr(float(fw)), "--width", "640", "--height", "360", "--supersample", "2",
        "--maxiter", str(auto_maxiter(float(fw))), "--palette", PALETTE,
        "--jpg-quality", str(JPG_Q), "--out", str(out), "--family", "multibrot4",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and out.exists()
    return ok, ("" if ok else r.stderr[-300:])


def main():
    assert ACTIVE_VERSION == "v7", f"expected live scorer v7, got {ACTIVE_VERSION}"
    rows = distinct_gather()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if CACHE.exists():
        for line in open(CACHE, encoding="utf-8"):
            d = json.loads(line)
            done.add((round(d["cx"], 10), round(d["cy"], 10), round(d["fw"], 12)))
    todo = [r for r in rows
            if (round(r["outcome_cx"], 10), round(r["outcome_cy"], 10),
                round(r["outcome_fw"], 12)) not in done]
    print(f"distinct gather mb4: {len(rows)}  already cached: {len(done)}  to do: {len(todo)}",
          flush=True)
    if not todo:
        print("nothing to do; cache complete.", flush=True)
        return

    TILES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # 1) render (parallel subprocess, capped at 4)
    tiles = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for i, r in enumerate(todo):
            tp = TILES / f"c{i:05d}.jpg"
            tiles[i] = tp
            futs[ex.submit(render, r["outcome_cx"], r["outcome_cy"], r["outcome_fw"], tp)] = i
        n = 0
        for fut in cf.as_completed(futs):
            ok, err = fut.result()
            i = futs[fut]
            n += 1
            if not ok:
                raise SystemExit(f"render failed [{i}]: {err}")
            if n % 50 == 0 or n == len(todo):
                dt = time.time() - t0
                print(f"  rendered {n}/{len(todo)}  ({dt:.1f}s, {dt/n*1000:.0f} ms/render)",
                      flush=True)
    render_dt = time.time() - t0
    print(f"render pass: {render_dt:.1f}s for {len(todo)} ({render_dt/len(todo)*1000:.0f} ms/ea)",
          flush=True)

    # 2) score with live v7
    print(f"loading v7 scorer: {ACTIVE_CKPT}", flush=True)
    scorer = Scorer(ACTIVE_CKPT)
    paths = [str(tiles[i]) for i in range(len(todo))]
    ts = time.time()
    triples = scorer.score_paths(paths, batch_size=64)
    print(f"score pass: {time.time()-ts:.1f}s", flush=True)

    # 3) append cache
    with open(CACHE, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            score, pnb, pg = (float(x) for x in triples[i])
            f.write(json.dumps({
                "cx": r["outcome_cx"], "cy": r["outcome_cy"], "fw": r["outcome_fw"],
                "v6_k3": r.get("k3"), "v6_decoded_class": r.get("decoded_class"),
                "v6_guard_pass": r.get("guard_pass"), "gather_id": r.get("id"),
                "v7_score": score, "v7_p_notbad": pnb, "v7_p_good": pg,
            }) + "\n")
    print(f"wrote {len(todo)} rows -> {CACHE.relative_to(ROOT)}", flush=True)
    print(f"TOTAL: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
