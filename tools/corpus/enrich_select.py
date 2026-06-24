"""Stage C — select the enriched set + the unbiased random reserve.

Reads scored.jsonl (Stage B). Two slices, both labeled at the SAME protocol
(center composition, argmax palette) so the positive-rate comparison is
apples-to-apples; only the *selection* differs:

  - **enriched**    : top ~N distinct locations by filter_score (v2 P(not-bad),
                      best-over-K). The point of the batch: enrich the okay/good
                      frontier so label budget isn't spent on bads.
  - **random_eval** : 100 locations drawn UNIFORMLY at random from the post-gate
                      pool, independent of score (spans the whole range). This is
                      the ONLY unbiased slice — the honest rev4 eval set for v3
                      and the calibration of how many true 2/3 the filter dropped.

Dedup: a reserve draw that also landed in the enriched top-N is kept once and
tagged `random_eval` (the unbiased role wins). Reports the implied cutoff tau and
the full filter_score distribution.

Emits:
  selection.jsonl       -> {image_id, cx, cy, fw, palette}  (drives `enrich --mode render`)
  selection_full.jsonl  -> everything the batch writer needs (provenance + v2 fields)

Run:
  uv run python tools/corpus/enrich_select.py \
      --scored data/enrich/run5/scored.jsonl \
      --n-enriched 1000 --n-reserve 100 --seed 12345 \
      --out-dir data/enrich/run5
"""
from __future__ import annotations

import argparse
import json
import os
import random


def safe_name(name: str) -> str:
    """Match the Rust enrich/present filename sanitizer exactly."""
    out = name
    for ch in '/\\ :*?"<>|':
        out = out.replace(ch, "_")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="data/enrich/run5/scored.jsonl")
    ap.add_argument("--n-enriched", type=int, default=1000)
    ap.add_argument("--n-reserve", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", default="data/enrich/run5")
    a = ap.parse_args()

    rows = []
    with open(a.scored, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    postgate = [r for r in rows if not r["gated"] and r.get("filter_score") is not None]
    n_gated = total - len([r for r in rows if not r["gated"]])
    print(f"scored locations: {total}  |  gated: {n_gated}  |  post-gate (scored): {len(postgate)}")
    if not postgate:
        raise SystemExit("no post-gate scored locations to select from")

    # --- enriched: top-N by filter_score (ties broken by idx for determinism) ---
    by_score = sorted(postgate, key=lambda r: (-r["filter_score"], r["idx"]))
    n_enr = min(a.n_enriched, len(by_score))
    enriched = by_score[:n_enr]
    tau = enriched[-1]["filter_score"]
    enriched_idx = {r["idx"] for r in enriched}

    # --- random reserve: uniform over the WHOLE post-gate pool (seeded) ---------
    rng = random.Random(a.seed)
    n_res = min(a.n_reserve, len(postgate))
    reserve = rng.sample(postgate, n_res)
    reserve_idx = {r["idx"] for r in reserve}

    overlap = enriched_idx & reserve_idx
    # role: random_eval wins on overlap (the unbiased role is the load-bearing one)
    role: dict[int, str] = {}
    for r in enriched:
        role[r["idx"]] = "enriched"
    for r in reserve:
        role[r["idx"]] = "random_eval"  # overrides enriched on overlap

    selected_idx = enriched_idx | reserve_idx
    selected = [r for r in postgate if r["idx"] in selected_idx]
    for r in selected:
        r["selection_role"] = role[r["idx"]]

    n_role_enr = sum(1 for r in selected if r["selection_role"] == "enriched")
    n_role_res = sum(1 for r in selected if r["selection_role"] == "random_eval")

    # --- distribution report ---------------------------------------------------
    fs = sorted(r["filter_score"] for r in postgate)
    def pct(p):
        return fs[min(len(fs) - 1, int(p * len(fs)))]
    print(f"\nfilter_score distribution over {len(fs)} post-gate locations:")
    print(f"  min {fs[0]:.3f}  p10 {pct(.10):.3f}  p25 {pct(.25):.3f}  p50 {pct(.50):.3f}  "
          f"p75 {pct(.75):.3f}  p90 {pct(.90):.3f}  p95 {pct(.95):.3f}  max {fs[-1]:.3f}")
    print(f"\nimplied cutoff tau (min enriched filter_score): {tau:.4f}")
    print(f"selected: {len(selected)}  =  enriched {n_role_enr}  +  random_eval {n_role_res}  "
          f"(overlap {len(overlap)} -> tagged random_eval)")
    res_fs = sorted(r["filter_score"] for r in selected if r["selection_role"] == "random_eval")
    if res_fs:
        print(f"random_eval filter_score spread: min {res_fs[0]:.3f} median "
              f"{res_fs[len(res_fs)//2]:.3f} max {res_fs[-1]:.3f}  "
              f"(below tau: {sum(1 for x in res_fs if x < tau)}/{len(res_fs)})")

    # --- write selection files -------------------------------------------------
    os.makedirs(a.out_dir, exist_ok=True)
    sel_path = os.path.join(a.out_dir, "selection.jsonl")
    full_path = os.path.join(a.out_dir, "selection_full.jsonl")
    n_pal = {}
    with open(sel_path, "w", encoding="utf-8") as fsel, \
         open(full_path, "w", encoding="utf-8") as ffull:
        for r in sorted(selected, key=lambda r: (-r["filter_score"], r["idx"])):
            pal = r["argmax_palette"]
            image_id = f"{r['idx']}_{safe_name(pal)}"
            r["image_id"] = image_id
            n_pal[pal] = n_pal.get(pal, 0) + 1
            fsel.write(json.dumps({
                "image_id": image_id, "cx": r["cx"], "cy": r["cy"], "fw": r["fw"], "palette": pal,
            }) + "\n")
            ffull.write(json.dumps(r) + "\n")

    print(f"\npalette diversity in batch: {len(n_pal)} distinct palettes")
    for pal, c in sorted(n_pal.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {c:>4}  {pal}")
    print(f"\nwrote {sel_path}  ({len(selected)} rows)")
    print(f"wrote {full_path}")


if __name__ == "__main__":
    main()
