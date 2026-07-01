"""Batch driver: generate N labeling queries (candidate generation only).

Thin wrapper over `query_sampler` (the reusable D + query composition). For each query
it draws a location uniformly from the q2+q3 pool, samples 6 candidates per the
query-type split, renders each through the shared `colormap.render_candidate` path, and
writes a durable **query record** with full recipes plus the 6 images, then a 3x2
contact sheet for the eye-check.

Field dumps (ss2) are cached per location under `out/fields/` and reused across the 6
candidates and across queries at the same location — the expensive Rust iterate runs
once per location, every recolor is pure Python.

This is candidate generation ONLY: no scorer, no labeling UI, no active-learning
selection. The eventual human-pick render is ss4 (separate, not built here).

    uv run python tools/queries/assemble_queries.py --n 20 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_sampler as qs  # noqa: E402

ROOT = qs.ROOT
EXE = ROOT / "target" / "release" / "fractal-generator.exe"

OUT_QUERIES = ROOT / "out" / "queries"
OUT_FIELDS = ROOT / "out" / "fields"
OUT_IMAGES = OUT_QUERIES / "images"
OUT_RECORDS = OUT_QUERIES / "records"


# ---------------------------------------------------------------------------
# Field cache — one ss2 dump per unique location, reused everywhere.
# ---------------------------------------------------------------------------

def _field_key(ref):
    """Stable filename stem for a location's ss2/eval field dump."""
    parts = [ref.kind, ref.cx, ref.cy, ref.fw, str(ref.maxiter),
             ref.c_re or "", ref.c_im or "",
             str(qs.CANDIDATE_SS), str(qs.EVAL_WIDTH), str(qs.EVAL_HEIGHT)]
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return f"{ref.kind}_{h}"


def ensure_field(ref):
    """Dump (or reuse) the ss2 smooth field for a location; return a loaded FieldData.

    Returns (FieldData, dump_seconds) where dump_seconds is 0.0 on a cache hit."""
    OUT_FIELDS.mkdir(parents=True, exist_ok=True)
    stem = _field_key(ref)
    bin_path = OUT_FIELDS / f"{stem}.bin"
    json_path = OUT_FIELDS / f"{stem}.json"
    dump_secs = 0.0
    if not (bin_path.exists() and json_path.exists()):
        cmd = [str(EXE), "render-one",
               "--cx", ref.cx, "--cy", ref.cy, "--fw", ref.fw,
               "--width", str(qs.EVAL_WIDTH), "--height", str(qs.EVAL_HEIGHT),
               "--supersample", str(qs.CANDIDATE_SS),
               "--maxiter", str(ref.maxiter),
               "--dump-field", str(bin_path)]
        if ref.kind == "julia":
            cmd += ["--julia", "--c", ref.c_re, ref.c_im]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        dump_secs = time.time() - t0
        if r.returncode != 0:
            raise RuntimeError(f"dump-field failed for {stem}:\n{r.stderr[-400:]}")
    return qs.cm.load_field(str(bin_path), str(json_path)), dump_secs


# ---------------------------------------------------------------------------
# Query record.
# ---------------------------------------------------------------------------

def candidate_record(cfg, sampler, image_rel):
    """Full recipe for one candidate — sufficient to re-render at any resolution."""
    ptype = sampler.library.palette_type(cfg.palette)
    return {
        "config": json.loads(cfg.to_json()),          # complete CandidateConfig
        "palette": cfg.palette,
        "palette_source": sampler.source_of(cfg.palette),
        "palette_type": ptype,
        # duplicated flat for the downstream degeneracy/instrumentation guards
        "reverse": cfg.reverse,
        "gamma": cfg.gamma,
        "log_premap": cfg.log_premap,
        "phase": cfg.phase,
        "n_cycles": cfg.n_cycles,
        "eval": [cfg.eval_width, cfg.eval_height],
        "ss": qs.CANDIDATE_SS,
        "image": image_rel,
    }


def query_record(qid, location, query_type, cands, sampler, image_rels):
    ref = location.ref
    return {
        "query_id": qid,
        "query_type": query_type,
        "location": {
            "family": ref.kind,
            "cx": ref.cx, "cy": ref.cy, "fw": ref.fw,
            "maxiter": ref.maxiter, "c_re": ref.c_re, "c_im": ref.c_im,
            "eval_width": qs.EVAL_WIDTH, "eval_height": qs.EVAL_HEIGHT, "ss": qs.CANDIDATE_SS,
            "qualifying_scores": sorted(location.scores),
            "source_batches": sorted(location.batch_ids),
        },
        "candidates": [candidate_record(c, sampler, rel) for c, rel in zip(cands, image_rels)],
    }


# ---------------------------------------------------------------------------
# Contact sheet — 3x2, the acceptance surface.
# ---------------------------------------------------------------------------

def contact_sheet(imgs, cands, qid, query_type, out_path, thumb_w=512, pad=8, bar=26):
    cols, rows = 3, 2
    tw = thumb_w
    th = round(tw * qs.EVAL_HEIGHT / qs.EVAL_WIDTH)
    W = cols * tw + (cols + 1) * pad
    H = rows * (th + bar) + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 2), f"{qid}  [{query_type}]", fill=(230, 230, 230))
    for i, (im, cfg) in enumerate(zip(imgs, cands)):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + bar + r * (th + bar + pad)
        thumb = Image.fromarray(im).resize((tw, th), Image.BILINEAR)
        sheet.paste(thumb, (x, y))
        lbl = f"{i}: {cfg.palette[:22]}"
        sub = f"g{cfg.gamma:.2f} ph{cfg.phase:.2f} n{cfg.n_cycles}{' rev' if cfg.reverse else ''}{' log' if cfg.log_premap=='log' else ''}"
        draw.text((x + 2, y + th + 2), lbl, fill=(220, 220, 160))
        draw.text((x + 2, y + th + 13), sub, fill=(160, 190, 220))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Assemble N labeling queries (candidate gen only).")
    ap.add_argument("--n", type=int, default=20, help="number of queries (default 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_QUERIES))
    ap.add_argument("--verify", action="store_true", default=True,
                    help="re-render one candidate per query from its stored recipe and diff")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    rec_dir = out_dir / "records"
    img_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pool = qs.LocationPool.from_corpus()
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)

    print(f"[pool] {pool.report()}")
    for st, names in sorted(sampler.strata.items()):
        print(f"[pool] palette stratum {st}: {len(names)}")

    t_wall = time.time()
    total_dump = 0.0
    color_times = []
    qtype_counts = {t: 0 for t in qs.QUERY_TYPES}
    verify_fails = 0
    field_cache = {}   # stem -> FieldData (in-memory reuse within the batch)

    for qi in range(args.n):
        qid = f"q{args.seed:03d}_{qi:04d}"
        loc = pool.sample(rng)
        stem = _field_key(loc.ref)
        if stem in field_cache:
            fld = field_cache[stem]
        else:
            fld, dsec = ensure_field(loc.ref)
            total_dump += dsec
            field_cache[stem] = fld

        query_type, cands = qs.compose_query(loc, rng, sampler)
        qtype_counts[query_type] += 1

        prep = qs.cm.stretch_field(fld)   # config-independent prefix, reused by all 6

        imgs = []
        image_rels = []
        for ci, cfg in enumerate(cands):
            t0 = time.time()
            im = qs.cm.render_candidate(fld, cfg, lib, prep=prep)
            color_times.append(time.time() - t0)
            rel = f"images/{qid}_{ci}.png"
            Image.fromarray(im).save(out_dir / rel)
            imgs.append(im)
            image_rels.append(rel)

        rec = query_record(qid, loc, query_type, cands, sampler, image_rels)
        (rec_dir / f"{qid}.json").write_text(json.dumps(rec, indent=1))
        contact_sheet(imgs, cands, qid, query_type, out_dir / f"{qid}.png")

        if args.verify:
            # Re-render candidate 0 from ITS STORED RECIPE and diff against the saved PNG.
            cfg0 = qs.cm.CandidateConfig.from_json(json.dumps(rec["candidates"][0]["config"]))
            im0 = qs.cm.render_candidate(fld, cfg0, lib)
            saved = np.asarray(Image.open(out_dir / image_rels[0]))
            if not np.array_equal(im0, saved):
                verify_fails += 1
                print(f"  [verify] MISMATCH on {qid} candidate 0")

    wall = time.time() - t_wall
    ct = np.array(color_times)
    print()
    print(f"[done] {args.n} queries -> {out_dir}")
    print(f"[split] {qtype_counts}  (target ~{dict(zip(qs.QUERY_TYPES, qs.QUERY_SPLIT))})")
    print(f"[fields] {len(field_cache)} unique locations, {total_dump:.1f}s dumping (ss{qs.CANDIDATE_SS})")
    print(f"[color] {len(ct)} candidates: mean {ct.mean()*1000:.0f}ms  "
          f"p95 {np.percentile(ct,95)*1000:.0f}ms  max {ct.max()*1000:.0f}ms")
    under = "OK" if ct.max() < 1.0 else "SLOW"
    print(f"[color] max per-candidate {ct.max():.3f}s  (target < 1s) [{under}]")
    print(f"[verify] {args.n - verify_fails}/{args.n} re-render-from-recipe matches exact")
    print(f"[wall] {wall:.1f}s total")


if __name__ == "__main__":
    main()
