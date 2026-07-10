"""Shared helpers for the permanent label corpus (data/label_corpus/).

See data/label_corpus/CORPUS_SCHEMA.md for the contract. This module owns the
*shape* of an images.jsonl row so every batch writer agrees on the field set:
`render` is version-invariant (identical keys across all batches), `provenance`
is version-tagged (free to be null/absent), `label` is the verdict.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys

# repo root = two levels up from tools/corpus/
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS_DIR = os.path.join(ROOT, "data", "label_corpus")
BATCHES_DIR = os.path.join(CORPUS_DIR, "batches")

# The version-invariant render field set — the ONLY thing the classifier sees
# (alongside the crop). Every batch's every row carries exactly these keys.
RENDER_KEYS = (
    "cx", "cy", "fw", "maxiter", "palette", "composition",
    "width", "height", "ss", "filter", "interior_mode",
)

# Provenance keys we currently model. A given generator version fills what it
# has and leaves the rest null — provenance is allowed to differ across batches.
PROVENANCE_KEYS = (
    "generator_version", "batch_id", "root_src", "branch", "depth",
    "target_depth", "walk_id", "placement", "focus_score",
    "draw_index", "seed_index", "black_fraction", "interior_frac",
    "occupancy", "void_guard",
    # v2-filtered-enrichment batch (2026-06-24): selection is v2-biased, recorded
    # so the bias is always recoverable (only `random_eval` rows are unbiased).
    "selection_role", "filter_score", "argmax_palette", "k_scores",
    "v2_est_class", "v2_model_id",
    # scale-controlled 2x2 batch (2026-06-25): the experiment factors. `cell` ∈
    # {A,B,C,D}; `center_proposer` ∈ {8k_content_focus, flat_acceptband};
    # `start_fw` ∈ {0.10 wide, 0.014093 narrow}; `rev4_fix` = True (occ-floor
    # skipped @ d1→d2). Bias-loop / analysis only — never enters training.
    "cell", "center_proposer", "start_fw", "rev4_fix",
    # v3-guided biased mining batch (2026-06-25): the full selection-bias trail.
    # `source` ∈ {landmark_mine, root_mine}; `seed_landmark_id` = the good this
    # walk perturbed (landmark only); `perturbation_frac` = |offset|/fw of the
    # seed perturbation; `beam_path` = the per-step top-k child indices; `loc_score`
    # = the v3 [0,2] score at the neutral location-scoring palette; `gate_kind`/
    # `gate_t2`/`gate_score` = the >=T2 gate (gate_score is the gated value);
    # `palette_family` ∈ {warm,cool,cyclic,diverging,mono}; `biased`=True (this
    # batch is biased-positive-enriched — NOT for unbiased eval). Bias loop only.
    "source", "seed_landmark_id", "perturbation_frac", "beam_path", "loc_score",
    "location_score_palette", "palette_family", "gate_kind", "gate_t2",
    "gate_score", "biased", "v3_model_id",
    # v6 gather-pool batch (2026-07-05): the guard-OFF gather harvest → label batch.
    # `family` = the ledger cloud partition / class (mandelbrot, multibrot{3,4,5},
    # julia:{mandelbrot,multibrot{3,4,5}}, phoenix); `k3` = the raw v5 E[ord] ∈ [0,2]
    # the pick was ranked on (also mirrored into `filter_score` so the UI orders
    # best-first); `decoded_class` = the CORN hard class ∈ {1,2,3} of the k3-winning
    # frame; `guard_verdict` = the degenerate-outcome guard's prior ∈
    # {pass,flat,interior,both} (logged, NOT gated — gather is guard-off);
    # `descend_mode` = the descent mode (cplane/phoenix, or Julia center/normal);
    # `parent_oid` = the c-plane parent outcome a Julia sub-descent hung off (null
    # for native families); `lineage` = "gather". `selection_role` (best /
    # random_eval / disagreement) and `filter_score` are reused from above. Bias
    # loop / analysis only — never enters training.
    "family", "k3", "decoded_class", "guard_verdict", "descend_mode",
    "parent_oid", "lineage",
)


# --- v6-decode stamp guard (discovery outcome ledgers) ---------------------
#
# A discovery-ledger row's `decoded_class` is the PERSISTED q3 hard-class verdict
# (corn_decode of raw_top3), stamped at harvest time by whichever classifier the
# seeder ran. production_seeder began writing `scorer_version="v6"` only partway
# through the gather runs, so a large body of older rows carry a v5-vintage
# `decoded_class` and NO stamp: the entire `gather/mandelbrot` and
# `gather/phoenix` partitions predate the stamp, as do the first chunks of
# multibrot{3,4,5} and ~237 rows of the main `outcome_ledger.jsonl`. Those v5
# verdicts must NEVER be consumed where a v6 readout is required (fresh-discovery
# emit, wallpaper-head "fresh machine-q3" selection).
#
# Discriminator: `scorer_version == "v6"`. A row without that exact stamp is
# v5-decoded (or pre-stamp) and its `decoded_class` is not a v6 verdict. This is
# an explicit stamp field — no path/source inference needed.
#
# READ-ONLY: this guard REJECTS v5 rows; it never re-decodes, re-stamps, or
# mutates a ledger. Re-running v6 on the v5 locations is a separate,
# compute-bearing project and out of scope here.
V6_SCORER_VERSION = "v6"   # must match tools/atlas/production_seeder.SCORER_VERSION


class V5DecodeError(ValueError):
    """A v5-decoded ledger row reached a path that requires a v6 verdict."""


def is_v6_decoded(row) -> bool:
    """Canonical predicate: True iff `row`'s decode verdict carries the v6 stamp.

    The ONE place the v6-stamp discriminator is defined. Consumers that require a
    v6 readout gate on this rather than open-coding the `scorer_version` check, so
    the stamp field can never drift out of sync across call sites."""
    return row.get("scorer_version") == V6_SCORER_VERSION


def v6_rows_only(rows):
    """Filter an iterable of ledger rows to v6-stamped ones.

    Returns `(kept, excluded)` — the kept rows plus the count of v5/unstamped rows
    dropped. For pool-builders that legitimately discard v5 rows and want to report
    how many they excluded."""
    kept, excluded = [], 0
    for r in rows:
        if is_v6_decoded(r):
            kept.append(r)
        else:
            excluded += 1
    return kept, excluded


def require_v6(row):
    """Return `row` if it is v6-stamped, else raise `V5DecodeError`.

    For single-row verdict-trust paths that must never proceed on a v5 verdict."""
    if not is_v6_decoded(row):
        raise V5DecodeError(
            f"ledger row {row.get('id')!r} is v5-decoded "
            f"(scorer_version={row.get('scorer_version')!r}); refusing to consume "
            f"its decoded_class={row.get('decoded_class')!r} as a v6 verdict"
        )
    return row


def hp_str(x) -> str:
    """Render a coordinate as a high-precision decimal string.

    The store keeps cx/cy/fw as strings (an f64 center is meaningless at deep
    zoom). For shallow f64-sourced batches the f64's shortest round-tripping
    decimal IS its full precision, so repr() is faithful; a future deep batch
    would pass an already-arbitrary-precision string straight through.
    """
    if isinstance(x, str):
        return x
    return repr(float(x))


def image_id_from_output(output_path: str) -> str:
    """The crop's basename without extension — the batch-unique, fs-safe stem.

    present writes `{seed_index}_{composition}_{palette}.{ext}`, unique per
    (seed_index, composition, palette) within a run; the store reuses it as
    `image_id`, and the crop lives at `crops/<image_id>.jpg`.
    """
    base = os.path.basename(str(output_path).replace("\\", "/"))
    stem, _ext = os.path.splitext(base)
    return stem


def render_block(*, cx, cy, fw, maxiter, palette, composition,
                 width, height, ss, filter, interior_mode) -> dict:
    """Build a version-invariant render block (coordinates → hi-prec strings)."""
    return {
        "cx": hp_str(cx),
        "cy": hp_str(cy),
        "fw": hp_str(fw),
        "maxiter": int(maxiter),
        "palette": str(palette),
        "composition": str(composition),
        "width": int(width),
        "height": int(height),
        "ss": int(ss),
        "filter": str(filter),
        "interior_mode": str(interior_mode),
    }


def provenance_block(generator_version: str, batch_id: str, **fields) -> dict:
    """Build a provenance block: every modeled key present, unspecified → null.

    Pass only the fields this generator version actually produced; the rest are
    explicitly null (never fabricated).
    """
    prov = {k: None for k in PROVENANCE_KEYS}
    prov["generator_version"] = generator_version
    prov["batch_id"] = batch_id
    for k, v in fields.items():
        if k not in PROVENANCE_KEYS:
            raise KeyError(f"unknown provenance key {k!r}; add it to PROVENANCE_KEYS first")
        prov[k] = v
    return prov


def label_block(score=None, labeler=None, labeled_at=None) -> dict:
    return {"score": score, "labeler": labeler, "labeled_at": labeled_at}


def make_row(image_id: str, render: dict, provenance: dict, label: dict) -> dict:
    missing = [k for k in RENDER_KEYS if k not in render]
    if missing:
        raise KeyError(f"render block missing keys {missing} for {image_id}")
    return {
        "image_id": image_id,
        "render": render,
        "provenance": provenance,
        "label": label,
    }


def write_jsonl(rows, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batch_dir(batch_id: str) -> str:
    return os.path.join(BATCHES_DIR, batch_id)


# ===========================================================================
# The canonical location-corpus label-crop render path.
#
# There is exactly ONE way a location-corpus label crop is produced: the native
# Rust colorer, `render-one --palette <name> --colormaps <library>`. The crop is
# a pure function of its version-invariant render block THROUGH this call (the
# "crops are rebuildable" contract) — geometry, ss, filter, maxiter, and the
# named palette fully determine the pixels.
#
# NEVER build a corpus crop from `render-one --dump-field` + `colormap.render_
# candidate`: that Python pct-stretch→LUT tail is a DIFFERENT recipe (measured
# mean Δ 16.2 / max 209 vs this path, ~75% of pixels differ), it is ~5–10× slower
# (59 MB field dump + GIL-serialized numpy), and it breaks cross-batch coloring
# consistency and reproducibility. The dump-field tail is correct only for
# ARBITRARY-PARAM coloring (gamma/phase/cycles/reverse) — e.g. the wallpaper-
# bootstrap / preference path, which is its own canonical recipe and out of scope.
#
# Route every location-corpus crop render through `render_corpus_crop` below so the
# wrong path is structurally unreachable for corpus code. (Named `render_corpus_crop`,
# NOT `render_label_crop`, to avoid colliding with the wallpaper-bootstrap module's
# own `render_label_crop`, which deliberately uses the render_candidate tail for its
# arbitrary-param preference recipe — a different, out-of-scope path.)
# ===========================================================================
CANONICAL_CROP_RECIPE = "render-one --palette --colormaps"
DEFAULT_CROP_JPGQ = 90


def default_bin() -> str:
    """The release engine binary (Windows exe; the .exe suffix is harmless on the
    path join even where absent — callers on this project run win32)."""
    return os.path.join(ROOT, "target", "release", "fractal-generator.exe")


def _location_mod():
    """Lazy import of the sibling `location` module (avoids a hard import-order
    dependency: callers insert tools/corpus on sys.path before importing us)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module("location")


def render_corpus_crop(render: dict, out_path, *, palette_source, bin_path=None,
                       jpg_quality: int = DEFAULT_CROP_JPGQ, cwd=None,
                       creationflags: int = 0, timeout=None) -> str:
    """Render ONE location-corpus label crop the canonical way and return `out_path`.

    `render` is a version-invariant render block (`RENDER_KEYS`, optionally the
    multi-family `fractal_type` + `c_re`/`c_im`); `palette_source` is the
    `--colormaps` library the `render["palette"]` name resolves in. Every pixel-
    affecting input is read straight off the block, so a rebuild from the same
    block + palette_source is byte-reproducible (this is what Guard B enforces).

    Raises RuntimeError on a non-zero exit or a missing output file. This is the
    ONLY sanctioned corpus-crop renderer — no raw `--dump-field`/`render_candidate`.
    """
    loc_mod = _location_mod()
    loc = loc_mod.from_render_block(render)
    binp = str(bin_path) if bin_path is not None else default_bin()
    cmd = [binp, "render-one", *loc_mod.render_one_flags(loc),
           "--cx", str(render["cx"]), "--cy", str(render["cy"]), "--fw", str(render["fw"]),
           "--width", str(render["width"]), "--height", str(render["height"]),
           "--supersample", str(render["ss"]), "--filter", str(render["filter"]),
           "--maxiter", str(render["maxiter"]),
           "--palette", str(render["palette"]), "--colormaps", str(palette_source),
           "--jpg-quality", str(jpg_quality), "--out", str(out_path)]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       creationflags=creationflags, timeout=timeout)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            f"render_corpus_crop failed for {out_path} "
            f"(rc={r.returncode}): {(r.stderr or '')[-400:]}")
    return str(out_path)


def render_recipe_stamp(palette_source, jpg_quality: int = DEFAULT_CROP_JPGQ) -> dict:
    """Self-identifying provenance for `batch.json`: the render path a batch's crops
    were produced through. Guard B asserts `path == CANONICAL_CROP_RECIPE`, so a
    batch built off-recipe (or hand-stamped wrong) fails the reproducibility check.
    `palette_source` is stored repo-relative when it lives under the repo."""
    src = str(palette_source)
    try:
        rel = os.path.relpath(src, ROOT)
        if not rel.startswith(".."):
            src = rel.replace("\\", "/")
    except ValueError:                       # different drive on win32 → keep absolute
        pass
    return {"path": CANONICAL_CROP_RECIPE, "palette_source": src,
            "jpg_quality": int(jpg_quality)}
