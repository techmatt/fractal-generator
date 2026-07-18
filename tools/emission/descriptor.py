"""descriptor.py — location intake: admitted rows → Location, morph embedding, and
incremental morph-cluster assignment.

The admitted-location loader enforces the current-decode predicate
(`corpus_common.is_current_decoded`) — a v6/v5/unstamped row is never consumed as a
current verdict. The canonical morph embedding is the LIBRARY recipe verbatim (a 640×360
ss2 smooth field → `library_annotate.morph_gray_image` robust-z tanh gray →
`colored_clip` CLIP `vit_base_patch16_clip_224.openai`). Clustering is incremental and
WITHIN fractal type (matching the established within-family CLIP dedup convention): a
location joins an existing cluster iff its cosine to the cluster medoid exceeds the strict
near-dup threshold (0.974), else it founds a new cluster.

The Location construction + admitted filter + clustering are pure (numpy only); the CLIP
model + `library_annotate` are imported lazily inside `embed_locations` so this module
loads without torch for unit tests.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "corpus"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import corpus_common as cc            # noqa: E402  is_current_decoded / require_current
from tools.corpus import location as loc_mod  # noqa: E402

# Strict near-dup cosine threshold — the established within-family morph-CLIP dedup knee
# (tools/studies/morphology_dedup.py DEFAULT_THRESHOLD). Join a cluster iff cos > this.
NEAR_DUP_THRESHOLD = 0.974

# auto_maxiter policy (mirror tools/scoring/active_ckpt.py — replicated here to keep this
# module torch-free; it is a pure function of fw).
_FW_HOME = 3.0
_MAXITER_BASE, _MAXITER_K, _MAXITER_MIN, _MAXITER_MAX = 500, 0.30, 200, 8000


def auto_maxiter(fw: float) -> int:
    ratio = _FW_HOME / fw if fw > 0 else 1.0
    lz = math.log2(ratio) if ratio > 0 else 0.0
    val = _MAXITER_BASE * (1.0 + _MAXITER_K * lz)
    return int(max(_MAXITER_MIN, min(_MAXITER_MAX, val)))


# --------------------------------------------------------------------------- #
# Partition (ledger `family`) → render family (mirror steered_frontier.render_family_of).
# --------------------------------------------------------------------------- #
def render_family_of(partition: str) -> str:
    if partition == "mandelbrot" or partition in ("multibrot3", "multibrot4", "multibrot5"):
        return partition
    if partition == "julia:mandelbrot":
        return "julia"
    if partition.startswith("julia:multibrot"):
        return "julia_" + partition.split(":", 1)[1]
    raise ValueError(f"unknown partition {partition!r}")


def location_of(row: dict) -> loc_mod.Location:
    """Ledger row → canonical Location. Coords are the reframed OUTCOME viewport; julia
    twins carry the parameter c from the row (`julia_c_re/im`)."""
    fam = render_family_of(row["family"])
    fw = float(row["outcome_fw"])
    kw = dict(family=fam, cx=str(row["outcome_cx"]), cy=str(row["outcome_cy"]),
              fw=str(fw), maxiter=auto_maxiter(fw))
    if row.get("julia_c_re") is not None:
        kw["c_re"] = str(row["julia_c_re"])
        kw["c_im"] = str(row["julia_c_im"])
    return loc_mod.Location(**kw)


# --------------------------------------------------------------------------- #
# Admitted-location loader (current-decode ENFORCED).
# --------------------------------------------------------------------------- #
def load_admitted(ledger_path: Path, require_current: bool = False) -> list:
    """Yield admitted rows from a run-scoped ledger: current-decode ∧ decoded_class==3 ∧
    guard_pass ∧ distinct. With `require_current=True` a stale-decoded row RAISES
    (`cc.StaleDecodeError`) instead of being skipped — the strict verdict-trust form used
    to prove old-ledger rows are rejected."""
    rows = []
    for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if require_current:
            cc.require_current(row)       # raises on stale decode
        elif not cc.is_current_decoded(row):
            continue
        if row.get("decoded_class") != 3 or not row.get("guard_pass") or not row.get("distinct"):
            continue
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Canonical morph embedding (library recipe).
# --------------------------------------------------------------------------- #
def embed_locations(rows: list, field_cache: Path, embs_path: Path) -> dict:
    """location_id → (L2-normalized morph-CLIP embedding, retained field bin/json paths).

    Renders each location's 640×360 ss2 smooth field once (retained under `field_cache`
    for reuse by the pref palette ranker), grays it via the library robust-z tanh transfer,
    and CLIP-embeds. Persists the embeddings atomically to `embs_path` (npz keyed by id)."""
    import torch  # noqa: F401  (ensures the CUDA context is up before the CLIP load)
    from tools.wallpaper import library_annotate as la
    from tools.curation.colored_clip import load_clip, embed_clip
    from tools import colormap as cm

    field_cache.mkdir(parents=True, exist_ok=True)
    model, tf = load_clip()
    out = {}
    fields = {}
    for row in rows:
        loc = location_of(row)
        field = la.ensure_field(loc, retain=True, tmp_dir=field_cache, cache_root=field_cache)
        gray = la.morph_gray_image(field)
        emb = embed_clip(model, tf, [gray])[0].astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        out[row["id"]] = emb
        # remember the retained field path (deterministic stem) for the palette ranker.
        from tools.wallpaper import library_store as store
        stem = store.field_stem(loc, "smooth", la.W, la.H, la.SS)
        fields[row["id"]] = (str(field_cache / f"{stem}.bin"), str(field_cache / f"{stem}.json"))
    _save_embs(out, embs_path)
    return out, fields


def _save_embs(embs: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(embs.keys())
    mat = np.stack([embs[i] for i in ids]) if ids else np.zeros((0, 768), np.float32)
    # tmp MUST end in .npz — np.savez_compressed appends .npz to any other suffix, which
    # would leave os.replace looking for a file numpy never wrote.
    tmp = path.parent / (path.stem + "_tmp.npz")
    np.savez_compressed(tmp, ids=np.array(ids, dtype=object), emb=mat.astype(np.float32))
    import os
    os.replace(tmp, path)


def load_embs(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    z = np.load(path, allow_pickle=True)
    return {str(i): e.astype(np.float32) for i, e in zip(z["ids"], z["emb"])}


# --------------------------------------------------------------------------- #
# Incremental medoid clustering (within type, at the strict near-dup threshold).
# --------------------------------------------------------------------------- #
def _cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cluster_incremental(items: list, threshold: float = NEAR_DUP_THRESHOLD) -> dict:
    """items: list of (id, emb) IN A STABLE ORDER. Returns id → cluster_index.

    Incremental: for each item, join the existing cluster whose medoid it is most similar
    to IF that cosine exceeds `threshold`, else found a new cluster. The medoid is the
    founding member's embedding (a deterministic incremental medoid)."""
    medoids: list = []          # cluster_index → founding embedding
    assign: dict = {}
    for cid, emb in items:
        best_i, best_cos = -1, -1.0
        for i, med in enumerate(medoids):
            c = _cos(emb, med)
            if c > best_cos:
                best_cos, best_i = c, i
        if best_i >= 0 and best_cos > threshold:
            assign[cid] = best_i
        else:
            medoids.append(emb)
            assign[cid] = len(medoids) - 1
    return assign


def assign_morph_clusters(rows: list, embs: dict,
                          threshold: float = NEAR_DUP_THRESHOLD) -> dict:
    """location_id → morph cluster tag `<type>#<k>`, clustering WITHIN each fractal type
    (the within-family dedup convention). Ledger order is the stable incremental order."""
    by_type: dict = {}
    for row in rows:
        by_type.setdefault(row["family"], []).append(row["id"])
    tags = {}
    for ftype, ids in by_type.items():
        items = [(i, embs[i]) for i in ids if i in embs]
        assign = cluster_incremental(items, threshold)
        for i, k in assign.items():
            tags[i] = f"{ftype}#{k}"
    return tags
