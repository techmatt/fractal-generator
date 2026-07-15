#!/usr/bin/env python
r"""Durable location-library store — records, crash-safe embedding shards, LRU field cache.

The persistence layer under the Phase-1 prospecting loop (`prospect_orchestrator.py`).
Everything here is **pure I/O + numpy** — NO GPU, NO torch, NO render — so it unit-tests
without a model. The GPU annotate tail (`library_annotate.py`) computes vectors/thumbnails
and hands them here to persist.

Three durable artifacts, all surviving `rm -r out/*` (they live under `data/`):

  data/library/records.jsonl                 append-only, one JSON record per LOCATION.
                                             `location_id` is the primary key; appends
                                             DEDUP on it, so re-running a cycle adds 0
                                             duplicates (resume idempotence).
  data/library/thumbs/<location_id>.jpg      384x216 grayscale thumbnail per location.
  data/library_embeddings/shards/            per-cycle npz shards (morph_uids + morph_clip),
    <run_id>__cycle_<NNN>.npz                written tmp+atomic-rename so a hard-kill mid-
                                             write never truncates a prior shard. A loader
                                             CONCATENATES the base embeddings.npz + every
                                             shard into one logical uid->vector store.

Embedding-append design (chosen: SHARD-PER-CYCLE, not in-place rewrite).
  The existing `data/library_embeddings/embeddings.npz` (62 curated morph rows + 564 colored)
  is the immutable BASE. New library locations append as one small npz per cycle under
  `shards/`. Rationale over "rewrite embeddings.npz in place each cycle": (1) crash-safety by
  construction — a kill can only corrupt the in-flight tmp shard (np.load fails -> loader
  skips it -> resume re-derives that one cycle), never the base or a prior shard; (2) O(cycle)
  append cost stays flat over a days-long run instead of rewriting an ever-growing monolith;
  (3) `by-reference keying` is preserved because records key descriptors by UID (== location_id),
  and the loader resolves uid->row across base+shards (last-writer-wins), so a global integer
  row index is never needed. morph_v6 is intentionally absent from shards (the grayscale v6
  prelogits were PROMOTED verbatim in the old store, not computed; recomputing them for a fresh
  location is not free -> skipped, per the build spec; v6 location-potential still rides the
  ledger join).

Field-cache retention (LRU under a GB cap). Retained smooth fields make Phase-2 colorize cheap.
  Kept under `data/library/field_cache/` (survives out/* wipes, unlike the disposable seeder
  scratch) but LRU-EVICTED under `--field-cache-gb` so it can never become an unbounded sink.
  Fields are deterministic from coordinates, so eviction costs re-dump time, never data. The
  cache key reuses the canonical `location.field_mode_token` + geometry tokens (see `field_stem`)
  — the same scheme deploy_tail keys its token'd dumps by — so no parallel cache is invented.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tools.corpus import location as loc_mod  # noqa: E402

# --- durable roots (data/ = survives `rm -r out/*`) ---
LIB_ROOT = ROOT / "data" / "library"
RECORDS_PATH = LIB_ROOT / "records.jsonl"
THUMBS_DIR = LIB_ROOT / "thumbs"
FIELD_CACHE_DIR = LIB_ROOT / "field_cache"
EMB_ROOT = ROOT / "data" / "library_embeddings"
EMB_BASE = EMB_ROOT / "embeddings.npz"          # immutable base (dim source of truth)
EMB_SHARDS = EMB_ROOT / "shards"

MORPH_CLIP_DIM = 768                            # asserted against the base store at append


# =========================================================================== #
# Records store (append-only JSONL, dedup by location_id).
# =========================================================================== #
def existing_location_ids(records_path: Path = RECORDS_PATH) -> set[str]:
    """The set of location_ids already persisted (the dedup / resume-idempotence key)."""
    if not records_path.exists():
        return set()
    ids = set()
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["location_id"])
    return ids


def append_records(records: list[dict], records_path: Path = RECORDS_PATH) -> list[dict]:
    """Append records whose location_id is NOT already present; return the ones written.

    Dedup-on-append is the whole resume-idempotence story: a re-run of a cycle produces the
    same location_ids, all of which already exist -> nothing is appended. Each row is written
    then flushed on the spot (durable append ledger, never buffered)."""
    have = existing_location_ids(records_path)
    fresh = [r for r in records if r["location_id"] not in have]
    if not fresh:
        return []
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with open(records_path, "a", encoding="utf-8") as f:
        for r in fresh:
            f.write(json.dumps(r) + "\n")
            f.flush()
    return fresh


# =========================================================================== #
# Embedding shards (crash-safe append; loader concatenates base + shards).
# =========================================================================== #
def base_morph_dim(emb_base: Path = EMB_BASE) -> int:
    """morph_clip column count from the immutable base store — the dim source of truth.
    Falls back to MORPH_CLIP_DIM only if the base store is absent (fresh checkout)."""
    if not emb_base.exists():
        return MORPH_CLIP_DIM
    z = np.load(emb_base, allow_pickle=True)
    if "morph_clip" not in z.files:
        return MORPH_CLIP_DIM
    return int(z["morph_clip"].shape[1])


def write_embedding_shard(run_id: str, cycle: int, uids: list[str], clip: np.ndarray,
                          shards_dir: Path = EMB_SHARDS, emb_base: Path = EMB_BASE) -> Path:
    """Write one cycle's morph embeddings to `<run_id>__cycle_<NNN>.npz`, crash-safely.

    Asserts the embedding dim against the base store BEFORE writing (the store's shapes are the
    source of truth — read, don't assume). Writes to a `.tmp` then os.replace's into place, so a
    kill mid-write leaves at most a stray tmp (ignored by the loader) and never a truncated shard.
    Overwriting the same (run_id, cycle) shard is idempotent — a resumed cycle rewrites identical
    content."""
    clip = np.asarray(clip, dtype=np.float32)
    if len(uids) == 0:
        # nothing to persist this cycle; ensure no stale shard lingers
        return shards_dir / f"{run_id}__cycle_{cycle:03d}.npz"
    dim = base_morph_dim(emb_base)
    assert clip.ndim == 2 and clip.shape[1] == dim, (
        f"morph_clip dim {clip.shape} != base store dim {dim} "
        f"(base {emb_base}); embedding space would be inconsistent")
    assert clip.shape[0] == len(uids), f"uids/clip length mismatch {len(uids)} vs {clip.shape[0]}"
    shards_dir.mkdir(parents=True, exist_ok=True)
    final = shards_dir / f"{run_id}__cycle_{cycle:03d}.npz"
    tmp = shards_dir / f".{run_id}__cycle_{cycle:03d}.npz.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, morph_uids=np.asarray(uids), morph_clip=clip)
    os.replace(tmp, final)               # atomic on Windows + POSIX
    return final


def load_library_embeddings(emb_base: Path = EMB_BASE, shards_dir: Path = EMB_SHARDS
                            ) -> dict[str, np.ndarray]:
    """Concatenate base morph rows + every shard into one uid -> morph_clip vector map.

    Later writers win (a re-run cycle's shard overwrites an earlier one on disk, and among
    shards the lexicographically-last — same run_id, higher cycle — is applied last). A shard
    that fails to load (a leftover partial tmp is never named `.npz`, but be defensive) is
    skipped, matching the crash-safety contract."""
    out: dict[str, np.ndarray] = {}
    if emb_base.exists():
        z = np.load(emb_base, allow_pickle=True)
        if "morph_uids" in z.files and "morph_clip" in z.files:
            for u, v in zip(z["morph_uids"].tolist(), z["morph_clip"]):
                out[u] = v
    if shards_dir.exists():
        for shard in sorted(shards_dir.glob("*.npz")):
            try:
                z = np.load(shard, allow_pickle=True)
                for u, v in zip(z["morph_uids"].tolist(), z["morph_clip"]):
                    out[u] = v
            except (OSError, ValueError, EOFError):
                continue
    return out


# =========================================================================== #
# Field-cache key (canonical: location.field_mode_token + geometry tokens).
# =========================================================================== #
def field_stem(loc, mode: str, w: int, h: int, ss: int) -> str:
    """Canonical retained-field stem — the SAME scheme deploy_tail keys its token'd dumps by
    (family + sha1(key|geom|maxiter|mode-token) + geom + mode). smooth-mode token is empty, so
    the smooth field keys identically to every other smooth-field consumer (no parallel cache)."""
    tok = loc_mod.field_mode_token(mode)
    suffix = f"|{tok}" if tok else ""
    h16 = hashlib.sha1(
        f"{loc.key()}|{w}x{h}ss{ss}|{loc.maxiter}{suffix}".encode()).hexdigest()[:16]
    return f"{loc_mod.family_of(loc)}_{h16}_{w}x{h}ss{ss}__{mode}"


# =========================================================================== #
# LRU field-cache eviction under a GB cap.
# =========================================================================== #
def field_cache_bytes(cache_dir: Path = FIELD_CACHE_DIR) -> int:
    if not cache_dir.exists():
        return 0
    return sum(f.stat().st_size for f in cache_dir.iterdir() if f.is_file())


def evict_field_cache_lru(cap_gb: float, cache_dir: Path = FIELD_CACHE_DIR,
                          log=None) -> tuple[int, int]:
    """Evict least-recently-USED (oldest atime, mtime fallback) field-cache entries until the
    directory is under `cap_gb`. Each field is a (`.bin`, `.json`) pair evicted together, keyed
    by stem. Deterministic-from-coords, so eviction costs re-dump time, never data. Returns
    (pairs_evicted, bytes_freed). No-op when the cache is absent or already under cap."""
    cap = int(cap_gb * 2**30)
    if not cache_dir.exists():
        return 0, 0
    # group files by stem into (bin, json) pairs; size + recency per pair
    pairs: dict[str, dict] = {}
    for f in cache_dir.iterdir():
        if not f.is_file():
            continue
        stem = f.stem
        st = f.stat()
        p = pairs.setdefault(stem, {"files": [], "size": 0, "atime": 0.0})
        p["files"].append(f)
        p["size"] += st.st_size
        # use the max access/mod time across the pair as its recency
        p["atime"] = max(p["atime"], st.st_atime, st.st_mtime)
    total = sum(p["size"] for p in pairs.values())
    if total <= cap:
        return 0, 0
    # evict oldest-first until under cap
    order = sorted(pairs.items(), key=lambda kv: kv[1]["atime"])
    evicted = freed = 0
    for _stem, p in order:
        if total <= cap:
            break
        for f in p["files"]:
            try:
                f.unlink()
            except OSError:
                pass
        total -= p["size"]
        freed += p["size"]
        evicted += 1
    if log and evicted:
        log(f"  field-cache LRU: evicted {evicted} field(s), freed ~{freed/2**30:.2f} GiB "
            f"(now ~{total/2**30:.2f}/{cap_gb} GiB)")
    return evicted, freed


# =========================================================================== #
# Store-wide summary (for the run report + smoke check).
# =========================================================================== #
def store_summary(records_path: Path = RECORDS_PATH, thumbs_dir: Path = THUMBS_DIR,
                  shards_dir: Path = EMB_SHARDS) -> dict:
    n_records = 0
    fams: dict[str, int] = {}
    if records_path.exists():
        with open(records_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n_records += 1
                fam = json.loads(line)["identity"]["family"]
                fams[fam] = fams.get(fam, 0) + 1
    n_thumbs = len(list(thumbs_dir.glob("*.jpg"))) if thumbs_dir.exists() else 0
    n_shards = len(list(shards_dir.glob("*.npz"))) if shards_dir.exists() else 0
    emb = load_library_embeddings(shards_dir=shards_dir)
    return {"records": n_records, "by_family": fams, "thumbs": n_thumbs,
            "shards": n_shards, "embeddings_total": len(emb)}
