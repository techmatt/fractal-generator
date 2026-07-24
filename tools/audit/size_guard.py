#!/usr/bin/env python3
"""Repo-size guard — the "stays small" constraint, as an enforced registry.

Sibling to `disk_audit.py`. Where that tool classifies data artifacts by
DELETION-SAFETY, this one enforces a different invariant: **the working tree
should stay ~what git tracks — source + irreplaceable metadata + `out/`.** Nothing
large lives in-tree without an explicit, written-down reason.

Two independent things live here:

  1. A WORKING-TREE size SCAN (`scan`). Walks the filesystem — not `git ls-files`,
     because a gitignored file can bloat the tree while invisible to git, which is
     the whole point. Flags:
       (a) any FILE >= FILE_THRESHOLD (1 MiB — matches the pre-commit blob hook), and
       (b) any DIRECTORY whose aggregate of SMALL files (< FILE_THRESHOLD) in its
           subtree >= DIR_THRESHOLD (~100 MB), reported at MINIMAL granularity (the
           leaf-most such dir). Rule (b) is deliberately keyed on small files only:
           big files are already caught one-by-one by rule (a), so keying (b) on the
           full aggregate would just re-flag every ancestor of a big file and force
           coarse mixed-disposition registry entries. Small-files-only isolates the
           many-small-file case (label crops, field caches) that no single-file rule
           can see.
     Excludes {out/, .venv/, target/, target-test/, .git/} from flagging. `.git` is
     a history-REWRITE target (git filter-repo), not a relocation one — its size is
     reported as an FYI line, never flagged.

  2. The REGISTRY (`REGISTRY`) — the deliverable. One explicit allowlist, same
     spirit as `tests/test_tracked_artifacts.py`'s `TRACKED_CANARIES`: the
     sanctioned-large-in-tree set. Every current violator is covered by exactly one
     entry, at a stable path-prefix granularity (so intra-dir churn — a new batch, a
     new crop — can't flake the guard). Each entry records size class, tracked-ness,
     and a DISPOSITION:
       KEEP     — legitimately stays in-tree (irreplaceable tracked metadata with no
                  smaller form). Being tracked is NOT an automatic pass; the reason
                  is the written-down "extremely good reason".
       RELOCATE -> <tier> — pending a move; delete the line when the move lands.
                  Tiers are disposition LABELS only (no dirs are created / paths
                  wired here — the precious-store *location* is still undecided):
                    artifacts      regenerable bulk (rebuildable render/cache output)
                    precious-store irreplaceable binaries (trained .pt weights)
                    trash          dead / superseded

The guard test (`tests/test_repo_size_guard.py`) fails on any flagged violator not
covered by a registry entry (new bloat caught from today), and REPORTS (does not
fail) any registry entry that no longer has over-threshold content (a nudge to
delete the line). As things relocate, their RELOCATE lines come out; when only KEEP
lines remain, every in-tree exception is explicit and reviewed.

This module MOVES / DELETES / COMMITS NOTHING. It scans and reports.

Usage
-----
  uv run python tools/audit/size_guard.py            # full report (inventory + guard status)
  uv run python tools/audit/size_guard.py --check    # terse pass/fail (what the test asserts)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Thresholds (constants — tune if the report is noisy or sparse).
# ---------------------------------------------------------------------------
FILE_THRESHOLD = 1 * 1024 * 1024        # 1 MiB — matches .git/hooks/pre-commit LIMIT
DIR_THRESHOLD = 100 * 1024 * 1024       # ~100 MB — many-small-file catch

# Excluded from FLAGGING (working-tree churn that is regenerable infra, not data
# bloat we relocate). `.git` is a history-rewrite target -> FYI size only.
EXCLUDE_PREFIXES = ("out", ".venv", "target", "target-test", ".git", ".pytest_cache")
GIT_DIR = ".git"

# ---------------------------------------------------------------------------
# Dispositions
# ---------------------------------------------------------------------------
KEEP = "KEEP"
RELOCATE = "RELOCATE"
ARTIFACTS = "artifacts"          # regenerable bulk
PRECIOUS = "precious-store"      # irreplaceable binaries (trained weights)
TRASH = "trash"                  # dead / superseded

# Report ordering for disposition groups.
GROUP_ORDER = [
    ("KEEP", None),
    ("RELOCATE", ARTIFACTS),
    ("RELOCATE", PRECIOUS),
    ("RELOCATE", TRASH),
]


@dataclass
class Entry:
    """One registry line. `prefix` is a repo-relative POSIX path; dir prefixes end
    with '/'. A violator is COVERED by this entry iff its path == prefix or starts
    with prefix (dir violators carry a trailing '/', so prefix matching is exact at
    the path-segment boundary)."""
    prefix: str
    disposition: str                 # KEEP | RELOCATE
    tier: str | None                 # None for KEEP; artifacts|precious-store|trash
    tracked: str                     # 'tracked' | 'ignored' | 'mixed'
    reason: str
    canary: bool = False             # covers a canaried path (move needs a canary update)

    def label(self) -> str:
        return self.disposition if self.disposition == KEEP else f"{RELOCATE} -> {self.tier}"


# ---------------------------------------------------------------------------
# THE REGISTRY — every current over-threshold violator, one covering line each.
# Stable path-prefix granularity: intra-dir churn (a new batch / crop / field) is
# absorbed by the prefix and never flakes the guard.
# ---------------------------------------------------------------------------
REGISTRY: list[Entry] = [
    # === KEEP — irreplaceable tracked metadata, legitimately in-tree ===========
    Entry("data/palettes/", KEEP, None, "mixed",
          "committed palette definitions (harvested 746-palette pool + features); "
          "load-bearing config for the palette system, tracked, no smaller form"),
    Entry("data/library_embeddings/", KEEP, None, "mixed",
          "prospect-library CLIP embeddings (embeddings.npz, tracked): unregenerable "
          "except value-approximate under a verdict-sensitive threshold. CANARY.",
          canary=True),

    # === RELOCATE -> artifacts — regenerable bulk (rebuildable render/cache) ====
    # Human-label corpora: the CROP JPGs are a pure function of render coords
    # (present/render-one). The tracked scores.json / images.jsonl labels + ledgers
    # are tiny and STAY in-tree (guarded by test_tracked_artifacts.py); only the crop
    # bulk relocates. `_work/` preview+staging subtrees are dead intermediates.
    Entry("data/label_corpus/", RELOCATE, ARTIFACTS, "mixed",
          "batch crops (regenerable via present/render-one) + dead `_work/` preview "
          "& crop-staging; tracked scores.json/images.jsonl labels stay in-tree"),
    Entry("data/wallpaper_corpus/", RELOCATE, ARTIFACTS, "mixed",
          "wallpaper batch crops (regenerable); tracked images.jsonl/ledgers stay"),
    Entry("data/render_mode_corpus/", RELOCATE, ARTIFACTS, "mixed",
          "render-mode batch crops (regenerable via present); tracked manifests stay"),
    Entry("data/label_crops/", RELOCATE, ARTIFACTS, "ignored",
          "early loose label-crop feed (loose0_v2/v3); regenerable render output"),
    Entry("data_large/label_crops/", RELOCATE, ARTIFACTS, "ignored",
          "loose0 crop feed; regenerable render output (tracked data_large/README stays)"),
    Entry("data/queries/", RELOCATE, ARTIFACTS, "mixed",
          "query-assembler field/colormap renders + scorer caches (regenerable via "
          "tools/queries); tracked queries/labels/*.json preference tiers stay"),
    Entry("data/library/", RELOCATE, ARTIFACTS, "mixed",
          "field_cache render bulk (regenerable); tracked library_records.jsonl stays"),
    Entry("data/root_field/", RELOCATE, ARTIFACTS, "ignored",
          "root8k f32 score-field cache (4x 256 MB); regenerable via the Rust dump "
          "(src/root_field.rs CACHE_DIR) — needs the Rust-side artifacts resolver first"),
    Entry("data/discovery/", RELOCATE, ARTIFACTS, "mixed",
          "regenerable run-state overlays (campaign*/steered*/shakeout* renders, "
          "logs); tracked ledgers/pools/outcome_feats provenance stays in-tree"),
    Entry("dramatic_palettes/", RELOCATE, ARTIFACTS, "mixed",
          "viz_render + viz_render_winners render sheets (regenerable); tracked "
          "palette definitions stay"),
    Entry("data/mining/", RELOCATE, ARTIFACTS, "mixed",
          "mining prospect renders (run1); regenerable via tools/mining"),
    Entry("data/guided_descend/", RELOCATE, ARTIFACTS, "mixed",
          "render/field caches (atlas_probe_step0, run5, julia_test_bulb); "
          "regenerable via present/enrich (tiny pool.jsonl pools stay)"),
    Entry("data/ranker/", RELOCATE, ARTIFACTS, "ignored",
          "frozen-feature location-ranker fits + feature caches (pref_loc_v0/v1, "
          "campaign1); regenerable — logistic on committed frozen features"),
    Entry("data/calibration/maxiter_diag/", RELOCATE, ARTIFACTS, "ignored",
          "maxiter diagnostic renders; regenerable (the frozen energy_calibration.json "
          "metric bins are tiny and stay tracked)"),
    # Build cache-manifests + plans: regenerable byte-identical via build_plan, but
    # read through hardcoded ROOT/data/vN/ paths (see repo_size_audit Phase 1).
    Entry("data/v4/", RELOCATE, ARTIFACTS, "ignored",
          "v4 build cache-manifest + plan + montage (regenerable via build_plan; "
          "superseded version)"),
    Entry("data/v5/", RELOCATE, ARTIFACTS, "ignored",
          "v5 build cache-manifest + julia plan (regenerable via build_plan)"),
    Entry("data/v6/", RELOCATE, ARTIFACTS, "ignored",
          "v6 build cache-manifest + gather plan (regenerable via build_plan)"),
    Entry("data/v7/", RELOCATE, ARTIFACTS, "ignored",
          "v7 build cache-manifest + plan (regenerable via build_plan; active version)"),

    # === RELOCATE -> precious-store — irreplaceable trained binaries (.pt) ======
    # Not GPU-reproducible (float nondeterminism), so no rebuild path. Active +
    # rollback anchors move to the precious store; the v5/v6/v7 weights are CANARY
    # paths — their eventual move needs a deliberate test_tracked_artifacts update.
    Entry("data/classifier/v7/", RELOCATE, PRECIOUS, "tracked",
          "v7 model_best.pt — LIVE deployed discovery-gate weight. CANARY.", canary=True),
    Entry("data/classifier/v6/", RELOCATE, PRECIOUS, "tracked",
          "v6 model_best.pt — one-flip rollback anchor. CANARY.", canary=True),
    Entry("data/classifier/v5/", RELOCATE, PRECIOUS, "tracked",
          "v5 model_best.pt — deeper rollback anchor. CANARY.", canary=True),
    Entry("data/wallpaper_head/", RELOCATE, PRECIOUS, "ignored",
          "trained wallpaper-quality heads (v1/v2/v3 .pt) — not GPU-reproducible; "
          "active + rollback -> precious-store, older versions curate to trash at move"),
    Entry("data/render_mode_head/", RELOCATE, PRECIOUS, "ignored",
          "trained render-mode (strange-mode gate) head v1 .pt — not GPU-reproducible"),

    # === RELOCATE -> trash — dead / superseded ================================
    Entry("data/classifier/v2/", RELOCATE, TRASH, "ignored",
          "superseded classifier v2 weight — won't be retrained"),
    Entry("data/classifier/v3/", RELOCATE, TRASH, "ignored",
          "superseded classifier v3 weight — won't be retrained"),
    Entry("data/classifier/v4/", RELOCATE, TRASH, "ignored",
          "superseded classifier v4 weight — won't be retrained"),
    Entry("data/classifier/v5_seed1/", RELOCATE, TRASH, "ignored",
          "v5 seed-1 diagnostic variant — not the live checkpoint, disposable"),
    Entry("data/focus_diag/", RELOCATE, TRASH, "ignored",
          "focus-diagnostic scratch (orbit-space field .npy dumps); dead, regenerable"),
    Entry("scratchpad/", RELOCATE, TRASH, "ignored",
          "canonical disposable temp dir — nothing large should persist here"),
]


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
@dataclass
class Violator:
    rel: str            # repo-relative POSIX; dir violators end with '/'
    size: int
    is_dir: bool


@dataclass
class ScanResult:
    file_violators: list[Violator]
    dir_violators: list[Violator]
    git_size: int
    # populated by check_registry:
    uncovered: list[Violator] = field(default_factory=list)
    stale: list[Entry] = field(default_factory=list)

    @property
    def violators(self) -> list[Violator]:
        return self.file_violators + self.dir_violators


def _excluded(rel_parts: tuple[str, ...]) -> bool:
    return bool(rel_parts) and rel_parts[0] in EXCLUDE_PREFIXES


def scan(repo: Path) -> ScanResult:
    """Walk the working tree; return file + directory violators and the .git FYI size."""
    small_sub: dict[Path, int] = {}     # subtree bytes of files < FILE_THRESHOLD
    file_viol: list[Violator] = []

    # one pruned top-down walk: collect each dir's own small-file bytes + its kept
    # children, and flag big files inline. Excluded top-level trees are never descended.
    small_own: dict[Path, int] = {}
    kid_map: dict[Path, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(repo, topdown=True):
        d = Path(dirpath)
        rel = d.relative_to(repo)
        if rel == Path("."):
            dirnames[:] = [n for n in dirnames if n not in EXCLUDE_PREFIXES]
        elif _excluded(rel.parts):
            dirnames[:] = []
            continue
        own = 0
        for fn in filenames:
            fp = d / fn
            try:
                sz = fp.stat().st_size
            except OSError:
                continue
            if sz >= FILE_THRESHOLD:
                file_viol.append(Violator(fp.relative_to(repo).as_posix(), sz, False))
            else:
                own += sz
        small_own[d] = own
        kid_map[d] = [d / n for n in dirnames]

    # bottom-up subtree small-file sums
    for d in sorted(kid_map, key=lambda p: len(p.parts), reverse=True):
        total = small_own.get(d, 0)
        for k in kid_map[d]:
            total += small_sub.get(k, 0)
        small_sub[d] = total

    # rule (b): MINIMAL dirs whose small-file subtree >= DIR_THRESHOLD (no child qualifies)
    dir_viol: list[Violator] = []
    for d, sz in small_sub.items():
        if sz < DIR_THRESHOLD:
            continue
        if any(small_sub.get(k, 0) >= DIR_THRESHOLD for k in kid_map.get(d, [])):
            continue
        rel = d.relative_to(repo)
        if rel == Path("."):
            continue
        dir_viol.append(Violator(rel.as_posix() + "/", sz, True))

    git_size = _dir_size(repo / GIT_DIR)
    file_viol.sort(key=lambda v: -v.size)
    dir_viol.sort(key=lambda v: -v.size)
    return ScanResult(file_viol, dir_viol, git_size)


def _dir_size(root: Path) -> int:
    total = 0
    for dirpath, _dn, filenames in os.walk(root):
        d = Path(dirpath)
        for fn in filenames:
            try:
                total += (d / fn).stat().st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------
def covering_entry(rel: str, registry: list[Entry] = REGISTRY) -> Entry | None:
    """Most-specific (longest-prefix) registry entry covering `rel`, or None."""
    best: Entry | None = None
    for e in registry:
        if rel == e.prefix or rel.startswith(e.prefix):
            if best is None or len(e.prefix) > len(best.prefix):
                best = e
    return best


def check_registry(res: ScanResult, registry: list[Entry] = REGISTRY) -> ScanResult:
    """Fill res.uncovered (violators no entry covers) and res.stale (entries covering
    no current violator)."""
    covered_prefixes: set[str] = set()
    uncovered: list[Violator] = []
    for v in res.violators:
        e = covering_entry(v.rel, registry)
        if e is None:
            uncovered.append(v)
        else:
            covered_prefixes.add(e.prefix)
    res.uncovered = sorted(uncovered, key=lambda v: -v.size)
    res.stale = [e for e in registry if e.prefix not in covered_prefixes]
    return res


def entry_size(res: ScanResult, entry: Entry) -> int:
    """Total violator bytes assigned (most-specifically) to this entry."""
    tot = 0
    for v in res.violators:
        if covering_entry(v.rel) is entry:
            tot += v.size
    return tot


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def human(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or u == "TB":
            return f"{int(x)}B" if u == "B" else f"{x:.1f}{u}"
        x /= 1024
    return f"{x:.1f}TB"


def _report(repo: Path) -> int:
    res = check_registry(scan(repo))
    print("=" * 78)
    print(f"REPO-SIZE GUARD   root={repo}")
    print(f"  file threshold >= {human(FILE_THRESHOLD)}   dir(small-file) threshold >= {human(DIR_THRESHOLD)}")
    print(f"  excluded from flagging: {', '.join(EXCLUDE_PREFIXES)}")
    print("=" * 78)
    n_v = len(res.violators)
    tot = sum(v.size for v in res.violators)
    print(f"\n{n_v} violators ({len(res.file_violators)} files + {len(res.dir_violators)} "
          f"small-file dirs), {human(tot)} flagged.")
    print(f".git FYI (history-rewrite target, not flagged): {human(res.git_size)}")

    # grouped by disposition
    for disp, tier in GROUP_ORDER:
        entries = [e for e in REGISTRY if e.disposition == disp and e.tier == tier]
        if not entries:
            continue
        head = disp if disp == KEEP else f"{RELOCATE} -> {tier}"
        gtot = sum(entry_size(res, e) for e in entries)
        print(f"\n--- {head}  ({human(gtot)}) ---")
        for e in sorted(entries, key=lambda e: -entry_size(res, e)):
            sz = entry_size(res, e)
            tag = " [CANARY]" if e.canary else ""
            print(f"  {human(sz):>9}  {e.prefix:<52} {e.tracked}{tag}")
            print(f"             -> {e.reason}")

    print("\n" + "=" * 78)
    if res.uncovered:
        print(f"UNCOVERED VIOLATORS ({len(res.uncovered)}) — new bloat, no registry entry:")
        for v in res.uncovered:
            print(f"  {human(v.size):>9}  {v.rel}")
    else:
        print("OK: every violator is covered by a registry entry.")
    if res.stale:
        print(f"\nSTALE REGISTRY ENTRIES ({len(res.stale)}) — no over-threshold content, "
              f"delete the line:")
        for e in res.stale:
            print(f"  {e.prefix}")
    else:
        print("OK: no stale registry entries.")
    print("=" * 78)
    return 1 if res.uncovered else 0


def _check(repo: Path) -> int:
    res = check_registry(scan(repo))
    if res.uncovered:
        print(f"FAIL: {len(res.uncovered)} uncovered violator(s):", file=sys.stderr)
        for v in res.uncovered:
            print(f"  {human(v.size):>9}  {v.rel}", file=sys.stderr)
        return 1
    print(f"PASS: {len(res.violators)} violators, all covered; "
          f"{len(res.stale)} stale entr{'y' if len(res.stale)==1 else 'ies'}.")
    return 0


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="terse pass/fail only (what the pytest guard asserts)")
    args = ap.parse_args()
    return _check(repo) if args.check else _report(repo)


if __name__ == "__main__":
    raise SystemExit(main())
