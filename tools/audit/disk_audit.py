#!/usr/bin/env python3
"""Disk audit + safe-delete tool for the fractal-generator data tree.

Scans large image/data artifacts, classifies each by DELETION-SAFETY using the
project's authoritative-vs-regenerable structure, reports reclaimable space, and
gates any actual deletion behind explicit --apply + --categories plus a hard
content-safety guard. Audit-first, dry-run by default. Human labels live in this
tree and are irreplaceable.

Categories
----------
  never       Authoritative / irreplaceable. Never deletable, ever.
  regenerable Safe to delete; carries a rebuild-cost note.
  scratch     Safe, cheap (out/ scratch, diag probes, logs).
  ambiguous   Report only. NEVER auto-deletable. Needs a human decision.

Hard content guard (non-negotiable): before any directory can be classified as
anything other than `never`, its subtree is scanned for images.jsonl / scores.json
carrying a populated (non-null) human `label.score`. If found -> forced to
`never`, regardless of path. Paths can surprise; label content is ground truth.

Cache-reuse guard (same philosophy, for reused-not-labeled artifacts): a cache dir
can be regenerable in isolation yet load-bearing because a *later* version reuses it
as a frozen base under a byte-identity recipe-parity gate (e.g. data/v4/aug_cache is
the frozen Mandelbrot base every v5/v6 location-classifier train reuses verbatim).
Two protections force such a dir + its whole subtree to `never`:
  (1) explicit: any dir containing an `.audit-keep` sentinel file (reason = its text);
  (2) best-effort auto-detect: any cache-like dir whose `"<ver>" / "<name>"` path
      segments are referenced in a tools/*/build_plan.py source (the parity gate /
      cache-manifest reuse site). Catches future caches nobody sentinel'd.
Protect by what the artifact *is* / how it's used, not by path-pattern version-guessing.

Usage
-----
  uv run python tools/audit/disk_audit.py                 # audit whole repo (data/ + out/)
  uv run python tools/audit/disk_audit.py --root data     # audit one root
  uv run python tools/audit/disk_audit.py --min-mb 50     # lower report threshold
  uv run python tools/audit/disk_audit.py --apply --categories scratch   # gated delete

Deletion is OFF unless BOTH --apply and --categories are given, and never touches
`never` or `ambiguous`. The content guard re-runs at delete time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
NEVER = "never"
REGEN = "regenerable"
SCRATCH = "scratch"
AMBIG = "ambiguous"

SAFE_CATEGORIES = {REGEN, SCRATCH}          # the only categories --apply may touch
CATEGORY_ORDER = [NEVER, AMBIG, REGEN, SCRATCH]

# ---------------------------------------------------------------------------
# Classification rules. First match wins. Evaluated against the repo-relative
# POSIX path (dirs have a trailing '/'). Encodes the project's real
# authoritative-vs-regenerable structure (see CLAUDE.md + prompts/disk_audit_safe_delete.md).
# ---------------------------------------------------------------------------
@dataclass
class Rule:
    pattern: str
    category: str
    reason: str
    rebuild: str = ""          # rebuild-cost note for regenerable items
    _rx: re.Pattern = field(init=False, repr=False)

    def __post_init__(self):
        self._rx = re.compile(self.pattern)


RULES: list[Rule] = [
    # -- provenance files, wherever they live: non-reproducible, tiny, keep ----
    Rule(r"(^|/)pool\.jsonl$", NEVER, "guided-descend location pool (stochastic, non-reproducible provenance)"),
    Rule(r"(^|/)(outcome_ledger|probe_rejects)\.jsonl$", NEVER, "append-only discovery ledger"),
    Rule(r"(^|/)outcome_feats\.npz$", NEVER, "discovery q3 outcome-feature cloud"),
    Rule(r"(^|/)[^/]*(manifest|ledger)[^/]*\.jsonl?$", NEVER, "run manifest / ledger (provenance)"),
    Rule(r"(^|/)scores\.json$", NEVER, "exported human label scores"),
    Rule(r"(^|/)images\.jsonl$", NEVER, "batch image manifest / label rows"),

    # -- human corpora: labels + the crops they reference ---------------------
    Rule(r"^data/label_corpus/", NEVER, "human location-label corpus"),
    Rule(r"^data/wallpaper_corpus/", NEVER, "human wallpaper-label corpus + batches"),

    # -- active + rollback model checkpoints ----------------------------------
    Rule(r"^data/classifier/v6/", NEVER, "active discovery-gate classifier v6"),
    Rule(r"^data/classifier/v5/", NEVER, "v5 classifier (live rollback checkpoint)"),
    Rule(r"^data/wallpaper_head/", NEVER, "wallpaper-quality head v1 (active)"),

    # -- config / palette / calibration artifacts (load-bearing definitions) --
    Rule(r"^data/palettes/", NEVER, "committed palette definitions"),
    Rule(r"^data/calibration/", NEVER, "frozen energy-calibration bins (metric definition)"),
    Rule(r"(^|/)(score3_colormaps|aug_roster)[^/]*$", NEVER, "score-3 palette / augmentation roster config"),
    Rule(r"(^|/)[^/]*_pool[^/]*\.json$", NEVER, "curated seed pool (661/777) config"),

    # -- AMBIGUOUS: superseded classifier versions v1-v4 + their aug caches ----
    #    (spec: likely the biggest reclaimable block, but keep-for-repro is Matt's call)
    Rule(r"^data/classifier/v2/", AMBIG, "superseded classifier v2 checkpoint (keep-for-repro?)"),
    Rule(r"^data/classifier/v3/", AMBIG, "superseded classifier v3 checkpoint (keep-for-repro?)"),
    Rule(r"^data/classifier/v4/", AMBIG, "superseded classifier v4 checkpoint (keep-for-repro?)"),
    Rule(r"^data/classifier/v5_seed1/", AMBIG, "v5 seed-1 diagnostic variant (not the live checkpoint)"),
    Rule(r"^data/v4/", AMBIG, "v4 per-version augmentation cache (superseded v4 -> keep-for-repro?)"),

    # -- AMBIGUOUS: atlas-era + stale crop feeds ------------------------------
    Rule(r"^data/atlas/", AMBIG, "atlas-era data (superseded discovery approach?)"),
    Rule(r"^data/atlas_probe/", AMBIG, "atlas probe-era data"),
    Rule(r"^data/label_crops/", AMBIG, "early loose label-crop feed (superseded by label_corpus?)"),
    Rule(r"^data/discovery/gather/", AMBIG, "raw gather crops beyond the label corpus (keep-for-mining?)"),

    # -- REGENERABLE: aug caches for the ACTIVE v5/v6 (rebuild = compute) ------
    Rule(r"^data/v5/aug_cache", REGEN, "v5 (active) augmentation cache",
         "rebuild via build_plan.py; byte-identical recipe-parity => full render compute"),
    Rule(r"^data/v6/aug_cache", REGEN, "v6 (active) augmentation cache",
         "rebuild via build_plan.py; byte-identical recipe-parity => full render compute"),
    Rule(r"(^|/)aug_cache[^/]*/", REGEN, "augmentation cache",
         "rebuild via build_plan.py; byte-identical recipe-parity => full render compute"),
    Rule(r"(^|/)_montage_tiles/", REGEN, "montage tile cache", "regenerate from source renders"),

    # -- REGENERABLE: field dumps + eval montages ------------------------------
    Rule(r"^data/root_field/", REGEN, "root-field cache", "regenerate via root_field dump"),
    Rule(r"(^|/)root_fields?/", REGEN, "field-dump cache", "regenerate via dump-field"),
    Rule(r"(^|/)fields?/", REGEN, "ss4 label-field / coarse-field cache", "regenerate via dump-field"),
    Rule(r"(^|/)montages?/", REGEN, "eval-visualization montages", "regenerate from eval scores"),

    # -- REGENERABLE: pipeline render caches ----------------------------------
    Rule(r"^data/queries/", REGEN, "query-assembler field/colormap renders", "regenerate via tools/queries"),
    Rule(r"^data/enrich/", REGEN, "enrich render cache", "regenerate via enrich --mode render"),
    Rule(r"^data/mining/", REGEN, "mining prospect renders", "regenerate via tools/mining"),
    Rule(r"^data/generated/", REGEN, "generate-subcommand render manifest views", "regenerate via generate"),
    Rule(r"^data/eda/", REGEN, "EDA field renders", "regenerate from source"),
    Rule(r"^data/guided_descend/", REGEN, "guided-descend render/field caches (pools kept separately)",
         "regenerate via present/enrich from the pool"),

    # -- SCRATCH: diag probes + logs + out/ tree ------------------------------
    Rule(r"^data/focus_diag/", SCRATCH, "focus-diagnostic scratch"),
    Rule(r"^data/gate_diag/", SCRATCH, "gate-diagnostic scratch"),
    Rule(r"^data/palette_probe/", SCRATCH, "palette-probe scratch"),
    Rule(r"^data/wallpaper_harvest/", SCRATCH, "wallpaper-harvest scratch (superseded by corpus)"),
    Rule(r"^data/discovery/runs/", NEVER, "per-run discovery ledgers (append-only provenance)"),
    Rule(r"^out/", SCRATCH, "disposable out/ scratch tree"),
    # scratchpad is the canonical disposable temp dir: image renders (contact
    # sheets, montages, sweep frames, dumped-field previews) are scratch. Data /
    # plan / script files (*.json, *.jsonl, *.py) get NO rule here on purpose, so
    # they fall through to `ambiguous` (report-only, never auto-deleted) — a
    # load-bearing plan or cache must be a human call, not a path guess.
    Rule(r"^scratchpad/.*\.(png|jpe?g)$", SCRATCH, "scratchpad render output (disposable temp dir)"),
    Rule(r"(^|/)[^/]*\.log$", SCRATCH, "log file"),
]

DEFAULT_UNMATCHED = Rule("", AMBIG, "unclassified by any rule -> needs a human decision")

# ---------------------------------------------------------------------------
# Content-safety guard: does a subtree hold a populated human label?
# ---------------------------------------------------------------------------
LABEL_FILES = {"images.jsonl", "scores.json"}


def _file_has_populated_label(path: Path) -> bool:
    """True if this label file carries a non-null label.score anywhere."""
    try:
        if path.name == "scores.json":
            # {"<image_id>": <score-or-obj>, ...}  score in {1,2,3} or {"score": n}
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                for v in obj.values():
                    s = v.get("score") if isinstance(v, dict) else v
                    if s is not None and s != 0:
                        return True
            return False
        # images.jsonl: one JSON row per line, look for label.score non-null
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lab = row.get("label")
                if isinstance(lab, dict) and lab.get("score") is not None:
                    return True
        return False
    except (OSError, json.JSONDecodeError):
        # Unreadable label file -> assume it MIGHT hold labels (fail safe).
        return True


# ---------------------------------------------------------------------------
# Cache-reuse guard: protect dirs that are reused as a frozen base (sentinel or
# referenced in a build_plan recipe-parity gate). Mirrors the content guard —
# protects by how the artifact is used, not by path-pattern version-guessing.
# ---------------------------------------------------------------------------
SENTINEL_FILE = ".audit-keep"
CACHE_DIR_RX = re.compile(r"aug_cache|_cache\b|cache$")   # cache-like dir names
# where cross-version reuse is declared (parity gate + verbatim cache-manifest reuse)
REUSE_SOURCE_GLOB = "tools/*/build_plan.py"


def _reuse_source_text(repo: Path) -> str:
    """Concatenated text of the build_plan sources that declare frozen-cache reuse."""
    chunks = []
    for p in sorted(repo.glob(REUSE_SOURCE_GLOB)):
        try:
            chunks.append(p.read_text(encoding="utf-8"))
        except OSError:
            pass
    return "\n".join(chunks)


def collect_protected(repo: Path, roots: list[Path], src_text: str) -> dict[str, str]:
    """Map {dir_rel_with_slash: reason} for dirs forced to NEVER by the cache-reuse
    guard. One pruned walk: a protected dir's whole subtree is protected, so we stop
    descending into it (also skips the huge per-location cache subtrees cheaply)."""
    protected: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            drel = d.relative_to(repo).as_posix() + "/"
            reason = None
            if SENTINEL_FILE in filenames:
                try:
                    txt = (d / SENTINEL_FILE).read_text(encoding="utf-8").strip()
                except OSError:
                    txt = ""
                first = txt.splitlines()[0].strip() if txt else "(.audit-keep sentinel)"
                reason = f"SENTINEL: {first}"
            elif src_text and CACHE_DIR_RX.search(d.name):
                parts = d.relative_to(repo).parts        # e.g. ('data','v4','aug_cache')
                if len(parts) >= 2 and re.search(
                        r'"%s"\s*/\s*"%s"' % (re.escape(parts[-2]), re.escape(parts[-1])),
                        src_text):
                    reason = ("AUTO-DETECT: referenced as a frozen/reused cache in a "
                              "build_plan recipe-parity gate; deleting breaks that build")
            if reason:
                protected[drel] = reason
                dirnames[:] = []                          # prune: subtree protected
    return protected


def protected_reason(drel: str, protected: dict[str, str]) -> str | None:
    """Reason if `drel` is a protected dir or lies under one, else None."""
    for pref, reason in protected.items():
        if drel == pref or drel.startswith(pref):
            return reason
    return None


# ---------------------------------------------------------------------------
# Walk + size + classify
# ---------------------------------------------------------------------------
@dataclass
class Node:
    path: Path
    rel: str                       # repo-relative POSIX, dirs end with '/'
    is_dir: bool
    size: int = 0                  # recursive byte size for dirs
    category: str = ""
    reason: str = ""
    rebuild: str = ""
    has_label: bool = False        # populated human label somewhere in subtree
    tracked: bool = False          # git-tracked anywhere in subtree
    synthetic: bool = False        # report-only aggregate (loose files in a mixed
                                   # dir); NEVER deletable — path points at the dir,
                                   # not the individual files it stands for.


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}B"
        x /= 1024
    return f"{x:.1f}TB"


def classify(rel: str) -> Rule:
    for r in RULES:
        if r._rx.search(rel):
            return r
    return DEFAULT_UNMATCHED


def git_tracked_set(repo: Path, roots: list[Path]) -> set[str]:
    """Set of repo-relative POSIX paths that git tracks under the given roots."""
    args = ["git", "-C", str(repo), "ls-files", "-z", "--"]
    args += [str(r.relative_to(repo)) for r in roots]
    try:
        out = subprocess.run(args, capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {p for p in out.decode("utf-8", "replace").split("\0") if p}


def recursive_size(root: Path) -> dict[Path, int]:
    """du-style recursive byte sizes for every directory under root."""
    sizes: dict[Path, int] = {}
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        d = Path(dirpath)
        total = 0
        for fn in filenames:
            try:
                total += (d / fn).stat().st_size
            except OSError:
                pass
        for sub in _dirnames:
            total += sizes.get(d / sub, 0)
        sizes[d] = total
    return sizes


# ---------------------------------------------------------------------------
# Reporting: pick a non-overlapping "cut" of category-homogeneous items
# ---------------------------------------------------------------------------
def audit(repo: Path, roots: list[Path]):
    """Returns (total_tree_bytes, cat_totals, cat_counts, items, per_dir).

    Classification is PER-FILE (path rule + label content-guard override), so
    `cat_totals` is an exact partition of the whole tree. A directory carries the
    SET of file-categories in its subtree; the display cut only collapses a dir
    into one item when that set is a singleton. This is the load-bearing safety
    property: a render dir that also holds a `never` file (pool.jsonl provenance,
    a ledger, a populated label) is NEVER collapsed into a safe deletable item —
    the cut recurses past it and the never-files never enter the manifest.
    """
    tracked = git_tracked_set(repo, roots)
    protected = collect_protected(repo, roots, _reuse_source_text(repo))
    total_tree = 0
    cat_totals: dict[str, int] = {c: 0 for c in CATEGORY_ORDER}
    cat_counts_files: dict[str, int] = {c: 0 for c in CATEGORY_ORDER}
    per_dir: dict[str, Node] = {}
    child_index: dict[str, list[str]] = {}
    sub_cats: dict[str, set[str]] = {}    # drel -> set of file categories in subtree
    own_loose: dict[str, dict[str, int]] = {}   # drel -> {cat: bytes of this dir's own files}

    for root in roots:
        if not root.exists():
            continue
        sizes = recursive_size(root)
        own_cats: dict[Path, set[str]] = {}
        own_label: dict[Path, bool] = {}
        own_tracked: dict[Path, bool] = {}

        for dirpath, _dn, filenames in os.walk(root):
            d = Path(dirpath)
            drel = d.relative_to(repo).as_posix() + "/"
            prot = protected_reason(drel, protected)
            cats: set[str] = set()
            loose: dict[str, int] = {}
            lab = trk = False
            for fn in filenames:
                fp = d / fn
                frel = fp.relative_to(repo).as_posix()
                try:
                    sz = fp.stat().st_size
                except OSError:
                    sz = 0
                cat = classify(frel).category
                # content guard: a populated human label forces `never` regardless
                # of path (paths can surprise; label content is ground truth).
                if fn in LABEL_FILES and _file_has_populated_label(fp):
                    cat, lab = NEVER, True
                # cache-reuse guard: a sentinel'd / build_plan-referenced frozen cache
                # (and its whole subtree) is load-bearing -> force `never`.
                if prot:
                    cat = NEVER
                cat_totals[cat] += sz
                cat_counts_files[cat] += 1
                total_tree += sz
                cats.add(cat)
                loose[cat] = loose.get(cat, 0) + sz
                if frel in tracked:
                    trk = True
            own_cats[d] = cats
            own_loose[drel] = loose
            own_label[d] = lab
            own_tracked[d] = trk

        # propagate cats/label/tracked up (deepest first)
        children: dict[Path, list[Path]] = {}
        for p in sizes:
            if p != root:
                children.setdefault(p.parent, []).append(p)
        agg_cats: dict[Path, set[str]] = {}
        has_label: dict[Path, bool] = {}
        has_tracked: dict[Path, bool] = {}
        for d in sorted(sizes, key=lambda p: len(p.parts), reverse=True):
            c = set(own_cats.get(d, set()))
            hl, ht = own_label.get(d, False), own_tracked.get(d, False)
            for sub in children.get(d, []):
                c |= agg_cats.get(sub, set())
                hl = hl or has_label.get(sub, False)
                ht = ht or has_tracked.get(sub, False)
            agg_cats[d], has_label[d], has_tracked[d] = c, hl, ht

        for d, sz in sizes.items():
            drel = d.relative_to(repo).as_posix() + "/"
            cats = agg_cats.get(d, set()) or {classify(drel).category}
            # per-dir headline category: the singleton if homogeneous, else the
            # most-authoritative category present (never > ambiguous > regen > scratch)
            headline = next(c for c in CATEGORY_ORDER if c in cats)
            r = classify(drel)
            reason = r.reason if r.category == headline else f"mixed subtree: {sorted(cats)}"
            rebuild = r.rebuild if r.category == headline else ""
            if has_label.get(d):
                reason = "CONTENT GUARD: populated human label in subtree"
            # cache-reuse guard: force the protected dir (and everything under it) to a
            # single `never` so the display cut collapses it and never recurses into a
            # child as a deletable item. Ancestors already carry `never` via propagation
            # (protected files were forced above), so they won't collapse over it either.
            prot = protected_reason(drel, protected)
            if prot:
                headline, reason, rebuild = NEVER, prot, ""
                cats = {NEVER}
            sub_cats[drel] = cats
            per_dir[drel] = Node(d, drel, True, size=sz, category=headline, reason=reason,
                                 rebuild=rebuild, has_label=has_label.get(d, False),
                                 tracked=has_tracked.get(d, False))
        for d in sizes:
            if d != root:
                child_index.setdefault(d.parent.relative_to(repo).as_posix() + "/", []).append(
                    d.relative_to(repo).as_posix() + "/")

    # display cut: descend until a dir's whole subtree is a single category.
    # For a mixed dir we recurse into child dirs AND surface this dir's own loose
    # files as synthetic per-category items, so the item set is a COMPLETE
    # partition of the tree (nothing — least of all ambiguous bytes — goes unseen).
    items: list[Node] = []

    def emit(node_rel: str):
        if len(sub_cats.get(node_rel, {AMBIG})) == 1:
            items.append(per_dir[node_rel])
            return
        node = per_dir[node_rel]
        for cat, nbytes in own_loose.get(node_rel, {}).items():
            if nbytes <= 0:
                continue
            r = classify(node_rel)
            reason = r.reason if r.category == cat else f"loose {cat} files in a mixed dir"
            items.append(Node(node.path, node_rel + f"(loose {cat} files)", True,
                              size=nbytes, category=cat, reason=reason,
                              rebuild=r.rebuild if r.category == cat else "",
                              tracked=node.tracked, synthetic=True))
        for k in sorted(child_index.get(node_rel, [])):
            emit(k)

    for root in roots:
        rrel = root.relative_to(repo).as_posix() + "/"
        if rrel in per_dir:
            emit(rrel)

    return total_tree, cat_totals, cat_counts_files, items, per_dir


# ---------------------------------------------------------------------------
# Deletion (gated)
# ---------------------------------------------------------------------------
def subtree_has_label(root: Path) -> bool:
    for dirpath, _dn, filenames in os.walk(root):
        d = Path(dirpath)
        for fn in filenames:
            if fn in LABEL_FILES and _file_has_populated_label(d / fn):
                return True
    return False


def subtree_has_sentinel(root: Path) -> bool:
    """True if any dir in the subtree carries an `.audit-keep` cache-reuse sentinel."""
    for _dirpath, _dn, filenames in os.walk(root):
        if SENTINEL_FILE in filenames:
            return True
    return False


def delete_items(items: list[Node], categories: set[str]):
    import shutil
    print(f"\n=== APPLY: deleting categories {sorted(categories)} ===")
    for n in items:
        if n.category not in categories:
            continue
        if n.synthetic:
            # report-only aggregate; its path is a mixed dir holding non-safe files.
            print(f"  SKIP (loose-file aggregate, delete individually): {n.rel}")
            continue
        if n.category not in SAFE_CATEGORIES:
            print(f"  REFUSE (unsafe category {n.category}): {n.rel}")
            continue
        # content guard re-runs at delete time
        if subtree_has_label(n.path):
            print(f"  ABORT (content guard tripped): {n.rel}")
            continue
        # cache-reuse guard re-runs at delete time (catch a sentinel'd frozen cache
        # that somehow reached a safe item — should never happen, but delete is final)
        if subtree_has_sentinel(n.path):
            print(f"  ABORT (cache-reuse sentinel in subtree): {n.rel}")
            continue
        try:
            if n.path.is_dir():
                shutil.rmtree(n.path)
            else:
                n.path.unlink()
            print(f"  removed {human(n.size):>9}  {n.rel}")
        except OSError as e:
            print(f"  ERROR removing {n.rel}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="root(s) to audit (repo-relative). Default: data out")
    ap.add_argument("--min-mb", type=float, default=100.0,
                    help="report threshold in MB (default 100)")
    ap.add_argument("--top", type=int, default=40, help="top-N largest items to list")
    ap.add_argument("--manifest", default=None,
                    help="write delete-manifest JSON here (default out/audit/delete_manifest.json)")
    ap.add_argument("--apply", action="store_true",
                    help="ACTUALLY delete (requires --categories; safe categories only)")
    ap.add_argument("--categories", default="",
                    help="comma list of categories to delete: regenerable,scratch")
    args = ap.parse_args()

    roots = [repo / r for r in (args.root or ["data", "out"])]
    min_bytes = int(args.min_mb * 1024 * 1024)

    total_tree, cat_totals, cat_counts, items, per_dir = audit(repo, roots)

    print("=" * 78)
    print(f"DISK AUDIT  roots={[str(r.relative_to(repo)) for r in roots]}  "
          f"total tree = {human(total_tree)}")
    print(f"(dry-run: this deletes nothing; report threshold {args.min_mb:.0f} MB)")
    print("=" * 78)
    print("\nReclaimable by category (whole tree):")
    label = {NEVER: "NEVER-DELETE (authoritative)", AMBIG: "AMBIGUOUS (your call)",
             REGEN: "REGENERABLE (safe; rebuild cost)", SCRATCH: "SCRATCH/LOGS (safe, cheap)"}
    for c in CATEGORY_ORDER:
        safe = "  <- safe to delete" if c in SAFE_CATEGORIES else ""
        print(f"  {label[c]:<34} {human(cat_totals[c]):>9}  ({cat_counts[c]} files){safe}")
    safe_total = cat_totals[REGEN] + cat_totals[SCRATCH]
    print(f"  {'-'*34} {'-'*9}")
    print(f"  {'RECLAIMABLE NOW (regen+scratch)':<34} {human(safe_total):>9}")
    print(f"  {'POTENTIAL (+ ambiguous, needs OK)':<34} {human(safe_total + cat_totals[AMBIG]):>9}")
    print("  (per-file partition; the delete-manifest below only lists dirs whose")
    print("   ENTIRE subtree is safe — mixed dirs holding provenance are excluded.)")

    # top-N largest items
    big = sorted([n for n in items if n.size >= min_bytes], key=lambda n: n.size, reverse=True)
    print(f"\nTop {min(args.top, len(big))} largest items (>= {args.min_mb:.0f} MB):")
    print(f"  {'size':>9}  {'category':<12} {'git':<5} path")
    for n in big[:args.top]:
        git = "trkd" if n.tracked else "IGN"
        note = f"  [rebuild: {n.rebuild}]" if n.rebuild else ""
        print(f"  {human(n.size):>9}  {n.category:<12} {git:<5} {n.rel}")
        print(f"             -> {n.reason}{note}")

    # ambiguous list (explicit, for the human decision)
    ambig = sorted([n for n in items if n.category == AMBIG and n.size >= min_bytes],
                   key=lambda n: n.size, reverse=True)
    print(f"\nAMBIGUOUS — needs your decision ({len(ambig)} items >= {args.min_mb:.0f} MB, "
          f"{human(cat_totals[AMBIG])} total):")
    for n in ambig:
        git = "git-tracked (recoverable)" if n.tracked else "GITIGNORED (permanent delete)"
        print(f"  {human(n.size):>9}  {n.rel}  [{git}]")
        print(f"             -> {n.reason}")

    # delete-manifest for the safe categories
    manifest_path = repo / (args.manifest or "out/audit/delete_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # only whole-clean-subtree dirs are truly rmtree-able; synthetic loose-file
    # aggregates are report-only (their path is a mixed dir with non-safe files).
    safe_items = [n for n in items if n.category in SAFE_CATEGORIES and not n.synthetic]
    loose_safe = sum(n.size for n in items if n.category in SAFE_CATEGORIES and n.synthetic)
    manifest_total = sum(n.size for n in safe_items)
    manifest = {
        "generated_by": "tools/audit/disk_audit.py",
        "roots": [str(r.relative_to(repo)) for r in roots],
        "note": "dry-run manifest; paths that WOULD be removed for safe categories "
                "(regenerable, scratch). Ambiguous/never are excluded by design.",
        "category_totals_bytes": cat_totals,
        "safe_reclaimable_bytes_perfile_max": safe_total,
        "safe_reclaimable_bytes_clean_subtrees": manifest_total,
        "items": [
            {"path": n.rel, "category": n.category, "bytes": n.size,
             "human": human(n.size), "git_tracked": n.tracked,
             "reason": n.reason, "rebuild": n.rebuild}
            for n in sorted(safe_items, key=lambda n: n.size, reverse=True)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDelete-manifest (safe categories, clean subtrees, dry-run) -> {manifest_path.relative_to(repo)}")
    print(f"  {len(safe_items)} whole-clean-dir items, {human(manifest_total)} rmtree-able")
    if loose_safe:
        print(f"  + {human(loose_safe)} safe bytes as loose files inside mixed dirs "
              f"(report-only; excluded from auto-delete)")

    # deletion gate
    cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    if args.apply:
        if not cats:
            print("\n--apply given with no --categories -> NO-OP (categories must be named).")
            return
        bad = cats - SAFE_CATEGORIES
        if bad:
            print(f"\nREFUSED: {sorted(bad)} not in safe set {sorted(SAFE_CATEGORIES)}. "
                  "Never deletes authoritative or ambiguous.")
            return
        delete_items(items, cats)
    else:
        print("\nDELETION OFF (dry-run). To delete safe items, re-run with e.g. "
              "--apply --categories scratch,regenerable")


if __name__ == "__main__":
    main()
