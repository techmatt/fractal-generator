#!/usr/bin/env python3
"""Generate specs/REGISTRY.md from specs/modes_registry.json (single source of truth).

Also validates that the registry's keys match exactly the .json spec files present
in specs/ (no missing, no extra), that every tier is valid, and — since specs are
tier-stamped — that each spec's stamped "tier" matches the registry. Raises on any
mismatch, so running this to regenerate the view also re-checks consistency. The
durable enforcement lives in the cargo test (tests/modes_registry.rs); this script
is the human-scannable-view generator.

Usage: uv run python tools/specs/gen_registry.py
"""
import json
import sys
from pathlib import Path

SPECS_DIR = Path(__file__).resolve().parents[2] / "specs"
REGISTRY_JSON = SPECS_DIR / "modes_registry.json"
REGISTRY_MD = SPECS_DIR / "REGISTRY.md"
VALID_TIERS = {"promoted", "niche"}


def load_registry():
    entries = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    by_spec = {}
    for e in entries:
        spec = e["spec"]
        if spec in by_spec:
            sys.exit(f"registry: duplicate entry for {spec!r}")
        by_spec[spec] = e
    return entries, by_spec


def validate(entries, by_spec):
    errs = []
    spec_files = {p.stem for p in SPECS_DIR.glob("*.json") if p.name != REGISTRY_JSON.name}
    reg_keys = set(by_spec)
    for missing in sorted(spec_files - reg_keys):
        errs.append(f"spec {missing!r} present in specs/ but absent from registry")
    for extra in sorted(reg_keys - spec_files):
        errs.append(f"registry key {extra!r} has no specs/{extra}.json")
    for e in entries:
        spec, tier = e["spec"], e.get("tier")
        if tier not in VALID_TIERS:
            errs.append(f"{spec!r}: invalid tier {tier!r}")
        if tier == "niche" and not e.get("reason", "").strip():
            errs.append(f"{spec!r}: niche entry requires a non-empty reason")
        if not e.get("identity", "").strip():
            errs.append(f"{spec!r}: missing identity")
        # stamped-tier parity
        path = SPECS_DIR / f"{spec}.json"
        if path.exists():
            stamped = json.loads(path.read_text(encoding="utf-8")).get("tier")
            if stamped != tier:
                errs.append(f"{spec!r}: stamped tier {stamped!r} != registry tier {tier!r}")
    if errs:
        sys.exit("registry validation FAILED:\n  " + "\n  ".join(errs))


def render_md(entries, by_spec):
    promoted = [e for e in entries if e["tier"] == "promoted"]
    niche = [e for e in entries if e["tier"] == "niche"]
    deletions = [e for e in entries if e.get("deletion_candidate")]

    def rows(items, with_reason):
        out = []
        for e in items:
            r = e.get("reason", "").strip()
            if with_reason and r:
                out.append(f"- **`{e['spec']}`** — {e['identity']} _{r}_")
            else:
                out.append(f"- **`{e['spec']}`** — {e['identity']}")
        return "\n".join(out)

    lines = [
        "# Render-mode registry",
        "",
        "**Generated from `specs/modes_registry.json` — do not hand-edit.**",
        "Regenerate: `uv run python tools/specs/gen_registry.py`. "
        "Consistency (keys match specs/, valid tiers, stamped-tier parity) is enforced "
        "by `cargo test --test modes_registry`.",
        "",
        f"Counts: **{len(promoted)} promoted**, **{len(niche)} niche** "
        f"({len(entries)} total). Deletion candidates: **{len(deletions)}**.",
        "",
        "## Promoted — standard / reference render modes",
        "",
        rows(promoted, with_reason=False),
        "",
        "## Niche — location-specialists, composite textures, exploration scaffolding",
        "",
        rows(niche, with_reason=True),
        "",
        "## Deletion candidates (flagged, not deleted)",
        "",
        (rows(deletions, with_reason=True) if deletions else "_(none)_"),
        "",
    ]
    REGISTRY_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    entries, by_spec = load_registry()
    validate(entries, by_spec)
    render_md(entries, by_spec)
    print(f"OK: {len(entries)} specs validated; wrote {REGISTRY_MD.relative_to(SPECS_DIR.parent)}")


if __name__ == "__main__":
    main()
