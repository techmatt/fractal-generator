"""Build the v4 augmentation palette roster: exactly 6 palettes —
neutral `twilight_shifted` + one each warm / cool / cyclic / diverging / mono,
drawn from the `palette_families` classification of the render library.

The `mono` family is absent from the curated score-3 set, so all picks are taken
from the full `clean_colormaps.json` (the library `render-one` / `v4-render-batch`
load by name). Each pick is a continuous, render-safe map (no quarantined
categoricals). Picks are hard-pinned here (not re-derived per run) so the cache is
reproducible across model versions.

  uv run python tools/v4/build_roster.py     # writes data/v4/aug_roster.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "palettes" / "clean_colormaps.json"
OUT = ROOT / "data" / "v4" / "aug_roster.json"

sys.path.insert(0, str(ROOT / "tools" / "mining"))
from palette_families import classify  # noqa: E402

# role -> palette name. `neutral` is twilight_shifted (the location-scoring map);
# the five families are one representative continuous member each.
ROSTER = [
    ("neutral", "twilight_shifted"),
    ("warm", "cmr.amber"),
    ("cool", "cmr.jungle"),
    ("cyclic", "cet_cyclic_mybm_20_100_c48_s25"),
    ("diverging", "coolwarm"),
    ("mono", "cet_linear_grey_10_95_c0"),
]


def main() -> None:
    lib = json.loads(CLEAN.read_text(encoding="utf-8"))
    by_name = {p["name"]: p for p in lib}
    fam = {p["name"]: classify(p) for p in lib}

    out = []
    for role, name in ROSTER:
        if name not in by_name:
            raise SystemExit(f"palette '{name}' (role {role}) not in {CLEAN}")
        out.append({
            "name": name,
            "role": role,
            # `palette_family` is the classifier's verdict on the actual stops;
            # for the neutral map that is 'cyclic', which is fine — its *role* is
            # neutral. The family tag is what the manifest carries.
            "palette_family": fam[name],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}  ({len(out)} palettes)")
    for r in out:
        print(f"  {r['role']:10s} {r['name']:38s} family={r['palette_family']}")


if __name__ == "__main__":
    main()
