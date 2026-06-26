"""Classify palettes into spread families and build a stratified spread roster.

No family metadata exists in the repo (only cycle/mirror_needed). We derive a
5-way family from each palette's sRGB stops + name tokens:

  cyclic     -- cycle == "cyclic" (seam-closed; the rendered band wraps)
  diverging  -- two-ended divergent map (cet_diverging_*, known mpl diverging,
                or stops whose hue flips ends through a low-chroma midpoint)
  mono       -- near-grayscale (low mean chroma): greys / single-luminance ramps
  warm       -- chroma-weighted mean hue in the red->yellow arc
  cool       -- chroma-weighted mean hue otherwise (green/cyan/blue/violet)

Priority cyclic > diverging > mono > warm/cool.

`build_spread_roster` emits a balanced subset (N per family) as a score3-shaped
JSON the frozen `enrich --colormaps` reads directly, always including
`twilight_shifted` (the neutral location-scoring palette).

  uv run python tools/mining/palette_families.py            # report + write roster
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "palettes" / "clean_colormaps.json"
# Roster is built from the curated score-3 set (no categoricals, already the
# trusted `enrich` roster); the broad clean set is classified only for reference.
SCORE3 = ROOT / "data" / "palettes" / "score3_colormaps.json"
ROSTER_OUT = ROOT / "data" / "mining" / "spread_roster.json"
NEUTRAL_OUT = ROOT / "data" / "mining" / "neutral_roster.json"  # twilight_shifted only
FAMILIES_OUT = ROOT / "data" / "mining" / "palette_families.json"
NEUTRAL = "twilight_shifted"

# known matplotlib diverging maps (name-tokened, not cet_diverging_*)
MPL_DIVERGING = {
    "coolwarm", "bwr", "seismic", "RdBu", "RdGy", "RdYlBu", "RdYlGn", "Spectral",
    "PiYG", "PRGn", "BrBG", "PuOr", "berlin", "managua", "vanimo",
}
MONO_CHROMA = 0.045   # mean chroma below -> grayscale/mono
WARM_HUE_LO, WARM_HUE_HI = -35.0, 95.0  # degrees: red..yellow arc = warm


def srgb_to_oklab(r, g, b):
    """sRGB8 -> OKLab (Bjorn Ottosson)."""
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = lin(r), lin(g), lin(b)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, a, bb


def palette_stats(stops):
    """Return (mean_chroma, hue_deg, end_hue_flip) over the stops in OKLab."""
    cx = cy = csum = 0.0
    hues = []
    chromas = []
    for _pos, (r, g, b) in stops:
        _L, a, bb = srgb_to_oklab(r, g, b)
        c = math.hypot(a, bb)
        chromas.append(c)
        csum += c
        cx += a
        cy += bb
        hues.append(math.degrees(math.atan2(bb, a)))
    n = max(1, len(stops))
    mean_chroma = csum / n
    hue = math.degrees(math.atan2(cy, cx))  # chroma-weighted mean hue
    # end-hue flip: the two ends point in clearly opposite hue directions and the
    # interior dips low-chroma -> diverging signature.
    _La, aa, ba = srgb_to_oklab(*stops[0][1])
    _Lb, ab, bb2 = srgb_to_oklab(*stops[-1][1])
    end_dot = aa * ab + ba * bb2
    end_mag = math.hypot(aa, ba) * math.hypot(ab, bb2) + 1e-9
    end_cos = end_dot / end_mag
    mid_chroma = min(chromas[len(chromas) // 4: 3 * len(chromas) // 4] or chromas)
    end_flip = (end_cos < -0.2) and (min(math.hypot(aa, ba), math.hypot(ab, bb2)) > 0.04) \
        and (mid_chroma < 0.5 * max(math.hypot(aa, ba), math.hypot(ab, bb2)))
    return mean_chroma, hue, end_flip


def classify(p) -> str:
    name = p["name"]
    if p.get("cycle") == "cyclic":
        return "cyclic"
    if name in MPL_DIVERGING or "diverging" in name:
        return "diverging"
    mean_chroma, hue, end_flip = palette_stats(p["stops"])
    if end_flip:
        return "diverging"
    if mean_chroma < MONO_CHROMA:
        return "mono"
    if WARM_HUE_LO <= hue <= WARM_HUE_HI:
        return "warm"
    return "cool"


def load_clean():
    return json.loads(CLEAN.read_text(encoding="utf-8"))


def classify_all():
    lib = load_clean()
    fam = {}
    for p in lib:
        fam[p["name"]] = classify(p)
    return lib, fam


def build_spread_roster(per_family: int = 6, seed: int = 0):
    """Balanced subset, deterministic, drawn from the curated score-3 set (no
    categoricals). Within each family pick spread-out members (prefer distinct
    sources, then by name) so the roster isn't all-cmr."""
    lib = json.loads(SCORE3.read_text(encoding="utf-8"))
    fam = {p["name"]: classify(p) for p in lib}
    by_name = {p["name"]: p for p in lib}
    groups: dict[str, list[str]] = {}
    for name, f in fam.items():
        groups.setdefault(f, []).append(name)

    roster_names: list[str] = []
    for f in ("warm", "cool", "cyclic", "diverging", "mono"):
        members = sorted(groups.get(f, []))
        # round-robin across sources for diversity
        bysrc: dict[str, list[str]] = {}
        for nm in members:
            bysrc.setdefault(by_name[nm]["source"], []).append(nm)
        order = []
        srcs = sorted(bysrc)
        i = 0
        while len(order) < len(members):
            s = srcs[i % len(srcs)]
            if bysrc[s]:
                order.append(bysrc[s].pop(0))
            i += 1
        roster_names.extend(order[:per_family])

    # always include the neutral scoring palette
    if NEUTRAL not in roster_names and NEUTRAL in by_name:
        roster_names.append(NEUTRAL)

    roster = [by_name[n] for n in roster_names]
    fam_of_roster = {n: fam[n] for n in roster_names}
    return roster, fam, fam_of_roster


def main():
    lib, fam = classify_all()
    from collections import Counter
    counts = Counter(fam.values())
    print("family distribution over clean_colormaps (224):")
    for f in ("warm", "cool", "cyclic", "diverging", "mono"):
        names = sorted(n for n, x in fam.items() if x == f)
        print(f"  {f:10s} {counts[f]:3d}   e.g. {names[:6]}")
    print(f"  neutral '{NEUTRAL}' -> family {fam.get(NEUTRAL)}")

    roster, fam_all, fam_roster = build_spread_roster()
    FAMILIES_OUT.parent.mkdir(parents=True, exist_ok=True)
    FAMILIES_OUT.write_text(json.dumps(fam_all, indent=2), encoding="utf-8")
    ROSTER_OUT.write_text(json.dumps(roster), encoding="utf-8")

    # neutral roster: twilight_shifted only, for fixed-palette location scoring
    score3 = json.loads(SCORE3.read_text(encoding="utf-8"))
    neutral = [p for p in score3 if p["name"] == NEUTRAL]
    NEUTRAL_OUT.write_text(json.dumps(neutral), encoding="utf-8")

    print(f"\nspread roster: {len(roster)} palettes "
          f"({Counter(fam_roster.values())})")
    print("  " + ", ".join(p["name"] for p in roster))
    print(f"wrote {FAMILIES_OUT}\nwrote {ROSTER_OUT}\nwrote {NEUTRAL_OUT} (neutral, {len(neutral)})")


if __name__ == "__main__":
    main()
