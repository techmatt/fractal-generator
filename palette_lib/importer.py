"""Import all sources into one stop-list representation; dedup + degenerate drop.

Common form per palette: dict(name, source, stops) where
    stops = list[(pos in [0,1), (r,g,b) 0-255)].

Sources:
  - colormaps: matplotlib built-ins, colorcet, cmasher -> sampled to stops
                (clean / committable backbone).
  - harvest:   parsed .ugr (multi-block) + .map (third-party, gitignored).

Dedup: bake a small OKLab signature ring and hash it; identical-appearance
palettes (e.g. colorcet's CET_L2 == its long alias) collapse to one.

Degenerate drop (palette-space, both tails):
  - sparse:  <=2 distinct stop colors, or all colors within a tiny OKLab radius
             (near one color).
  - busy:    OKLab total-variation around the baked ring far above the smooth
             population (essentially random noise).
"""

from __future__ import annotations

import numpy as np

from . import coloring

# Signature ring size — small, just for dedup + degeneracy stats.
_SIG_N = 96
# OKLab radius under which a palette counts as "near one color".
_SPARSE_RADIUS = 0.06
# OKLab ring total-variation above which a palette is flagged as noise/busy.
# Calibrated from the observed population (smooth maps sit well below this;
# see REPORT.md for the measured distribution).
_BUSY_TV = 18.0


def _sample_mpl(name, n_stops=33):
    import matplotlib

    cmap = matplotlib.colormaps[name]
    xs = np.linspace(0.0, 1.0, n_stops)
    rgb8 = (np.asarray(cmap(xs))[:, :3] * 255.0 + 0.5).astype(int)
    return [(i / n_stops, tuple(int(v) for v in rgb8[i])) for i in range(n_stops)]


def from_colormaps(n_stops=33):
    """matplotlib + colorcet + cmasher -> palette dicts (forward maps only)."""
    import matplotlib.pyplot as plt
    import colorcet  # registers cet_* into mpl
    import cmasher    # registers cmr.* into mpl

    allnames = plt.colormaps()
    cc_keys = set(colorcet.palette.keys())
    out = []
    for name in allnames:
        if name.endswith("_r"):
            continue
        if name.startswith("cmr."):
            source = "cmasher"
        elif name.startswith("cet_") or name in cc_keys:
            source = "colorcet"
        else:
            source = "matplotlib"
        try:
            stops = _sample_mpl(name, n_stops)
        except Exception:
            continue
        out.append({"name": name, "source": source, "stops": stops})
    return out


def from_harvest(files):
    """Parse harvested .ugr (multi-block) and .map files -> palette dicts."""
    out = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        ext = path.suffix.lower()
        if ext == ".ugr":
            for blockname, stops in coloring.parse_ugr(text):
                if stops:
                    out.append({"name": f"{path.stem}:{blockname}", "source": "ugr", "stops": stops})
        elif ext == ".map":
            stops = coloring.parse_map(text)
            if stops:
                out.append({"name": path.stem, "source": "map", "stops": stops})
    return out


# ---------------------------------------------------------------------------
# Signatures + degeneracy
# ---------------------------------------------------------------------------


def _ring_oklab(pal):
    """Bake a short OKLab ring for dedup/degeneracy. Returns (_SIG_N,3) or None."""
    try:
        lut = coloring.bake_lut(pal["stops"], lut_size=_SIG_N)
    except ValueError:
        return None
    return coloring.linear_srgb_to_oklab(lut)


def _signature(ring):
    q = np.round(ring / 0.04).astype(np.int16)
    return q.tobytes()


def _ring_tv(ring):
    """Cyclic OKLab total variation around the ring (sum of adjacent steps)."""
    d = np.diff(ring, axis=0, append=ring[:1])
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def _distinct_colors(stops):
    return {tuple(c) for _, c in stops}


def classify(pal):
    """Return (ring, status) where status in {'ok','sparse','busy','bad'}."""
    ring = _ring_oklab(pal)
    if ring is None:
        return None, "bad"
    distinct = _distinct_colors(pal["stops"])
    if len(distinct) <= 2:
        return ring, "sparse"
    # near-one-color: tight OKLab bounding radius around the ring centroid.
    radius = float(np.linalg.norm(ring - ring.mean(axis=0), axis=1).max())
    if radius < _SPARSE_RADIUS:
        return ring, "sparse"
    if _ring_tv(ring) > _BUSY_TV:
        return ring, "busy"
    return ring, "ok"


def build_library(harvest_files, n_stops=33, verbose=True):
    """Unify + classify + dedup. Returns (survivors, report_dict)."""
    raw = from_colormaps(n_stops) + from_harvest(harvest_files)

    per_source = {}
    for p in raw:
        per_source.setdefault(p["source"], {"total": 0})
        per_source[p["source"]]["total"] += 1

    survivors = []
    seen = set()
    counts = {"sparse": 0, "busy": 0, "bad": 0, "dup": 0, "ok": 0}
    drop_by_source = {}
    for p in raw:
        ring, status = classify(p)
        ds = drop_by_source.setdefault(p["source"], {"sparse": 0, "busy": 0, "bad": 0, "dup": 0, "ok": 0})
        if status != "ok":
            counts[status] += 1
            ds[status] += 1
            continue
        sig = _signature(ring)
        if sig in seen:
            counts["dup"] += 1
            ds["dup"] += 1
            continue
        seen.add(sig)
        counts["ok"] += 1
        ds["ok"] += 1
        survivors.append(p)

    report = {
        "raw_total": len(raw),
        "per_source_total": {k: v["total"] for k, v in per_source.items()},
        "drop_by_source": drop_by_source,
        "counts": counts,
        "survivors": len(survivors),
    }
    if verbose:
        _print_report(report)
    return survivors, report


def _print_report(r):
    print(f"\n[importer] raw palettes: {r['raw_total']}")
    print("  per source (raw):", r["per_source_total"])
    print(f"  dropped: sparse={r['counts']['sparse']} busy={r['counts']['busy']} "
          f"bad={r['counts']['bad']} dup={r['counts']['dup']}")
    print("  per-source ok/sparse/busy/dup:")
    for src, d in sorted(r["drop_by_source"].items()):
        print(f"    {src:11s} ok={d['ok']:4d} sparse={d['sparse']:3d} "
              f"busy={d['busy']:3d} dup={d['dup']:3d} bad={d['bad']:3d}")
    print(f"  SURVIVORS: {r['survivors']}\n")
