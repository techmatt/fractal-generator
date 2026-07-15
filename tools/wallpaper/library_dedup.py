#!/usr/bin/env python
r"""Identity-aware coordinate dedup for the Phase-1 library store.

Phase 1's thesis is "mostly stop discarding": EVERY fresh q3 becomes a library record, with the
ONE legitimate exception being a true coordinate-duplicate of a record already in the store. This
module owns that single exception — the coordinate test — so the orchestrator's reconciliation
assert has an authoritative, unit-tested definition of "is this the same spot as something we
already kept".

The identity rules (must NOT drift from build_fresh_discovery._spatially_in / the corpus contract):

  * A location matches a stored one iff SAME render-family, SAME parameter `c` (exact string
    match on c_re/c_im), and viewport within `DEDUP_FRAC * min(fw, kfw)`.
  * Julia (julia / julia_multibrot{d}) identity is viewport AND c — ignoring c would falsely
    collide different fractals that happen to share a z-plane viewport, so a near-viewport match
    with a DIFFERENT c is NOT a dup.
  * Phoenix carries the fixed Ushiki c/p, so its c always matches and the test reduces to a
    scale-aware z-plane viewport proximity. `min(fw, kfw)` is load-bearing here: phoenix fw spans
    ~3 decades, and a flat `1.5 * max(fw)` radius over-merges distinct deep spots under a shallow
    neighbour (the exact failure the corpus contract warns against). min() keeps the tolerance
    tied to the TIGHTER of the two frames.
  * c-plane mandelbrot/multibrot carry c = None on both sides, so None == None matches and the
    test is pure viewport proximity.

Pure I/O + float math — no torch, no render — so it unit-tests without a model.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# Two same-family coords within DEDUP_FRAC * min(fw) at a MATCHING c are "the same spot".
# Shared with build_fresh_discovery.DEDUP_FRAC (kept in lockstep by intent — the head-corpus
# proximity guard and the store dedup use the same notion of "same place").
DEDUP_FRAC = 0.5


def coord_of_identity(identity: dict):
    """A store record's `identity` block -> (family, cx, cy, fw, c_re, c_im).

    c_re/c_im are the exact strings the record was written with (None for a c-plane family).
    Phoenix's fixed Ushiki c IS stamped into identity["c"], so it flows through as a real c."""
    c = identity.get("c")
    c_re = c["re"] if c else None
    c_im = c["im"] if c else None
    return (identity["family"], str(identity["cx"]), str(identity["cy"]),
            str(identity["fw"]), c_re, c_im)


def coord_of_location(loc):
    """A canonical Location -> the same (family, cx, cy, fw, c_re, c_im) coord tuple used to
    key the dedup index. `loc.family` is the RENDER family (julia / julia_multibrot{d} / phoenix /
    mandelbrot / multibrot{d}), matching the store identity's `family`."""
    c_re = None if loc.c_re is None else str(loc.c_re)
    c_im = None if loc.c_im is None else str(loc.c_im)
    return (loc.family, str(loc.cx), str(loc.cy), str(loc.fw), c_re, c_im)


class StoreIndex:
    """Family-bucketed coordinate index for O(matches-per-family) dup tests. Seed it from the
    store's records, then `add()` each location accepted THIS cycle so within-batch coord
    collisions (two fresh q3 at the same spot in one cycle) collapse to a single record too."""

    def __init__(self):
        self._by_family: dict[str, list] = defaultdict(list)

    @classmethod
    def from_records(cls, records_path: Path) -> "StoreIndex":
        idx = cls()
        records_path = Path(records_path)
        if not records_path.exists():
            return idx
        with open(records_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                idn = rec.get("identity")
                if not idn:
                    continue
                idx.add_coord(*coord_of_identity(idn))
        return idx

    def add_coord(self, family, cx, cy, fw, c_re, c_im):
        try:
            self._by_family[family].append((float(cx), float(cy), float(fw), c_re, c_im))
        except (TypeError, ValueError):
            pass  # a coord we can't parse can't be dedup-matched; drop it from the index

    def add_location(self, loc):
        self.add_coord(*coord_of_location(loc))

    def is_dup(self, family, cx, cy, fw, c_re, c_im, frac: float = DEDUP_FRAC) -> bool:
        """True iff (family, cx, cy, fw, c) is a coordinate-duplicate of an indexed location:
        same family, exact-string-matching c, viewport within frac * min(fw, kfw)."""
        try:
            cxf, cyf, fwf = float(cx), float(cy), float(fw)
        except (TypeError, ValueError):
            return False
        for kcx, kcy, kfw, kc_re, kc_im in self._by_family.get(family, []):
            if kc_re != c_re or kc_im != c_im:      # julia/phoenix identity: c must match
                continue
            tol = frac * min(fwf, kfw)               # scale-aware (phoenix fw spans decades)
            if abs(cxf - kcx) < tol and abs(cyf - kcy) < tol:
                return True
        return False

    def is_location_dup(self, loc, frac: float = DEDUP_FRAC) -> bool:
        return self.is_dup(*coord_of_location(loc), frac=frac)
