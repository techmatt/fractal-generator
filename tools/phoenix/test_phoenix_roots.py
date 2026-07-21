"""End-to-end smoke for phoenix root plumbing (Phase A item 4): propose a small seeded
batch, descend a few seeds briefly, and confirm every ledger row carries the parameter
point (c, p, z_{-1}) and that two seeds with DISTINCT parameters do not dup-collide even
at an identical viewport. Skipped when the release binary is absent.

  uv run pytest tools/phoenix/test_phoenix_roots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))
sys.path.insert(0, str(HERE))

import production_seeder as ps  # noqa: E402
import phoenix_sampler as psamp  # noqa: E402
import phoenix_roots as pr  # noqa: E402

_IDENT_KEYS = ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re", "phoenix_p_im",
               "phoenix_zm1_re", "phoenix_zm1_im")


def _pick_distinct(seed, n_want=3, pool=24):
    """A few proposals with pairwise-distinct parameter points."""
    out, seen = [], set()
    for s in psamp.propose_batch(seed, pool):
        key = (round(s.c.real, 9), round(s.c.imag, 9), round(s.p.real, 9),
               round(s.p.imag, 9), round(s.z_m1.real, 9), round(s.z_m1.imag, 9))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) == n_want:
            break
    return out


def test_phoenix_roots_smoke(tmp_path):
    binary = pr.default_binary()
    if not binary.exists():
        import pytest
        pytest.skip("release binary not built")

    seeds = _pick_distinct(seed=0, n_want=3)
    assert len(seeds) == 3
    # sanity: the proposed parameter points really are pairwise distinct.
    keys = [ps.row_phoenix_key(ps.phoenix_ident_fields(
        (s.c.real, s.c.imag), (s.p.real, s.p.imag), (s.z_m1.real, s.z_m1.imag)))
        for s in seeds]
    assert len(set(keys)) == 3

    rows = pr.run_batch(seeds, binary=binary, scratch=tmp_path / "scratch",
                        ledger_path=tmp_path / "ledger.jsonl", run_ts="smoke",
                        n_walks=2, depth_min=1, depth_max=3)
    assert rows, "descent produced no outcome rows"

    # (1) every ledger row carries the full parameter-point identity.
    for r in rows:
        assert r["family"] == "phoenix"
        for k in _IDENT_KEYS:
            assert k in r, f"row {r['id']} missing {k}"
    # the ledger file round-trips the same rows.
    import json
    disk = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines() if x.strip()]
    assert len(disk) == len(rows)

    # (2) rows from DIFFERENT seeds carry different identities.
    by_seed = {}
    for r in rows:
        by_seed.setdefault(ps.row_phoenix_key(r), []).append(r)
    assert len(by_seed) >= 2, "expected outcomes from at least two distinct seeds"

    # (3) two distinct-parameter outcomes do NOT dup-collide even at the IDENTICAL viewport
    #     (the whole point of keying phoenix identity on (c, p, z_{-1})).
    idents = list(by_seed)
    ra = dict(by_seed[idents[0]][0], outcome_cx=0.0, outcome_cy=0.0, outcome_fw=3.0,
              decoded_class=3, guard_pass=True)
    rb = dict(by_seed[idents[1]][0], outcome_cx=0.0, outcome_cy=0.0, outcome_fw=3.0,
              decoded_class=3, guard_pass=True)
    assert ps.near_dup(0.0, 0.0, 3.0, 0.0, 0.0, 3.0, a_c=ps.row_ident(ra),
                       b_c=ps.row_ident(rb)) is False
    cloud = ps.build_cloud([ra, rb], "phoenix")
    assert len(cloud) == 2, "distinct-param phoenix seeds must be distinct cloud places"

    # (4) a same-parameter revisit at a near viewport DOES collapse (identity dedup still works).
    rb_same = dict(ra, id="revisit", outcome_cx=0.01)
    cloud2 = ps.build_cloud([ra, rb_same], "phoenix")
    assert len(cloud2) == 1
