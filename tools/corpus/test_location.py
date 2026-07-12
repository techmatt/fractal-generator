"""Tests for the canonical location + family-params slot (tools/corpus/location.py).

The linchpin is KEY-STABILITY: adding the general family-params slot must not move a
single existing mandelbrot/julia key, or it would break the corpus splits/manifest and
orphan every cached field. These tests pin that byte-identity, the cache-key
non-collision for the new families, the five-family render-one arg builder, the
render-block/sidecar parse round-trip (Phoenix's `p` survives), and that a synthetic
new-family location keys/loads without tripping the v5 manifest guard.

Run either way:
  uv run pytest tools/corpus/test_location.py
  uv run python tools/corpus/test_location.py     # prints PASS/FAIL summary (+ optional
                                                    binary-backed Phoenix round-trip)
"""
import hashlib
import itertools
import os
import subprocess
import sys
from pathlib import Path

_TOOLS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _TOOLS)
import _bootstrap  # noqa: E402,F401  (tools/{palettes,corpus,queries} on sys.path)

import location as loc_mod  # noqa: E402
import colormap as cm  # noqa: E402
import corpus_reader as cr  # noqa: E402
import assemble_queries as aq  # noqa: E402  (the field-cache key under test)
import query_sampler as qs  # noqa: E402

ROOT = Path(_TOOLS).parent


# --------------------------------------------------------------------------- #
# The pre-slot field-cache key formula, frozen here as the byte-identity oracle.
# --------------------------------------------------------------------------- #
def _old_field_key(ref):
    parts = [ref.kind, ref.cx, ref.cy, ref.fw, str(ref.maxiter),
             ref.c_re or "", ref.c_im or "",
             str(qs.CANDIDATE_SS), str(qs.EVAL_WIDTH), str(qs.EVAL_HEIGHT)]
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return f"{ref.kind}_{h}"


def _old_loc_key(ref):
    """The pre-slot dedup key (the 6-tuple query_sampler/query_batch_gen used)."""
    return (ref.kind, ref.cx, ref.cy, ref.fw, ref.c_re, ref.c_im)


def _sample_existing_locations(limit=200):
    """A representative sample of existing (mandelbrot+julia) corpus locations as
    canonical Locations, deduped by key."""
    out = {}
    for lc in itertools.islice(cr.iter_labeled(), limit * 4):
        canon = loc_mod.from_render_block(lc.render)
        out.setdefault(canon.key(), canon)
        if len(out) >= limit:
            break
    return list(out.values())


# --------------------------------------------------------------------------- #
# 1. Arg builder — the five-family location -> flags mapping.
# --------------------------------------------------------------------------- #
def test_render_one_flags_five_families():
    L = loc_mod.Location
    assert loc_mod.render_one_flags(
        L(family="mandelbrot", cx="0", cy="0", fw="3")) == ["--family", "mandelbrot"]

    assert loc_mod.render_one_flags(
        L(family="julia", cx="0", cy="0", fw="0.75", c_re="-0.8", c_im="0.156")) == \
        ["--family", "mandelbrot", "--julia", "--c", "-0.8", "0.156"]

    for n in ("multibrot3", "multibrot4", "multibrot5"):
        assert loc_mod.render_one_flags(L(family=n, cx="0", cy="0", fw="3")) == ["--family", n]

    assert loc_mod.render_one_flags(
        L(family="phoenix", cx="0", cy="0", fw="3", c_re="0.5667", c_im="0.0",
          family_params={"p_re": "-0.5", "p_im": "0.0"})) == \
        ["--family", "phoenix", "--c", "0.5667", "0.0", "--p", "-0.5", "0.0"]

    # Phoenix with the constants omitted (acceptance "test the Rust default" case):
    assert loc_mod.render_one_flags(L(family="phoenix", cx="0", cy="0", fw="3")) == \
        ["--family", "phoenix"]

    # Unknown family is a loud error, not a silent mandelbrot.
    try:
        loc_mod.render_one_flags(L(family="nope", cx="0", cy="0", fw="1"))
        assert False, "expected ValueError for unknown family"
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# 2. Key-stability (HARD GATE) — every existing m/j key + field-cache filename
#    is byte-identical to the pre-slot scheme.
# --------------------------------------------------------------------------- #
def test_existing_keys_byte_identical():
    locs = _sample_existing_locations()
    assert locs, "no corpus locations found"
    for canon in locs:
        # identity key == the pre-slot 6-tuple, joined the same way (empty params append nothing)
        old = "|".join("" if v is None else str(v) for v in _old_loc_key(canon))
        assert canon.key() == old, (canon.family, canon.key(), old)
        # field-cache filename unchanged (this is what would orphan the cache if it moved)
        assert aq._field_key(canon) == _old_field_key(canon), canon.key()


def test_existing_families_are_only_m_j():
    """Sanity: the current corpus is mandelbrot+julia only, so the sample above really
    is exercising the empty-params path."""
    fams = {canon.family for canon in _sample_existing_locations()}
    assert fams <= {"mandelbrot", "julia"}, fams


# --------------------------------------------------------------------------- #
# 3. Cache-key non-collision for the new families (and m/j still stable).
# --------------------------------------------------------------------------- #
def test_cache_key_noncollision():
    L = loc_mod.Location
    vp = dict(cx="0.0", cy="0.0", fw="3.0", maxiter=400)

    mand = L(family="mandelbrot", **vp)
    mb3 = L(family="multibrot3", **vp)
    assert mand.key() != mb3.key()
    assert aq._field_key(mand) != aq._field_key(mb3)

    ph_a = L(family="phoenix", c_re="0.5667", c_im="0.0",
             family_params={"p_re": "-0.5", "p_im": "0.0"}, **vp)
    ph_b = L(family="phoenix", c_re="0.5667", c_im="0.0",
             family_params={"p_re": "-0.4", "p_im": "0.0"}, **vp)   # differs ONLY in p
    assert ph_a.key() != ph_b.key()
    assert aq._field_key(ph_a) != aq._field_key(ph_b)

    # ...and the multibrot key/filename does NOT accidentally collide with a julia at
    # the same viewport carrying a c that stringifies into the same slot.
    jul = L(family="julia", c_re="0.0", c_im="0.0", **vp)
    assert len({mand.key(), mb3.key(), ph_a.key(), jul.key()}) == 4


# --------------------------------------------------------------------------- #
# 3b. Field-mode token — the render-mode / field-identity token that keeps a
#     strange pure-field dump (tia/stripe/…) from colliding with the cached
#     SMOOTH field. Parity gate: the smooth key is byte-identical to its
#     pre-token value (both beam + emit paths), and distinct modes key distinctly.
# --------------------------------------------------------------------------- #
def test_field_mode_token_semantics():
    # smooth (default / None / explicit "smooth") -> empty token; strange -> itself.
    assert loc_mod.field_mode_token(None) == ""
    assert loc_mod.field_mode_token("smooth") == ""
    assert loc_mod.field_mode_token("tia") == "tia"
    assert loc_mod.field_mode_token("stripe") == "stripe"


# Known input, frozen pre-token stems (computed on the live smooth path). These
# literals are the invariant: if a key edit moves the smooth stem, every cached
# field is orphaned — this test must fail before that ships.
_KNOWN = loc_mod.Location(family="mandelbrot", cx="-0.743643887",
                          cy="0.131825904", fw="1e-6", maxiter=2000)
_BEAM_SMOOTH = "mandelbrot_90d714081a89180f"
_EMIT_SMOOTH = "mandelbrot_3c882a9fb29412d4_2560x1440ss4"


def test_field_key_smooth_parity_beam():
    # default and explicit "smooth" both reproduce the frozen pre-token stem.
    assert aq._field_key(_KNOWN) == _BEAM_SMOOTH
    assert aq._field_key(_KNOWN, "smooth") == _BEAM_SMOOTH
    assert aq._field_key(_KNOWN, None) == _BEAM_SMOOTH
    # distinct modes -> distinct stems, pairwise disjoint (incl. vs smooth).
    stems = {aq._field_key(_KNOWN, m) for m in (None, "smooth", "tia", "stripe", "curvature")}
    assert len(stems) == 4  # {smooth, tia, stripe, curvature}
    assert _BEAM_SMOOTH in stems


def test_field_key_smooth_parity_emit():
    import importlib
    sys.path.insert(0, os.path.join(_TOOLS, "wallpaper"))
    ev = importlib.import_module("emit_v1")   # lazy: keeps torch off the collection path
    assert ev._emit_field_stem(_KNOWN) == _EMIT_SMOOTH
    assert ev._emit_field_stem(_KNOWN, "smooth") == _EMIT_SMOOTH
    assert ev._emit_field_stem(_KNOWN, None) == _EMIT_SMOOTH
    stems = {ev._emit_field_stem(_KNOWN, m) for m in (None, "smooth", "tia", "stripe", "curvature")}
    assert len(stems) == 4
    assert _EMIT_SMOOTH in stems


# --------------------------------------------------------------------------- #
# 4. Render-block / sidecar parse round-trip — the new families load correctly,
#    Phoenix's p survives parse -> key -> flags.
# --------------------------------------------------------------------------- #
def test_from_render_block_families():
    m = loc_mod.from_render_block({"cx": "0", "cy": "0", "fw": "3", "maxiter": 400})
    assert m.family == "mandelbrot" and m.c_re is None and m.family_params == ()

    j = loc_mod.from_render_block({"fractal_type": "julia", "cx": "0", "cy": "0",
                                   "fw": "0.75", "maxiter": 800,
                                   "c_re": "0.27", "c_im": "0.48"})
    assert j.family == "julia" and (j.c_re, j.c_im) == ("0.27", "0.48")

    p = loc_mod.from_render_block({"fractal_type": "phoenix", "cx": "0", "cy": "0",
                                   "fw": "3", "maxiter": 400, "c_re": "0.5667",
                                   "c_im": "0.0", "p_re": "-0.5", "p_im": "0.0"})
    assert p.family == "phoenix" and p.params == {"p_re": "-0.5", "p_im": "0.0"}


def test_sidecar_phoenix_p_survives():
    """render -> sidecar -> parse -> key -> re-render flags: Phoenix's p is never lost."""
    meta = {"location": {"kind": "phoenix", "cx": "0.0", "cy": "0.0", "fw": "3.0",
                         "maxiter": 400, "c_re": "0.5667", "c_im": "0.0",
                         "p_re": "-0.5", "p_im": "0.0"}}
    loc = loc_mod.from_sidecar(meta)
    assert loc.params == {"p_re": "-0.5", "p_im": "0.0"}
    assert "p_re" in loc.key() or loc.key().endswith("-0.5|0.0")
    flags = loc_mod.render_one_flags(loc)
    assert flags[-4:] == ["--p", "-0.5", "0.0"] or ("--p" in flags and "-0.5" in flags)
    # and the key is stable under a re-parse of an equivalent sidecar
    assert loc_mod.from_sidecar(meta).key() == loc.key()


# --------------------------------------------------------------------------- #
# 5. Coloring bridge — m/j Location -> LocationRef leaves the recipe byte-identical.
# --------------------------------------------------------------------------- #
def test_to_location_ref_recipe_stable():
    canon = loc_mod.Location(family="julia", cx="0.0", cy="0.0", fw="0.75",
                             maxiter=800, c_re="0.27", c_im="0.48")
    ref_direct = cm.LocationRef(kind="julia", cx="0.0", cy="0.0", fw="0.75",
                                maxiter=800, c_re="0.27", c_im="0.48")
    ref_bridge = loc_mod.to_location_ref(canon)
    # A CandidateConfig built from either serializes byte-identically (recipe schema).
    def _recipe(ref):
        return cm.CandidateConfig(palette="twilight", location=ref,
                                  eval_width=1024, eval_height=576).to_json()
    assert _recipe(ref_bridge) == _recipe(ref_direct)
    # An already-LocationRef passes straight through.
    assert loc_mod.to_location_ref(ref_direct) is ref_direct


# --------------------------------------------------------------------------- #
# 6. Manifest — the v5 guard still passes; a synthetic new-family location neither
#    changes the julia count nor trips assert_matches_v5.
# --------------------------------------------------------------------------- #
def test_manifest_untripped_by_new_family():
    pool = qs.LocationPool.from_corpus(verbose=False)
    base_julia = pool.family_counts().get("julia", 0)
    assert pool.assert_matches_v5() == base_julia

    # Inject a synthetic phoenix location; it must load, key, and leave the guard green.
    ph = loc_mod.Location(family="phoenix", cx="0.0", cy="0.0", fw="3.0", maxiter=400,
                          c_re="0.5667", c_im="0.0",
                          family_params={"p_re": "-0.5", "p_im": "0.0"})
    pool.locations.append(qs.PooledLocation(ref=ph, scores=[3], batch_ids={"synthetic"}))
    assert pool.family_counts().get("phoenix", 0) == 1
    assert pool.family_counts().get("julia", 0) == base_julia   # unchanged
    assert pool.assert_matches_v5() == base_julia               # still green
    assert aq._field_key(ph)                                    # keys/loads fine


# --------------------------------------------------------------------------- #
# Optional binary-backed Phoenix round-trip (Step-6 ≤1 LSB): render -> dump-field ->
# sidecar -> color-in-Python == Rust ref. Script-only (skipped when the binary is
# absent), mirroring the reframe GPU check being kept out of the pytest gate.
# --------------------------------------------------------------------------- #
def _phoenix_acceptance():
    import colormap_acceptance as ca
    if not ca.BIN.exists():
        print("SKIP  phoenix_acceptance (release binary not built)")
        return True
    m = ca.run_gate("test_06", palette="twilight", filt="box", width=640, height=360, ss=2)
    ok = m["passed"]
    print(f"{'PASS' if ok else 'FAIL'}  phoenix_acceptance "
          f"(max_diff={m['max_diff']} frac_gt1={m['frac_gt1']:.2e})")
    return ok


def main():
    tests = [
        test_render_one_flags_five_families,
        test_existing_keys_byte_identical,
        test_existing_families_are_only_m_j,
        test_cache_key_noncollision,
        test_field_mode_token_semantics,
        test_field_key_smooth_parity_beam,
        test_field_key_smooth_parity_emit,
        test_from_render_block_families,
        test_sidecar_phoenix_p_survives,
        test_to_location_ref_recipe_stable,
        test_manifest_untripped_by_new_family,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL  %s: %s" % (t.__name__, e))
    if not _phoenix_acceptance():
        failed += 1
    print("\n%d/%d python tests passed" % (len(tests) - failed if failed <= len(tests)
                                           else 0, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
