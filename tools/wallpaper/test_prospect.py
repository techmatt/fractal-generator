"""Tests for the Phase-1 prospecting tail — record shape, per-family identity, crash-safe
embedding append, LRU field-cache eviction, resume idempotence.

GPU-free: exercises only the pure record-building + store I/O (no torch, no render). The CLIP
embed / grayscale render are validated end-to-end by the orchestrator smoke run, not here.

Run either way:
  uv run pytest tools/wallpaper/test_prospect.py
  uv run python tools/wallpaper/test_prospect.py     # prints PASS/FAIL summary
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))
import library_store as store          # noqa: E402
import library_annotate as ann         # noqa: E402
from tools.corpus import location as loc_mod  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures — synthetic pool rows + ledger, one per family kind.
# --------------------------------------------------------------------------- #
def _pool_row(oid, family, fractal_type, cx="0.1", cy="0.2", fw="0.01",
              c_re=None, c_im=None):
    render = {"cx": cx, "cy": cy, "fw": fw, "maxiter": 1500, "fractal_type": fractal_type}
    if c_re is not None:
        render["c_re"], render["c_im"] = c_re, c_im
    return {
        "image_id": f"{oid}_00",
        "render": render,
        "provenance": {"family": family, "source_oid": oid,
                       "seeder_decoded_class": 3, "seeder_p_good": 0.7,
                       "source_ledger": "data/discovery/fresh_runs/RUN/outcome_ledger.jsonl"},
        "label": {"score": None},
    }


def _ledger_row(oid, family):
    return {"id": oid, "family": family, "scorer_version": "v6", "k3": 0.31,
            "raw_top3": [0.3, 0.31, 0.32], "decoded_class": 3, "p_good": 0.42,
            "p_notbad": 0.8, "t_good": 0.24, "reached_depth": 9, "guard_pass": True}


def _record(oid, family, fractal_type, **kw):
    row = _pool_row(oid, family, fractal_type, **kw)
    led = {oid: _ledger_row(oid, family)}
    return ann.build_record(oid, row["render"], row["provenance"], led,
                            run_id="RUN", cycle=3, source_ledger="LED")


# --------------------------------------------------------------------------- #
# Record shape + per-family identity.
# --------------------------------------------------------------------------- #
def test_record_shape_dense_and_reserved():
    r = _record("m_1", "mandelbrot", "mandelbrot")
    # dense blocks present
    assert r["record_version"] == "0.1"
    assert r["location_id"] == "m_1"
    assert r["run_id"] == "RUN" and r["cycle"] == 3
    assert r["identity"]["family"] == "mandelbrot"
    assert r["location_potential"]["k3"] == 0.31          # JOINED from ledger, not recomputed
    assert r["location_potential"]["decoded_class"] == 3
    assert r["descriptors"]["uid"] == "m_1"
    assert r["descriptors"]["morph_producer"] == ann.MORPH_PRODUCER   # seam marker present
    assert r["descriptors"]["morph_v6"] is None            # skipped (not free)
    assert r["descriptors"]["thumbnail"] == "thumbs/m_1.jpg"
    # reserved null/empty — demand-driven at Phase 2, NOT filled here
    assert r["palette_candidates"] == []
    assert r["mode_candidacy"] is None
    assert r["descriptors"]["colored_clip"] is None
    assert r["wallpaper_quality"]["predicted_p_ge3"] is None
    assert r["wallpaper_quality"]["actual_p_ge3"] is None


def test_identity_mandelbrot():
    idn = _record("m_1", "mandelbrot", "mandelbrot")["identity"]
    assert idn["c"] is None and idn["p"] is None
    assert idn["coord_kind"] == "c_plane"
    assert idn["source_oid"] == "m_1"


def test_identity_julia_carries_c():
    idn = _record("j_1", "julia", "julia", c_re="0.233", c_im="0.538")["identity"]
    assert idn["c"] == {"re": "0.233", "im": "0.538"}
    assert idn["p"] is None
    assert idn["coord_kind"] == "julia_c_fixed"


def test_identity_julia_multibrot_carries_c():
    idn = _record("jm3_1", "julia_multibrot3", "julia_multibrot3",
                  c_re="-0.387", c_im="-0.629")["identity"]
    assert idn["c"] == {"re": "-0.387", "im": "-0.629"}
    assert idn["coord_kind"] == "julia_c_fixed"
    assert idn["family"] == "julia_multibrot3"


def test_identity_phoenix_stamps_ushiki():
    # phoenix pool render block leaves c/p NULL — identity must STAMP the fixed Ushiki c/p.
    idn = _record("ph_1", "phoenix", "phoenix")["identity"]
    assert idn["c"] == ann.PHOENIX_C
    assert idn["p"] == ann.PHOENIX_P
    assert idn["coord_kind"] == "z_viewport"


def test_render_location_phoenix_flags():
    # the Location built for the field dump must recover c + p so render-one gets --c AND --p.
    row = _pool_row("ph_2", "phoenix", "phoenix")
    loc = ann.render_location(row["render"])
    flags = loc_mod.render_one_flags(loc)
    assert "--family" in flags and "phoenix" in flags
    assert "--c" in flags and "--p" in flags
    ci = flags.index("--c"); pi = flags.index("--p")
    assert flags[ci + 1:ci + 3] == [ann.PHOENIX_C["re"], ann.PHOENIX_C["im"]]
    assert flags[pi + 1:pi + 3] == [ann.PHOENIX_P["re"], ann.PHOENIX_P["im"]]


def test_render_location_julia_multibrot_degree_survives():
    row = _pool_row("jm4_1", "julia_multibrot4", "julia_multibrot4",
                    c_re="0.45", c_im="0.65")
    loc = ann.render_location(row["render"])
    flags = loc_mod.render_one_flags(loc)
    assert flags[:2] == ["--family", "multibrot4"]        # degree kept, flipped to dynamical twin
    assert "--julia" in flags and "--c" in flags


# --------------------------------------------------------------------------- #
# unique_locations — one row per source_oid.
# --------------------------------------------------------------------------- #
def test_unique_locations_dedup(tmp_path):
    p = tmp_path / "images.jsonl"
    with open(p, "w") as f:
        for r in [_pool_row("a", "mandelbrot", "mandelbrot"),
                  {**_pool_row("a", "mandelbrot", "mandelbrot"), "image_id": "a_01"},
                  _pool_row("b", "julia", "julia", c_re="0", c_im="0")]:
            f.write(json.dumps(r) + "\n")
    rows = ann.unique_locations(p)
    assert [r["provenance"]["source_oid"] for r in rows] == ["a", "b"]


# --------------------------------------------------------------------------- #
# Crash-safe embedding append + dim assert + concatenating loader.
# --------------------------------------------------------------------------- #
def _tmp_base(tmp_path, dim=768):
    base = tmp_path / "embeddings.npz"
    np.savez(base, morph_uids=np.asarray(["base_0"]),
             morph_clip=np.zeros((1, dim), np.float32))
    return base


def test_embedding_shard_roundtrip_and_dim_source_of_truth(tmp_path):
    base = _tmp_base(tmp_path, dim=768)
    shards = tmp_path / "shards"
    assert store.base_morph_dim(base) == 768               # read from base, not assumed
    clip = np.random.rand(3, 768).astype(np.float32)
    shard = store.write_embedding_shard("RUN", 1, ["x", "y", "z"], clip,
                                        shards_dir=shards, emb_base=base)
    assert shard.exists()
    emb = store.load_library_embeddings(emb_base=base, shards_dir=shards)
    assert set(emb) == {"base_0", "x", "y", "z"}
    assert np.allclose(emb["y"], clip[1])


def test_embedding_dim_assert_rejects_mismatch(tmp_path):
    base = _tmp_base(tmp_path, dim=768)
    shards = tmp_path / "shards"
    bad = np.zeros((2, 512), np.float32)                   # wrong width
    try:
        store.write_embedding_shard("RUN", 1, ["a", "b"], bad,
                                    shards_dir=shards, emb_base=base)
        assert False, "expected dim assert to fire"
    except AssertionError as e:
        assert "512" in str(e) or "dim" in str(e)


def test_embedding_append_crash_safe(tmp_path):
    # a stray leftover .tmp (interrupted write) must NOT be loaded; the atomic .npz must.
    base = _tmp_base(tmp_path)
    shards = tmp_path / "shards"
    store.write_embedding_shard("RUN", 1, ["ok"], np.ones((1, 768), np.float32),
                                shards_dir=shards, emb_base=base)
    (shards / ".RUN__cycle_002.npz.tmp").write_bytes(b"garbage partial write")
    emb = store.load_library_embeddings(emb_base=base, shards_dir=shards)
    assert set(emb) == {"base_0", "ok"}                    # tmp ignored, no crash


def test_embedding_shard_rewrite_idempotent(tmp_path):
    base = _tmp_base(tmp_path)
    shards = tmp_path / "shards"
    v1 = np.ones((1, 768), np.float32)
    store.write_embedding_shard("RUN", 1, ["k"], v1, shards_dir=shards, emb_base=base)
    v2 = np.full((1, 768), 2.0, np.float32)                # a resumed cycle recomputes same key
    store.write_embedding_shard("RUN", 1, ["k"], v2, shards_dir=shards, emb_base=base)
    emb = store.load_library_embeddings(emb_base=base, shards_dir=shards)
    assert len(list(shards.glob("*.npz"))) == 1            # overwrote, not duplicated
    assert np.allclose(emb["k"], 2.0)


# --------------------------------------------------------------------------- #
# LRU field-cache eviction.
# --------------------------------------------------------------------------- #
def test_lru_eviction_under_cap(tmp_path):
    cache = tmp_path / "field_cache"
    cache.mkdir()
    # 4 fields x 1 MiB each = 4 MiB; cap at ~2.5 MiB -> evict the 2 oldest.
    stems = ["f0", "f1", "f2", "f3"]
    for i, s in enumerate(stems):
        (cache / f"{s}.bin").write_bytes(b"\0" * (1024 * 1024))
        (cache / f"{s}.json").write_text("{}")
        t = 1000.0 + i                                     # f0 oldest ... f3 newest
        os.utime(cache / f"{s}.bin", (t, t))
        os.utime(cache / f"{s}.json", (t, t))
    evicted, freed = store.evict_field_cache_lru(2.5 / 1024, cache_dir=cache)
    assert evicted == 2
    remaining = {f.stem for f in cache.glob("*.bin")}
    assert remaining == {"f2", "f3"}                       # oldest two gone, pair evicted together
    assert not (cache / "f0.json").exists()


def test_lru_noop_under_cap(tmp_path):
    cache = tmp_path / "field_cache"
    cache.mkdir()
    (cache / "f.bin").write_bytes(b"\0" * 1024)
    (cache / "f.json").write_text("{}")
    evicted, freed = store.evict_field_cache_lru(10.0, cache_dir=cache)
    assert evicted == 0 and freed == 0


# --------------------------------------------------------------------------- #
# Resume idempotence — re-appending a cycle's records adds 0 duplicates.
# --------------------------------------------------------------------------- #
def test_append_records_idempotent(tmp_path):
    rp = tmp_path / "records.jsonl"
    recs = [_record("m_1", "mandelbrot", "mandelbrot"),
            _record("j_1", "julia", "julia", c_re="0", c_im="0")]
    w1 = store.append_records(recs, rp)
    assert len(w1) == 2
    w2 = store.append_records(recs, rp)                    # re-run same cycle
    assert len(w2) == 0                                    # 0 duplicates
    assert store.existing_location_ids(rp) == {"m_1", "j_1"}
    # one extra new location appends cleanly alongside
    w3 = store.append_records([_record("m_2", "mandelbrot", "mandelbrot")] + recs, rp)
    assert len(w3) == 1 and w3[0]["location_id"] == "m_2"


def test_field_stem_smooth_token_empty():
    loc = loc_mod.Location(family="mandelbrot", cx="0", cy="0", fw="1", maxiter=100)
    stem = store.field_stem(loc, "smooth", 640, 360, 2)
    assert stem.endswith("640x360ss2__smooth")
    assert loc_mod.field_mode_token("smooth") == ""        # smooth token empty (no collision key)


# --------------------------------------------------------------------------- #
# Grayscale morphology transfer — locks the RECOVERED robust-z tanh (K=2) formula.
# Any drift in MORPH_K / MORPH_MAD_SCALE / the tanh form / the linear box-downsample
# breaks the 62 curated morph_clip rows' parity (cosine 1.0), so pin it here (GPU-free).
# --------------------------------------------------------------------------- #
def _synthetic_field(ss=2):
    from tools import colormap as cm
    # 4x4 super-res (ss2 -> 2x2 out); one interior (NaN) pixel, skewed exterior for a real MAD.
    v = np.array([[0.0, 1.0, 2.0, 3.0],
                  [1.0, np.nan, 4.0, 2.0],
                  [2.0, 3.0, 10.0, 1.0],
                  [0.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    loc = cm.LocationRef(kind="mandelbrot", cx="0", cy="0", fw="1", maxiter=100)
    return cm.FieldData(values=v, supersample=ss, location=loc)


def test_morph_gray_transfer_robustz():
    field = _synthetic_field()
    out = np.asarray(ann.morph_gray_image(field))          # (2,2,3) uint8, RGB-replicated

    # reference: the documented transform, computed independently
    v = field.values
    fin = np.isfinite(v)
    m = np.median(v[fin])
    mad = np.median(np.abs(v[fin] - m)) * ann.MORPH_MAD_SCALE + 1e-12
    t = 0.5 * (1.0 + np.tanh((v - m) / (ann.MORPH_K * mad)))
    t = np.where(fin, t, 0.0)
    g = t.reshape(2, 2, 2, 2).mean(axis=(1, 3))            # linear ss2 block-mean
    ref = np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8)

    assert out.shape == (2, 2, 3)
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    assert np.array_equal(out[..., 0], ref)                # exact match to the formula
    # constants are the recovered original (median/MAD tanh, K=2)
    assert ann.MORPH_K == 2.0 and abs(ann.MORPH_MAD_SCALE - 1.4826) < 1e-9


def test_morph_gray_interior_is_black_and_deterministic():
    field = _synthetic_field()
    a = np.asarray(ann.morph_gray_image(field))
    b = np.asarray(ann.morph_gray_image(field))
    assert np.array_equal(a, b)                            # deterministic
    # a fully-interior (all-NaN) block downsamples to pure black
    field2 = _synthetic_field()
    field2.values[:2, :2] = np.nan
    out = np.asarray(ann.morph_gray_image(field2))
    assert out[0, 0, 0] == 0


# --------------------------------------------------------------------------- #
# --rerun-failed — a deferred cycle drains, records land, a second drain adds zero.
# GPU-free: the annotate subprocess is replaced with a store-append stub that reuses the REAL
# library_store dedup, so the idempotence under test is the production dedup, not a mock.
# --------------------------------------------------------------------------- #
import argparse  # noqa: E402
import prospect_orchestrator as po  # noqa: E402
import overnight_orchestrator as oo  # noqa: E402


def _fake_annotate_pool(batch_dir, ledger, watermark, run_id, cycle, sinks, field_cache_gb,
                        retain_fields, est_annotate_s, log, baseline_gpu, tag):
    """Store-side of library_annotate WITHOUT the GPU embed/thumbnail: dedup the pool against the
    store, append the survivors, write a schema-faithful annotate_report. Real store dedup -> a
    re-drain over the same pool appends 0 (the whole idempotence claim)."""
    rows = ann.unique_locations(Path(batch_dir) / "images.jsonl")
    led = ann.load_ledger(Path(ledger))
    have = store.existing_location_ids(sinks.records)
    records, n_dup = [], 0
    for r in rows:
        oid = r["provenance"]["source_oid"]
        if oid in have:
            n_dup += 1
            continue
        records.append(ann.build_record(oid, r["render"], r["provenance"], led,
                                         run_id, cycle, str(ledger)))
    written = store.append_records(records, sinks.records)
    (Path(batch_dir) / "annotate_report.json").write_text(json.dumps(
        {"cycle": cycle, "pool_unique_locations": len(rows), "dropped_coord_dup": n_dup,
         "dropped_field_fail": 0, "records_written": len(written)}), encoding="utf-8")
    return True, {"ok": True}


def _setup_deferred_run(tmp_path, oids=("c1_a", "c1_b")):
    """A run tree with ONE deferred failed cycle: retained pool + ledger + state.failed_cycles."""
    run_dir = tmp_path / "out" / "RUN"
    disc_dir = tmp_path / "disc" / "RUN"
    (run_dir / "pools" / "cycle_001").mkdir(parents=True)
    disc_dir.mkdir(parents=True)
    ledger = disc_dir / "outcome_ledger.jsonl"
    with open(ledger, "w", encoding="utf-8") as f:
        for oid in oids:
            f.write(json.dumps(_ledger_row(oid, "mandelbrot")) + "\n")
    with open(run_dir / "pools" / "cycle_001" / "images.jsonl", "w", encoding="utf-8") as f:
        for oid in oids:
            f.write(json.dumps(_pool_row(oid, "mandelbrot", "mandelbrot")) + "\n")
    oo.save_state(run_dir / "state.json", {
        "run_id": "RUN", "deadline_epoch": 0, "cycles_done": 1,
        "failed_cycles": [{"cycle": 1, "q3_deferred": len(oids), "records_salvaged": 0,
                           "reason": "annotate failed twice (test)", "ledger_watermark": 0}]})
    args = argparse.Namespace(
        run_id="RUN", out_root=str(tmp_path / "out"), discovery_root=str(tmp_path / "disc"),
        field_cache_gb=20.0, retain_fields=True, est_loc_s=6.0,
        store_records=str(tmp_path / "store" / "records.jsonl"),
        store_thumbs=str(tmp_path / "store" / "thumbs"),
        store_emb_shards=str(tmp_path / "store" / "shards"),
        store_field_cache=str(tmp_path / "store" / "field_cache"))
    return run_dir, args


class _patch_annotate:
    """Manual patch of po._annotate_pool (works under both pytest AND the tmp_path-only standalone
    runner, which doesn't provide the monkeypatch fixture)."""
    def __enter__(self):
        self._orig = po._annotate_pool
        po._annotate_pool = _fake_annotate_pool
        return self

    def __exit__(self, *exc):
        po._annotate_pool = self._orig


def test_rerun_failed_drains_then_idempotent(tmp_path):
    with _patch_annotate():
        run_dir, args = _setup_deferred_run(tmp_path)
        rp = Path(args.store_records)

        # First drain: the 2 deferred q3 land as records; reconcile balances (unexplained==0).
        r1 = po.rerun_failed(args)
        assert r1["records_added"] == 2
        assert store.existing_location_ids(rp) == {"c1_a", "c1_b"}
        assert r1["drained"][0]["records_written"] == 2
        assert r1["drained"][0]["unexplained"] == 0
        assert r1["still_failed"] == []
        # state drained: the failed cycle is gone from the resume ledger.
        assert oo.load_state(run_dir / "state.json")["failed_cycles"] == []

        # Re-inject the same failed cycle (operator re-run / crash before state saved) and drain
        # AGAIN: store-dedup makes it a no-op — 0 records added, reconcile still balances.
        st = oo.load_state(run_dir / "state.json")
        st["failed_cycles"] = [{"cycle": 1, "q3_deferred": 2, "records_salvaged": 2,
                                "reason": "re-injected", "ledger_watermark": 0}]
        oo.save_state(run_dir / "state.json", st)

        r2 = po.rerun_failed(args)
        assert r2["records_added"] == 0                       # store-dedup: nothing new lands
        assert store.existing_location_ids(rp) == {"c1_a", "c1_b"}   # store unchanged
        d2 = r2["drained"][0]
        assert d2["records_written"] == 0 and d2["dropped_coord_dup"] == 2
        assert d2["unexplained"] == 0                         # reconciles clean on the second pass


def test_rerun_failed_no_deferred_cycles_is_noop(tmp_path):
    run_dir, args = _setup_deferred_run(tmp_path)
    st = oo.load_state(run_dir / "state.json")
    st["failed_cycles"] = []
    oo.save_state(run_dir / "state.json", st)
    r = po.rerun_failed(args)
    assert r["records_added"] == 0 and r["drained"] == [] and r["still_failed"] == []


def test_rerun_failed_missing_pool_stays_deferred(tmp_path):
    with _patch_annotate():
        run_dir, args = _setup_deferred_run(tmp_path)
        # A cycle whose retained pool was (wrongly) purged cannot be re-derived — left deferred,
        # never rebuilt from the watermark (the pool builder reads watermark..EOF, spanning later
        # cycles).
        import shutil
        shutil.rmtree(run_dir / "pools" / "cycle_001")
        r = po.rerun_failed(args)
        assert r["records_added"] == 0
        assert len(r["still_failed"]) == 1 and r["drained"] == []
        assert oo.load_state(run_dir / "state.json")["failed_cycles"][0]["cycle"] == 1


def test_embedding_shard_carries_producer(tmp_path):
    dim = store.base_morph_dim()
    shard = store.write_embedding_shard("RUN", 1, ["u0", "u1"],
                                        np.ones((2, dim), np.float32),
                                        shards_dir=tmp_path, emb_base=tmp_path / "none.npz",
                                        producer=ann.MORPH_PRODUCER)
    z = np.load(shard, allow_pickle=True)
    assert "morph_producer" in z.files
    assert list(z["morph_producer"]) == [ann.MORPH_PRODUCER, ann.MORPH_PRODUCER]


# --------------------------------------------------------------------------- #
# Standalone runner.
# --------------------------------------------------------------------------- #
def _run_standalone():
    import tempfile, traceback
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    npass = 0
    for name, fn in tests:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"PASS {name}")
            npass += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{npass}/{len(tests)} passed")
    return npass == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_standalone() else 1)
