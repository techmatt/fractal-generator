"""Tests for the within-flavor deficit palette pick (tools/emission/palette_deficit.py)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.emission import palette_deficit as pd


def _solid_lut(rgb, n=64):
    return np.tile(np.array(rgb, float), (n, 1))


def test_signature_parity_with_realized_stats():
    """_hsv_signature must match realized_palette_stats' HSV convention on a real image."""
    from tools.emission import build_emission_diversity_v1 as bd
    from PIL import Image
    import tempfile, os
    # a deterministic 2-hue image (green top / orange bottom)
    im = np.zeros((32, 32, 3), np.uint8)
    im[:16] = (40, 200, 60)      # green
    im[16:] = (230, 140, 30)     # orange
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jpg")
        Image.fromarray(im).save(p, quality=100)
        realized = bd.realized_palette_stats(p)
    sig = pd._hsv_signature((np.asarray(im, float) / 255.0).reshape(-1, 3))
    # green top half ⇒ realized hue_hist also carries green-bin mass
    assert np.array(realized["hue_hist"])[list(pd.GREEN_BINS)].sum() > 0.2
    # green share should agree closely (jpeg at q100 is near-lossless); both put mass in green bins
    assert sig["green"] > 0.2
    assert abs(sig["hue"].sum() - 1.0) < 1e-6 and abs(sig["chroma"].sum() - 1.0) < 1e-6


def test_empty_start_discriminates_green():
    """Non-uniform target ⇒ a green palette out-gains a red one with an EMPTY corpus."""
    t = pd.DeficitTracker()
    green = pd._hsv_signature(_solid_lut([0.2, 0.9, 0.3]))
    red = pd._hsv_signature(_solid_lut([0.9, 0.2, 0.2]))
    assert t.gain(green) > t.gain(red)


def test_deficit_self_balances():
    """After the corpus fills with green, a fresh green candidate loses its advantage."""
    t = pd.DeficitTracker()
    green = pd._hsv_signature(_solid_lut([0.2, 0.9, 0.3]))
    red = pd._hsv_signature(_solid_lut([0.9, 0.2, 0.2]))
    g0 = t.gain(green)
    # ingest 200 green renders (realized ~ the green intrinsic signature)
    realized_green = {"hue_hist": list(green["hue"]), "chroma_hist": list(green["chroma"])}
    for _ in range(200):
        t.ingest(realized_green)
    assert t.gain(green) < g0                    # green deficit consumed
    assert t.gain(red) > t.gain(green)           # pull has moved to the starved red


def test_ingest_is_order_independent():
    """Resume safety: deficit depends only on the multiset of realized rows (sums)."""
    rows = [{"hue_hist": list(np.random.default_rng(i).random(12)),
             "chroma_hist": list(np.random.default_rng(100 + i).random(8))} for i in range(20)]
    a, b = pd.DeficitTracker(), pd.DeficitTracker()
    for r in rows:
        a.ingest(r)
    for r in reversed(rows):
        b.ingest(r)
    assert np.allclose(a.H, b.H) and np.allclose(a.C, b.C) and a.n == b.n


def test_pick_recovers_pref_argmax_when_deficit_flat():
    """Identical signatures ⇒ zero gain spread ⇒ v3-gvo (pref) decides = argmax baseline."""
    t = pd.DeficitTracker()
    members = ["a", "b", "c"]
    sig = pd._hsv_signature(_solid_lut([0.5, 0.5, 0.9]))
    sigs = {m: sig for m in members}
    pref = [1.0, 5.0, 2.0]
    assert pick_name(members, pref, sigs, t) == "b"          # highest pref


def test_pick_overrides_pref_for_green_under_deficit():
    """A mid-pref green palette wins over a top-pref grey one when green is in deficit."""
    t = pd.DeficitTracker()
    members = ["grey_best", "green_mid"]
    sigs = {"grey_best": pd._hsv_signature(_solid_lut([0.55, 0.55, 0.55])),
            "green_mid": pd._hsv_signature(_solid_lut([0.2, 0.9, 0.3]))}
    pref = [1.0, 0.6]                  # grey is the pref argmax
    assert pick_name(members, pref, sigs, t, lam=1.5) == "green_mid"


def test_pref_z_floor_drops_clear_losers():
    """A palette far below the pref distribution is excluded even if it fills the deficit."""
    t = pd.DeficitTracker()
    members = ["ok0", "ok1", "ok2", "ok3", "green_trash"]
    base = pd._hsv_signature(_solid_lut([0.55, 0.55, 0.9]))
    sigs = {m: base for m in members[:4]}
    sigs["green_trash"] = pd._hsv_signature(_solid_lut([0.2, 0.9, 0.3]))
    pref = [10.0, 10.1, 9.9, 10.0, 2.0]         # green_trash is a clear outlier-low
    assert pick_name(members, pref, sigs, t, lam=3.0, pref_z_floor=-1.0) != "green_trash"


def pick_name(members, pref, sigs, t, **kw):
    return members[pd.pick(members, pref, sigs, t, **kw)]
