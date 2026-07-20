"""Executable check for the deploy-transform invariant (invariant-audit part 1).

CLAUDE.md documents the classifier's deploy preprocessing in prose:

    Deploy transform = classifier.data.Transform(train=False): the deterministic
    1280x720 -> 384x224 bicubic stretch + normalize mirror of present.rs's JPG path.

This turns that prose into an assertion. It pins two things that could silently drift:

  1. The ACTUAL production scoring path (tools/mining/score_lib.Scorer, reached from
     production_seeder -> guard.make_guarded_scorer -> probe.make_scorer) builds its
     transform from the deployed checkpoint's own config. We assert that config decodes
     to the documented recipe: geometry == "stretch", interp == "bicubic", output 3x224x384.
  2. Transform(train=False) on a sample 1280x720 image is BIT-IDENTICAL to an independent
     re-implementation of the documented steps (PIL BICUBIC resize -> /255 -> (t-mean)/std).

No GPU / no Rust binary needed — pure preprocessing parity.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classifier"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

from classifier.data import Transform, SRC_W, SRC_H, TARGET_W, TARGET_H  # noqa: E402

ACTIVE_CKPT = ROOT / "data" / "classifier" / "v7" / "model_best.pt"


def _sample_1280x720() -> Image.Image:
    """A deterministic non-trivial 1280x720 RGB image (structured, not flat — so a
    resize/interp mismatch actually shows up)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:SRC_H, 0:SRC_W]
    r = ((xx * 255) // SRC_W).astype(np.uint8)
    g = ((yy * 255) // SRC_H).astype(np.uint8)
    b = (rng.integers(0, 256, size=(SRC_H, SRC_W))).astype(np.uint8)
    return Image.fromarray(np.dstack([r, g, b]), "RGB")


def _documented_transform(img: Image.Image, mean, std) -> torch.Tensor:
    """Independent re-implementation of the DOCUMENTED deploy recipe, from the prose:
    1280x720 -> 384x224 bicubic stretch, /255, normalize. Intentionally does NOT import
    resize_core — it is a second witness, not a re-call of the code under test."""
    assert img.size == (SRC_W, SRC_H)
    resized = img.resize((TARGET_W, TARGET_H), Image.BICUBIC)          # stretch, bicubic
    arr = np.array(resized, dtype=np.uint8)  # copy -> writable (matches data._to_tensor)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0         # CHW in [0,1]
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (t - m) / s


def _deployed_cfg() -> dict:
    if not ACTIVE_CKPT.exists():
        pytest.skip(f"deployed checkpoint absent: {ACTIVE_CKPT}")
    ckpt = torch.load(ACTIVE_CKPT, map_location="cpu", weights_only=False)
    return ckpt["config"]


def test_deployed_checkpoint_config_is_documented_recipe():
    """The production scoring path reads geometry/interp/mean/std from THIS config; assert
    they decode to the documented 384x224 bicubic-stretch recipe."""
    cfg = _deployed_cfg()
    assert cfg["geometry"] == "stretch", f"deploy geometry drifted: {cfg['geometry']!r}"
    assert cfg["interpolation"] == "bicubic", f"deploy interp drifted: {cfg['interpolation']!r}"
    assert len(cfg["mean"]) == 3 and len(cfg["std"]) == 3
    # output geometry the model actually consumes
    assert (TARGET_W, TARGET_H) == (384, 224)


def test_transform_matches_documented_steps_bit_for_bit():
    """Transform(train=False) == the independent documented re-implementation, exactly."""
    cfg = _deployed_cfg()
    tf = Transform(geometry=cfg["geometry"], interp=cfg["interpolation"],
                   mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False)
    img = _sample_1280x720()
    got = tf(img)
    want = _documented_transform(img, cfg["mean"], cfg["std"])
    assert got.shape == (3, TARGET_H, TARGET_W), got.shape
    assert torch.equal(got, want), (
        f"deploy transform diverged from documented recipe: max|Δ|={ (got-want).abs().max().item() }")


def test_production_scorer_uses_the_same_transform():
    """The live production scorer (score_lib.Scorer, the one guard wraps) builds a transform
    that is byte-identical on a sample input to Transform(train=False) from the same cfg —
    i.e. no scorer re-wraps or re-implements preprocessing."""
    if not ACTIVE_CKPT.exists():
        pytest.skip("deployed checkpoint absent")
    sys.path.insert(0, str(ROOT / "tools" / "mining"))
    from score_lib import Scorer  # noqa: E402
    sc = Scorer(model_path=str(ACTIVE_CKPT), device="cpu")
    cfg = sc.cfg
    ref = Transform(geometry=cfg["geometry"], interp=cfg["interpolation"],
                    mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False)
    img = _sample_1280x720()
    assert torch.equal(sc.transform(img), ref(img))
    # and that ref equals the documented independent witness
    assert torch.equal(ref(img), _documented_transform(img, cfg["mean"], cfg["std"]))


if __name__ == "__main__":
    for fn in (test_deployed_checkpoint_config_is_documented_recipe,
               test_transform_matches_documented_steps_bit_for_bit,
               test_production_scorer_uses_the_same_transform):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
