"""Deterministic deploy-time scorer for the v1 aesthetic classifier.

The bias loop imports `load_scorer()` / `Scorer.score_paths()`. The deploy
transform is the **exact deterministic mirror** of the train resize (no aug,
no flips, center) and matches `present.rs`'s 1280x720 JPG path, so scores stay
on the trained distribution.

    from classifier.inference import load_scorer
    scorer = load_scorer("data/classifier/v1/model_best.pt")
    scores = scorer.score_paths(["a.jpg", "b.jpg"])   # higher = nicer
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .data import Transform
from .model import build_model, score_from_logits


class Scorer:
    def __init__(self, model, transform: Transform, target: str, config: dict, device: str):
        self.model = model.eval().to(device)
        self.transform = transform   # train=False deploy mirror
        self.target = target
        self.config = config
        self.device = device

    @torch.no_grad()
    def score_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,3,H,W) already normalized. Returns (N,) monotone scores."""
        x = x.to(self.device)
        with torch.autocast(device_type=self.device.split(":")[0], enabled=(self.device != "cpu")):
            logits = self.model(x)
        return score_from_logits(logits.float(), self.target).cpu()

    @torch.no_grad()
    def score_paths(self, paths, batch_size: int = 32) -> list[float]:
        out: list[float] = []
        batch: list[torch.Tensor] = []
        for p in paths:
            with Image.open(p) as im:
                im.load()
                batch.append(self.transform(im.convert("RGB")))
            if len(batch) == batch_size:
                out.extend(self.score_tensor(torch.stack(batch)).tolist())
                batch = []
        if batch:
            out.extend(self.score_tensor(torch.stack(batch)).tolist())
        return out


def load_scorer(ckpt_path: str | Path, device: str | None = None) -> Scorer:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    target = cfg["target"]
    model = build_model(target=target, drop_rate=cfg.get("drop_rate", 0.2),
                        drop_path_rate=cfg.get("drop_path_rate", 0.1), pretrained=False,
                        num_classes=cfg.get("num_classes", 3))  # 3 for v1..v6, 4 for wallpaper head
    model.load_state_dict(ckpt["state_dict"])
    transform = Transform(
        geometry=cfg["geometry"], interp=cfg["interpolation"],
        mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False,
    )
    return Scorer(model, transform, target, cfg, device)
