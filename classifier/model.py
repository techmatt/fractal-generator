"""Model + ordinal head/loss for the v1 aesthetic classifier.

Backbone: timm mobilenetv4_conv_medium.e250_r384_in12k (in12k checkpoint).
Two targets, both producing a single monotone scalar rank score:

  ordinal (default) : CORN rank-consistent ordinal, K-1=2 binary logits.
                      score = Σ σ(logit_k)  (the prompt's specified scalar).
  binary            : 1-vs-{2,3} BCE on a single logit. score = σ(logit).

mean/std/interpolation are pulled from the checkpoint's own data config — NOT
ImageNet-1k defaults (the in12k checkpoint differs).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "mobilenetv4_conv_medium.e250_r384_in12k"


def build_model(target: str = "ordinal", drop_rate: float = 0.2,
                drop_path_rate: float = 0.1, pretrained: bool = True):
    n_out = 2 if target == "ordinal" else 1  # K-1=2 ordinal logits, else 1 binary logit
    model = timm.create_model(
        BACKBONE, pretrained=pretrained, num_classes=n_out,
        drop_rate=drop_rate, drop_path_rate=drop_path_rate,
    )
    return model


def data_config(model) -> dict:
    """Resolved mean/std/interpolation/input_size from the checkpoint."""
    cfg = timm.data.resolve_model_data_config(model)
    return {
        "mean": tuple(float(x) for x in cfg["mean"]),
        "std": tuple(float(x) for x in cfg["std"]),
        "interpolation": cfg["interpolation"],
        "input_size": tuple(int(x) for x in cfg["input_size"]),
    }


# --------------------------------------------------------------------------- #
# CORN ordinal loss (rank-consistent, conditional training). Inlined to avoid a
# coral-pytorch dependency. Labels here are RANKS 0..K-1 (class-1).
# --------------------------------------------------------------------------- #
def corn_loss(logits: torch.Tensor, ranks: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """logits: (N, K-1). ranks: (N,) in 0..K-1. Conditional-subset BCE per task."""
    total = logits.new_zeros(())
    n_tasks = num_classes - 1
    for k in range(n_tasks):
        mask = ranks > (k - 1)               # examples with rank >= k
        if mask.sum() < 1:
            continue
        tgt = (ranks[mask] > k).float()      # 1 if rank > k
        pred = logits[mask, k]
        # numerically stable BCE-with-logits, summed then mean over the subset
        ls = F.logsigmoid(pred)
        loss = -torch.sum(ls * tgt + (ls - pred) * (1.0 - tgt))
        total = total + loss / mask.sum()
    return total / n_tasks


def binary_loss(logits: torch.Tensor, labels123: torch.Tensor) -> torch.Tensor:
    """logits: (N,1). labels123: raw 1/2/3. Target = (label>=2)."""
    tgt = (labels123 >= 2).float().view(-1, 1)
    return F.binary_cross_entropy_with_logits(logits, tgt)


def score_from_logits(logits: torch.Tensor, target: str) -> torch.Tensor:
    """Monotone scalar rank score (higher = better/nicer). One value per sample."""
    if target == "ordinal":
        return torch.sigmoid(logits).sum(dim=1)   # Σ σ(logit_k), in [0, K-1]
    return torch.sigmoid(logits).view(-1)         # binary: σ(logit), in [0, 1]


def compute_loss(logits: torch.Tensor, labels123: torch.Tensor, target: str) -> torch.Tensor:
    if target == "ordinal":
        ranks = (labels123 - 1).long()            # {1,2,3} -> {0,1,2}
        return corn_loss(logits, ranks, num_classes=3)
    return binary_loss(logits, labels123)
