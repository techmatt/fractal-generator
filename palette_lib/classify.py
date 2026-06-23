"""Single source of truth for palette quarantine + cyclic/sequential labelling.

Operationalizes the eye-validated pass from `prompts/palette-quarantine-and-seam.md`
(quarantine the internally-rough categorical maps; flag sequential maps for the
render mirror fix) so `build_sheet` reproduces the hand-verified split on every
rebuild instead of clobbering it. The metrics are computed **directly from the
stops** (no render); the bake matches `coloring.bake_lut` exactly so
`internal_max_step` is the true dense-LUT roughness the renderer would show.

Thresholds are the eye-validated constants below — this module is imported by
`build_sheet` (and any bench), never reimplemented. Operationalize, don't re-tune:
a surprise gets reported, not re-thresholded.
"""

from __future__ import annotations

import numpy as np

from .coloring import LUT_SIZE, srgb8_to_oklab

# --- eye-validated thresholds (NOT re-tuned here) ----------------------------
N_JUMP_QUARANTINE = 3      # quarantine when n_jump (# body jumps) >= this
JUMP_OKLAB = 0.25          # a body stop-step above this OKLab dist counts as a "jump"
SEAM_CYCLIC = 0.10         # seam (endpoint OKLab dist) below this => cyclic, else sequential


def classify_palette(stops):
    """stops: list[(pos, (r,g,b))] (the `bake_lut` input form).

    Returns a dict::

        {seam, internal_max_step, n_jump, max_stop_step, mean_stop_step,
         cycle: "cyclic"|"sequential", mirror_needed: bool, quarantine: bool}

    - `n_jump`           = # adjacent body stop-steps with OKLab dist > JUMP_OKLAB
                           (the wrap pair excluded). The quarantine separator.
    - `seam`             = OKLab dist between the last and first stop (the wrap pair).
                           Small => genuinely cyclic; large => sequential / mirror-needed.
    - `internal_max_step`= max adjacent OKLab step in the dense LUT_SIZE bake,
                           EXCLUDING the wrap (seam) segment — the visible body
                           roughness (descriptor; not the quarantine criterion).
    - `max/mean_stop_step` = body stop-step extremes (descriptors).

    Decisions are taken on the raw values; the stored metrics are rounded to the
    sidecar's precision.
    """
    pos = np.array([p % 1.0 for p, _ in stops], dtype=np.float64)
    lab = srgb8_to_oklab(np.array([c for _, c in stops], dtype=np.float64))
    order = np.argsort(pos, kind="stable")
    pos = pos[order]
    lab = lab[order]

    # Body stop-steps: consecutive OKLab distances, NOT wrapping (the seam pair
    # is handled separately).
    steps = np.linalg.norm(np.diff(lab, axis=0), axis=1)
    n_jump = int(np.count_nonzero(steps > JUMP_OKLAB))

    # Seam = endpoint OKLab distance (the wrap pair) -> mirror flag.
    seam = float(np.linalg.norm(lab[-1] - lab[0]))

    # internal_max_step: roughness of the dense bake, excluding the seam segment.
    # Cyclically extend exactly as bake_lut does, sample LUT_SIZE points, take the
    # max adjacent step among samples whose segment lies in the body [pos0, posN].
    ext_pos = np.concatenate(([pos[-1] - 1.0], pos, [pos[0] + 1.0]))
    ext_lab = np.concatenate((lab[-1:], lab, lab[:1]), axis=0)
    t = np.arange(LUT_SIZE, dtype=np.float64) / LUT_SIZE
    lab_t = np.empty((LUT_SIZE, 3), dtype=np.float64)
    for ch in range(3):
        lab_t[:, ch] = np.interp(t, ext_pos, ext_lab[:, ch])
    d = np.linalg.norm(np.diff(lab_t, axis=0), axis=1)
    body = (t[:-1] >= pos[0]) & (t[1:] <= pos[-1])
    internal_max_step = float(d[body].max())

    cyclic = seam < SEAM_CYCLIC
    return {
        "seam": round(seam, 4),
        "internal_max_step": round(internal_max_step, 5),
        "n_jump": n_jump,
        "max_stop_step": round(float(steps.max()), 4),
        "mean_stop_step": round(float(steps.mean()), 4),
        "cycle": "cyclic" if cyclic else "sequential",
        "mirror_needed": not cyclic,
        "quarantine": n_jump >= N_JUMP_QUARANTINE,
    }


def criterion_text():
    """The reproducible-threshold description written into the audit sidecar."""
    return {
        "quarantine": f"n_jump(body OKLab stop-step > {JUMP_OKLAB}) >= {N_JUMP_QUARANTINE}",
        "seam_cyclic": (
            f"seam (endpoint OKLab distance) < {SEAM_CYCLIC} -> cyclic, "
            "else sequential(mirror_needed)"
        ),
        "note": "seam is a mirror flag for the render path, NOT a quarantine reason",
    }
