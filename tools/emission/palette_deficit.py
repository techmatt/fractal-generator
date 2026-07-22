"""palette_deficit.py — within-flavor palette pick that serves a running realized
chroma×hue DEFICIT, restoring the green / high-chroma / spectral variety the v3-gvo
argmax collapses.

Diagnosis that motivates this (out/first_release, 1387-loc whole-library colorize):

  * The within-flavor palette pick is v3-gvo argmax. It systematically leaves variety
    on the table: 64% of picks were >0.15 lower-chroma than an available flavor-mate
    (median headroom 0.206), and 86% passed over a >0.2-greener flavor-mate (median
    green headroom 0.392).
  * It is NOT a supply gap — the 987-palette pool holds 128 high-chroma (>0.5),
    205 green-present, 57 special:spectral, 181 rainbow (≥6 hue bins).
  * It is NOT recipe muting — realized mean_chroma p50 0.310 ≈ chosen-palette intrinsic
    p50 0.332 (chroma survives the canonical recipe).
  * The head GATES do not reject green/high-chroma — wallpaper corr(green,p_ge3)=-0.07≈0,
    corr(chroma,p_ge3)=0.00; mining corr(green,p_ge3)=+0.14 (green passes MORE often).

So the fix belongs at the pick: within a flavor, choose the concrete palette to fill the
running realized-color deficit (spread hue — incl. green — and chroma, nudged toward
rainbow), with v3-gvo quality as the within-deficit tiebreaker. Head floors are unchanged
and still gate quality after the render, so a deficit-serving pick that is genuinely poor
simply fails to gate — it never lowers the bar.

Signature convention MIRRORS `build_emission_diversity_v1.realized_palette_stats`: HSV
max−min chroma, RGB-wheel hue, 12 chroma-weighted hue bins + 8 chroma bins. Intrinsic
signatures are computed off each palette's sRGB LUT (the same stops the renderer bakes),
so a palette's intrinsic signature is a faithful proxy for the hue/chroma its render emits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

HUE_BINS, CHROMA_BINS = 12, 8
GREEN_BINS = (3, 4, 5)                 # RGB-wheel 90–180° (chartreuse→green→spring-green)


# --------------------------------------------------------------------------- #
# Intrinsic palette signature (same HSV convention as realized_palette_stats).
# --------------------------------------------------------------------------- #
def _hsv_signature(rgb: np.ndarray) -> dict:
    """rgb: (N,3) in [0,1] sampled along a palette -> intrinsic signature."""
    rgb = np.asarray(rgb, dtype=np.float64)
    mx = rgb.max(1); mn = rgb.min(1); chroma = mx - mn
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = np.where(mx == r, (g - b) / np.where(chroma == 0, 1, chroma), 0)
        gc = np.where(mx == g, 2.0 + (b - r) / np.where(chroma == 0, 1, chroma), 0)
        bc = np.where(mx == b, 4.0 + (r - g) / np.where(chroma == 0, 1, chroma), 0)
    h6 = np.where(mx == r, rc, np.where(mx == g, gc, bc))
    hue = (h6 / 6.0) % 1.0
    nz = chroma > 1e-6
    hh = np.histogram(hue[nz], bins=HUE_BINS, range=(0, 1), weights=chroma[nz])[0]
    hh = hh / (hh.sum() + 1e-9)
    ch = np.histogram(chroma, bins=CHROMA_BINS, range=(0, 1))[0]
    ch = ch / (ch.sum() + 1e-9)
    spread = float((hh > 0.02).sum()) / HUE_BINS      # rainbow-ness in [0,1]
    return {"hue": hh, "chroma": ch, "spread": spread,
            "mean_chroma": float(chroma.mean()), "green": float(hh[list(GREEN_BINS)].sum())}


def lut_from_stops(stops, n: int = 64) -> np.ndarray:
    xs = np.array([s[0] for s in stops], dtype=np.float64)
    cols = np.array([s[1] for s in stops], dtype=np.float64) / 255.0
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, xs, cols[:, k]) for k in range(3)], axis=1)


def signatures_from_lib(lib) -> dict:
    """{palette_name -> signature} over every loadable pool palette (from its stops)."""
    sigs = {}
    for name, cm in lib.colormaps.items():
        stops = cm.get("stops")
        if not stops:
            continue
        sigs[name] = _hsv_signature(lut_from_stops(stops))
    return sigs


# --------------------------------------------------------------------------- #
# Aspiration target (what a well-spread corpus should look like).
# --------------------------------------------------------------------------- #
def target_hue(green_boost: float = 1.6) -> np.ndarray:
    """Uniform hue with the green bins up-weighted — green is the deepest realized deficit,
    so a mild standing over-weight pulls the corpus back toward it without starving others."""
    w = np.ones(HUE_BINS)
    for b in GREEN_BINS:
        w[b] = green_boost
    return w / w.sum()


def target_chroma() -> np.ndarray:
    """Mid-high chroma aspiration — the realized corpus piles into the low-chroma bins."""
    w = np.array([0.5, 0.8, 1.0, 1.2, 1.3, 1.3, 1.0, 0.6])
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Running deficit tracker (resume-safe: rebuilt by replaying pool_log realized stats).
# --------------------------------------------------------------------------- #
@dataclass
class DeficitTracker:
    green_boost: float = 1.6
    wc: float = 1.0                    # chroma-deficit weight
    ws: float = 0.05                   # spread (rainbow) nudge weight
    H: np.ndarray = field(default_factory=lambda: np.zeros(HUE_BINS))
    C: np.ndarray = field(default_factory=lambda: np.zeros(CHROMA_BINS))
    n: int = 0

    def __post_init__(self):
        self._th = target_hue(self.green_boost)
        self._tc = target_chroma()

    def ingest(self, realized: dict | None):
        """Fold one render's realized_palette dict into the running histograms."""
        if realized and realized.get("hue_hist"):
            self.H = self.H + np.asarray(realized["hue_hist"], dtype=np.float64)
            self.C = self.C + np.asarray(realized["chroma_hist"], dtype=np.float64)
            self.n += 1

    def deficit(self) -> tuple[np.ndarray, np.ndarray]:
        Hn = self.H / self.n if self.n else np.zeros(HUE_BINS)
        Cn = self.C / self.n if self.n else np.zeros(CHROMA_BINS)
        return np.maximum(self._th - Hn, 0.0), np.maximum(self._tc - Cn, 0.0)

    def gain(self, sig: dict) -> float:
        """Deficit-fill gain of a candidate palette given the corpus so far. Non-uniform
        target ⇒ discriminates even at n=0 (empty start), so the very first picks already
        favor green/high-chroma; as those bins fill their deficit falls and the pull moves on."""
        dh, dc = self.deficit()
        return float(dh @ sig["hue"] + self.wc * (dc @ sig["chroma"]) + self.ws * sig["spread"])


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def pick(members: list[str], pref_scores, sigs: dict, tracker: DeficitTracker,
         lam: float = 1.5, pref_z_floor: float = -1.5) -> int:
    """Index into `members` of the deficit-serving pick.

    obj = z(pref) + lam · z(deficit_gain), argmax over members whose pref z-score clears
    `pref_z_floor` (drops only clear quality losers — ~bottom 7% of the flavor by v3-gvo —
    so green/high-chroma mid-pref members stay in contention). v3-gvo is the within-deficit
    tiebreaker: with a flat deficit the gain z-scores are ~0 and pref decides, recovering the
    argmax baseline. When pref is unavailable (None), pick pure deficit gain."""
    gains = np.array([tracker.gain(sigs[m]) if m in sigs else 0.0 for m in members])
    if pref_scores is None:
        return int(np.argmax(gains))
    pref = np.asarray(pref_scores, dtype=np.float64)
    zp, zg = _z(pref), _z(gains)
    obj = zp + lam * zg
    allowed = zp >= pref_z_floor - 1e-9            # epsilon: keep the exactly-N-std member
    if not allowed.any():
        allowed = np.ones_like(zp, dtype=bool)
    obj = np.where(allowed, obj, -np.inf)
    return int(np.argmax(obj))
