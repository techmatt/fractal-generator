"""Canonical location identity + general family-params slot + render-one arg builder.

The version-invariant location representation shared by the corpus pool, the field
cache, reframing, and every `render-one` invocation. A *location* is:

    family + primary constant c=(c_re,c_im) + viewport (cx,cy,fw) + family_params

where `family_params` is the **general per-family extra-constant slot** — empty for
`mandelbrot`, `julia`, `multibrot3/4/5`; `{p_re,p_im}` for `phoenix`. The slot exists
NOW (only Phoenix's `p` occupies it) so a future `(c,p)` sweep or the deferred
multi-Julia variants never force a corpus/cache **re-key**.

Degree stays in the family NAME (`multibrot3/4/5`) — Python treats `family` as an
opaque string; degree is Rust's concern. There is no numeric degree field.

This module owns:
  * FAMILY_PARAM_KEYS   — per-family extra-constant registry (canonical order).
  * Location            — the canonical dataclass (+ `.kind` alias, `.key()`).
  * location_key(loc)   — the stable identity string. Duck-typed: works on a
                          `Location` OR the coloring path's `colormap.LocationRef`.
                          For empty-params families the appended part is empty, so
                          existing mandelbrot/julia keys serialize byte-identically.
  * render_one_flags    — the ONE `location -> render-one CLI flags` builder. Every
                          render-one call site routes through this (no ad-hoc flags).
  * from_render_block / from_sidecar — parse a corpus `images.jsonl` render block or
                          a `--dump-field` sidecar into a Location (family-conditional
                          p_re/p_im).
  * to_location_ref     — canonical Location -> colormap.LocationRef, for the coloring
                          recipe (family_params are dropped: the dumped field already
                          bakes the family's dynamics, so coloring is fractal-agnostic).

It deliberately does NOT touch `colormap.py`: `LocationRef` stays the recipe's
location type, and `family_params` never enter a `CandidateConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Per-family extra-constant registry. Default (unlisted family) -> no extra params.
# Values are the canonical field order the key/sidecar/flags all follow.
# ---------------------------------------------------------------------------
FAMILY_PARAM_KEYS = {
    "phoenix": ("p_re", "p_im"),
}

# Parameter-plane z^d families whose flag is just `--family <name>`.
_MULTIBROT = ("multibrot3", "multibrot4", "multibrot5")

# Dynamical Julia-multibrot families (`--julia --family multibrot{d}`, fixed c). The
# degree stays in the family NAME (mirrors _MULTIBROT); each maps to the parameter-
# plane `--family` value its `--julia` twin reuses. Kept in sync with Rust
# `Family::kind_str` (`julia_multibrot{d}`).
_JULIA_MULTIBROT = {
    "julia_multibrot3": "multibrot3",
    "julia_multibrot4": "multibrot4",
    "julia_multibrot5": "multibrot5",
}


def family_param_keys(family: str) -> tuple:
    """Canonical-ordered extra-constant keys for `family` (empty tuple if none)."""
    return FAMILY_PARAM_KEYS.get(family, ())


# ---------------------------------------------------------------------------
# Duck-typed accessors — work on a Location OR a colormap.LocationRef.
# ---------------------------------------------------------------------------
def family_of(loc) -> str:
    """The family string, whether `loc` carries it as `.family` (Location) or `.kind`
    (colormap.LocationRef / reframe Location namedtuple both expose one of these)."""
    fam = getattr(loc, "family", None)
    if fam is not None:
        return fam
    return loc.kind


def params_of(loc) -> dict:
    """`{key: value}` extra-constants of `loc` (empty for a plain LocationRef)."""
    fp = getattr(loc, "family_params", ()) or ()
    return dict(fp) if not isinstance(fp, dict) else dict(fp)


# ---------------------------------------------------------------------------
# Canonical location.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Location:
    """Canonical, version-invariant location. Coords are decimal strings (an f64 is
    meaningless at deep zoom). `family_params` is normalized to a canonical-order
    tuple of `(key, value_str)` pairs (so the dataclass stays hashable) — pass a dict
    or a tuple of pairs; either is normalized against the family registry."""
    family: str
    cx: str
    cy: str
    fw: str
    maxiter: int = 0
    c_re: Optional[str] = None
    c_im: Optional[str] = None
    family_params: tuple = ()

    def __post_init__(self):
        keys = family_param_keys(self.family)
        src = dict(self.family_params) if not isinstance(self.family_params, dict) \
            else dict(self.family_params)
        fp = tuple((k, str(src[k])) for k in keys if src.get(k) is not None)
        object.__setattr__(self, "family_params", fp)

    # `.kind` alias so canonical Locations are drop-in wherever LocationRef.kind is read.
    @property
    def kind(self) -> str:
        return self.family

    @property
    def params(self) -> dict:
        return dict(self.family_params)

    def param(self, key):
        return dict(self.family_params).get(key)

    def key(self) -> str:
        return location_key(self)


# ---------------------------------------------------------------------------
# Stable identity key. Fixed per-family field order; family_params appended in
# canonical order. Empty-params families append nothing -> byte-identical to the
# pre-slot key for mandelbrot/julia (the hard stability gate).
# ---------------------------------------------------------------------------
def location_key(loc) -> str:
    fam = family_of(loc)
    p = params_of(loc)
    parts = [
        fam,
        str(loc.cx), str(loc.cy), str(loc.fw),
        "" if loc.c_re is None else str(loc.c_re),
        "" if loc.c_im is None else str(loc.c_im),
    ]
    for k in family_param_keys(fam):
        v = p.get(k)
        parts.append("" if v is None else str(v))
    return "|".join(parts)


# ---------------------------------------------------------------------------
# The ONE render-one flag builder. Five cases (Step 3 of the prompt):
#   mandelbrot   -> --family mandelbrot
#   julia        -> --family mandelbrot --julia --c <c_re> <c_im>   (unchanged mechanism)
#   multibrot3/4/5 -> --family multibrotN
#   phoenix      -> --family phoenix --c <c_re> <c_im> --p <p_re> <p_im>
# Phoenix's constants are emitted only when present, so the acceptance harness's
# "test the Rust default" entries (no --c/--p) also route through here unchanged.
# ---------------------------------------------------------------------------
def render_one_flags(loc) -> list:
    fam = family_of(loc)
    p = params_of(loc)
    if fam == "mandelbrot":
        return ["--family", "mandelbrot"]
    if fam == "julia":
        if loc.c_re is None or loc.c_im is None:
            raise ValueError("julia location requires c_re/c_im")
        return ["--family", "mandelbrot", "--julia", "--c", str(loc.c_re), str(loc.c_im)]
    if fam in _JULIA_MULTIBROT:
        # Dynamical z^d+c at the fixed parameter c: the multibrot degree in --family,
        # flipped to its dynamical twin by --julia.
        if loc.c_re is None or loc.c_im is None:
            raise ValueError(f"{fam} location requires c_re/c_im")
        return ["--family", _JULIA_MULTIBROT[fam], "--julia", "--c", str(loc.c_re), str(loc.c_im)]
    if fam in _MULTIBROT:
        return ["--family", fam]
    if fam == "phoenix":
        flags = ["--family", "phoenix"]
        if loc.c_re is not None and loc.c_im is not None:
            flags += ["--c", str(loc.c_re), str(loc.c_im)]
        if p.get("p_re") is not None and p.get("p_im") is not None:
            flags += ["--p", str(p["p_re"]), str(p["p_im"])]
        return flags
    raise ValueError(f"unknown family {fam!r}")


# ---------------------------------------------------------------------------
# Parsers — corpus render block / dump-field sidecar -> Location.
# ---------------------------------------------------------------------------
def from_render_block(render: dict, *, maxiter=None) -> Location:
    """A corpus `images.jsonl` render block -> Location. Family is `fractal_type`
    (mandelbrot when absent); `family_params` are read family-conditionally."""
    fam = render.get("fractal_type") or render.get("family") or "mandelbrot"
    fp = {k: render[k] for k in family_param_keys(fam) if render.get(k) is not None}
    mi = render.get("maxiter", maxiter)
    return Location(
        family=fam, cx=render["cx"], cy=render["cy"], fw=render["fw"],
        maxiter=int(mi) if mi is not None else 0,
        c_re=render.get("c_re"), c_im=render.get("c_im"),
        family_params=fp,
    )


def from_sidecar(meta: dict) -> Location:
    """A `--dump-field` JSON sidecar (`meta["location"]`) -> Location. `kind` is the
    family; Phoenix's `p_re/p_im` are recovered here (family-conditional)."""
    loc = meta["location"]
    fam = loc.get("kind") or loc.get("family") or "mandelbrot"
    fp = {k: loc[k] for k in family_param_keys(fam) if loc.get(k) is not None}
    return Location(
        family=fam, cx=loc["cx"], cy=loc["cy"], fw=loc["fw"],
        maxiter=int(loc.get("maxiter", 0)),
        c_re=loc.get("c_re"), c_im=loc.get("c_im"),
        family_params=fp,
    )


# ---------------------------------------------------------------------------
# Coloring bridge — canonical Location -> colormap.LocationRef (recipe type).
# family_params are dropped: the dumped field already bakes the family's dynamics.
# ---------------------------------------------------------------------------
def to_location_ref(loc):
    """Return `loc` unchanged if it is already a `colormap.LocationRef`, else build one
    from a canonical Location (or any duck-typed location). Imported lazily so this
    module has no hard dependency on the coloring tail."""
    import colormap as cm
    if isinstance(loc, cm.LocationRef):
        return loc
    return cm.LocationRef(
        kind=family_of(loc), cx=loc.cx, cy=loc.cy, fw=loc.fw,
        maxiter=int(getattr(loc, "maxiter", 0) or 0),
        c_re=loc.c_re, c_im=loc.c_im,
    )
