#!/usr/bin/env python
"""Mandelbrot + Julia HTML explorer (v1).

Mandelbrot left, Julia right. The Julia parameter `c` tracks the current
Mandelbrot frame's center. Pure Flask + HTML/JS; renders by shelling out to the
`fractal-generator render-one` binary. Server-authoritative state, single
user. All coordinate math in Decimal (never round-tripped through JS floats).

Run:  uv run python tools/explorer/app.py
"""
import base64
import hashlib
import subprocess
import sys
import tempfile
from decimal import Decimal, getcontext
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Plenty of guard digits for deep-zoom decimal coordinate math.
getcontext().prec = 60

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
# Canonical binary: the v4 precompute is done and the sidecar was folded back
# into the normal release build (Julia + v4-render-batch now live here too, and
# the Mandelbrot/Julia kernels are byte-identical to the retired sidecar).
RENDER_BIN = REPO_ROOT / "target" / "release" / "fractal-generator.exe"
COLORMAPS = REPO_ROOT / "data" / "palettes" / "clean_colormaps.json"

# Navigation render (fast, aliased — quality irrelevant while navigating).
NAV_W, NAV_H = 700, 394
NAV_SS = 1

# Full-quality wallpaper render (render-one locked defaults).
FULL_W, FULL_H = 2560, 1440
FULL_SS = 4

DEFAULT_PALETTE = "twilight_shifted"
DEFAULT_ZOOM = 2

# Home frames.
M_HOME = (Decimal("-0.5"), Decimal("0"), Decimal("3.0"))
J_HOME = (Decimal("0"), Decimal("0"), Decimal("3.0"))

# maxiter escalation: maxiter = base * (1 + k * log2(fw_home / fw_current)).
MAXITER_BASE = 500
MAXITER_K = 0.30
MAXITER_MIN = 200
MAXITER_MAX = 8000

TMPDIR = Path(tempfile.gettempdir()) / "fractal_explorer"
TMPDIR.mkdir(exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Server-authoritative state
# ---------------------------------------------------------------------------
# Cursor model: each panel is an ordered frame list + a current-index pointer
# (instead of a stack whose tail is "current"). Breadcrumb clicks move the index
# without truncating; only a fresh descend truncates the forward path.
STATE = {
    "m_frames": [M_HOME],           # list of (cx, cy, fw)
    "m_index": 0,                   # currently-displayed Mandelbrot level
    "j_frames": [J_HOME],           # list of (jx, jy, jfw) at current c
    "j_index": 0,                   # currently-displayed Julia level
    "palette": DEFAULT_PALETTE,
    "zoom": DEFAULT_ZOOM,
    "maxiter_override": None,       # None => depth-aware auto
}

_render_cache = {}  # param-hash -> data URL


def m_frame():
    return STATE["m_frames"][STATE["m_index"]]


def j_frame():
    return STATE["j_frames"][STATE["j_index"]]


def current_c():
    cx, cy, _ = m_frame()
    return (cx, cy)


# ---------------------------------------------------------------------------
# Palette catalog
# ---------------------------------------------------------------------------
def load_palette_names():
    import json
    data = json.loads(COLORMAPS.read_text())
    return [c["name"] for c in data]


PALETTES = load_palette_names()


# ---------------------------------------------------------------------------
# Coordinate math (Decimal)
# ---------------------------------------------------------------------------
def click_to_world(px, py, ctr_x, ctr_y, fw, w=NAV_W, h=NAV_H):
    """Pixel (top-left origin) on a panel -> complex-plane point (Decimal)."""
    fw = Decimal(fw)
    W, H = Decimal(w), Decimal(h)
    fh = fw * H / W
    fx = Decimal(px) / W - Decimal("0.5")
    fy = Decimal(py) / H - Decimal("0.5")
    world_x = Decimal(ctr_x) + fx * fw
    world_y = Decimal(ctr_y) - fy * fh   # screen-y down, imaginary up
    return world_x, world_y


# ---------------------------------------------------------------------------
# maxiter (depth-aware)
# ---------------------------------------------------------------------------
def auto_maxiter(fw):
    if STATE["maxiter_override"] is not None:
        return STATE["maxiter_override"]
    fw = Decimal(fw)
    fw_home = M_HOME[2]
    ratio = fw_home / fw if fw > 0 else Decimal(1)
    # log2(ratio)
    import math
    lz = math.log2(float(ratio)) if ratio > 0 else 0.0
    val = MAXITER_BASE * (1.0 + MAXITER_K * lz)
    return int(max(MAXITER_MIN, min(MAXITER_MAX, val)))


# ---------------------------------------------------------------------------
# Render (shell out to the render-one binary)
# ---------------------------------------------------------------------------
def _dec(x):
    """Decimal -> plain decimal string (no scientific notation)."""
    return format(Decimal(x), "f")


def render(panel, ctr_x, ctr_y, fw, maxiter, w, h, ss, julia_c=None):
    """Render one frame, return base64 data URL. Cached by full param set."""
    palette = STATE["palette"]
    key_src = (
        panel, _dec(ctr_x), _dec(ctr_y), _dec(fw), maxiter, w, h, ss, palette,
        _dec(julia_c[0]) if julia_c else None,
        _dec(julia_c[1]) if julia_c else None,
    )
    key = hashlib.sha1(repr(key_src).encode()).hexdigest()
    if key in _render_cache:
        return _render_cache[key]

    out = TMPDIR / f"{key}.png"
    cmd = [
        str(RENDER_BIN), "render-one",
        "--cx", _dec(ctr_x), "--cy", _dec(ctr_y), "--fw", _dec(fw),
        "--width", str(w), "--height", str(h), "--supersample", str(ss),
        "--palette", palette, "--colormaps", str(COLORMAPS),
        "--maxiter", str(maxiter), "--out", str(out),
    ]
    if julia_c is not None:
        cmd += ["--julia", "--c", _dec(julia_c[0]), _dec(julia_c[1])]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"render failed: {proc.stderr or proc.stdout}")

    b64 = base64.b64encode(out.read_bytes()).decode()
    url = f"data:image/png;base64,{b64}"
    _render_cache[key] = url
    return url


def render_mandelbrot():
    cx, cy, fw = m_frame()
    return render("M", cx, cy, fw, auto_maxiter(fw), NAV_W, NAV_H, NAV_SS)


def render_julia():
    jx, jy, jfw = j_frame()
    return render("J", jx, jy, jfw, auto_maxiter(jfw), NAV_W, NAV_H, NAV_SS,
                  julia_c=current_c())


# ---------------------------------------------------------------------------
# c membership classification
# ---------------------------------------------------------------------------
def classify_c(cx, cy, maxiter=1000):
    """Iterate c under z->z^2+c. Classify inside / dust / boundary."""
    cre, cim = float(cx), float(cy)
    zr, zi = 0.0, 0.0
    escaped_at = None
    for n in range(maxiter):
        zr2, zi2 = zr * zr, zi * zi
        if zr2 + zi2 > 4.0:
            escaped_at = n
            break
        zi = 2.0 * zr * zi + cim
        zr = zr2 - zi2 + cre
    if escaped_at is None:
        return "inside M"
    # Escaped: slow escape (deep n) => near boundary; fast => dust.
    if escaped_at > 80:
        return "near boundary"
    return "outside (dust)"


# ---------------------------------------------------------------------------
# State serialization for the client
# ---------------------------------------------------------------------------
def coord_str(frame):
    x, y, fw = frame
    return {"x": _dec(x), "y": _dec(y), "fw": f"{float(fw):.6e}"}


def breadcrumbs(frames):
    out = []
    for i, fr in enumerate(frames):
        label = "home" if i == 0 else f"L{i}"
        out.append({"level": i, "label": label})
    return out


def state_payload(m_img=None, j_img=None):
    cx, cy = current_c()
    return {
        "m_img": m_img,
        "j_img": j_img,
        "m_coord": coord_str(m_frame()),
        "j_coord": coord_str(j_frame()),
        "m_breadcrumbs": breadcrumbs(STATE["m_frames"]),
        "j_breadcrumbs": breadcrumbs(STATE["j_frames"]),
        "m_index": STATE["m_index"],
        "j_index": STATE["j_index"],
        "c": {"re": _dec(cx), "im": _dec(cy),
              "membership": classify_c(cx, cy)},
        "palette": STATE["palette"],
        "zoom": STATE["zoom"],
        "maxiter": auto_maxiter(m_frame()[2]),
        "maxiter_override": STATE["maxiter_override"],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", palettes=PALETTES,
                           default_palette=DEFAULT_PALETTE)


@app.route("/initial", methods=["POST"])
def initial():
    return jsonify(state_payload(render_mandelbrot(), render_julia()))


@app.route("/click", methods=["POST"])
def click():
    d = request.get_json()
    panel = d["panel"]
    px, py = float(d["px"]), float(d["py"])
    zoom = Decimal(str(d.get("zoom", 1)))  # 1 = recenter only

    if panel == "M":
        cx, cy, fw = m_frame()
        wx, wy = click_to_world(px, py, cx, cy, fw)
        new_fw = fw / zoom
        # Descend anew: truncate the forward path, then push + advance index.
        i = STATE["m_index"]
        STATE["m_frames"] = STATE["m_frames"][: i + 1] + [(wx, wy, new_fw)]
        STATE["m_index"] = len(STATE["m_frames"]) - 1
        # c becomes the new M center -> reset Julia to home at new c.
        STATE["j_frames"] = [J_HOME]
        STATE["j_index"] = 0
        return jsonify(state_payload(render_mandelbrot(), render_julia()))
    else:  # Julia panel
        jx, jy, jfw = j_frame()
        wx, wy = click_to_world(px, py, jx, jy, jfw)
        new_fw = jfw / zoom
        i = STATE["j_index"]
        STATE["j_frames"] = STATE["j_frames"][: i + 1] + [(wx, wy, new_fw)]
        STATE["j_index"] = len(STATE["j_frames"]) - 1
        return jsonify(state_payload(j_img=render_julia()))


@app.route("/breadcrumb", methods=["POST"])
def breadcrumb():
    d = request.get_json()
    panel, level = d["panel"], int(d["level"])
    if panel == "M":
        # Move to the level, preserving deeper frames (forward path intact).
        STATE["m_index"] = max(0, min(level, len(STATE["m_frames"]) - 1))
        # c changes to frames[level]'s center -> reset Julia to home at new c.
        STATE["j_frames"] = [J_HOME]
        STATE["j_index"] = 0
        return jsonify(state_payload(render_mandelbrot(), render_julia()))
    else:
        STATE["j_index"] = max(0, min(level, len(STATE["j_frames"]) - 1))
        return jsonify(state_payload(j_img=render_julia()))


@app.route("/palette", methods=["POST"])
def palette():
    STATE["palette"] = request.get_json()["palette"]
    return jsonify(state_payload(render_mandelbrot(), render_julia()))


@app.route("/zoom", methods=["POST"])
def zoom():
    STATE["zoom"] = int(request.get_json()["zoom"])
    return jsonify(state_payload())


@app.route("/maxiter", methods=["POST"])
def maxiter():
    v = request.get_json()["value"]
    STATE["maxiter_override"] = None if v in (None, "", "auto") else int(v)
    return jsonify(state_payload(render_mandelbrot(), render_julia()))


@app.route("/reset", methods=["POST"])
def reset():
    STATE["m_frames"] = [M_HOME]
    STATE["m_index"] = 0
    STATE["j_frames"] = [J_HOME]
    STATE["j_index"] = 0
    STATE["maxiter_override"] = None
    return jsonify(state_payload(render_mandelbrot(), render_julia()))


@app.route("/fullres", methods=["POST"])
def fullres():
    panel = request.get_json()["panel"]
    if panel == "M":
        cx, cy, fw = m_frame()
        url = render("M_full", cx, cy, fw, auto_maxiter(fw),
                     FULL_W, FULL_H, FULL_SS)
    else:
        jx, jy, jfw = j_frame()
        url = render("J_full", jx, jy, jfw, auto_maxiter(jfw),
                     FULL_W, FULL_H, FULL_SS, julia_c=current_c())
    return jsonify({"img": url})


def smoke_test():
    """Programmatic round-trip through the render binary for each panel."""
    print("Smoke test: rendering Mandelbrot + Julia via render binary...")
    m = render_mandelbrot()
    j = render_julia()
    assert m.startswith("data:image/png;base64,") and len(m) > 1000
    assert j.startswith("data:image/png;base64,") and len(j) > 1000
    print(f"  Mandelbrot OK ({len(m)} bytes b64)")
    print(f"  Julia OK      ({len(j)} bytes b64)")
    print("Smoke test PASSED.")


if __name__ == "__main__":
    if not RENDER_BIN.exists():
        sys.exit(f"render binary not found: {RENDER_BIN}")
    smoke_test()
    print(f"\nRender binary: {RENDER_BIN}")
    print("Explorer running at: http://127.0.0.1:5005\n")
    app.run(host="127.0.0.1", port=5005, debug=False, threaded=True)
