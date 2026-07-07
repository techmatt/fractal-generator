#!/usr/bin/env python3
"""
grid_runner.py -- palette grid-runner (v3.1, API generation, design A: local validation).

    ############################################################################
    ## NOT CURRENTLY ACTIVE -- DO NOT USE.                                     ##
    ## The API-driven generation path is parked; palettes are not being        ##
    ## generated via the Anthropic API for now. The pipeline is complete and   ##
    ## smoke-tested (grid/skip/prompt-fill/parse/validate) but no live run has  ##
    ## been made. Requires ANTHROPIC_API_KEY if ever revived.                   ##
    ############################################################################

Generates results/{mood}_{cband}_{ver}.json by calling the Anthropic API across the
mood x complexity grid. Each cell:
  1. sample 3 random calibration images (artist-fractals/*.jpg), base64 them;
  2. fill the v3.1 prompt's Run-conditioning block for this cell -> system prompt;
  3. call the API (default claude-opus-4-8, max_tokens ~16k); parse the LAST ```json block;
  4. validate LOCALLY, in-process, via validate_palettes.validate_batch (never shell out);
  5. optional --fix-retries N loop: feed back ONLY the errors, re-call, re-validate;
  6. write the final array to results/<cell>.json regardless of remaining errors;
  7. append a run-log line to results/_runlog.jsonl.

Cells whose results/ file already exists are skipped. --limit N caps new cells.

Auth: ANTHROPIC_API_KEY in the environment.

Run:
  uv run python grid_runner.py --limit 1 --fix-retries 0     # first inspection run
  uv run python grid_runner.py --dry-run                     # print cells, no API
"""

import argparse, base64, datetime, glob, json, os, random, re, sys
from pathlib import Path

import validate_palettes as vp

HERE = Path(__file__).resolve().parent
PROMPT_PATH = HERE / "dramatic-palette-generator-prompt-v3.1.md"
IMAGES_DIR = HERE / "artist-fractals"
RESULTS_DIR = HERE / "results"
RUNLOG_PATH = RESULTS_DIR / "_runlog.jsonl"

PROMPT_VERSION = "v3.1"
VER = "v3_1"
DEFAULT_MODEL = "claude-opus-4-8"
MAX_TOKENS = 16000
N_CAL_IMAGES = 3

# cband token -> (MOOD-facing "COMPLEXITY BAND" text, human band label)
CBANDS = {
    "c1-2": "1-2",
    "c3-4": "3-4",
    "c5": "5",
    "c6-ultra": "6-ultra",
}

MOODS = [
    "fire-ice", "jewel-earth", "atmospheric-deep", "antique-faded",
    "high-key-luminous", "pastel-iridescent", "autumn-ember", "oceanic",
    "orchid-twilight", "verdigris-copper", "ember-in-ash", "tonal-restrained",
]

# c5 subset per the task spec.
C5_MOODS = ["fire-ice", "jewel-earth", "orchid-twilight", "oceanic"]


def build_grid():
    """Ordered config list of (mood, cband). All 12 moods at c3-4, then the c5 subset,
    then c1-2 and c6-ultra for all moods (the 'later' bands)."""
    cells = []
    for m in MOODS:
        cells.append((m, "c3-4"))
    for m in C5_MOODS:
        cells.append((m, "c5"))
    for m in MOODS:
        cells.append((m, "c1-2"))
    for m in MOODS:
        cells.append((m, "c6-ultra"))
    return cells


def cell_filename(mood, cband):
    return f"{mood}_{cband}_{VER}.json"


def batch_size_for(cband):
    # 6-ultra emits ~6 by design regardless of BATCH SIZE.
    return 6 if cband == "c6-ultra" else 20


# ---- Run-conditioning block fill -------------------------------------------
# The v3.1 prompt has a fenced block of four lines under "## Run conditioning".
# Regex-replace the four value lines (BATCH SIZE / MOOD FAMILY / COMPLEXITY BAND / VALUE KEY),
# preserving any trailing "# comment".
def _replace_line(text, key, value):
    # matches e.g.  "MOOD FAMILY:     fire-ice        # comment"
    pat = re.compile(rf"^(?P<lead>{re.escape(key)}:\s*)(?P<val>\S+)(?P<rest>.*)$", re.MULTILINE)
    n = [0]
    def sub(m):
        n[0] += 1
        return f"{m.group('lead')}{value}{m.group('rest')}"
    out = pat.sub(sub, text)
    if n[0] != 1:
        raise RuntimeError(f"Run-conditioning fill: expected exactly 1 match for {key!r}, got {n[0]}")
    return out


def fill_prompt(prompt_text, mood, cband, value_key="span"):
    t = prompt_text
    t = _replace_line(t, "BATCH SIZE", str(batch_size_for(cband)))
    t = _replace_line(t, "MOOD FAMILY", mood)
    t = _replace_line(t, "COMPLEXITY BAND", CBANDS[cband])
    t = _replace_line(t, "VALUE KEY", value_key)
    return t


# ---- image sampling ---------------------------------------------------------
def sample_images(rng, n=N_CAL_IMAGES):
    paths = sorted(glob.glob(str(IMAGES_DIR / "*.jpg")))
    if len(paths) < n:
        raise RuntimeError(f"need >={n} calibration images in {IMAGES_DIR}, found {len(paths)}")
    chosen = rng.sample(paths, n)
    blocks, names = [], []
    for p in chosen:
        data = Path(p).read_bytes()
        b64 = base64.standard_b64encode(data).decode("ascii")
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
        names.append(Path(p).name)
    return blocks, names


# ---- response parsing -------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)

def parse_last_json_array(text):
    """Return the array parsed from the LAST ```json fenced block."""
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        raise RuntimeError("no fenced code block found in model reply")
    raw = blocks[-1].strip()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("palettes", data.get("data", [data]))
    if not isinstance(data, list):
        raise RuntimeError(f"parsed JSON is not an array (got {type(data).__name__})")
    return data


def errors_only(results):
    """Flatten validate_batch output into a list of 'name: error' strings (errors only)."""
    lines = []
    for r in results:
        for e in r["errors"]:
            lines.append(f"[{r['name']}] {e}")
    return lines


def total_errors(results):
    return sum(len(r["errors"]) for r in results)


def total_warnings(results):
    return sum(len(r["warnings"]) for r in results)


# ---- the per-cell pipeline --------------------------------------------------
def run_cell(client, prompt_text, mood, cband, rng, fix_retries, model, verbose=True):
    system = fill_prompt(prompt_text, mood, cband)
    image_blocks, image_names = sample_images(rng)

    user_content = image_blocks + [
        {"type": "text", "text": "These three images are calibration only. Generate the batch now."}
    ]
    messages = [{"role": "user", "content": user_content}]

    def call():
        resp = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=system, messages=messages,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    if verbose:
        print(f"  [api] calling {model} (max_tokens={MAX_TOKENS}) ...", flush=True)
    reply = call()
    palettes = parse_last_json_array(reply)
    results = vp.validate_batch(palettes)
    retries_used = 0

    while total_errors(results) > 0 and retries_used < fix_retries:
        errs = errors_only(results)
        if verbose:
            print(f"  [fix] {len(errs)} error(s); retry {retries_used + 1}/{fix_retries}", flush=True)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content":
            "validate_palettes.py reports these ERRORS:\n" + "\n".join(errs) +
            "\n\nReturn the corrected full array as a single ```json block — fix exactly these, "
            "leave valid palettes unchanged."})
        reply = call()
        palettes = parse_last_json_array(reply)
        results = vp.validate_batch(palettes)
        retries_used += 1

    return palettes, results, image_names, retries_used


def write_outputs(mood, cband, palettes, results, image_names, retries_used, model):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = cell_filename(mood, cband)
    out_path = RESULTS_DIR / fname
    out_path.write_text(json.dumps(palettes, indent=2, ensure_ascii=False), encoding="utf-8")

    spread = vp.batch_spread(palettes)
    remaining = errors_only(results)
    logrec = {
        "cell": f"{mood}_{cband}",
        "filename": fname,
        "model": model,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "images": image_names,
        "n_palettes": len(palettes),
        "retries_used": retries_used,
        "remaining_errors": len(remaining),
        "error_reasons": remaining,
        "warning_count": total_warnings(results),
        "spread": {
            "architecture": spread["architecture"],
            "skeleton": spread["skeleton"],
            "value_key": spread["value_key"],
            "complexity": spread["complexity"],
            "vivid": spread["vivid"],
            "muted": spread["muted"],
        },
    }
    with RUNLOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(logrec, ensure_ascii=False) + "\n")
    return out_path, spread


def print_validator_report(palettes, results):
    nerr, nwarn = total_errors(results), total_warnings(results)
    print(f"\n=== validator: {len(palettes)} palettes | {nerr} error(s), {nwarn} warning(s) ===")
    for r in results:
        if r["errors"] or r["warnings"]:
            print(f"\n[{r['name']}]  ({r['architecture']} / {r['skeleton']})")
            for x in r["errors"]:   print(f"   ERROR  {x}")
            for x in r["warnings"]: print(f"   warn   {x}")
    sp = vp.batch_spread(palettes)
    print("\n--- batch spread ---")
    print("  architecture:", sp["architecture"])
    print("  skeleton    :", sp["skeleton"])
    print("  value_key   :", sp["value_key"])
    print("  complexity  :", sp["complexity"])
    print(f"  chroma      : {sp['vivid']} vivid (peakC>={vp.VIVID_PEAKC}), "
          f"{sp['muted']} muted (peakC<{vp.MUTED_PEAKC})")


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Palette grid-runner (v3.1, API generation, local validation)")
    ap.add_argument("--limit", type=int, default=None, help="max new cells to generate (default: all missing)")
    ap.add_argument("--fix-retries", type=int, default=0, help="max error-feedback retries per cell (default 0)")
    ap.add_argument("--dry-run", action="store_true", help="print cells that would run; no API call")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for image sampling (default: nondeterministic)")
    args = ap.parse_args()

    grid = build_grid()
    missing = [(m, c) for (m, c) in grid if not (RESULTS_DIR / cell_filename(m, c)).exists()]

    if args.limit is not None:
        todo = missing[:args.limit]
    else:
        todo = missing

    print(f"grid: {len(grid)} cells | existing: {len(grid) - len(missing)} | missing: {len(missing)} | "
          f"this run: {len(todo)}")

    if args.dry_run:
        print("\n[dry-run] cells that would run:")
        for m, c in todo:
            print(f"  {m}_{c}  (batch {batch_size_for(c)}) -> {cell_filename(m, c)}")
        if not todo:
            print("  (none — all target cells already exist)")
        return 0

    if not todo:
        print("nothing to do — all target cells already exist.")
        return 0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment.", file=sys.stderr)
        return 2

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    rng = random.Random(args.seed)

    for i, (mood, cband) in enumerate(todo, 1):
        print(f"\n--- cell {i}/{len(todo)}: {mood}_{cband} ---", flush=True)
        palettes, results, image_names, retries_used = run_cell(
            client, prompt_text, mood, cband, rng, args.fix_retries, args.model)
        out_path, _ = write_outputs(mood, cband, palettes, results, image_names, retries_used, args.model)
        print_validator_report(palettes, results)
        clean = total_errors(results) == 0
        print(f"\n  images: {image_names}")
        print(f"  retries used: {retries_used}   zero-errors: {clean}")
        print(f"  wrote: {out_path}")
        print(f"  logged: {RUNLOG_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
