#!/usr/bin/env python
"""Build the worked-instance library_records.jsonl for the 47 curated locations.

Design-first validation: populate the proposed location-library record schema
(see docs/findings/library_record_schema.md) entirely from existing artifacts. No new
producers. Every field is sourced from one of:

  emit/curated/manifest.jsonl   identity, emitted coloring + gate, wallpaper qual
  pools/<cycle>/images.jsonl    palette-candidate beam (K siblings) + prov->oid
  fresh_runs/.../outcome_ledger location-potential (v6 k3/p_good/decoded_class)
  data/palettes/palette_categories.json   color_category (nested k=8/12/16)
  out/mining/deploy_tail/report.json      mode-candidacy
  data/library_embeddings/embeddings.npz  descriptor reference (morph_clip/v6, by uid)

Gaps (fields with no producer) are emitted as null with a comment in the schema.

Promoted out of scratchpad (2026-07-15 hygiene pass) — it is the sole producer of
data/library/library_records.jsonl, which the colored_clip curation tools consume.
NOTE: EMB below points at the promoted base store, whose key names are
morph_uids/morph_clip/morph_v6 (the wiped gray embeddings.npz used uids/clip/v6); the
`load()` reader keys off `uids` and would need remapping if this producer is re-run.
"""
import json, os

ROOT = "out/wallpaper/overnight/overnight_20260713_001420"
LEDGER = "data/discovery/fresh_runs/overnight_20260713_001420/outcome_ledger.jsonl"
GATHER = ["phoenix", "mandelbrot", "multibrot3", "multibrot4", "multibrot5"]
PC = "data/palettes/palette_categories.json"
DT = "out/mining/deploy_tail/report.json"
EMB = "data/library_embeddings/embeddings.npz"
OUT = "data/library/library_records.jsonl"

# Fixed Ushiki phoenix constants (src/v4_cache.rs:44-45) -- implicit per-record.
PHOENIX_C = {"re": "0.5667", "im": "0.0"}
PHOENIX_P = {"re": "-0.5", "im": "0.0"}


def load():
    cur = [json.loads(l) for l in open(f"{ROOT}/emit/curated/manifest.jsonl")]
    pools = {}
    for c in range(1, 6):
        cyc = f"cycle_00{c}"
        for l in open(f"{ROOT}/pools/{cyc}/images.jsonl"):
            r = json.loads(l)
            pools[(cyc, r["image_id"])] = r
    led = {}
    for l in open(LEDGER):
        r = json.loads(l)
        led[r["id"]] = r
    for fam in GATHER:
        p = f"data/discovery/gather/{fam}/outcome_ledger.jsonl"
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                led.setdefault(r["id"], r)
    pc = json.load(open(PC))["palettes"]
    dt = {r["loc_id"]: r for r in json.load(open(DT))["per_location"]}
    import numpy as np
    e = np.load(EMB)
    uidx = {u: i for i, u in enumerate(e["uids"].tolist())}
    return cur, pools, led, pc, dt, uidx


def color_category(pc, name):
    p = pc.get(name)
    if p is None:
        return None
    cl = p.get("cluster", {})
    return {
        "k8": cl.get("8"), "k12": cl.get("12"), "k16": cl.get("16"),
        "special": p.get("special"), "leaf_pos": p.get("leaf_pos"),
    }


def palette_candidates(pools, pc, cyc, emitted_img):
    base = "_".join(emitted_img.split("_")[:2])  # wfd_000
    sibs = sorted(k for k in pools if k[0] == cyc and k[1].startswith(base + "_"))
    out = []
    for _, img in sibs:
        prov = pools[(cyc, img)]["provenance"]
        params = prov.get("params", {})
        out.append({
            "variant_id": img,
            "emitted": img == emitted_img,
            "pref_rank": prov.get("pref_v2_rank"),
            "pref_score": prov.get("pref_v2_score"),
            "selection_role": prov.get("selection_role"),
            "palette_ref": {                       # reference, NOT a copy of stops
                "name": prov.get("palette"),
                "source": params.get("palette_source"),
                "type": params.get("palette_type"),
            },
            "color_category": color_category(pc, prov.get("palette")),
            "mood": None,                          # GAP: no mood producer exists
            "coloring": {k: params[k] for k in (
                "reverse", "log_premap", "gamma", "phase", "n_cycles",
                "transfer", "transfer_gamma", "interior_color") if k in params},
        })
    out.sort(key=lambda c: (c["pref_rank"] is None, c["pref_rank"]))
    return out


def mode_candidacy(dt, loc_id):
    r = dt.get(loc_id)
    if r is None:
        return None
    return [{k: c.get(k) for k in (
        "mode", "kind", "p_ge3", "p_ge2", "E_ord", "passed", "kept")}
        for c in r.get("candidates", [])]


def build():
    cur, pools, led, pc, dt, uidx = load()
    rows = []
    for r in cur:
        cyc, img = r["curated_from"].split("/")
        prov = pools[(cyc, img)]["provenance"]
        oid = prov.get("source_oid")
        lr = led.get(oid, {})
        loc = r["location"]
        fam = r["family"]
        is_phoenix = loc.get("fractal_type") == "phoenix"
        is_dyn = loc.get("c_re") is not None or is_phoenix

        identity = {
            "family": fam,
            "fractal_type": loc.get("fractal_type"),
            "cx": loc["cx"], "cy": loc["cy"], "fw": loc["fw"],
            "maxiter": loc.get("maxiter"),
            # dynamical additive constant c (julia = the fractal; phoenix = Ushiki)
            "c": ({"re": loc["c_re"], "im": loc["c_im"]}
                  if loc.get("c_re") is not None
                  else (PHOENIX_C if is_phoenix else None)),
            # phoenix z_{n-1} coefficient p (Ushiki); null for all others
            "p": PHOENIX_P if is_phoenix else None,
            # for phoenix, cx/cy/fw is a z-plane viewport of one fixed system
            "coord_kind": "z_viewport" if is_phoenix else (
                "c_plane" if not is_dyn else "julia_c_fixed"),
            "source_oid": oid,
        }

        location_potential = {
            "scorer_version": lr.get("scorer_version"),
            "k3": lr.get("k3"),
            "raw_top3": lr.get("raw_top3"),
            "decoded_class": lr.get("decoded_class"),
            "p_good": lr.get("p_good"),
            "p_notbad": lr.get("p_notbad"),
            "t_good": lr.get("t_good"),
            "reached_depth": lr.get("reached_depth"),
            "guard_pass": lr.get("guard_pass"),
            # seeder-time snapshot carried in provenance (may predate ledger score)
            "seeder_decoded_class": prov.get("seeder_decoded_class"),
            "seeder_p_good": prov.get("seeder_p_good"),
            "source_ledger": prov.get("source_ledger"),
        }

        descriptors = {                            # references-not-copies
            "npz": EMB,
            "uid": r["curated_from"],
            "clip_vitb16_row": uidx.get(r["curated_from"]),
            "clip_dim": 768,
            "v6_prelogits_row": uidx.get(r["curated_from"]),
            "v6_dim": 1280,
            "v6_note": "UNFIT for fine grayscale dedup (saturates); CLIP is primary",
            "colored_clip": None,                  # GAP: slot, no producer yet
        }

        wallpaper_quality = {
            "emitted_variant": img,
            "emitted_palette": r["palette"],
            "emitted_coloring": r.get("coloring"),
            "render_mode": r.get("render_mode"),
            "render_spec": r.get("render_spec"),
            "wallpaper_canon": r.get("wallpaper_canon"),
            "gate": r.get("gate"),
            # FORK #1: gate p_ge3 is PREDICTED (pre-render, smooth 1280x720 crop).
            "predicted_p_ge3": (r.get("gate") or {}).get("p_ge3"),
            "actual_p_ge3": None,                  # GAP: post-render rescore not stored
        }

        rows.append({
            "record_version": "0.1",
            "location_id": r["image_id"],
            "curated_from": r["curated_from"],
            "identity": identity,
            "location_potential": location_potential,
            "palette_candidates": palette_candidates(pools, pc, cyc, img),
            "mode_candidacy": mode_candidacy(dt, r["image_id"]),
            "descriptors": descriptors,
            "wallpaper_quality": wallpaper_quality,
        })

    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} records -> {OUT}")

    # quick coverage summary
    def cov(fn):
        return sum(1 for r in rows if fn(r))
    print("dense coverage:")
    print("  location_potential.k3 present:",
          cov(lambda r: r["location_potential"]["k3"] is not None), "/", len(rows))
    print("  descriptors.clip row present:",
          cov(lambda r: r["descriptors"]["clip_vitb16_row"] is not None), "/", len(rows))
    print("  mode_candidacy present:",
          cov(lambda r: r["mode_candidacy"] is not None), "/", len(rows))
    print("  palette_candidates avg K:",
          round(sum(len(r["palette_candidates"]) for r in rows) / len(rows), 1))
    print("gap fields (all null by design):",
          "mood, colored_clip, actual_p_ge3")


if __name__ == "__main__":
    build()
