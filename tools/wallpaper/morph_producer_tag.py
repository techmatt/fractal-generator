"""Re-run through the FIXED morph producer + non-destructive base-store producer tag.

1. Re-embed the 47-location fixture through the production morph_gray_image (now robust-z;
   see docs/findings/morph_parity.md) and assert self-cos == 1.0 vs the stored morph_clip.
2. Add a `morph_producer` array to the base store (additive; morph_clip/uids untouched),
   since parity proves the base rows ARE this producer's output.

Promoted out of scratchpad (2026-07-15 hygiene pass): it is the sole producer of the
`morph_producer` provenance field in data/library_embeddings/embeddings.npz. Idempotent —
the store-tag write is guarded by an `if "morph_producer" not in arrays` check. FIELDS is a
regenerable smooth-field cache; re-embedding re-dumps any missing tiles.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import colormap as cm
from tools.curation.colored_clip import location_from_record, load_clip, embed_clip
from tools.wallpaper import library_annotate as la

STORE = ROOT / "data/library_embeddings/embeddings.npz"
RECORDS = ROOT / "data/library/library_records.jsonl"
FIELDS = ROOT / "out/curation/morph_fields"


def stem(u): return u.replace("/", "__")
def unit(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def main():
    z = np.load(STORE, allow_pickle=True)
    uids_all = z["morph_uids"].tolist()
    stored = {u: v for u, v in zip(uids_all, z["morph_clip"])}
    recs = {r["curated_from"]: r for r in
            (json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip())
            if r.get("curated_from")}
    fix = [u for u in uids_all
           if u in recs and (FIELDS / f"{stem(u)}.bin").exists()]

    model, tf = load_clip()
    imgs = [la.morph_gray_image(cm.load_field(str(FIELDS / f"{stem(u)}.bin"))) for u in fix]
    E = unit(embed_clip(model, tf, imgs).astype(np.float32))
    S = unit(np.stack([stored[u] for u in fix]).astype(np.float32))
    sc = np.sum(S * E, axis=1)
    print(f"[step2-rerun] FIXED producer self-cos over {len(fix)} fixture locations: "
          f"min={sc.min():.6f} median={np.median(sc):.6f} max={sc.max():.6f}")
    assert sc.min() > 0.9999, f"parity NOT reached: min self-cos {sc.min()}"
    print("[step2-rerun] PARITY CONFIRMED (min self-cos > 0.9999)")

    # --- non-destructive base-store producer tag ---
    arrays = {k: z[k] for k in z.files}
    if "morph_producer" not in arrays:
        arrays["morph_producer"] = np.asarray([la.MORPH_PRODUCER] * len(uids_all))
        np.savez(STORE, **arrays)
        print(f"[tag] base store: added morph_producer='{la.MORPH_PRODUCER}' "
              f"for {len(uids_all)} rows (morph_clip/uids untouched)")
    else:
        print(f"[tag] base store already tagged: {set(arrays['morph_producer'].tolist())}")


if __name__ == "__main__":
    main()
