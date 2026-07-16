"""Shared loader for the provenance x label EDA (prompts/provenance-label-eda.md).

Read-only. Separates the three label-corpus batches into the analysis cohorts the
prompt mandates:

  - UNBIASED DESCENT  = rev4 (run4, all rows) + batch2 `random_eval` (100 rows).
                        The only cohort allowed into "where good things live" trends.
  - BIASED ENRICHED   = batch2 `enriched` (v2-selected, warm/high-predicted). Used in
                        exactly ONE place: the palette-bias test (Panel E).
  - LOOSE0 CONTRAST   = flat_generate loose0_v3. A different (flat, shallow) generator,
                        shown only as contrast, never pooled into descent trends.

Label convention: score in {1 bad, 2 okay, 3 good}; not-bad := score>=2; good := score==3.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BATCH_DIR = os.path.join(ROOT, "data", "label_corpus", "batches")
EDA_DIR = os.path.join(ROOT, "data", "eda")

B_REV4 = "2026-06-24_guided_descend_rev4"            # run4, unbiased descent
B_ENR = "2026-06-24_guided_descend_rev4occfix_v2filtered"  # enriched + random_eval
B_LOOSE0 = "2026-06-23_flat_generate_loose0_v3"      # flat contrast

POOL_RUN4 = os.path.join(ROOT, "data", "guided_descend", "run4", "pool.jsonl")
POOL_RUN5 = os.path.join(ROOT, "data", "guided_descend", "run5", "pool.jsonl")

FIELD_F32 = os.path.join(ROOT, "data", "root_field", "field_8192x8192_m1000.f32")
FIELD_JSON = os.path.join(ROOT, "data", "root_field", "field_8192x8192_m1000.json")

# rev4 / batch2 sampling draw weights for the descent target branch (batch.json target_mix).
TARGET_MIX = {"foci": 0.7, "density": 0.1, "random": 0.2}
# present guards in force for the descent batches.
GUARD_BLACK_CAP = 0.30
GUARD_OCC_FLOOR = 0.321


def _rows(batch):
    p = os.path.join(BATCH_DIR, batch, "images.jsonl")
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def crop_path(batch, image_id):
    return os.path.join(BATCH_DIR, batch, "crops", image_id + ".jpg")


def load_cohorts():
    """Return dict of cohort name -> list of flattened records.

    Each record: image_id, batch, cx, cy, fw, palette, composition, score (or None),
    plus all provenance fields, plus selection_role (None for rev4/loose0).
    """
    def flat(r, batch):
        rec = dict(r["provenance"])
        rec["image_id"] = r["image_id"]
        rec["batch"] = batch
        rec["score"] = r["label"]["score"]
        rec["palette"] = r["render"]["palette"]
        rec["composition"] = r["render"]["composition"]
        rec["cx"] = float(r["render"]["cx"])
        rec["cy"] = float(r["render"]["cy"])
        rec["fw"] = float(r["render"]["fw"])
        return rec

    rev4 = [flat(r, B_REV4) for r in _rows(B_REV4)]
    enr_all = [flat(r, B_ENR) for r in _rows(B_ENR)]
    loose0 = [flat(r, B_LOOSE0) for r in _rows(B_LOOSE0)]

    random_eval = [r for r in enr_all if r.get("selection_role") == "random_eval"]
    enriched = [r for r in enr_all if r.get("selection_role") == "enriched"]

    cohorts = {
        "rev4": rev4,
        "random_eval": random_eval,
        "enriched": enriched,
        "loose0": loose0,
        # the sanctioned union for natural "where good things live" trends:
        "unbiased": rev4 + random_eval,
    }
    return cohorts


def load_pool(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def labeled(records):
    return [r for r in records if r["score"] in (1, 2, 3)]


def rate(records, pred):
    """(count matching pred, n labeled, rate). pred over a labeled record."""
    lab = labeled(records)
    n = len(lab)
    k = sum(1 for r in lab if pred(r))
    return k, n, (k / n if n else float("nan"))


def notbad(r):
    return r["score"] is not None and r["score"] >= 2


def good(r):
    return r["score"] == 3


def load_field_downsampled(target=1024):
    """Load the 8k smooth-iter field, downsample by striding to ~target px.

    Returns (img2d, extent) with extent = [re_lo, re_hi, im_lo, im_hi] for imshow.
    Interior sentinel (NaN) stays NaN.
    """
    import numpy as np
    meta = json.load(open(FIELD_JSON))
    w, h = meta["w"], meta["h"]
    a = np.fromfile(FIELD_F32, dtype="<f4").reshape(h, w)
    step = max(1, w // target)
    a = a[::step, ::step]
    extent = [meta["re_lo"], meta["re_hi"], meta["im_lo"], meta["im_hi"]]
    return a, extent


if __name__ == "__main__":
    c = load_cohorts()
    for name in ["rev4", "random_eval", "enriched", "loose0", "unbiased"]:
        recs = c[name]
        lab = labeled(recs)
        kg = sum(1 for r in lab if good(r))
        kn = sum(1 for r in lab if notbad(r))
        print(f"{name:12s} rows={len(recs):5d} labeled={len(lab):5d} "
              f"notbad={kn:4d} ({100*kn/len(lab):4.1f}%) good={kg:3d} ({100*kg/len(lab):4.1f}%)")
