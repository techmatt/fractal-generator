#!/usr/bin/env python
"""Palette-preference labeling server for the cold-start query batch.

This is a NEW, distinct tool from the location-quality 0-3 labeler
(`tools/viz/corpus_label.html` + `tools/corpus/merge_scores.py`) — do not confuse
the two. That one tiers single crops for the *location* pool; this one collects
*relative palette/recipe preference* per query (6 candidates on one shared
location, ranked against each other) for a later single-tower image-utility net.

Scope: collect human rank tiers into a trainer-ready store. No scorer, no training,
no pair derivation here — we store the raw per-candidate tiers; deriving pairs is a
downstream job.

Serving mirrors `tools/explorer/app.py` (Flask, single local user,
server-authoritative). Persistence mirrors the still_extractor pattern: every label
write is an atomic temp-file + `os.replace` over the authoritative store, so a crash
never corrupts or loses prior work; on launch we load existing labels and resume.

Run:
  uv run python tools/queries/launch_query_label_server.py
  uv run python tools/queries/launch_query_label_server.py --batch coldstart_v2 --port 5099
  uv run python tools/queries/launch_query_label_server.py --selftest   # headless, temp store

Store:   data/queries/labels/<batch_id>.json   (authoritative, incremental)
Export:  data/queries/labels/<batch_id>_export_<UTCstamp>.json  (stamped, trainer hand-off)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

TIERS = ("bad", "okay", "good")
RANK = {"bad": 0, "okay": 1, "good": 2}
MAX_PER_SIDE = 3            # <=3 good and <=3 bad per query
REPEAT_FRACTION = 0.10     # ~10% of queries shown a second time (consistency probe)
REPEAT_MIN_GAP = 12        # a repeat lands at least this many presentations after its first

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Server-authoritative config/state (set in configure())
# ---------------------------------------------------------------------------
CFG = {
    "batch_id": None,
    "batch_dir": None,
    "images_dir": None,
    "store_path": None,
    "seed": None,
}
STORE = None                # the in-memory store dict (see build_store schema)
CANDIDATES = {}             # cid -> {img, query_id, pos_in_record, recipe}
_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Batch loading — one record = one query with 6 candidates on a shared location
# ---------------------------------------------------------------------------
def load_records(batch_dir: Path):
    """Return (queries, candidates).

    queries:    ordered list of {query_id, query_type, family, cids:[cid,...]}
    candidates: cid -> {img, query_id, recipe}
    Canonical candidate id = the image filename stem (e.g. 'q001_0125_0'), the
    durable identity; images are referenced batch-relative in the record.
    """
    records_dir = batch_dir / "records"
    queries, cands = [], {}
    for path in sorted(records_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        qid = rec["query_id"]
        loc = rec.get("location", {})
        cids = []
        for cand in rec["candidates"]:
            cid = Path(cand["image"]).stem            # 'images/q001_0125_0.png' -> 'q001_0125_0'
            cids.append(cid)
            cands[cid] = {
                "img": Path(cand["image"]).name,      # 'q001_0125_0.png'
                "query_id": qid,
                "recipe": {
                    "palette": cand.get("palette"),
                    "palette_source": cand.get("palette_source"),
                    "palette_type": cand.get("palette_type"),
                    "reverse": cand.get("reverse"),
                    "gamma": cand.get("gamma"),
                    "phase": cand.get("phase"),
                    "n_cycles": cand.get("n_cycles"),
                    "log_premap": cand.get("log_premap"),
                    "family": loc.get("family"),
                },
            }
        queries.append({
            "query_id": qid,
            "query_type": rec.get("query_type"),
            "family": loc.get("family"),
            "cids": cids,
        })
    return queries, cands


# ---------------------------------------------------------------------------
# Presentation queue — deterministic from seed, persisted in the store so it is
# stable across restarts (resume + progress stay meaningful). Includes the
# consistency-probe repeats interspersed later in the queue.
# ---------------------------------------------------------------------------
def build_queue(queries, seed: int):
    rng = random.Random(seed)

    def shuffled(xs):
        ys = list(xs)
        rng.shuffle(ys)
        return ys

    # pass-1 presentations in a shuffled query order, each with its own shuffled
    # candidate display order.
    order = shuffled([q["query_id"] for q in queries])
    cids_by_q = {q["query_id"]: q["cids"] for q in queries}
    queue = [{
        "pres_id": f"{qid}#p1",
        "query_id": qid,
        "pass": 1,
        "display_order": shuffled(cids_by_q[qid]),
    } for qid in order]

    # pick ~10% of queries for a second showing.
    n_rep = max(1, round(REPEAT_FRACTION * len(queries)))
    repeat_qids = set(rng.sample([q["query_id"] for q in queries], n_rep))

    # insert each repeat at a random position strictly later than its first
    # showing (>= REPEAT_MIN_GAP after it, never adjacent), re-shuffling positions
    # so the repeat is not visually identical. Insert one at a time (indices shift).
    for qid in shuffled(sorted(repeat_qids)):
        first = next(i for i, p in enumerate(queue) if p["query_id"] == qid)
        lo = min(first + REPEAT_MIN_GAP, len(queue))
        pos = rng.randint(lo, len(queue))
        queue.insert(pos, {
            "pres_id": f"{qid}#p2",
            "query_id": qid,
            "pass": 2,
            "display_order": shuffled(cids_by_q[qid]),
        })
    return queue


def build_store(queries, seed: int):
    return {
        "batch_id": CFG["batch_id"],
        "schema_version": 1,
        "seed": seed,
        "created_at": now_iso(),
        "queue": build_queue(queries, seed),
        # pres_id -> {query_id, pass, display_order, tiers:{cid:tier}, confirmed, labeled_at}
        "labels": {},
    }


# ---------------------------------------------------------------------------
# Atomic persistence (still_extractor pattern: temp file + os.replace)
# ---------------------------------------------------------------------------
def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)                          # atomic on Windows + POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def persist():
    atomic_write_json(CFG["store_path"], STORE)


def load_or_init_store(queries):
    """Load an existing store, else build one. If a store exists but its queue was
    built from a different seed/batch, keep it (labels are the ground truth) — never
    silently regenerate a queue that would strand collected labels."""
    global STORE
    sp = CFG["store_path"]
    if sp.exists():
        STORE = json.loads(sp.read_text(encoding="utf-8"))
        STORE.setdefault("labels", {})
        return
    STORE = build_store(queries, CFG["seed"])
    persist()


# ---------------------------------------------------------------------------
# Consistency-probe agreement (computed over queries whose BOTH passes are
# confirmed). Two numbers, per the spec.
# ---------------------------------------------------------------------------
def compute_agreement():
    labels = STORE["labels"]
    # queries that have both a confirmed p1 and confirmed p2
    repeats = []
    for q in {p["query_id"] for p in STORE["queue"] if p["pass"] == 2}:
        l1, l2 = labels.get(f"{q}#p1"), labels.get(f"{q}#p2")
        if l1 and l2 and l1.get("confirmed") and l2.get("confirmed"):
            repeats.append((q, l1["tiers"], l2["tiers"]))

    same, same_tot = 0, 0
    pair_agree, pair_tot = 0, 0
    per_query = []
    for q, t1, t2 in repeats:
        cids = sorted(set(t1) | set(t2))
        # (1) per-candidate same-tier rate
        q_same = sum(1 for c in cids if t1.get(c, "okay") == t2.get(c, "okay"))
        same += q_same
        same_tot += len(cids)
        # (2) cross-tier pair-direction agreement: over pairs strictly ordered in
        #     BOTH passes, does the direction (the training constraint) hold?
        q_pa, q_pt = 0, 0
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                d1 = RANK[t1.get(a, "okay")] - RANK[t1.get(b, "okay")]
                d2 = RANK[t2.get(a, "okay")] - RANK[t2.get(b, "okay")]
                if d1 == 0 or d2 == 0:
                    continue                          # not a constraint in both passes
                q_pt += 1
                if (d1 > 0) == (d2 > 0):
                    q_pa += 1
        pair_agree += q_pa
        pair_tot += q_pt
        per_query.append({
            "query_id": q,
            "same_tier": q_same, "n_candidates": len(cids),
            "pair_agree": q_pa, "pair_total": q_pt,
        })

    return {
        "n_repeat_queries": len(repeats),
        "same_tier_rate": (same / same_tot) if same_tot else None,
        "same_tier_num": same, "same_tier_den": same_tot,
        "pair_direction_rate": (pair_agree / pair_tot) if pair_tot else None,
        "pair_direction_num": pair_agree, "pair_direction_den": pair_tot,
        "per_query": per_query,
    }


def progress():
    q1 = [p for p in STORE["queue"] if p["pass"] == 1]
    q2 = [p for p in STORE["queue"] if p["pass"] == 2]
    conf = lambda ps: sum(1 for p in ps if STORE["labels"].get(p["pres_id"], {}).get("confirmed"))
    return {
        "total": len(STORE["queue"]),
        "confirmed": conf(STORE["queue"]),
        "base_total": len(q1), "base_confirmed": conf(q1),
        "repeat_total": len(q2), "repeat_confirmed": conf(q2),
    }


# ---------------------------------------------------------------------------
# Label validation + write
# ---------------------------------------------------------------------------
def valid_tiers(tiers, display_order):
    if set(tiers) != set(display_order):
        return "tiers must cover exactly the query's 6 candidates"
    if any(v not in TIERS for v in tiers.values()):
        return "tier values must be one of bad/okay/good"
    g = sum(1 for v in tiers.values() if v == "good")
    b = sum(1 for v in tiers.values() if v == "bad")
    if g > MAX_PER_SIDE or b > MAX_PER_SIDE:
        return f"at most {MAX_PER_SIDE} good and {MAX_PER_SIDE} bad per query"
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(str(HERE), "query_label.html")


@app.route("/img/<path:name>")
def img(name):
    return send_from_directory(str(CFG["images_dir"]), name)


@app.route("/api/state")
def api_state():
    """Everything the client needs in one shot: batch id, the presentation queue,
    the candidate catalog (id -> image + recipe), and existing labels for resume."""
    return jsonify({
        "batch_id": STORE["batch_id"],
        "queue": STORE["queue"],
        "candidates": CANDIDATES,
        "labels": STORE["labels"],
        "progress": progress(),
        "config": {"max_per_side": MAX_PER_SIDE},
    })


@app.route("/api/label", methods=["POST"])
def api_label():
    d = request.get_json(force=True)
    pres_id = d["pres_id"]
    with _LOCK:
        pres = next((p for p in STORE["queue"] if p["pres_id"] == pres_id), None)
        if pres is None:
            return jsonify({"ok": False, "error": f"unknown pres_id {pres_id}"}), 400
        tiers = {c: d["tiers"].get(c, "okay") for c in pres["display_order"]}
        err = valid_tiers(tiers, pres["display_order"])
        if err:
            return jsonify({"ok": False, "error": err}), 400
        confirmed = bool(d.get("confirmed", False))
        # A confirmed entry must contain a real ranking signal: at least one good
        # and one bad. An all-okay (or one-sided) entry carries no training
        # constraint, so it is never allowed to count as confirmed — drafts are fine.
        if confirmed:
            g = sum(1 for v in tiers.values() if v == "good")
            b = sum(1 for v in tiers.values() if v == "bad")
            if g < 1 or b < 1:
                return jsonify({"ok": False,
                                "error": "confirm needs at least 1 good and 1 bad"}), 400
        STORE["labels"][pres_id] = {
            "query_id": pres["query_id"],
            "pass": pres["pass"],
            "display_order": pres["display_order"],   # order actually shown (audit)
            "tiers": tiers,
            "confirmed": confirmed,
            "labeled_at": now_iso(),
        }
        persist()                                     # atomic write on every change
        return jsonify({"ok": True, "progress": progress()})


@app.route("/api/agreement")
def api_agreement():
    return jsonify(compute_agreement())


@app.route("/api/export", methods=["POST"])
def api_export():
    with _LOCK:
        prog = progress()
        agree = compute_agreement()
        stamp = now_iso().replace(":", "").replace("-", "")
        out_path = CFG["store_path"].with_name(f"{CFG['batch_id']}_export_{stamp}.json")
        dump = {
            "batch_id": CFG["batch_id"],
            "exported_at": now_iso(),
            "schema_version": STORE["schema_version"],
            "seed": STORE["seed"],
            "complete": prog["base_confirmed"] == prog["base_total"],
            "progress": prog,
            "agreement": agree,
            # trainer-consumable: per presentation, per-candidate tier (all 6),
            # display order shown, pass, timestamp. Both passes of a repeat kept.
            "labels": STORE["labels"],
        }
        atomic_write_json(out_path, dump)
        return jsonify({
            "ok": True,
            "path": str(out_path),
            "complete": dump["complete"],
            "progress": prog,
            "agreement": agree,
        })


# ---------------------------------------------------------------------------
# Configuration / startup
# ---------------------------------------------------------------------------
def configure(batch_id: str, seed: int, store_dir: Path | None = None):
    global CANDIDATES
    batch_dir = REPO_ROOT / "data" / "queries" / batch_id
    if not (batch_dir / "records").is_dir():
        sys.exit(f"batch records not found: {batch_dir / 'records'}")
    labels_dir = store_dir or (REPO_ROOT / "data" / "queries" / "labels")
    CFG.update({
        "batch_id": batch_id,
        "batch_dir": batch_dir,
        "images_dir": batch_dir / "images",
        "store_path": labels_dir / f"{batch_id}.json",
        "seed": seed,
    })
    queries, CANDIDATES = load_records(batch_dir)
    load_or_init_store(queries)
    return queries


# ---------------------------------------------------------------------------
# Self-test — headless, temp store, exercises GET page + label POSTs + repeat
# preservation + agreement + export, then leaves the REAL store untouched.
# ---------------------------------------------------------------------------
def selftest():
    tmp = Path(tempfile.mkdtemp(prefix="qlabel_selftest_"))
    print(f"[selftest] temp store dir: {tmp}")
    try:
        queries = configure("coldstart_v2", seed=1234, store_dir=tmp)
        assert CFG["store_path"].exists(), "store not created on init"
        client = app.test_client()

        # GET the page
        r = client.get("/")
        assert r.status_code == 200 and b"<html" in r.data.lower(), "index did not serve"
        print(f"[selftest] GET / -> {r.status_code}, {len(r.data)} bytes")

        # GET state
        st = client.get("/api/state").get_json()
        assert st["batch_id"] == "coldstart_v2"
        assert len(st["queue"]) > len(queries), "queue missing consistency repeats"
        n_repeat = sum(1 for p in st["queue"] if p["pass"] == 2)
        print(f"[selftest] queue={len(st['queue'])} (base={len(queries)}, repeats={n_repeat})")

        # find a query that has BOTH passes so we can exercise repeat preservation
        rep_q = next(p["query_id"] for p in st["queue"] if p["pass"] == 2)
        p1 = f"{rep_q}#p1"
        p2 = f"{rep_q}#p2"
        order1 = next(p["display_order"] for p in st["queue"] if p["pres_id"] == p1)
        order2 = next(p["display_order"] for p in st["queue"] if p["pres_id"] == p2)

        # POST pass 1: mark first candidate good, last bad, rest okay
        t1 = {c: "okay" for c in order1}
        t1[order1[0]] = "good"
        t1[order1[-1]] = "bad"
        r = client.post("/api/label", json={"pres_id": p1, "tiers": t1, "confirmed": True})
        assert r.get_json()["ok"], r.get_json()
        # POST pass 2: a DIFFERENT labeling (flip which candidate is bad) to show
        # both are preserved. Still one good + one bad so the confirm rule passes.
        t2 = {c: "okay" for c in order2}
        t2[order2[0]] = "good"
        t2[order2[1]] = "bad"
        r = client.post("/api/label", json={"pres_id": p2, "tiers": t2, "confirmed": True})
        assert r.get_json()["ok"], r.get_json()

        # also label one plain pass-1 presentation (needs a good + a bad to confirm)
        plain = next(p for p in st["queue"] if p["pass"] == 1 and p["query_id"] != rep_q)
        tp = {c: "okay" for c in plain["display_order"]}
        tp[plain["display_order"][0]] = "good"
        tp[plain["display_order"][-1]] = "bad"
        client.post("/api/label", json={"pres_id": plain["pres_id"], "tiers": tp, "confirmed": True})

        # confirm-rule enforcement: an all-okay confirm must be rejected
        allokay = {c: "okay" for c in order1}
        r = client.post("/api/label", json={"pres_id": p1, "tiers": allokay, "confirmed": True})
        assert r.status_code == 400, "all-okay confirm not rejected"
        # but the same all-okay tiers as a DRAFT (confirmed=False) is allowed
        r = client.post("/api/label", json={"pres_id": p1, "tiers": allokay, "confirmed": False})
        assert r.get_json()["ok"], "all-okay draft wrongly rejected"
        # restore p1's real confirmed labeling for the agreement/repeat checks below
        client.post("/api/label", json={"pres_id": p1, "tiers": t1, "confirmed": True})
        print("[selftest] confirm-rule OK (all-okay confirm rejected, draft allowed)")

        # verify store on disk kept BOTH passes distinctly (atomic write landed)
        disk = json.loads(CFG["store_path"].read_text(encoding="utf-8"))
        assert p1 in disk["labels"] and p2 in disk["labels"], "both passes not persisted"
        assert disk["labels"][p1]["tiers"] != disk["labels"][p2]["tiers"], "second pass overwrote first"
        print(f"[selftest] both passes preserved for {rep_q}; store has {len(disk['labels'])} labels")

        # cap enforcement: 4 good must be rejected
        bad = {c: "good" for c in order1[:4]}
        for c in order1[4:]:
            bad[c] = "okay"
        r = client.post("/api/label", json={"pres_id": p1, "tiers": bad, "confirmed": True})
        assert r.status_code == 400, "4-good cap not enforced"
        print("[selftest] cap enforcement OK (4 good rejected)")

        # agreement
        ag = client.get("/api/agreement").get_json()
        assert ag["n_repeat_queries"] >= 1, "agreement saw no completed repeat"
        print(f"[selftest] agreement: n_repeat={ag['n_repeat_queries']} "
              f"same_tier={ag['same_tier_rate']:.3f} "
              f"pair_dir={ag['pair_direction_rate']}")

        # export
        ex = client.post("/api/export", json={}).get_json()
        assert ex["ok"] and Path(ex["path"]).exists(), "export did not write"
        print(f"[selftest] export -> {ex['path']} (complete={ex['complete']})")

        print("[selftest] PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[selftest] cleaned temp store dir (real store under "
              f"data/queries/labels/ untouched)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="coldstart_v2")
    ap.add_argument("--seed", type=int, default=7, help="queue/shuffle seed (persisted in store)")
    ap.add_argument("--port", type=int, default=5099)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    configure(a.batch, a.seed)
    prog = progress()
    print(f"Batch:  {a.batch}")
    print(f"Store:  {CFG['store_path']}")
    print(f"Queue:  {prog['total']} presentations "
          f"({prog['base_total']} queries + {prog['repeat_total']} consistency repeats)")
    print(f"Done:   {prog['confirmed']}/{prog['total']} confirmed (resuming)")
    print(f"\nLabeling UI:  http://127.0.0.1:{a.port}/\n")
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
