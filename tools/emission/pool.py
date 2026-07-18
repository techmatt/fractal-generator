"""pool.py — resume-safe, append-only persistent pool of colorized wallpapers.

Same kill-and-resume discipline as the steered frontier driver
(`tools/atlas/steered_frontier.py`): the append-only log is the DURABLE source of
truth, replayed on load to rebuild every count; the state file (RNG + heuristic
cursor) is atomic-replaced and never trusted for anything a lost/stale copy could
corrupt.

One line per colorize ATTEMPT is appended to `pool_log.jsonl` AFTER the render+score
completes — a single write, so a kill mid-render loses at most the in-flight attempt
(never a partial line, never a duplicate of a logged one). Each attempt row carries
`passed` and the full descriptor + realized palette statistics + provenance, so:

  * the GATED POOL is `[r for r in log if r["passed"]]` (inventory that persists
    across releases — unselected wallpapers are inventory, not waste);
  * joint FILL counts rebuild from the passing rows;
  * per-cell ATTEMPT counts (for the attempt cap) rebuild from ALL rows.

Attempt ids are `em_<seq>` with seq = number of already-logged attempts, so a resume
continues the sequence with no id collision and no re-append of a logged row.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class Pool:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.log_path = self.dir / "pool_log.jsonl"
        self.state_path = self.dir / "pool_state.json"
        self.rows: list = []
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    # ---- ids ------------------------------------------------------------ #
    def next_id(self) -> str:
        return f"em_{len(self.rows):06d}"

    def n_attempts(self) -> int:
        return len(self.rows)

    # ---- append (durable) ---------------------------------------------- #
    def append(self, row: dict):
        """Append one completed colorize attempt. The single write is the commit
        point; nothing derived is persisted separately, so there is no window where
        the log and a count file could disagree."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)

    # ---- derived views (rebuilt from the durable log) ------------------- #
    def gated(self) -> list:
        return [r for r in self.rows if r.get("passed")]

    def attempts_per_location(self) -> dict:
        out: dict = {}
        for r in self.rows:
            out[r["location_id"]] = out.get(r["location_id"], 0) + 1
        return out

    # ---- heuristic state (atomic; never trusted for counts) ------------- #
    def save_state(self, state: dict):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def load_state(self) -> dict | None:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return None
