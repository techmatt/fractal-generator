#!/usr/bin/env python
r"""Embedding-shard crash-safety harness (SLOW / opt-in) — a REAL kill mid-`np.savez`.

`write_embedding_shard` guards the ONE artifact in this project that isn't cheaply
regenerable: the accumulating `data/library_embeddings/shards/`. Its whole safety claim is
crash-safety BY CONSTRUCTION — write a `.tmp`, then atomic `os.replace` into place, so a
kill can only ever orphan the in-flight `.tmp`, never truncate a prior shard or the base.
The GPU-free unit tests (`test_prospect.py`) simulate that with a hand-planted stray `.tmp`;
this harness proves it against an ACTUAL kill of a process caught inside the shard write.

The kill is DETERMINISTIC. A child process writes the base + a prior shard for real, then
enters `write_embedding_shard` for a new cycle with `numpy.savez` patched to write a partial
(BadZipFile) `.tmp`, fsync it, drop a sentinel, and block. The parent waits for the sentinel
— so the `.tmp` is guaranteed on disk, mid-write, no `os.replace` yet — then hard-kills the
tree (Windows `taskkill /F /T`, POSIX SIGKILL). Everything else is the REAL store code: the
base/prior writes, the `.tmp` open, the loader, and the resume write all go through
production functions; only the single crashing `savez` is intercepted, which is exactly the
"killed mid-savez" instant under test.

Asserts, post-kill: the partial `.tmp` is the ONLY casualty (a BadZipFile, no final `.npz`);
the base and prior shard load intact; `load_library_embeddings` concatenates base+prior and
IGNORES the orphan with no error; and a resume write of the same cycle completes, clears the
orphan, and the loader then includes it.

EXERCISE THIS on any change to `library_store.write_embedding_shard` or
`load_library_embeddings` (the tmp+atomic-replace sequence, the loader's shard glob / skip).

  uv run pytest tools/wallpaper/test_shard_crash.py         # this harness (opt-in)
  uv run pytest tools/wallpaper -m "not slow"               # fast gate only, skips this
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))
import library_store as store          # noqa: E402

pytestmark = pytest.mark.slow

RUN_ID = "RUN"
DIM = store.MORPH_CLIP_DIM
PARTIAL_NPZ = b"PK\x03\x04" + b"\x00" * 4096   # a truncated zip -> BadZipFile on np.load


# --------------------------------------------------------------------------- #
# Child: real base + prior shard, then a shard write that stalls mid-savez.
# --------------------------------------------------------------------------- #
def _crash_child(shards_dir: Path, base_path: Path, sentinel: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    # Base + prior shard: the REAL write path (real np.savez) — these MUST survive the kill.
    np.savez(base_path, morph_uids=np.asarray(["base_0", "base_1"]),
             morph_clip=np.zeros((2, DIM), np.float32))
    store.write_embedding_shard(RUN_ID, 1, ["p0", "p1"], np.ones((2, DIM), np.float32),
                                shards_dir=shards_dir, emb_base=base_path)

    # Now crash the NEXT cycle's write mid-savez: patch numpy.savez to write a partial .tmp
    # (into the handle write_embedding_shard already opened), fsync it, signal, and block. The
    # .tmp exists, no os.replace has run — the exact killed-mid-write state.
    def _stall_savez(file, *args, **kwargs):
        file.write(PARTIAL_NPZ)
        file.flush()
        os.fsync(file.fileno())
        sentinel.write_text("mid-savez", encoding="utf-8")
        while True:
            time.sleep(3600)

    np.savez = _stall_savez  # store's `np` is this same module object -> its np.savez resolves here
    store.write_embedding_shard(RUN_ID, 2, ["x0", "x1", "x2"], np.ones((3, DIM), np.float32),
                                shards_dir=shards_dir, emb_base=base_path)


def _kill_tree(pid: int) -> None:
    """Hard, unblockable kill of the whole child tree (kill -9 equivalent)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True, timeout=30)
    else:
        os.kill(pid, signal.SIGKILL)


# --------------------------------------------------------------------------- #
# Parent: spawn, wait for the mid-savez sentinel, kill, then assert.
# --------------------------------------------------------------------------- #
def test_kill_mid_savez_orphans_only_the_tmp(tmp_path):
    shards_dir = tmp_path / "shards"
    base_path = tmp_path / "embeddings.npz"
    sentinel = tmp_path / "mid_savez.flag"
    final2 = shards_dir / f"{RUN_ID}__cycle_002.npz"
    tmp2 = shards_dir / f".{RUN_ID}__cycle_002.npz.tmp"

    proc = subprocess.Popen(
        [sys.executable, str(_HERE / "test_shard_crash.py"), "--crash-child",
         str(shards_dir), str(base_path), str(sentinel)],
        cwd=str(_ROOT))
    try:
        deadline = time.time() + 60
        while not sentinel.exists():
            if proc.poll() is not None:
                raise AssertionError(f"child exited early rc={proc.returncode} before mid-savez")
            if time.time() > deadline:
                raise AssertionError("child never reached the mid-savez window")
            time.sleep(0.02)
        # caught inside the shard write: kill -9 before os.replace runs.
        _kill_tree(proc.pid)
    finally:
        try:
            proc.wait(timeout=30)
        except Exception:
            _kill_tree(proc.pid)

    # 1. only the .tmp is orphaned — the partial cycle-2 write never became a real shard.
    assert tmp2.exists(), "expected the in-flight .tmp on disk after the mid-savez kill"
    assert not final2.exists(), "os.replace must NOT have run — no final cycle-002 shard"
    npz_shards = sorted(p.name for p in shards_dir.glob("*.npz"))
    assert npz_shards == [f"{RUN_ID}__cycle_001.npz"], f"only the prior shard should exist: {npz_shards}"

    # 2. the orphan is a genuinely-corrupt partial (BadZipFile), not a truncated-but-loadable npz.
    with pytest.raises(Exception):
        np.load(tmp2, allow_pickle=True)

    # 3. base + prior shard are intact and load independently.
    zbase = np.load(base_path, allow_pickle=True)
    assert list(zbase["morph_uids"]) == ["base_0", "base_1"]
    zprior = np.load(shards_dir / f"{RUN_ID}__cycle_001.npz", allow_pickle=True)
    assert list(zprior["morph_uids"]) == ["p0", "p1"]

    # 4. the loader concatenates base+prior and IGNORES the orphan .tmp (glob skips it; the
    #    try/except is the backstop) with no error.
    emb = store.load_library_embeddings(emb_base=base_path, shards_dir=shards_dir)
    assert set(emb) == {"base_0", "base_1", "p0", "p1"}

    # 5. resume: the real write of the same cycle completes, clears the orphan, and the loader
    #    then includes it — the crash cost re-derive time, never data.
    store.write_embedding_shard(RUN_ID, 2, ["x0", "x1", "x2"], np.full((3, DIM), 2.0, np.float32),
                                shards_dir=shards_dir, emb_base=base_path)
    assert final2.exists() and not tmp2.exists(), "resume must land the final shard and clear the tmp"
    emb2 = store.load_library_embeddings(emb_base=base_path, shards_dir=shards_dir)
    assert set(emb2) == {"base_0", "base_1", "p0", "p1", "x0", "x1", "x2"}
    assert np.allclose(emb2["x1"], 2.0)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--crash-child":
        _crash_child(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        # convenience: run the harness directly (equivalent to `pytest -m slow` on this file)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            test_kill_mid_savez_orphans_only_the_tmp(Path(d))
            print("PASS test_kill_mid_savez_orphans_only_the_tmp")
