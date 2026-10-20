import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cwd", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--gpu")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    root, out = a.cwd.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = a.command[1:] if a.command[:1] == ["--"] else a.command

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    before = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    env = dict(
        os.environ,
        WARP_CUDA_PATH="/usr/local/cuda-13.0",
        WARP_CACHE_PATH=str(root / "_build" / "contribution-refresh-cache"),
        PYTHONPATH=str(root),
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = a.gpu
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tick = time.monotonic()
    with (out / (a.name + ".log")).open("w") as stream:
        r = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    result = {
        "name": a.name,
        "command": command,
        "source_commit": before,
        "source_commit_after": git("rev-parse", "HEAD"),
        "tracked_tree_dirty_before": dirty,
        "tracked_tree_dirty_after": bool(git("status", "--porcelain", "--untracked-files=no")),
        "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seconds": time.monotonic() - tick,
        "exit_code": r.returncode,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "private_kernel_cache": True,
        "native_library_sha256": {x.name: digest(x) for x in sorted((root / "warp/bin").glob("*.so"))},
    }
    (out / (a.name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
