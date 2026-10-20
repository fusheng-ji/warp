"""Run one experiment, retaining sanitized output, source state and exit status."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[0] == "--" else args.command
    checkout = Path.cwd()
    output = Path(__file__).resolve().parent
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    substitutions = [(str(checkout), "<CHECKOUT>"), (str(output), "<RESULTS>")]
    cache = os.environ.get("WARP_CACHE_PATH")
    if cache:
        substitutions.insert(0, (cache, "<PRIVATE_CACHE>"))
    substitutions.append((str(Path.home()), "<HOME>"))

    def clean(value):
        for before, after in substitutions:
            value = value.replace(before, after)
        return value

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], text=True).strip()

    document = {
        "name": args.name,
        "command": [clean(part) for part in command],
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "dirty_before": git("status", "--porcelain"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "private_cache": bool(cache),
        "experiment_scripts_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.py"))
        },
    }
    start = time.monotonic()
    with (logs / f"{args.name}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            value = clean(line)
            log.write(value)
            log.flush()
            print(value, end="", flush=True)
        document["exit_code"] = process.wait()
    document["elapsed_seconds"] = time.monotonic() - start
    document["dirty_after"] = git("status", "--porcelain")
    (logs / f"{args.name}.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document), flush=True)
    sys.exit(document["exit_code"])


if __name__ == "__main__":
    main()
