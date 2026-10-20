"""Run one source-frozen rendering experiment in the allocated Slurm step."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    checkout = args.checkout.resolve()
    if not (output / "finish_report.py").exists():
        shutil.copy2(output.parents[1] / "finish_report.py", output / "finish_report.py")
    job = os.environ["SLURM_JOB_ID"]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if source != "cb86abb75d123876d80c6832d8d360f4cda0cc10":
        raise RuntimeError("Feature source changed since scheduling")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip():
        raise RuntimeError("Source worktree is dirty")
    info = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    fields = dict(item.split("=", 1) for item in info.split() if "=" in item)
    selected = {
        key: fields.get(key)
        for key in (
            "JobId",
            "JobState",
            "Partition",
            "NumNodes",
            "NumCPUs",
            "ReqTRES",
            "AllocTRES",
            "TresPerNode",
            "TresPerTask",
            "OverSubscribe",
        )
    }
    if selected["Partition"] != "overflow" or "gres/gpu=1" not in selected["AllocTRES"]:
        raise RuntimeError("Expected one whole GPU from overflow")
    selected.update(
        source_commit=source,
        status="running",
        started_unix=time.time(),
        restart_count=os.environ.get("SLURM_RESTART_COUNT", "0"),
    )
    (output / "allocation.json").write_text(json.dumps(selected, indent=2) + "\n")

    def run(name, script, *arguments):
        command = [
            "uv",
            "run",
            "--no-sync",
            str(output / "run_record.py"),
            "--name",
            name,
            "--",
            "uv",
            "run",
            "--no-sync",
            str(output / script),
            *map(str, arguments),
        ]
        subprocess.run(command, cwd=checkout, check=True)

    try:
        run("device-selection", "select_device.py", "--checkout", checkout, "--output", output / "device.json")
        egl = json.loads((output / "device.json").read_text())["egl_device_index"]
        run(
            "filled-final",
            "validate_filled.py",
            "--checkout",
            checkout,
            "--output",
            output / "filled",
            "--mode",
            "switch",
            "--device",
            "cuda:0",
            "--egl-device",
            egl,
        )
        run(
            "benchmark-final",
            "benchmark_capsules.py",
            "--checkout",
            checkout,
            "--output",
            output / "benchmark",
            "--egl-device",
            egl,
        )
        run("analysis-final", "summarize.py", "--input", output)
        run("report-final", "finish_report.py", "--input", output)
        selected["status"] = "passed"
    except BaseException as error:
        selected["status"] = "failed"
        selected["error_type"] = type(error).__name__
        raise
    finally:
        selected["finished_unix"] = time.time()
        (output / "allocation.json").write_text(json.dumps(selected, indent=2) + "\n")
        for path in output.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".log", ".csv", ".txt"):
                text = path.read_text()
                redacted = re.sub(r"GPU-[a-fA-F0-9-]{36}", "<ALLOCATED_GPU_UUID>", text)
                if text != redacted:
                    path.write_text(redacted)
        files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (output / "SHA256SUMS").write_text(
            "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output)}\n" for p in files)
        )


if __name__ == "__main__":
    main()
