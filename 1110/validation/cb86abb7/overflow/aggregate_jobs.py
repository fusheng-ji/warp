"""Verify independent Slurm runs and summarize variation between allocations."""

import argparse
import csv
import hashlib
import json
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "independent-summary"
    output.mkdir(exist_ok=True)
    check = unittest.TestCase()
    submissions = json.loads((root / "independent-submissions.json").read_text())
    accounting = {row["JobID"]: row for row in load_csv(root / "independent-accounting.csv")}
    check.assertEqual(len(submissions), 3)
    check.assertEqual(len({row["node_group"] for row in submissions}), 3)
    check.assertEqual(len({row["seed"] for row in submissions}), 3)
    all_rows = []
    manifest_checks = []
    environment_keys = []
    frames_total = 0
    filled_total = 0
    restored_total = 0
    jobs = []
    for submission in submissions:
        job = submission["job_id"]
        case = root / "jobs" / job
        account = accounting[job]
        check.assertEqual(account["State"], "COMPLETED")
        check.assertEqual(account["ExitCode"], "0:0")
        lines = (case / "SHA256SUMS").read_text().splitlines()
        for line in lines:
            expected, name = line.split("  ", 1)
            check.assertEqual(hashlib.sha256((case / name).read_bytes()).hexdigest(), expected, f"{job}: {name}")
        manifest_checks.append({"job_id": job, "verified_entries": len(lines)})
        allocation = json.loads((case / "allocation.json").read_text())
        analysis = json.loads((case / "analysis.json").read_text())
        device = json.loads((case / "device.json").read_text())
        env = json.loads((case / "benchmark/environment.json").read_text())
        check.assertEqual(allocation["status"], "passed")
        check.assertEqual(env["status"], "passed")
        check.assertEqual(allocation["source_commit"], submission["source"])
        check.assertEqual(env["arguments"]["seed"], submission["seed"])
        check.assertEqual(analysis["performance_frames"], 2280)
        check.assertEqual(analysis["performance_blocks"], 90)
        check.assertEqual(analysis["filled_frames"], 40)
        check.assertEqual(analysis["restored_images_exact"], 10)
        check.assertTrue(device["native_cuda_gl_interop_verified"])
        check.assertEqual(device["preexisting_compute_or_graphics_processes"], 0)
        check.assertEqual(device["visible_cuda_devices"], 1)
        check.assertEqual(device["mig_mode"], "Disabled")
        check.assertEqual(env["source_dirty_before"], "")
        check.assertEqual(env["source_dirty_after"], "")
        environment_keys.append(
            {
                "source_commit": env["source_commit"],
                "source_sha256": env["source_sha256"],
                "library_sha256": env["library_sha256"],
                "script_sha256": env["script_sha256"],
                "versions": {
                    key: env["environment"][key]
                    for key in (
                        "python",
                        "warp",
                        "numpy",
                        "pyglet",
                        "cuda_device",
                        "gl_vendor",
                        "gl_renderer",
                        "gl_version",
                        "thread_environment",
                    )
                },
            }
        )
        for record in (case / "logs").glob("*.json"):
            check.assertEqual(json.loads(record.read_text())["exit_code"], 0)
        rows = load_csv(case / "performance_summary.csv")
        check.assertEqual(len(rows), 18)
        for row in rows:
            all_rows.append({"job_id": job, "node_group": submission["node_group"], "seed": submission["seed"], **row})
        frames_total += analysis["performance_frames"]
        filled_total += analysis["filled_frames"]
        restored_total += analysis["restored_images_exact"]
        jobs.append(
            {**submission, "state": account["State"], "exit_code": account["ExitCode"], "elapsed": account["Elapsed"]}
        )
    for environment in environment_keys[1:]:
        check.assertEqual(environment, environment_keys[0], "Environment or measured source differs between jobs")
    write_csv(output / "per_job_summary.csv", all_rows)
    aggregates = []
    for mode in ("steady", "switch_all"):
        for count in (64, 512, 4096):
            for low in (3, 8, 16, 32) if mode == "steady" else (3, 8):
                selected = [
                    row
                    for row in all_rows
                    if (row["mode"], int(row["instances"]), int(row["low_segments"])) == (mode, count, low)
                ]
                check.assertEqual(len(selected), 3)
                values = np.array([float(row["wall_median_of_block_medians_ms"]) for row in selected])
                ratios = (
                    np.array([float(row["steady_32_relative_ratio"]) for row in selected]) if mode == "steady" else None
                )
                row = {
                    "mode": mode,
                    "instances": count,
                    "low_segments": low,
                    "independent_jobs": len(values),
                    "median_job_latency_ms": float(np.median(values)),
                    "minimum_job_latency_ms": float(values.min()),
                    "maximum_job_latency_ms": float(values.max()),
                    "job_range_relative_percent": float((values.max() - values.min()) / np.median(values) * 100),
                    "median_paired_32_ratio": float(np.median(ratios)) if ratios is not None else "",
                    "minimum_paired_32_ratio": float(ratios.min()) if ratios is not None else "",
                    "maximum_paired_32_ratio": float(ratios.max()) if ratios is not None else "",
                }
                aggregates.append(row)
    write_csv(output / "across_jobs.csv", aggregates)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    colors = {3: "#0072B2", 8: "#009E73", 16: "#D55E00", 32: "#CC79A7"}
    for low in (3, 8, 16, 32):
        selected = [row for row in aggregates if row["mode"] == "steady" and row["low_segments"] == low]
        values = np.array([row["median_job_latency_ms"] for row in selected])
        error = np.array(
            [
                [row["median_job_latency_ms"] - row["minimum_job_latency_ms"] for row in selected],
                [row["maximum_job_latency_ms"] - row["median_job_latency_ms"] for row in selected],
            ]
        )
        axes[0].errorbar(
            (64, 512, 4096), values, yerr=error, color=colors[low], marker="o", capsize=4, label=f"segments={low}"
        )
        if low != 32:
            values = np.array([row["median_paired_32_ratio"] for row in selected])
            error = np.array(
                [
                    [row["median_paired_32_ratio"] - row["minimum_paired_32_ratio"] for row in selected],
                    [row["maximum_paired_32_ratio"] - row["median_paired_32_ratio"] for row in selected],
                ]
            )
            axes[1].errorbar(
                (64, 512, 4096), values, yerr=error, color=colors[low], marker="o", capsize=4, label=f"32 / {low}"
            )
    for low in (3, 8):
        selected = [row for row in aggregates if row["mode"] == "switch_all" and row["low_segments"] == low]
        values = np.array([row["median_job_latency_ms"] for row in selected])
        error = np.array(
            [
                [row["median_job_latency_ms"] - row["minimum_job_latency_ms"] for row in selected],
                [row["maximum_job_latency_ms"] - row["median_job_latency_ms"] for row in selected],
            ]
        )
        axes[2].errorbar(
            (64, 512, 4096),
            values,
            yerr=error,
            color=colors[low],
            marker="o",
            capsize=4,
            label=f"All instances: 32 ↔ {low}",
        )
    axes[0].set(title="Static filled scene", ylabel="Completed frame latency (ms)", yscale="log")
    axes[1].set(title="Paired static latency ratios", ylabel="32-segment latency / lower-LOD latency")
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[2].set(title="Cached switching every frame", ylabel="Completed frame latency (ms)", yscale="log")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks((64, 512, 4096), ("64", "512", "4096"))
        axis.set_xlabel("Capsule instances")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Three independent Slurm overflow allocations on three nodes; one exclusive B200 each\nPoints: median of job summaries. Bars: full range across jobs, not confidence intervals.",
        fontsize=11,
    )
    figure.savefig(output / "across_jobs.png", dpi=170)
    figure.savefig(output / "across_jobs.svg")
    plt.close(figure)
    report = [
        "Warp #1110: three additional independent Slurm allocations",
        "Source: cb86abb75d123876d80c6832d8d360f4cda0cc10",
        "No code, native artifacts or measurement thresholds were changed.",
        "Only the random case-order seed differs between the three benchmark runs.",
        "Partition overflow; one full B200 GPU per allocation, eight CPU cores; nodes may be shared.",
        "Three distinct nodes were selected; public records label them A/B/C instead of disclosing hostnames.",
        "",
        *[
            f"Job {job['job_id']}, node group {job['node_group']}, seed {job['seed']}: {job['state']}, exit {job['exit_code']}, elapsed {job['elapsed']}"
            for job in jobs
        ],
        "",
        f"New totals: {filled_total} filled frames, {restored_total} exact restored images, {frames_total} timed frames, 270 measurement blocks.",
        "The earlier allocation 1143896 and shared-GPU measurements are excluded from these new totals.",
        "Source, native-library hashes, benchmark script, versions and thread settings match across all three jobs.",
        "Each job independently verifies native CUDA/EGL interop, one visible CUDA device, MIG disabled and no preexisting GPU process.",
        "",
        "Static latency: median across three job summaries (ms/frame), with job minimum/maximum:",
    ]
    for row in aggregates:
        if row["mode"] == "steady":
            report.append(
                f"{row['instances']} instances, {row['low_segments']} segments: {row['median_job_latency_ms']:.6f} [{row['minimum_job_latency_ms']:.6f}, {row['maximum_job_latency_ms']:.6f}]"
            )
    report.extend(["", "All-instance dynamic switching (ms/frame):"])
    for row in aggregates:
        if row["mode"] == "switch_all":
            report.append(
                f"{row['instances']} instances, 32/{row['low_segments']}: {row['median_job_latency_ms']:.6f} [{row['minimum_job_latency_ms']:.6f}, {row['maximum_job_latency_ms']:.6f}]"
            )
    report.extend(
        [
            "",
            "Method: each job performs five seeded randomized repeats, eight warmup frames per case, 32 static or 12 dynamic measured frames per block.",
            "Wall-clock timing waits for GL completion on every frame. Pixel reads, creation, initial upload and warmup are excluded.",
            "A job summary is the median of five block medians. Across-jobs results use the median and full observed range of the three job summaries.",
            "No frame is removed for being slow. Raw per-frame and block data, including tails, remain in the individual job directories.",
            "There are three independent allocations, not 6840 independent statistical samples. Error bars are not confidence intervals.",
            "Nodes and other GPUs can be shared; CPU/node activity, device variation, clocks and thermal state can affect results. No clocks were locked.",
            "Lower resolution changes fidelity. Reciprocals of frame latency do not establish application or display FPS.",
            "No GitHub publications are performed. Full Slurm output with hostnames stays in private-slurm; public accounting uses node-group labels.",
            "",
            "Files: per_job_summary.csv, across_jobs.csv, across_jobs.png/.svg, validation.json, ACCOUNTING.csv, SHA256SUMS.",
        ]
    )
    (output / "REPORT.txt").write_text("\n".join(report) + "\n")
    (output / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "jobs": jobs,
                "filled_frames": filled_total,
                "restored_images_exact": restored_total,
                "timed_frames": frames_total,
                "measurement_blocks": 270,
                "source_environment_identical": True,
                "job_manifests_verified": manifest_checks,
                "three_distinct_nodes_verified": True,
            },
            indent=2,
        )
        + "\n"
    )
    write_csv(output / "ACCOUNTING.csv", [accounting[row["job_id"]] for row in submissions])
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files)
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "jobs": [row["job_id"] for row in submissions],
                "timed_frames": frames_total,
                "filled_frames": filled_total,
            }
        )
    )


if __name__ == "__main__":
    main()
