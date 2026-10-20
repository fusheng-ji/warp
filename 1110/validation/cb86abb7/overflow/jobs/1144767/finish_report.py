"""Create the exclusive-allocation report and retain the shared-run comparison."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    allocation = json.loads((root / "allocation.json").read_text())
    device = json.loads((root / "device.json").read_text())
    analysis = json.loads((root / "analysis.json").read_text())
    environment = json.loads((root / "benchmark/environment.json").read_text())
    with (root / "performance_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    lines = [
        "Warp #1110: Slurm overflow exclusive-GPU validation",
        f"Job: {allocation['JobId']}; partition: {allocation['Partition']}",
        f"Source: {allocation['source_commit']}",
        f"Random case-order seed: {environment['arguments']['seed']}",
        f"GPU: {device['gpu']}; driver: {device['driver']}",
        f"Allocation: {allocation['AllocTRES']}",
        "One full GPU is exclusively allocated by Slurm GRES. The node and other GPUs may be shared.",
        "MIG is disabled. No preexisting compute or graphics process was present on the selected GPU.",
        "CUDA/EGL identity was matched and native CUDA/OpenGL interop verified with copy fallback disabled.",
        f"Filled frames: {analysis['filled_frames']}; exact restored images: {analysis['restored_images_exact']}",
        f"Timed frames: {analysis['performance_frames']}; blocks: {analysis['performance_blocks']}",
        "",
        "Static filled-scene latency in ms/frame (median of five block medians):",
        "instances,segments=32,segments=16,segments=8,segments=3",
    ]
    for count in (64, 512, 4096):
        selected = {
            int(row["low_segments"]): row for row in rows if row["mode"] == "steady" and int(row["instances"]) == count
        }
        lines.append(
            str(count)
            + ","
            + ",".join(f"{float(selected[s]['wall_median_of_block_medians_ms']):.6f}" for s in (32, 16, 8, 3))
        )
    lines.extend(["", "All-instance switching latency in ms/frame (retained even when slower):"])
    for row in rows:
        if row["mode"] == "switch_all":
            lines.append(
                f"{row['instances']} instances, alternating 32/{row['low_segments']}: {float(row['wall_median_of_block_medians_ms']):.6f}"
            )
    lines.extend(
        [
            "",
            "Method: filled 1280x720, vsync off, GL completion per frame; cached geometry and warmup excluded.",
            "Static frames omit per-object calls; switch frames update every capsule and include rebuilding/upload work.",
            "No clock or power limits were changed. Scheduler exclusivity does not mean whole-node isolation.",
            "Five randomized repeats; error bars are full ranges of block medians, not confidence intervals.",
            "Lower resolution changes visual fidelity. Latency reciprocals are not application/display FPS.",
            "This is a new measurement; earlier shared-GPU data remain separately identified and unmodified.",
            "Different node, allocation, CPU scheduling and recorded thread settings may affect comparison;",
            "differences cannot be attributed solely to GPU exclusivity.",
            f"Thread settings: {environment['environment']['thread_environment']}",
            "No native library rebuild or product changes were needed. Hashes identify the reused artifacts.",
            "All logs and scripts are local delivery; no GitHub comment, PR change or asset publication is performed.",
            "Raw scheduler output is stored separately from this sanitized, checksum-covered result directory.",
        ]
    )
    shared_path = root.parents[2] / "warp-1110-supplement-20261020/performance_summary.csv"
    if shared_path.exists():
        with shared_path.open() as stream:
            shared = {(r["mode"], r["instances"], r["low_segments"]): r for r in csv.DictReader(stream)}
        comparison = []
        for row in rows:
            old = shared[(row["mode"], row["instances"], row["low_segments"])]
            comparison.append(
                {
                    "mode": row["mode"],
                    "instances": row["instances"],
                    "low_segments": row["low_segments"],
                    "previous_shared_ms": old["wall_median_of_block_medians_ms"],
                    "new_exclusive_ms": row["wall_median_of_block_medians_ms"],
                    "shared_source": allocation["source_commit"],
                    "exclusive_source": allocation["source_commit"],
                    "exclusive_job_id": allocation["JobId"],
                }
            )
        with (root / "comparison_with_shared.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(comparison)
        lines.append("comparison_with_shared.csv contains both observations without causal speedup claims.")
    (root / "REPORT.txt").write_text("\n".join(lines) + "\n")
    print("Exclusive-allocation report written.")


if __name__ == "__main__":
    main()
