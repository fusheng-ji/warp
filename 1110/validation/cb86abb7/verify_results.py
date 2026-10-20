# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recompute saved frame statistics and publication tables without running Warp."""

import argparse
import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCE = "cb86abb75d123876d80c6832d8d360f4cda0cc10"
PRIMARY_JOBS = ("1144767", "1144768", "1144769")


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def compare_numeric(check, actual, expected):
    check.assertEqual(set(actual), set(expected))
    for key, value in expected.items():
        if isinstance(value, (int, float)):
            np.testing.assert_allclose(float(actual[key]), value, rtol=1e-12, atol=1e-12, err_msg=key)
        else:
            check.assertEqual(actual[key], value, key)


def recompute(base, check):
    env = json.loads((base / "benchmark/environment.json").read_text())
    check.assertEqual(env["status"], "passed")
    check.assertEqual(env["source_commit"], SOURCE)
    check.assertEqual(env["source_dirty_before"], "")
    check.assertEqual(env["source_dirty_after"], "")
    args = env["arguments"]
    blocks = defaultdict(list)
    frames = read_csv(base / "benchmark/frames.csv")
    for row in frames:
        blocks[(row["mode"], int(row["instances"]), int(row["low_segments"]), int(row["repeat"]))].append(row)
        check.assertTrue(np.isfinite([float(row["wall_ms"]), float(row["gl_ms"])]).all())
        check.assertGreater(float(row["wall_ms"]), 0)
        check.assertGreater(float(row["gl_ms"]), 0)
        target = int(row["target_segments"])
        triangles = 2 * (target + 1) ** 2
        check.assertEqual(int(row["triangles_per_instance"]), triangles)
        check.assertEqual(int(row["submitted_triangles"]), triangles * int(row["instances"]))
    expected_blocks = args["repeats"] * len(args["counts"]) * (len(args["segments"]) + 2)
    check.assertEqual(len(blocks), expected_blocks)
    medians = {}
    for key, rows in blocks.items():
        mode, _count, low, _repeat = key
        expected_count = args["frames"] if mode == "steady" else args["dynamic_frames"]
        check.assertEqual([int(row["frame"]) for row in rows], list(range(expected_count)))
        check.assertEqual(
            [int(row["target_segments"]) for row in rows],
            [low if mode == "steady" else (32 if index % 2 == 0 else low) for index in range(expected_count)],
        )
        wall = float(np.median([float(row["wall_ms"]) for row in rows]))
        gl = float(np.median([float(row["gl_ms"]) for row in rows]))
        medians[key] = wall, gl
        metadata = next(
            case
            for case in env["cases"]
            if (case["mode"], case["instances"], case["low_segments"], case["repeat"]) == key
        )
        check.assertTrue(metadata["passed"])
        np.testing.assert_allclose(
            [metadata["wall_median_ms"], metadata["gl_median_ms"]], [wall, gl], rtol=1e-12, atol=1e-12
        )
        check.assertEqual(metadata["measured_frames"], expected_count)
    saved = {
        (row["mode"], int(row["instances"]), int(row["low_segments"])): row
        for row in read_csv(base / "performance_summary.csv")
    }
    recomputed = {}
    for mode in ("steady", "switch_all"):
        for count in args["counts"]:
            for low in args["segments"] if mode == "steady" else (3, 8):
                keys = [(mode, count, low, repeat) for repeat in range(args["repeats"])]
                wall = np.array([medians[key][0] for key in keys])
                gl = np.array([medians[key][1] for key in keys])
                selected = [float(row["wall_ms"]) for key in keys for row in blocks[key]]
                row = {
                    "mode": mode,
                    "instances": count,
                    "low_segments": low,
                    "repeats": args["repeats"],
                    "measured_frames": len(selected),
                    "wall_median_of_block_medians_ms": float(np.median(wall)),
                    "wall_block_min_ms": float(wall.min()),
                    "wall_block_max_ms": float(wall.max()),
                    "wall_p95_all_frames_ms": float(np.percentile(selected, 95)),
                    "gl_median_of_block_medians_ms": float(np.median(gl)),
                    "steady_32_relative_ratio": "",
                    "paired_ratio_min": "",
                    "paired_ratio_max": "",
                }
                if mode == "steady":
                    ratios = np.array(
                        [
                            medians[(mode, count, 32, repeat)][0] / medians[(mode, count, low, repeat)][0]
                            for repeat in range(args["repeats"])
                        ]
                    )
                    row.update(
                        steady_32_relative_ratio=float(np.median(ratios)),
                        paired_ratio_min=float(ratios.min()),
                        paired_ratio_max=float(ratios.max()),
                    )
                compare_numeric(check, saved[(mode, count, low)], row)
                recomputed[(mode, count, low)] = row
    check.assertEqual(len(saved), len(recomputed))
    return recomputed, len(frames), len(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.input.resolve()
    check = unittest.TestCase()
    computed = {}
    totals = {"filled_frames": 0, "restored_images_exact": 0, "timed_frames": 0, "measurement_blocks": 0}
    datasets = []
    for relative, primary in [
        ("shared-gpu", False),
        ("overflow/jobs/1143896", False),
        *[(f"overflow/jobs/{job}", True) for job in PRIMARY_JOBS],
    ]:
        base = root / relative
        rows, frames, blocks = recompute(base, check)
        filled = read_csv(base / "filled/results.csv")
        check.assertEqual(len(filled), 40)
        check.assertTrue(all(row["passed"] == "True" for row in filled))
        groups = defaultdict(list)
        for row in filled:
            groups[(row["primitive"], row["target_pre_cached"])].append(row)
            check.assertGreater(int(row["left_foreground_pixels"]), 32)
            check.assertGreater(int(row["right_foreground_pixels"]), 32)
            check.assertEqual(int(row["instance_count"]), 2)
        check.assertEqual(len(groups), 10)
        for group in groups.values():
            check.assertEqual([int(row["segments"]) for row in group], [32, 8, 3, 32])
            check.assertEqual(
                [int(row["cached_meshes"]) for row in group],
                [3, 3, 3, 3] if group[0]["target_pre_cached"] == "True" else [1, 2, 3, 3],
            )
            # PNG streams are lossless; compare decoded RGB pixels directly.
            from PIL import Image  # noqa: PLC0415

            with (
                Image.open(base / "filled" / group[0]["image"]) as first,
                Image.open(base / "filled" / group[-1]["image"]) as last,
            ):
                np.testing.assert_array_equal(np.asarray(first), np.asarray(last))
        datasets.append(
            {
                "directory": relative,
                "primary": primary,
                "frames": frames,
                "blocks": blocks,
                "filled_frames": len(filled),
                "restored_exact": len(groups),
            }
        )
        if primary:
            computed[base.name] = rows
            totals["filled_frames"] += len(filled)
            totals["restored_images_exact"] += len(groups)
            totals["timed_frames"] += frames
            totals["measurement_blocks"] += blocks
    check.assertEqual(
        totals, {"filled_frames": 120, "restored_images_exact": 30, "timed_frames": 6840, "measurement_blocks": 270}
    )
    across = {
        (row["mode"], int(row["instances"]), int(row["low_segments"])): row
        for row in read_csv(root / "overflow/independent-summary/across_jobs.csv")
    }
    medians = {}
    for key in computed[PRIMARY_JOBS[0]]:
        mode, count, low = key
        values = np.array([computed[job][key]["wall_median_of_block_medians_ms"] for job in PRIMARY_JOBS])
        ratios = (
            np.array([computed[job][key]["steady_32_relative_ratio"] for job in PRIMARY_JOBS])
            if mode == "steady"
            else None
        )
        row = {
            "mode": mode,
            "instances": count,
            "low_segments": low,
            "independent_jobs": 3,
            "median_job_latency_ms": float(np.median(values)),
            "minimum_job_latency_ms": float(values.min()),
            "maximum_job_latency_ms": float(values.max()),
            "job_range_relative_percent": float((values.max() - values.min()) / np.median(values) * 100),
            "median_paired_32_ratio": float(np.median(ratios)) if ratios is not None else "",
            "minimum_paired_32_ratio": float(ratios.min()) if ratios is not None else "",
            "maximum_paired_32_ratio": float(ratios.max()) if ratios is not None else "",
        }
        compare_numeric(check, across[key], row)
        medians[key] = row
    check.assertEqual(len(across), 18)

    def cell(key):
        row = medians[key]
        return f"{row['median_job_latency_ms']:.3f} [{row['minimum_job_latency_ms']:.3f}, {row['maximum_job_latency_ms']:.3f}]"

    text = [
        "# Independent allocation results",
        "",
        "Source: `cb86abb7`. Jobs: 1144767, 1144768, 1144769 on three distinct nodes, one exclusive B200 GPU each.",
        "",
        "Values are ms/frame: median of three job summaries [minimum, maximum]. Each job summary is the median of five block medians. Ranges are observations, not confidence intervals.",
        "",
        "## Static filled rendering",
        "",
        "| Capsules | 32 segments | 16 segments | 8 segments | 3 segments |",
        "|---:|---:|---:|---:|---:|",
    ]
    for count in (64, 512, 4096):
        text.append(f"| {count} | " + " | ".join(cell(("steady", count, low)) for low in (32, 16, 8, 3)) + " |")
    text += [
        "",
        "## All-instance switching on every frame",
        "",
        "| Capsules | Alternating 32/8 | Alternating 32/3 |",
        "|---:|---:|---:|",
    ]
    for count in (64, 512, 4096):
        text.append(f"| {count} | " + " | ".join(cell(("switch_all", count, low)) for low in (8, 3)) + " |")
    text += [
        "",
        "These 120 filled frames, 30 exact restored images, 6840 timed frames and 270 blocks exclude all previous allocations and shared-GPU runs. The statistical units here are three allocations, not 6840 independent samples.",
        "",
        "Lower resolution changes visual fidelity. Timings wait for GL completion; they are not application/display FPS. Nodes and other GPUs can be shared; no clocks were locked. Dynamic switching includes all per-object calls and rebuilding/upload work.",
        "",
        "[Full-precision CSV](overflow/independent-summary/across_jobs.csv) · [Per-job CSV](overflow/independent-summary/per_job_summary.csv) · [Raw frames by job](overflow/jobs/)",
        "",
    ]
    (root / "TABLES.md").write_text("\n".join(text))
    (root / "recomputation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "source": SOURCE,
                "primary_jobs": PRIMARY_JOBS,
                "primary_totals": totals,
                "datasets": datasets,
                "statistics": "Recomputed from every raw timing row; all saved block, job, percentile, paired-ratio and across-job summaries checked at 1e-12 absolute/relative tolerance; all restored PNG pixels checked exactly.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": "passed", "primary_totals": totals, "verified_datasets": len(datasets)}))


if __name__ == "__main__":
    main()
