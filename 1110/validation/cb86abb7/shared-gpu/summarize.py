"""Check the archived measurements and create standalone figures and CSVs."""

import argparse
import csv
import json
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    check = unittest.TestCase()
    environment = json.loads((root / "benchmark/environment.json").read_text())
    check.assertEqual(environment["status"], "passed")
    check.assertEqual(environment["source_commit"], "cb86abb75d123876d80c6832d8d360f4cda0cc10")
    check.assertEqual(environment["source_dirty_before"], "")
    check.assertEqual(environment["source_dirty_after"], "")
    frames = read_csv(root / "benchmark/frames.csv")
    check.assertEqual(len(frames), 2280)
    check.assertEqual(len(environment["cases"]), 90)
    counts = sorted(environment["arguments"]["counts"])
    segments = sorted(environment["arguments"]["segments"])
    repeats = environment["arguments"]["repeats"]
    summaries = []
    for mode in ("steady", "switch_all"):
        for count in counts:
            for low in segments if mode == "steady" else (3, 8):
                cases = [
                    case
                    for case in environment["cases"]
                    if (case["mode"], case["instances"], case["low_segments"]) == (mode, count, low)
                ]
                check.assertEqual(len(cases), repeats)
                selected = [
                    row
                    for row in frames
                    if (row["mode"], int(row["instances"]), int(row["low_segments"])) == (mode, count, low)
                ]
                check.assertEqual(len(selected), repeats * (32 if mode == "steady" else 12))
                block_wall = np.array([case["wall_median_ms"] for case in cases])
                block_gl = np.array([case["gl_median_ms"] for case in cases])
                row = {
                    "mode": mode,
                    "instances": count,
                    "low_segments": low,
                    "repeats": repeats,
                    "measured_frames": len(selected),
                    "wall_median_of_block_medians_ms": float(np.median(block_wall)),
                    "wall_block_min_ms": float(block_wall.min()),
                    "wall_block_max_ms": float(block_wall.max()),
                    "wall_p95_all_frames_ms": float(np.percentile([float(r["wall_ms"]) for r in selected], 95)),
                    "gl_median_of_block_medians_ms": float(np.median(block_gl)),
                    "steady_32_relative_ratio": "",
                    "paired_ratio_min": "",
                    "paired_ratio_max": "",
                }
                if mode == "steady":
                    baseline = [
                        next(
                            case["wall_median_ms"]
                            for case in environment["cases"]
                            if (case["mode"], case["instances"], case["low_segments"], case["repeat"])
                            == ("steady", count, 32, r)
                        )
                        for r in range(repeats)
                    ]
                    by_repeat = [
                        next(case["wall_median_ms"] for case in cases if case["repeat"] == r) for r in range(repeats)
                    ]
                    ratios = np.array(baseline) / np.array(by_repeat)
                    row.update(
                        steady_32_relative_ratio=float(np.median(ratios)),
                        paired_ratio_min=float(ratios.min()),
                        paired_ratio_max=float(ratios.max()),
                    )
                summaries.append(row)
    write_csv(root / "performance_summary.csv", summaries)
    colors = {3: "#0072B2", 8: "#009E73", 16: "#D55E00", 32: "#CC79A7"}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for low in segments:
        rows = [r for r in summaries if r["mode"] == "steady" and r["low_segments"] == low]
        y = np.array([r["wall_median_of_block_medians_ms"] for r in rows])
        error = np.array(
            [
                [r["wall_median_of_block_medians_ms"] - r["wall_block_min_ms"] for r in rows],
                [r["wall_block_max_ms"] - r["wall_median_of_block_medians_ms"] for r in rows],
            ]
        )
        axes[0].errorbar(counts, y, yerr=error, marker="o", capsize=3, label=f"segments={low}", color=colors[low])
        if low != 32:
            ratio = np.array([r["steady_32_relative_ratio"] for r in rows])
            spread = np.array(
                [
                    [r["steady_32_relative_ratio"] - r["paired_ratio_min"] for r in rows],
                    [r["paired_ratio_max"] - r["steady_32_relative_ratio"] for r in rows],
                ]
            )
            axes[1].errorbar(counts, ratio, yerr=spread, marker="o", capsize=3, label=f"32 / {low}", color=colors[low])
    for low in (3, 8):
        rows = [r for r in summaries if r["mode"] == "switch_all" and r["low_segments"] == low]
        y = np.array([r["wall_median_of_block_medians_ms"] for r in rows])
        spread = np.array(
            [
                [r["wall_median_of_block_medians_ms"] - r["wall_block_min_ms"] for r in rows],
                [r["wall_block_max_ms"] - r["wall_median_of_block_medians_ms"] for r in rows],
            ]
        )
        axes[2].errorbar(
            counts, y, yerr=spread, marker="o", capsize=3, label=f"32 ↔ {low}, all instances", color=colors[low]
        )
    steady = [
        r["wall_median_of_block_medians_ms"] for r in summaries if r["mode"] == "steady" and r["low_segments"] == 32
    ]
    axes[2].plot(counts, steady, "--", label="static 32 reference", color=colors[32])
    axes[0].set(title="Static filled scene", ylabel="Completed frame latency (ms)", yscale="log")
    axes[1].set(title="Paired static latency ratios", ylabel="32-segment latency / lower-LOD latency")
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[2].set(title="Cached switching is additional work", ylabel="Completed frame latency (ms)", yscale="log")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(counts, [str(n) for n in counts])
        axis.set_xlabel("Capsule instances")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "B200 · filled 1280x720 · GL completion per frame · cb86abb7\nFive randomized repeats; bars show full range of block medians, not confidence intervals. Shared GPU; no fixed clocks.",
        fontsize=11,
    )
    figure.savefig(root / "capsule_performance.png", dpi=170)
    figure.savefig(root / "capsule_performance.svg")
    plt.close(figure)

    filled = read_csv(root / "filled/results.csv")
    check.assertEqual(len(filled), 40)
    kinds = list(dict.fromkeys(row["primitive"] for row in filled))
    figure, axes = plt.subplots(5, 4, figsize=(14, 11), layout="constrained")
    restored = 0
    for index, kind in enumerate(kinds):
        for cached in ("False", "True"):
            selected = [row for row in filled if row["primitive"] == kind and row["target_pre_cached"] == cached]
            check.assertEqual([int(row["segments"]) for row in selected], [32, 8, 3, 32])
            check.assertTrue(all(row["passed"] == "True" for row in selected))
            np.testing.assert_array_equal(
                plt.imread(root / "filled" / selected[0]["image"]), plt.imread(root / "filled" / selected[-1]["image"])
            )
            restored += 1
        selected = [row for row in filled if row["primitive"] == kind and row["target_pre_cached"] == "False"]
        for column, row in enumerate(selected):
            axes[index, column].imshow(plt.imread(root / "filled" / row["image"]))
            axes[index, column].set(
                xticks=[], yticks=[], xlabel=f"{row['vertices']} vertices · {row['triangles']} triangles"
            )
            if column == 0:
                axes[index, column].set_ylabel(kind.capitalize())
            if index == 0:
                axes[index, column].set_title(
                    ["32 segments", "8 segments", "3 segments (minimum)", "32 restored"][column]
                )
    figure.suptitle(
        "Filled primitive resolution: changing left object, unchanged right object\nFive shapes x cached/uncached targets x four frames; all ten restored images exactly match."
    )
    figure.savefig(root / "filled_comparison.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    for axis, low in zip(axes, segments, strict=True):
        axis.imshow(plt.imread(root / f"benchmark/capsules-512-segments-{low}.png"))
        axis.set(xticks=[], yticks=[], title=f"segments={low}")
    figure.suptitle("512 capsules, filled rendering: lower resolution changes visual fidelity")
    figure.savefig(root / "capsule_quality.png", dpi=160)
    plt.close(figure)
    summary = {
        "filled_frames": len(filled),
        "restored_images_exact": restored,
        "performance_frames": len(frames),
        "performance_blocks": len(environment["cases"]),
        "status": "passed",
    }
    (root / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
