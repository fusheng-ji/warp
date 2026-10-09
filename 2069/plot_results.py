# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render reproducible static figures from GH-2069 measurement CSV files."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(
            r[k]
            for k in (
                "nrow",
                "length",
                "storage",
                "compute",
                "distribution",
                "block_rows",
                "block_cols",
                "mode",
                "path",
            )
        )
        groups[key].append(r)
    result = []
    for key, samples in groups.items():
        t = np.array([float(r["us"]) for r in samples])
        q = np.percentile(t, [25, 50, 75])
        result.append(
            dict(
                zip(
                    (
                        "nrow",
                        "length",
                        "storage",
                        "compute",
                        "distribution",
                        "block_rows",
                        "block_cols",
                        "mode",
                        "path",
                    ),
                    key,
                    strict=True,
                )
            )
            | dict(
                q25_us=q[0],
                median_us=q[1],
                q75_us=q[2],
                samples=len(t),
                max_abs_error=max(float(r["max_abs_error"]) for r in samples),
            )
        )
    return result


def main(args):
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    device_name = json.loads((root / "final-uniform.json").read_text())["device"]
    rows = read(root / "final-uniform.csv")
    summary = summarize(rows)
    with (root / "timing-summary.csv").open("w") as f:
        out = csv.DictWriter(f, fieldnames=list(summary[0]))
        out.writeheader()
        out.writerows(summary)
    table = {tuple(r[k] for k in ("nrow", "length", "storage", "compute", "mode", "path")): r for r in summary}
    precisions = [("float32", "float32"), ("float64", "float64"), ("float32", "float64")]
    lengths = sorted({int(r["length"]) for r in rows})
    sizes = sorted({int(r["nrow"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
    speedups = []
    for ax, (s, c) in zip(axes, precisions, strict=True):
        a = np.empty((len(sizes), len(lengths)))
        for i, n in enumerate(sizes):
            for j, length in enumerate(lengths):
                t = {
                    p: table[(str(n), str(length), s, c, "graph", p)]["median_us"]
                    for p in ["auto", "scalar", "packed128t4"]
                }
                a[i, j] = min(t["auto"], t["scalar"]) / t["packed128t4"]
                speedups.append(dict(nrow=n, length=length, storage=s, compute=c, speedup=a[i, j]))
                ax.text(j, i, f"{a[i, j]:.2f}", ha="center", va="center", fontsize=9)
        im = ax.imshow(np.log2(a), cmap="RdYlGn", vmin=-3, vmax=3, aspect="auto")
        ax.set(
            xticks=range(len(lengths)),
            xticklabels=lengths,
            yticks=range(len(sizes)),
            yticklabels=[f"{n:,}" for n in sizes],
            xlabel="Entries per row",
            title=f"{s} matrix / {c} vector",
        )
    axes[0].set_ylabel("Scalar rows")
    fig.colorbar(im, ax=axes, shrink=0.8, label="log2(speedup)")
    fig.suptitle(
        f"Row packing vs faster of auto/scalar — {device_name}, CUDA Graph\n128 threads/block, 4 threads/row; values below 1 are slower",
        fontsize=12,
    )
    for ext in ("png", "svg"):
        fig.savefig(root / f"row-packing-speedup.{ext}")
    plt.close(fig)
    with (root / "speedup-summary.csv").open("w") as f:
        out = csv.DictWriter(f, fieldnames=list(speedups[0]))
        out.writeheader()
        out.writerows(speedups)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    n = "32768"
    for ax, mode in zip(axes, ["graph", "launch"], strict=True):
        for path, label in [
            ("auto", "Automatic"),
            ("scalar", "Scalar"),
            ("tile64", "One row / block (64 threads)"),
            ("packed128t4", "32 rows / block (128 threads)"),
        ]:
            t = [table[(n, str(l), "float32", "float64", mode, path)] for l in lengths]
            y = np.array([r["median_us"] for r in t])
            low = np.array([r["q25_us"] for r in t])
            high = np.array([r["q75_us"] for r in t])
            ax.plot(lengths, y, "o-", label=label, linewidth=1.5, markersize=4)
            ax.fill_between(lengths, low, high, alpha=0.12)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Entries per row",
            ylabel="Microseconds / product",
            title="CUDA Graph" if mode == "graph" else "Ordinary launch (stream elapsed)",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("FP32 matrix / FP64 vectors, 32,768 rows — median and interquartile range")
    for ext in ("png", "svg"):
        fig.savefig(root / f"latency.{ext}")
    plt.close(fig)
    fem_samples = defaultdict(list)
    for row in read(root / "final-fem.csv"):
        fem_samples[(row["matrix"], row["storage"], row["compute"], row["path"])].append(float(row["us"]))
    matrices = ["BsrMvLinearGridMatrix", "BsrMvQuadraticTetmeshMatrix", "BsrMvAlmostDense"]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for i, (storage, compute) in enumerate(precisions):
        ratios = []
        for matrix in matrices:
            medians = {
                path: np.median(fem_samples[(matrix, storage, compute, path)])
                for path in ["auto", "scalar", "packed128t4"]
            }
            ratios.append(min(medians["auto"], medians["scalar"]) / medians["packed128t4"])
        bars = ax.bar(np.arange(3) + (i - 1) * 0.24, ratios, width=0.23, label=f"{storage} / {compute}")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
    ax.set(
        xticks=np.arange(3),
        xticklabels=["Linear grid", "Quadratic tetmesh", "Almost dense"],
        ylabel="Speedup vs faster of auto/scalar",
        title=f"FEM matrices — {device_name}, CUDA Graph\n128 threads/block, 4 threads/row; values below 1 are slower",
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=9, loc="upper right")
    for ext in ("png", "svg"):
        fig.savefig(root / f"fem-speedup.{ext}")
    plt.close(fig)
    before = read(root / "numerics-before.csv")
    after = read(root / "numerics-after.csv")
    devices = ["cpu", "cuda:0", "cuda:1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    x = np.arange(3)
    for i, (name, data) in enumerate([("Before", before), ("After", after)]):
        axes[0].bar(
            x + (i - 0.5) * 0.32,
            [sum(r["status"] == "pass" for r in data if r["device"] == d) for d in devices],
            width=0.32,
            label=name,
        )
    axes[0].set(xticks=x, xticklabels=devices, ylabel="Passing cases", title="Mixed-precision acceptance")
    axes[0].legend()
    axes[1].axis("off")
    axes[1].text(
        0.05,
        0.8,
        f"Before: {sum(r['status'] == 'pass' for r in before)} / {len(before)} passed\nAfter: {sum(r['status'] == 'pass' for r in after)} / {len(after)} passed\n\nExact diagnostic cases: max absolute error = "
        + str(max(float(r["max_abs_error"]) for r in after if r["max_abs_error"]))
        + "\nRandom-matrix errors are reported in timing-summary.csv.\nReference uses the actual stored matrix coefficients.",
        va="top",
        linespacing=1.7,
    )
    for ext in ("png", "svg"):
        fig.savefig(root / f"mixed-precision-correctness.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    main(parser.parse_args())
