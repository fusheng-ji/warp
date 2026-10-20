# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a contact sheet and mesh counts from validate_render.py output."""

import argparse
import csv
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.input / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    check = unittest.TestCase()
    check.assertEqual(len(rows), 30)
    check.assertTrue(all(row["passed"] == "True" for row in rows))
    kinds = list(dict.fromkeys(row["primitive"] for row in rows))
    figure, axes = plt.subplots(len(kinds), 3, figsize=(13, 12), layout="constrained")
    counts = []
    for row_index, kind in enumerate(kinds):
        selected = [row for row in rows if row["primitive"] == kind and row["target_pre_cached"] == "False"]
        check.assertEqual([row["segments"] for row in selected], ["32", "8", "32"])
        np.testing.assert_array_equal(
            plt.imread(args.input / selected[0]["image"]), plt.imread(args.input / selected[2]["image"])
        )
        for column, row in enumerate(selected):
            axis = axes[row_index, column]
            axis.imshow(plt.imread(args.input / row["image"]))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel(f"{row['vertices']} vertices / {row['triangles']} triangles", fontsize=10)
            if column == 0:
                axis.set_ylabel(kind.capitalize(), fontsize=12)
            if row_index == 0:
                axis.set_title(("32 segments", "8 segments", "32 segments restored")[column], fontsize=13)
        counts.append(
            {
                "primitive": kind,
                "vertices_32": selected[0]["vertices"],
                "vertices_8": selected[1]["vertices"],
                "triangles_32": selected[0]["triangles"],
                "triangles_8": selected[1]["triangles"],
                "restored_pixels_exact": True,
            }
        )
    figure.suptitle(
        "OpenGL primitive resolution: 32 → 8 → 32\n"
        "Left object changes resolution; right object stays at 32. NVIDIA B200, wireframe rendering.",
        fontsize=14,
    )
    figure.savefig(args.output / "resolution_comparison.png", dpi=140)
    plt.close(figure)
    with (args.output / "mesh_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(counts[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(counts)


if __name__ == "__main__":
    main()
