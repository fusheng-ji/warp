# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the recorded nonfinite SVD3 evaluations without rerunning Warp."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    """Write a summary CSV and figures from the adjacent per-evaluation CSV."""
    root = Path(__file__).resolve().parent
    with (root / "nonfinite.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    summary = []
    for device in dict.fromkeys(row["device"] for row in rows):
        for dtype in ("float16", "float32", "float64"):
            group = [
                row for row in rows if row["device"] == device and row["dtype"] == dtype
            ]
            invalid = [row for row in group if row["zero_control"] == "False"]
            zeros = [row for row in group if row["zero_control"] == "True"]
            summary.append(
                {
                    "device": device,
                    "dtype": dtype,
                    "evaluations": len(group),
                    "invalid_passed": sum(row["passed"] == "True" for row in invalid),
                    "invalid_total": len(invalid),
                    "zero_passed": sum(row["passed"] == "True" for row in zeros),
                    "zero_total": len(zeros),
                    "failed": sum(row["passed"] != "True" for row in group),
                }
            )
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    labels = [f"{row['device']}\n{row['dtype']}" for row in summary]
    invalid = [row["invalid_passed"] for row in summary]
    zeros = [row["zero_passed"] for row in summary]
    ax.bar(labels, invalid, color="#287D8E", label="Invalid-input propagation passed")
    ax.bar(labels, zeros, bottom=invalid, color="#80C7B9", label="Zero controls passed")
    for i, row in enumerate(summary):
        ax.text(
            i,
            invalid[i] + zeros[i] + 0.7,
            f"{invalid[i] + zeros[i]}/{row['evaluations']}",
            ha="center",
        )
    ax.set_ylim(0, max(row["evaluations"] for row in summary) * 1.22)
    ax.set_ylabel("Evaluations (both SVD calling forms)")
    passed = sum(row["passed"] == "True" for row in rows)
    ax.set_title(
        f"SVD3 invalid-input validation: {passed}/{len(rows)} passed\n"
        "Source 400ff598 | CPU + one NVIDIA B200 | current run only"
    )
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "svg"):
        fig.savefig(
            root / f"nonfinite-validation.{extension}",
            dpi=180,
            metadata={"Creator": "Matplotlib"} if extension == "svg" else None,
        )
    plt.close(fig)
    svg_path = root / "nonfinite-validation.svg"
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
