# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build public tables, plots, and a report from the two measurement CSVs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def save_figure(figure, root, name):
    figure.savefig(root / f"{name}.png", dpi=170, facecolor="white", metadata={"Software": "Matplotlib"})
    figure.savefig(root / f"{name}.svg", facecolor="white", metadata={"Date": None, "Creator": "Matplotlib"})
    plt.close(figure)


def main():
    root = Path(__file__).resolve().parent
    before, after = (read_rows(root / f"{phase}.csv") for phase in ("before", "after"))
    metadata = [json.loads((root / f"{phase}-metadata.json").read_text()) for phase in ("before", "after")]
    keys = ("device", "dtype", "n", "overload", "scenario", "sample", "seed")
    if [tuple(row[k] for k in keys) for row in before] != [tuple(row[k] for k in keys) for row in after]:
        raise ValueError("Before and after evaluations must have identical inputs and ordering")
    summaries = []
    for device in ("cpu", "cuda:0", "cuda:1"):
        for dtype in ("float16", "float32", "float64"):
            old = [r for r in before if r["device"] == device and r["dtype"] == dtype]
            new = [r for r in after if r["device"] == device and r["dtype"] == dtype]
            summaries.append(
                {
                    "device": device,
                    "dtype": dtype,
                    "evaluations_per_phase": len(new),
                    "failed_before": sum(r["passed"] != "True" for r in old),
                    "failed_after": sum(r["passed"] != "True" for r in new),
                    "nonfinite_gradients_before": sum(r["finite_gradient"] != "True" for r in old),
                    "nonfinite_gradients_after": sum(r["finite_gradient"] != "True" for r in new),
                    "max_abs_gradient_error_after": max(float(r["max_abs_gradient_error"]) for r in new),
                    "max_gradient_error_ratio_after": max(float(r["gradient_error_ratio"]) for r in new),
                    "max_loss_error_ratio_after": max(float(r["loss_error_ratio"]) for r in new),
                }
            )
    with (root / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle("SVD3 FP16: finite gradients and independent losses", fontsize=18, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.915,
        "A = 2I  |  CUDA:0 (NVIDIA B200)  |  return-value overload  |  output seed = 1",
        ha="center",
        color="#475569",
    )
    for row_index, scenario in enumerate(("norm", "diagnostic_trace")):
        records = [
            next(
                r
                for r in rows
                if r["device"] == "cuda:0"
                and r["dtype"] == "float16"
                and r["n"] == "3"
                and r["overload"] == "return_values"
                and r["scenario"] == scenario
                and r["sample"] == "1"
            )
            for rows in (before, after)
        ]
        matrices = [np.array(json.loads(r["gradient"]), dtype=float) for r in records]
        matrices.append(np.array(json.loads(records[1]["expected_gradient"]), dtype=float))
        for column, values in enumerate(matrices):
            ax = axes[row_index, column]
            cmap = plt.get_cmap("RdBu").copy()
            cmap.set_bad("#e2e8f0")
            maximum = 4.0 if row_index == 0 else 1.0
            ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-maximum, vmax=maximum)
            for i in range(3):
                for j in range(3):
                    value = values[i, j]
                    label = f"{value:.4g}" if np.isfinite(value) else "NaN"
                    small_value = np.isfinite(value) and 0.0 < abs(value) < 0.01
                    if small_value:
                        label = f"{value:.1e}".replace("e-0", "e-")
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=10 if small_value else 13,
                        color="white" if np.isfinite(value) and value > maximum * 0.6 else "#0f172a",
                    )
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            if row_index == 0:
                ax.set_title(("Before", "After", "Analytic reference")[column], fontsize=14, pad=12)
            if column == 0:
                ax.set_ylabel(
                    (
                        "Singular-value squared norm\nExpected gradient: 4I",
                        "Trace + unused SVD diagnostic\nExpected gradient: I",
                    )[row_index],
                    labelpad=15,
                )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.1, top=0.83, hspace=0.42, wspace=0.28)
    fig.text(
        0.5,
        0.025,
        "Gray cells are non-finite results, not zero. Both CPU and the second GPU show the same pass/fail behavior.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    save_figure(fig, root, "svd-gradient-comparison")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("GH-2054 validation: 540 matched evaluations per version", fontsize=18, fontweight="bold", y=0.97)
    y = np.arange(len(summaries))
    left.barh(y - 0.18, [s["failed_before"] for s in summaries], height=0.32, color="#dc654d", label="Before")
    left.barh(y + 0.18, [s["failed_after"] for s in summaries], height=0.32, color="#148c99", label="After")
    for i, s in enumerate(summaries):
        for offset, key in ((-0.18, "failed_before"), (0.18, "failed_after")):
            left.text(s[key] + 1, i + offset, str(s[key]), va="center", fontsize=10)
    labels = [s["device"] + " / " + s["dtype"].replace("float", "FP") for s in summaries]
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlim(0, 60)
    left.set_xlabel("Failed evaluations / 60 in each group")
    left.set_title("Non-finite gradients eliminated", pad=14)
    left.legend(loc="lower right", frameon=False)
    ratios = [s["max_gradient_error_ratio_after"] for s in summaries]
    right.barh(y, ratios, height=0.5, color="#148c99")
    for i, value in enumerate(ratios):
        right.text(value + 0.025, i, f"{value:.3g}", va="center", fontsize=10)
    right.axvline(1.0, color="#475569", linestyle="--", linewidth=1)
    right.text(1.0, -0.65, "Pass limit", ha="center", fontsize=10, color="#475569")
    right.set_yticks(y, labels)
    right.invert_yaxis()
    right.set_xlim(0, 1.15)
    right.set_xlabel("Maximum gradient error / allowed error")
    right.set_title("After: every group within tolerance", pad=14)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.18, top=0.84, wspace=0.65)
    fig.text(
        0.5,
        0.075,
        "2 dimensions x 2 calling forms x 5 scenarios x 3 batch elements = 60 evaluations per device/dtype",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.text(
        0.5,
        0.035,
        "Repeated inputs are A=2I; these are API/scenario evaluations, not 540 independent random matrices.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    save_figure(fig, root, "svd-gradient-summary")

    lines = [
        "GH-2054: SVD zero-adjoint gradient validation",
        "===========================================",
        "",
        f"Before: {metadata[0]['commit']}",
        f"After:  {metadata[1]['commit']}",
        "Both revisions were built independently with the standard build_lib.py command.",
        "Neither revision includes the GH-2055 SVD3 forward change.",
        "",
        "Environment: Linux x86_64; Python 3.12.13; NumPy 2.5.3; Warp 1.19.0.dev0.",
        "CUDA Toolkit 13.0 (nvcc 13.0.88); NVIDIA driver 580.126.20.",
        "Devices: CPU, cuda:0 and cuda:1 (two NVIDIA B200 GPUs, sm_100).",
        "",
        "Standalone measurement results",
        "------------------------------",
        "Device | Dtype | Evaluations/phase | Failed before | Failed after | Max gradient error ratio after",
    ]
    lines.extend(
        f"{s['device']} | {s['dtype']} | {s['evaluations_per_phase']} | {s['failed_before']} | {s['failed_after']} | {s['max_gradient_error_ratio_after']:.9g}"
        for s in summaries
    )
    lines.extend(
        [
            "",
            "Method: A=2I for SVD2 and SVD3; return-value and output-argument forms.",
            "Scenarios: squared singular-value norm; norm plus trace; norm with seeds",
            "[0, 1, -0.5]; trace alone; trace preceded by an unused SVD diagnostic.",
            "Each scenario uses three batch elements. Unused diagnostic outputs do not require gradients.",
            "Counts are evaluations across APIs, devices, and seeds, not independent random matrices.",
            "",
            "References: norm loss = 4n and gradient = 4I; adding trace gives loss 6n",
            "and gradient 5I; trace gives loss 2n and gradient I, with or without the diagnostic.",
            "Gradient references are multiplied by the corresponding output seed.",
            "Comparisons use float64 references computed from the actual input values.",
            "",
            "Error ratio = max(abs(actual-expected)/(atol+rtol*abs(expected))).",
            "Both atol and rtol are 1e-2 / 1e-5 / 1e-10 for FP16 / FP32 / FP64.",
            "An evaluation passes only if both loss and gradient are finite and each ratio <= 1.",
            "Non-finite results remain nan in CSV data; heatmaps display them as gray NaN cells.",
            "",
            "Repository validation",
            "---------------------",
            "Baseline with added regression tests: 60 tests, 48 failed FP16 subtests in 12 methods.",
            "After: all 78 SVD tests and all 295 matrix-module tests passed on CPU and both GPUs.",
            "Includes existing FP32/FP64 finite differences and small nonzero U/V adjoint checks.",
            "Applicable pre-commit hooks and git diff --check passed.",
            "",
            "Scope: exact-zero U/V adjoints bypass singular-vector derivatives; zero sigma",
            "adjoints preserve the existing input gradient. Forward implementations and the",
            "general nonzero U/V adjoint formula are unchanged. This does not address all",
            "decomposition findings in GH-2030 or the forward NaN defect in GH-2055.",
            "",
            "Reproduce",
            "---------",
            "Run measure.py from a checkout of the corresponding source revision, after",
            "uv run build_lib.py. Save this asset directory outside the source checkout.",
            "Use a separate kernel cache for each build. Example arguments:",
            "uv run python /path/to/assets/2054/measure.py --phase before --expected-commit "
            + metadata[0]["commit"]
            + " --output /path/to/results --cache /tmp/warp-2054-before",
            "uv run python /path/to/assets/2054/measure.py --phase after --expected-commit "
            + metadata[1]["commit"]
            + " --output /path/to/results --cache /tmp/warp-2054-after",
            "The defaults require CPU and two CUDA devices; --devices can select available devices.",
            "Copy build_assets.py beside both CSVs and metadata JSONs, then run:",
            "uv run --with matplotlib --with numpy python /path/to/results/build_assets.py",
            "",
            "Published files deliberately omit raw logs and machine-specific filesystem paths.",
        ]
    )
    (root / "REPORT.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
