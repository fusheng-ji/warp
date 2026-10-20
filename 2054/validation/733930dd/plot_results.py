"""Plot current GH-2054 gradient error ratios from the published CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = [(device, dtype) for device in ("cpu", "cuda:0", "cuda:1") for dtype in ("float16", "float32", "float64")]
    values = [
        max(float(r["gradient_error_ratio"]) for r in rows if (r["device"], r["dtype"]) == group) for group in groups
    ]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.bar(range(len(groups)), values, color=["#2563eb", "#059669", "#9333ea"] * 3)
    ax.axhline(1.0, color="#dc2626", linestyle="--", label="Pass threshold")
    ax.set_yscale("log")
    ax.set_ylim(1.0e-6, 2.0)
    ax.set_xticks(range(len(groups)), [f"{d}\n{t}" for d, t in groups])
    ax.set_ylabel("Maximum gradient error / allowance")
    ax.set_title("GH-2054: current-build gradient validation (540 evaluations)")
    ax.legend()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "gradient-validation.png", dpi=170, metadata={"Software": "Matplotlib"})
    fig.savefig(args.output / "gradient-validation.svg", metadata={"Date": None, "Creator": "Matplotlib"})
    plt.close(fig)


if __name__ == "__main__":
    main()
