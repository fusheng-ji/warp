# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the review report from raw measurements, without private local logs."""

import csv
import html
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
validation = json.loads((root / "validation.json").read_text())
metadata = json.loads((root / "final-uniform.json").read_text())
conversion = json.loads((root / "conversion.json").read_text())


def read(name):
    with (root / name).open() as f:
        return list(csv.DictReader(f))


def summarize(rows, keys):
    groups = defaultdict(lambda: defaultdict(list))
    rounds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups[key][r["path"]].append(float(r["us"]))
        rounds[key][r["round"]][r["path"]].append(float(r["us"]))
    results = []
    for key, paths in groups.items():
        medians = {p: statistics.median(v) for p, v in paths.items()}
        ratio = min(medians["auto"], medians["scalar"]) / medians["packed128t4"]
        round_ratios = []
        for samples in rounds[key].values():
            m = {p: statistics.median(v) for p, v in samples.items()}
            round_ratios.append(min(m["auto"], m["scalar"]) / m["packed128t4"])
        results.append(
            dict(zip(keys, key, strict=True))
            | dict(
                auto_us=medians["auto"],
                scalar_us=medians["scalar"],
                packed_us=medians["packed128t4"],
                speedup=ratio,
                min_round_speedup=min(round_ratios),
                max_round_speedup=max(round_ratios),
            )
        )
    return results


keys = ["nrow", "length", "storage", "compute", "distribution", "block_rows", "block_cols", "mode"]
all_summary = []
raw = []
for filename in ["final-uniform.csv", "final-irregular.csv", "final-blocks.csv"]:
    rows = read(filename)
    raw.extend(rows)
    all_summary.extend(summarize(rows, keys))
with (root / "all-speedups.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_summary[0]))
    writer.writeheader()
    writer.writerows(all_summary)
fem_raw = read("final-fem.csv")
fem = summarize(fem_raw, ["matrix", "nrow", "nnz_blocks", "block_rows", "block_cols", "storage", "compute"])
with (root / "fem-summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(fem[0]))
    writer.writeheader()
    writer.writerows(fem)
core = [
    r
    for r in all_summary
    if r["mode"] == "graph"
    and r["distribution"] == "uniform"
    and r["block_rows"] == "1"
    and r["block_cols"] == "1"
    and r["nrow"] in ["32768", "131072"]
    and r["length"] in ["32", "64"]
]
errors = {}
for r in raw + fem_raw:
    precision = r["storage"] + "/" + r["compute"]
    errors[precision] = max(errors.get(precision, 0), float(r["max_abs_error"]))
# Same-type old/new comparison on configurations measured in both sweeps.
baseline = defaultdict(list)
after = defaultdict(list)
comparison_keys = keys + ["path"]
for r in read("final-baseline.csv"):
    baseline[tuple(r[k] for k in comparison_keys)].append(float(r["us"]))
for r in read("final-uniform.csv"):
    key = tuple(r[k] for k in comparison_keys)
    if key in baseline:
        after[key].append(float(r["us"]))
compatibility = []
for key, values in after.items():
    compatibility.append(
        dict(zip(comparison_keys, key, strict=True))
        | dict(
            before_us=statistics.median(baseline[key]),
            after_us=statistics.median(values),
            after_over_before=statistics.median(values) / statistics.median(baseline[key]),
        )
    )
with (root / "same-type-comparison.csv").open("w") as f:
    out = csv.DictWriter(f, fieldnames=list(compatibility[0]))
    out.writeheader()
    out.writerows(compatibility)
summary = dict(
    validation=validation,
    device=metadata["device"],
    precision_max_abs_errors=errors,
    main_cases=core,
    fem_cases=fem,
    measurement_rows=len(raw) + len(fem_raw),
    performance_configurations=len(all_summary) + len(fem),
    conversion=conversion,
)
controls = []
for mode in ["graph", "launch"]:
    ratios = [r["after_over_before"] for r in compatibility if r["path"] == "auto" and r["mode"] == mode]
    controls.append(
        dict(mode=mode, median_ratio=statistics.median(ratios), min_ratio=min(ratios), max_ratio=max(ratios))
    )
summary["same_type_auto_controls"] = controls
(root / "results-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def table(rows, columns):
    def cell(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    return (
        "<table><thead><tr>"
        + "".join("<th>" + html.escape(c) + "</th>" for c in columns)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + html.escape(cell(row[c])) + "</td>" for c in columns) + "</tr>" for row in rows
        )
        + "</tbody></table>"
    )


body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Warp GH-2069 validation</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#202a35}}h1,h2{{line-height:1.25}}table{{border-collapse:collapse;font-size:13px;margin:20px 0;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#f3f6f9}}img{{max-width:100%;height:auto}}code{{background:#f3f6f9;padding:2px 5px}}.note{{padding:16px;background:#f3f6f9}}a{{color:#1659a5}}</style>
<h1>Warp #2069 — correctness and performance</h1>
<p>FP32 matrix storage with FP64 vectors and arithmetic, plus explicit row packing for non-transposed CUDA products. Existing automatic dispatch is retained.</p>
<p>Base: <code>{validation["base_commit"]}</code><br>Implementation: <code>{validation["implementation_commit"]}</code><br>Device: {html.escape(metadata["device"])}; CUDA Toolkit 13.0; physical GPU 1 isolated with <code>CUDA_VISIBLE_DEVICES=1</code>. Both correctness devices are B200; this is not evidence across architectures.</p>
<h2>Correctness</h2><p>80 diagnostic cases change from unsupported-type errors to passing results on CPU and both GPUs. The exact diagnostic references match with zero absolute error. Random-matrix maximum absolute errors are {html.escape(str(errors))}; references use the actual stored coefficients.</p>
<img src="mixed-precision-correctness.png" alt="Before and after correctness results">
<h2>Row packing</h2><p>Configuration fixed before the validation sweep: 128 threads per block, 32 scalar rows per block (4 threads per row). Speedup is relative to the faster of automatic dispatch and the forced scalar path. Values below 1 represent slowdowns.</p>
<img src="row-packing-speedup.png" alt="Speedup heatmaps including short-row regressions">
{table(core, ["nrow", "length", "storage", "compute", "auto_us", "scalar_us", "packed_us", "speedup", "min_round_speedup"])}
<img src="latency.png" alt="Median latency with interquartile ranges">
<h2>FEM matrices</h2><p>These are the repository's existing linear-grid, quadratic-tetmesh and almost-dense benchmark matrices. Promoted FP64 matrices contain the same coefficients as the FP32 originals.</p>
<img src="fem-speedup.png" alt="FEM speedups, including quadratic-tetmesh and almost-dense regressions">
{table(fem, ["matrix", "storage", "compute", "auto_us", "scalar_us", "packed_us", "speedup"])}
<p>Linear-grid gains do not generalize to quadratic-tetmesh or almost-dense matrices with this fixed configuration. The measurements support an explicit option, not a universal replacement or a new automatic dispatch rule.</p>
<h2>Existing same-type calls</h2><p>The table compares the automatic path against the unmodified base for matched FP32/FP32 and FP64/FP64 cases. Ratios are after/before latency, so values above 1 are slower. Ordinary-launch measurements include variable host-submission gaps and do not establish an end-to-end improvement.</p>
{table(controls, ["mode", "median_ratio", "min_ratio", "max_ratio"])}
<h2>Storage and conversion</h2><p>For {conversion["nnz"]:,} coefficients, matrix value storage is {conversion["matrix_value_bytes_fp32"]:,} bytes in FP32 and {conversion["matrix_value_bytes_fp64"]:,} bytes in FP64. Index storage is {conversion["index_bytes"]:,} bytes in either case. The warm conversion-plus-allocation median is {conversion["median_ms"]:.3f} ms (range {min(conversion["conversion_with_allocation_ms"]):.3f}–{max(conversion["conversion_with_allocation_ms"]):.3f} ms); this variable host-side cost is excluded from steady-state SpMV comparisons.</p>
<h2>Validation</h2><ul><li>139 sparse tests: CPU and both B200 GPUs.</li><li>180 linear-solver and tile-reduction tests: CPU and isolated second B200.</li><li>8 ASV setup/time combinations.</li><li>Standard native build; pre-commit passed; Sphinx passed with warnings treated as errors.</li><li>DCO sign-off retained.</li></ul>
<h2>Method and limits</h2><p>Compilation, allocation, reference generation and at least 50 ms of warmup are outside timing. Five batches of six samples use randomized path order. GPU events time 50 products per sample. Ordinary-launch stream times include host-submission gaps; graph times measure repeated captured execution. All products overwrite the output (beta=0).</p>
<p>The option is explicit because short rows and some small matrices with long rows can regress, and only one GPU architecture has been measured. No default heuristic, solver dtype inference, or autodiff support is changed. CPU forced-tile behavior is a pre-existing limitation documented in README.txt; CPU tests cover automatic/scalar and transpose paths.</p>
<p class="note">This is an independent correctness and performance evaluation. No downstream production deployment or maintainer assignment is claimed. PR organization and publication remain undecided.</p>
<h2>Reproduction and raw data</h2><ul>{"".join('<li><a href="' + name + '">' + name + "</a></li>" for name in ["README.txt", "validation.json", "results-summary.json", "final-uniform.csv", "final-irregular.csv", "final-blocks.csv", "final-fem.csv", "final-baseline.csv", "same-type-comparison.csv", "all-speedups.csv", "fem-summary.csv", "numerics-before.csv", "numerics-after.csv", "benchmark.py", "fem_benchmark.py", "numerics.py", "conversion.py", "plot_results.py", "report_results.py"])}</ul></html>"""
(root / "report.html").write_text(body)
print(
    json.dumps(
        {
            "main_speedup_range": [min(r["speedup"] for r in core), max(r["speedup"] for r in core)],
            "min_main_round_speedup": min(r["min_round_speedup"] for r in core),
            "max_errors": errors,
            "measurements": len(raw) + len(fem_raw),
        },
        indent=2,
    )
)
