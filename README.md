<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR assets

Numerical reports, figures, data, and reproduction scripts for contributions to NVIDIA/warp.

| Issue | Assets | Report | Figure |
|---|---|---|---|
| [NVIDIA/warp#1110](https://github.com/NVIDIA/warp/issues/1110) | [OpenGL primitive resolution](1110/validation/cb86abb7/) | [Validation and provenance](1110/validation/cb86abb7/README.md), [Data tables](1110/validation/cb86abb7/TABLES.md) | [Independent B200 allocations](1110/validation/cb86abb7/overflow/independent-summary/across_jobs.png), [Filled geometry](1110/validation/cb86abb7/overflow/jobs/1144767/filled_comparison.png) |
| [NVIDIA/warp#2054](https://github.com/NVIDIA/warp/issues/2054) | [SVD gradient validation](2054/) | [Numerical report](2054/REPORT.txt) | [Before / after gradients](2054/svd-gradient-comparison.png) |
| [NVIDIA/warp#2055](https://github.com/NVIDIA/warp/issues/2055) | [SVD3 forward validation](2055/) | [Numerical report](2055/REPORT.txt) | [Before / after comparison](2055/svd3-validation.png) |
| [NVIDIA/warp#2069](https://github.com/NVIDIA/warp/issues/2069) | [Mixed-precision and row-packed BSR products](2069/) | [Validation report](2069/report.html), [Split validation](2069/split/REPORT.txt) | [Row packing](2069/row-packing-speedup.png), [FEM comparison](2069/fem-speedup.png) |
| [NVIDIA/warp#2120](https://github.com/NVIDIA/warp/issues/2120) | [CPU/OpenGL host buffer fix](2120/validation/d619a625/) | [Runtime validation](2120/validation/d619a625/REPORT.txt), [Warning-state cleanup](2120/validation/8be105eb/REPORT.txt) | [Actual CPU/CUDA captures](2120/validation/d619a625/render-comparison.png) |

Keep each contribution in its issue-number directory. PR descriptions should use links pinned to an asset commit so their evidence remains reproducible as this branch grows. Published evidence excludes private environment identifiers; newer bundles include complete diagnostic output with filesystem paths redacted.

The [current GH-2054 validation](2054/validation/733930dd/REPORT.txt) covers source `733930dd`, rebuilt on official main `0382a97e`. The original GH-2054 CSVs and figures above remain historical evidence for `c47fe4c8` / `e6185ac3`.

For #2069, separate [mixed-precision](2069/pr-drafts/mixed-precision.md) and [row-packing](2069/pr-drafts/row-packed.md) PR body drafts are available, with [titles and branch dependency notes](2069/pr-drafts/README.txt).

For #1110, the [proposed PR body](1110/pr-drafts/render-shape-segments.md) and [title/status notes](1110/pr-drafts/README.txt) accompany the frozen feature validation. No feature PR has been opened; #2121 remains the separate CPU/OpenGL fix.
