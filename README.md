<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR assets

Numerical reports, figures, data, and reproduction scripts for contributions to NVIDIA/warp.

| Issue | Assets | Report | Figure |
|---|---|---|---|
| [NVIDIA/warp#2054](https://github.com/NVIDIA/warp/issues/2054) | [SVD gradient validation](2054/) | [Numerical report](2054/REPORT.txt) | [Before / after gradients](2054/svd-gradient-comparison.png) |
| [NVIDIA/warp#2055](https://github.com/NVIDIA/warp/issues/2055) | [SVD3 forward validation](2055/) | [Numerical report](2055/REPORT.txt) | [Before / after comparison](2055/svd3-validation.png) |
| [NVIDIA/warp#2069](https://github.com/NVIDIA/warp/issues/2069) | [Mixed-precision and row-packed BSR products](2069/) | [Validation report](2069/report.html), [Split validation](2069/split/REPORT.txt) | [Row packing](2069/row-packing-speedup.png), [FEM comparison](2069/fem-speedup.png) |

Keep each contribution in its issue-number directory. PR descriptions should use links pinned to an asset commit so their evidence remains reproducible as this branch grows. Published assets exclude raw diagnostic logs and private environment identifiers.

For #2069, separate [mixed-precision](2069/pr-drafts/mixed-precision.md) and [row-packing](2069/pr-drafts/row-packed.md) PR body drafts are available, with [titles and branch dependency notes](2069/pr-drafts/README.txt).
