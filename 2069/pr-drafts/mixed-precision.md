## What problem are you trying to solve?

Part of #2069.

`bsr_mv` currently requires the matrix and vectors to have the same scalar type. Applying an FP32 matrix to FP64 vectors therefore requires allocating a converted matrix. This change supports FP32 coefficient storage with FP64 multiplication, accumulation, and scaling directly.

This proposal comes from an independent correctness and performance investigation of #2069. No downstream production deployment or maintainer assignment is claimed; upstream coordination remains pending.

## What does this change?

Allow `wp.float32` matrices with `wp.float64` input/output vectors in the existing scalar, CUDA tiled, and transpose paths. Convert coefficients before multiplication and allocate an omitted output using the input vector's scalar type and the matrix's output block shape. Preserve existing same-type combinations and reject unsupported mixed types before modifying the output.

```python
import warp as wp
from warp.sparse import bsr_identity, bsr_mv

A = bsr_identity(4, block_type=wp.float32)
x = wp.ones(4, dtype=wp.float64, device=A.device)
y = bsr_mv(A, x)  # FP64 output; no FP64 matrix copy.
```

The same rules cover BSR blocks and scaled matrix expressions. Double-precision arithmetic uses the coefficients actually stored in FP32; it cannot recover precision lost when constructing the matrix. This contribution does not add FP16/BF16 combinations, solver dtype inference, or autodiff support. Row packing is a separate contribution.

## Existing pull requests

- [x] I searched open pull requests for duplicate or overlapping work.
- Related pull requests: [#1791](https://github.com/NVIDIA/warp/pull/1791) changes sparse nonzero-count synchronization; [#1975](https://github.com/NVIDIA/warp/pull/1975) includes BSR transpose and tile fixes. Neither implements mixed-precision `bsr_mv`. A separate [row-packing branch](https://github.com/fusheng-ji/warp/tree/perf/bsr-mv-row-packed) depends on this contribution.

Checked the titles and descriptions of all 60 open upstream PRs on 2026-10-10 and inspected the sparse changes in the related PRs above. No direct duplicate was found in this search.

## Validation

- Regression validation: N/A, feature extension. All 80 mixed-precision acceptance cases reject unsupported scalar types on official base `c47fe4c8`; all 80 pass on this branch at `4db534d6`, with finite results and zero maximum absolute error for these deliberately exact diagnostic inputs.
- NVIDIA GPU: two NVIDIA B200 GPUs, Ubuntu 24.04.3 LTS, CUDA Toolkit 13.0, driver 580.126.20; Python 3.12.13 and Warp 1.19.0.dev0. Sparse and numerical checks cover CPU and both GPUs. Solver and tile-reduction checks cover CPU and the second physical GPU.
- Other validation: 132 sparse tests plus 180 solver/tile-reduction tests passed (312 total); four ASV setup/time smoke combinations passed. The branch had an independent standard `uv run build_lib.py` build and private kernel cache. Pre-commit and documentation with warnings treated as errors passed.

Coverage includes multiplication before accumulation in FP64, scaling, transpose, block shapes, padded topology, aliasing, strided vectors, output allocation, rejected type combinations, and CUDA graph capture. CPU coverage uses automatic/scalar and transpose paths. Forced CPU tiling has a separately documented baseline limitation; forced tiling is validated on CUDA.

| Numerical acceptance | Official base | Mixed-only branch |
|---|---:|---:|
| Accepted and correct cases | 0/80 | 80/80 |
| Finite results | Unsupported types | 80/80 |
| Maximum absolute error on exact diagnostics | N/A | 0 |

[Fresh branch-specific numerical CSV](https://github.com/fusheng-ji/warp/blob/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069/split/mixed-numerics.csv) · [Exact validation metadata](https://github.com/fusheng-ji/warp/blob/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069/split/validation.json) · [Build/test commands and integration checks](https://github.com/fusheng-ji/warp/blob/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069/split/REPORT.txt).

![Mixed-precision acceptance on CPU and two B200 GPUs](https://raw.githubusercontent.com/fusheng-ji/warp/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069/mixed-precision-correctness.png)

The figure records the original combined implementation's acceptance run. The fresh mixed-only run linked above reproduces the same 80/80 result. Row-packing performance measurements are not attributed to this contribution.

## Checklist

- [ ] I have read and followed the [Contributing Guidelines](https://github.com/NVIDIA/warp/blob/main/CONTRIBUTING.md).
- [x] This pull request does not bundle unrelated changes unless a maintainer agreed to the broader scope.
- [x] My branch is up to date with `main` and contains one commit, unless a maintainer agreed to keep the change as multiple commits.
- [x] I included any required [changelog fragments](https://github.com/NVIDIA/warp/blob/main/changelog/README.md) and used a `+` prefix only for changes that do not require a GitHub issue.

Checklist status is recorded against `main` at `c47fe4c8` on 2026-10-10. The contribution-guidelines item remains pending upstream coordination; the commit includes DCO sign-off and `changelog/2069.added.md`.
