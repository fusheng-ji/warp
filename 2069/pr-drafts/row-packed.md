## What problem are you trying to solve?

Part of #2069. Depends on [#2072: Support mixed-precision BSR products](https://github.com/NVIDIA/warp/pull/2072).

This is a stacked draft for the row-packing contribution. Please review the [incremental diff above #2072](https://github.com/fusheng-ji/warp/compare/4db534d616d5ff16b31c8b851d93690dfc296fd8...589fd1216f21ef500aa0d7bfe2e4dc8d6c3086ce). Before this PR is ready to merge, #2072 must land and this branch must be rebased onto upstream `main` so its diff contains only row packing.

The existing tiled `bsr_mv` path assigns a thread block to one output row. For some matrices with medium-length rows, processing several rows together improves throughput. This change exposes an explicit CUDA row-packing option while retaining the existing default dispatch.

This proposal comes from an independent correctness and performance investigation. No downstream production deployment or maintainer assignment is claimed; upstream coordination remains pending.

## What does this change?

Add `rows_per_block` to `bsr_mv`. Values greater than one select the new non-transposed CUDA path; the default of one preserves current dispatch. For example, the following assigns four threads to each of 32 scalar output rows:

```python
import warp as wp
from warp.sparse import bsr_identity, bsr_mv

A = bsr_identity(64, block_type=wp.float32, device="cuda:0")
x = wp.ones(64, dtype=wp.float64, device=A.device)
y = bsr_mv(A, x, tile_size=128, rows_per_block=32)
```

`tile_size` is the total threads per block; `rows_per_block` counts scalar rows, including subrows inside BSR blocks. Explicit packing requires a power-of-two tile size from 32 to 1024 and 1, 2, 4, 8, 16, or 32 threads per row. CPU and transpose requests are rejected. The cached kernel specializes the block shape, masks tails and padded entries, and keeps all lanes in the collective reduction.

## Existing pull requests

- [x] I searched open pull requests for duplicate or overlapping work.
- Related pull requests: [#2072](https://github.com/NVIDIA/warp/pull/2072) provides the mixed-precision foundation and is included in this branch's history. [#1791](https://github.com/NVIDIA/warp/pull/1791) changes sparse nonzero-count synchronization; [#1975](https://github.com/NVIDIA/warp/pull/1975) includes BSR transpose and tile fixes. Neither of those two implements row-packed `bsr_mv`.

The initial search on 2026-10-10 checked the titles and descriptions of all 60 then-open upstream PRs and inspected the sparse changes in #1791 and #1975. No direct duplicate was found. The subsequently opened #2072 is the explicit dependency for this contribution.

The upstream target is `NVIDIA/warp:main`. While #2072 is open, GitHub's full diff against `main` includes both contributions. The scope described here is the single incremental commit `589fd121` above #2072's head `4db534d6`; mixed-precision behavior and its API contract belong to #2072.

## Validation

- Regression validation: N/A, optional performance feature. Numerical comparisons use a float64 reference built from the actual stored coefficients. All 80 mixed-precision diagnostic cases pass after the split with finite results and zero error on the exact diagnostic inputs.
- NVIDIA GPU: two NVIDIA B200 GPUs, Ubuntu 24.04.3 LTS, CUDA Toolkit 13.0, driver 580.126.20; Python 3.12.13 and Warp 1.19.0.dev0. Sparse tests run on CPU and both GPUs; solver/tile-reduction tests use CPU and physical GPU 1. Timing runs use only physical GPU 1.
- Other validation: 139 sparse tests plus 180 solver/tile-reduction tests passed (319 total); eight ASV setup/time smoke combinations passed. Independent standard native build, private kernel cache, pre-commit, and documentation with warnings treated as errors all passed. Coverage includes same/mixed types, BSR blocks, tails and padding, alpha/beta semantics, aliasing, strided vectors, graph capture, and invalid configurations.

[Branch-specific numerical CSV](https://github.com/fusheng-ji/warp/blob/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069/split/packed-numerics.csv) · [Exact validation metadata](https://github.com/fusheng-ji/warp/blob/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069/split/validation.json) · [Integration and reproduction notes](https://github.com/fusheng-ji/warp/blob/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069/split/REPORT.txt).

Performance uses a fixed 128-thread block with four threads per row. CUDA events time batches of 50 products after at least 50 ms of untimed warmup; each product uses `beta=0`. Five rounds with six samples each randomize path order. Compilation, allocation, and reference computation are excluded. Speedup is relative to the faster of automatic and scalar execution, not every possible tuned configuration.

For uniform matrices with 32,768 or 131,072 rows and 32 or 64 entries per row, median CUDA Graph speedups range from **2.27x to 4.25x** across FP32/FP32, FP32/FP64, and FP64/FP64. The full sweep also contains regressions:

![Uniform matrix sweep, including configurations slower than the existing paths](https://raw.githubusercontent.com/fusheng-ji/warp/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069/row-packing-speedup.png)

| FEM matrix | FP32 / FP32 | FP32 / FP64 | FP64 / FP64 |
|---|---:|---:|---:|
| Linear grid | 1.53x | 1.62x | 1.70x |
| Quadratic tetmesh | 0.77x | 0.81x | 1.02x |
| Almost dense | 0.21x | 0.14x | 0.12x |

![FEM speedups and slowdowns](https://raw.githubusercontent.com/fusheng-ji/warp/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069/fem-speedup.png)

Values below one are slower. The 1.02x tetmesh aggregate is not a stable gain across rounds. These results support an explicit option that users benchmark for their matrices. They establish neither an end-to-end solver speedup nor results on other GPU architectures.

Across the sampled timing workloads, maximum absolute error against the stored-coefficient reference is `2.84e-14` for FP32/FP64 and FP64/FP64, and `1.55e-5` for FP32/FP32. [Raw samples, figures, and reproduction scripts](https://github.com/fusheng-ji/warp/tree/b43287be27bc7980ae65668bc8ce18fbd8ab9297/2069) include all 31,680 non-baseline timing records and negative results.

Timings were collected on the original combined implementation before the mechanical split. This branch's complete file tree matches that benchmarked tree exactly. Numerical and unit tests were rerun independently after splitting; performance timings were not. Sequential patch application and merge-tree checks reproduce the original tree without conflicts at base `c47fe4c8`.

## Checklist

- [ ] I have read and followed the [Contributing Guidelines](https://github.com/NVIDIA/warp/blob/main/CONTRIBUTING.md).
- [x] This pull request does not bundle unrelated changes unless a maintainer agreed to the broader scope.
- [ ] My branch is up to date with `main` and contains one commit, unless a maintainer agreed to keep the change as multiple commits.
- [x] I included any required [changelog fragments](https://github.com/NVIDIA/warp/blob/main/changelog/README.md) and used a `+` prefix only for changes that do not require a GitHub issue.

Upstream coordination and rebasing after #2072 merges remain pending. The incremental commit includes DCO sign-off and `changelog/2069.added.1.md`. The single-commit checklist item remains unchecked while this branch contains both contributions above `main`.
