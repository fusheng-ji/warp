# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import cache
from typing import Any

import warp as wp


@cache
def make_packed_kernel(block_dim, threads_per_row, block_rows, block_cols):
    rows_per_block = block_dim // threads_per_row

    @wp.kernel(enable_backward=False, module="unique")
    def packed(
        alpha: Any,
        offsets: wp.array[int],
        counts: wp.array[int],
        columns: wp.array[int],
        values: wp.array3d[Any],
        x: wp.array[Any],
        beta: Any,
        y: wp.array[Any],
    ):
        thread = wp.tid()
        group = thread // block_dim
        lane = thread % threads_per_row
        scalar_row = thread // threads_per_row
        lane_sum = y.dtype(0)
        if scalar_row < y.shape[0] and alpha != y.dtype(0):
            row = scalar_row // block_rows
            subrow = scalar_row % block_rows
            begin = offsets[row]
            count = offsets[row + 1] - begin
            if counts:
                count = counts[row]
            for index in range(lane, count * block_cols, threads_per_row):
                block = begin + index // block_cols
                col = index % block_cols
                lane_sum += y.dtype(values[block, subrow, col]) * x[columns[block] * block_cols + col]
        partials = wp.tile_reshape(wp.tile(lane_sum), shape=(rows_per_block, threads_per_row))
        sums = alpha * wp.tile_sum(partials, axis=1)
        if beta != y.dtype(0):
            sums += beta * wp.tile_load(y, shape=rows_per_block, offset=group * rows_per_block)
        wp.tile_store(y, sums, offset=group * rows_per_block)

    return packed


def packed_mv(A, x, y, block_dim=128, threads_per_row=4, alpha=1.0, beta=0.0):
    rows = block_dim // threads_per_row
    wp.launch(
        make_packed_kernel(block_dim, threads_per_row, *A.block_shape),
        dim=((y.size + rows - 1) // rows) * block_dim,
        block_dim=block_dim,
        inputs=[y.dtype(alpha), A.offsets, A.row_counts, A.columns, A.scalar_values, x, y.dtype(beta), y],
        device=y.device,
    )
