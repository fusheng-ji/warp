# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys
from time import perf_counter
import numpy as np
import warp as wp
import warp.sparse as wps
from benchmark import make_case

wp.init()
device = wp.get_device("cuda:0")
A, x, y, ref = make_case(32768, 32, "float32", "float64", device)
B = wps.bsr_copy(A, scalar_type=wp.float64)
wp.synchronize_device(device)
times = []
for _ in range(10):
    start = perf_counter()
    B = wps.bsr_copy(A, scalar_type=wp.float64)
    wp.synchronize_device(device)
    times.append((perf_counter() - start) * 1000)
# Compact copies may reorder columns; compare coefficients by their coordinates.
source_columns = A.columns.numpy().reshape(A.nrow, -1)
order = np.argsort(source_columns, axis=1)
np.testing.assert_array_equal(B.offsets.numpy(), A.offsets.numpy())
np.testing.assert_array_equal(B.columns.numpy().reshape(A.nrow, -1), np.take_along_axis(source_columns, order, axis=1))
np.testing.assert_array_equal(
    B.values.numpy().reshape(A.nrow, -1),
    np.take_along_axis(A.values.numpy().reshape(A.nrow, -1), order, axis=1).astype(np.float64),
)
wps.bsr_mv(B, x, y)
np.testing.assert_allclose(y.numpy(), ref, rtol=1e-12, atol=1e-12)
data = dict(
    nrow=A.nrow,
    nnz=A.nnz,
    matrix_value_bytes_fp32=A.values.numpy().nbytes,
    matrix_value_bytes_fp64=B.values.numpy().nbytes,
    index_bytes=A.columns.numpy().nbytes + A.offsets.numpy().nbytes,
    conversion_with_allocation_ms=times,
    median_ms=float(np.median(times)),
    note="Warm conversion plus allocation and synchronization; compilation excluded. Matrix coefficients are identical after promotion and column reordering.",
)
with open(sys.argv[1], "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
