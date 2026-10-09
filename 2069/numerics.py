# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record mixed-precision results against references using stored coefficients."""

import argparse
import csv
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--checkout", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
sys.path.insert(0, str(Path(a.checkout).resolve()))
import numpy as np
import warp as wp
import warp.sparse as wps

wp.init()
fields = ["device", "case", "transpose", "tile_size", "status", "max_abs_error", "finite", "detail"]
with open(a.output, "w") as f:
    out = csv.DictWriter(f, fields)
    out.writeheader()
    for device in wp.get_devices():
        for case in ["cancellation", "small_increment", "scaling", "alias", "unused_capacity"]:
            for transpose in [False, True]:
                for tile_size in [-1, 0, 64] if device.is_cuda else [-1, 0]:
                    dense = np.array([[2**24, 1, -(2**24)], [0, 2, 0], [1, 0, -1]], dtype=np.float32)
                    if case != "cancellation":
                        dense = np.eye(3, dtype=np.float32)
                    rows, cols = np.nonzero(dense)
                    A = wps.bsr_zeros(
                        3, 3, wp.float32, device=device, row_capacity=4 if case == "unused_capacity" else None
                    )
                    wps.bsr_set_from_triplets(
                        A,
                        wp.array(rows, dtype=int, device=device),
                        wp.array(cols, dtype=int, device=device),
                        wp.array(dense[rows, cols], device=device),
                        topology="padded" if case == "unused_capacity" else "compact",
                    )
                    x_np = np.ones(3, dtype=np.float64)
                    if case == "small_increment":
                        x_np += np.array([1, -1, 1]) * 2.0**-40
                    x = wp.array(x_np, device=device)
                    y = x if case == "alias" else wp.ones(3, dtype=wp.float64, device=device)
                    alpha = 1.0 + 2.0**-35 if case == "scaling" else 1.0
                    beta = 0.5 if case == "alias" else 0.0
                    ref = alpha * ((dense.T if transpose else dense).astype(np.float64) @ x_np) + beta * y.numpy()
                    r = dict(device=str(device), case=case, transpose=transpose, tile_size=tile_size)
                    try:
                        wps.bsr_mv(A, x, y, alpha=alpha, beta=beta, transpose=transpose, tile_size=tile_size)
                        actual = y.numpy()
                        finite = bool(np.isfinite(actual).all())
                        error = float(np.max(np.abs(actual - ref)))
                        np.testing.assert_allclose(actual, ref, rtol=1e-15, atol=0.0)
                        r.update(status="pass", max_abs_error=error, finite=finite, detail="")
                    except Exception as e:
                        r.update(
                            status="fail",
                            max_abs_error="",
                            finite="",
                            detail=f"{type(e).__name__}: {str(e).splitlines()[0]}",
                        )
                    out.writerow(r)
