# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and time existing FEM benchmark matrices using public bsr_mv paths."""

import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import bsr_matrix
import warp as wp
import warp.sparse as wps

sys.path.insert(0, str(Path("asv").resolve()))
from benchmarks.sparse.matrix_vector import BsrMvAlmostDense, BsrMvLinearGridMatrix, BsrMvQuadraticTetmeshMatrix

wp.init()
wp.config.quiet = True
rng = np.random.default_rng(2069)
fields = [
    "matrix",
    "nrow",
    "nnz_blocks",
    "block_rows",
    "block_cols",
    "storage",
    "compute",
    "path",
    "round",
    "sample",
    "us",
    "max_abs_error",
]
with open(sys.argv[1], "w") as f:
    out = csv.DictWriter(f, fields)
    out.writeheader()
    for factory in [BsrMvLinearGridMatrix, BsrMvQuadraticTetmeshMatrix, BsrMvAlmostDense]:
        fixture = factory()
        fixture.setup()
        A32 = fixture._mat
        device = A32.device
        A64 = wps.bsr_copy(A32, scalar_type=wp.float64)
        offsets = A32.offsets.numpy()
        nnz = int(offsets[-1])
        values = A32.values.numpy().reshape(-1, *A32.block_shape)[:nnz]
        ref_A = bsr_matrix((values.astype(np.float64), A32.columns.numpy()[:nnz], offsets), shape=A32.shape)
        x_data = rng.normal(size=A32.shape[1])
        for A, compute in [(A32, wp.float32), (A32, wp.float64), (A64, wp.float64)]:
            x = wp.array(x_data, dtype=compute, device=device)
            y = wp.empty(A.shape[0], dtype=compute, device=device)
            ref = ref_A @ x.numpy().astype(np.float64)
            paths = {
                "auto": lambda: wps.bsr_mv(A, x, y),
                "scalar": lambda: wps.bsr_mv(A, x, y, tile_size=-1),
                "tile64": lambda: wps.bsr_mv(A, x, y, tile_size=64),
                "packed128t4": lambda: wps.bsr_mv(A, x, y, tile_size=128, rows_per_block=32),
            }
            graphs, errors = {}, {}
            for name, fn in paths.items():
                fn()
                actual = y.numpy()
                np.testing.assert_allclose(
                    actual,
                    ref,
                    atol=3e-5 if compute == wp.float32 else 1e-12,
                    rtol=3e-5 if compute == wp.float32 else 1e-12,
                )
                errors[name] = float(np.max(np.abs(actual - ref)))
                with wp.ScopedCapture(device=device) as cap:
                    for _ in range(50):
                        fn()
                graphs[name] = cap.graph
                wp.capture_launch(cap.graph)
            wp.synchronize_device(device)
            warmup_start = perf_counter()
            while perf_counter() - warmup_start < 0.05:
                for graph in graphs.values():
                    wp.capture_launch(graph)
                wp.synchronize_device(device)
            start, end = wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)
            for round_id in range(5):
                for sample in range(6):
                    order = list(paths)
                    rng.shuffle(order)
                    for name in order:
                        with wp.ScopedDevice(device):
                            wp.record_event(start)
                            wp.capture_launch(graphs[name])
                            wp.record_event(end)
                        us = wp.get_event_elapsed_time(start, end) * 1000 / 50
                        out.writerow(
                            dict(
                                matrix=factory.__name__,
                                nrow=A.shape[0],
                                nnz_blocks=nnz,
                                block_rows=A.block_shape[0],
                                block_cols=A.block_shape[1],
                                storage=A.scalar_type.__name__,
                                compute=compute.__name__,
                                path=name,
                                round=round_id,
                                sample=sample,
                                us=us,
                                max_abs_error=errors[name],
                            )
                        )
                f.flush()
            print(factory.__name__, A.scalar_type.__name__, compute.__name__, "done", flush=True)
