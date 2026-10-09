# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce GH-2069 experiments; run from a Warp checkout with uv run.

GPU event times exclude JIT, allocation and matrix assembly. Graphs contain
identical independent products (beta=0); ordinary launches are reported separately.
"""

import argparse
import csv
import json
import platform
import os
import hashlib
import sys
import subprocess
from pathlib import Path
from time import perf_counter

if "--checkout" in sys.argv:
    sys.path.insert(0, str(Path(sys.argv[sys.argv.index("--checkout") + 1]).resolve()))

import numpy as np
import warp as wp
import warp.sparse as wps

from packed_prototype import packed_mv


def make_case(nrow, length, storage, compute, device, distribution="uniform", shape=(1, 1)):
    rng = np.random.default_rng(2069)
    counts = np.full(nrow, length, dtype=np.int32)
    if distribution == "irregular":
        counts = rng.integers(0, 2 * length + 1, nrow, dtype=np.int32)
    capacity = counts + (length if distribution == "padded" else 0)
    offsets = np.concatenate(([0], np.cumsum(capacity))).astype(np.int32)
    owner = np.repeat(np.arange(nrow), capacity)
    index = np.arange(int(offsets[-1])) - np.repeat(offsets[:-1], capacity)
    active = index < counts[owner]
    columns = ((owner * 17 + index * 13) % nrow).astype(np.int32)
    values = rng.uniform(-1, 1, size=(len(columns), *shape)).astype(np.float32).astype(storage)
    values[~active] = np.nan
    columns[~active] = -1
    block_type = getattr(wp, storage) if shape == (1, 1) else wp.types.matrix(shape, getattr(wp, storage))
    A = wps.bsr_zeros(nrow, nrow, block_type, device=device)
    A.nnz = len(columns)
    A.offsets = wp.array(offsets, dtype=int, device=device)
    A.row_counts = wp.array(counts, dtype=int, device=device) if distribution == "padded" else None
    A.columns = wp.array(columns, dtype=int, device=device)
    A.values = wp.array(values, dtype=block_type, device=device).flatten()
    x_np = rng.uniform(-1, 1, size=(nrow, shape[1])).astype(compute)
    x = wp.array(x_np.ravel(), device=device)
    y = wp.zeros(nrow * shape[0], dtype=getattr(wp, compute), device=device)
    ref = np.zeros((nrow, shape[0]), dtype=np.float64)
    products = np.einsum("bij,bj->bi", values[active].astype(np.float64), x_np[columns[active]].astype(np.float64))
    np.add.at(ref, owner[active], products)
    return A, x, y, ref.ravel()


def run(args):
    wp.init()
    wp.config.quiet = True
    device = wp.get_device(args.device)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "commit": subprocess.check_output(
            ["git", "-C", str(Path(wp.__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": device.name,
        "device_alias": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "arch": device.arch,
        "sparse_sha256": hashlib.sha256((Path(wp.__file__).parent / "_src/sparse.py").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "system": platform.system(),
        "warp": wp.__version__,
        "arguments": vars(args),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,utilization.gpu,clocks.sm,power.draw",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    rng = np.random.default_rng(2069)
    fields = [
        "nrow",
        "length",
        "storage",
        "compute",
        "distribution",
        "block_rows",
        "block_cols",
        "path",
        "mode",
        "round",
        "sample",
        "us",
        "wall_us",
        "max_abs_error",
    ]
    with prefix.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for nrow in args.rows:
            for length in args.lengths:
                for precision in args.precisions:
                    storage, compute = precision.split(":")
                    for distribution in args.distributions:
                        for block in args.blocks:
                            shape = tuple(map(int, block.split("x")))
                            A, x, y, ref = make_case(nrow, length, storage, compute, device, distribution, shape)
                            paths = {
                                "auto": lambda: wps.bsr_mv(A, x, y),
                                "scalar": lambda: wps.bsr_mv(A, x, y, tile_size=-1),
                            }
                            for t in (32, 64):
                                paths[f"tile{t}"] = lambda t=t: wps.bsr_mv(A, x, y, tile_size=t)
                            if args.packed:
                                for b in args.block_dims:
                                    for t in args.threads:
                                        paths[f"packed{b}t{t}"] = (
                                            (lambda b=b, t=t: wps.bsr_mv(A, x, y, tile_size=b, rows_per_block=b // t))
                                            if args.public_api
                                            else (lambda b=b, t=t: packed_mv(A, x, y, b, t))
                                        )
                            graphs, errors = {}, {}
                            for name, fn in paths.items():
                                fn()
                                result = y.numpy()
                                np.testing.assert_allclose(
                                    result,
                                    ref,
                                    atol=2e-5 if compute == "float32" else 1e-12,
                                    rtol=2e-5 if compute == "float32" else 1e-12,
                                )
                                errors[name] = float(np.max(np.abs(result - ref), initial=0))
                                with wp.ScopedCapture(device=device) as capture:
                                    for _ in range(args.iterations):
                                        fn()
                                graphs[name] = capture.graph
                                wp.capture_launch(capture.graph)
                            wp.synchronize_device(device)
                            warmup_start = perf_counter()
                            while perf_counter() - warmup_start < 0.05:
                                for graph in graphs.values():
                                    wp.capture_launch(graph)
                                wp.synchronize_device(device)
                            start, end = wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)
                            for round_id in range(args.rounds):
                                for sample in range(args.samples):
                                    order = list(paths)
                                    rng.shuffle(order)
                                    for name in order:
                                        for mode in args.modes:
                                            wp.synchronize_device(device)
                                            wall = perf_counter()
                                            with wp.ScopedDevice(device):
                                                wp.record_event(start)
                                                if mode == "graph":
                                                    wp.capture_launch(graphs[name])
                                                else:
                                                    for _ in range(args.iterations):
                                                        paths[name]()
                                                wp.record_event(end)
                                            us = wp.get_event_elapsed_time(start, end) * 1000 / args.iterations
                                            wall_us = (perf_counter() - wall) * 1e6 / args.iterations
                                            writer.writerow(
                                                dict(
                                                    nrow=nrow,
                                                    length=length,
                                                    storage=storage,
                                                    compute=compute,
                                                    distribution=distribution,
                                                    block_rows=shape[0],
                                                    block_cols=shape[1],
                                                    path=name,
                                                    mode=mode,
                                                    round=round_id,
                                                    sample=sample,
                                                    us=us,
                                                    wall_us=wall_us,
                                                    max_abs_error=errors[name],
                                                )
                                            )
                                stream.flush()
                            print(nrow, length, precision, distribution, block, "done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkout")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    p.add_argument("--rows", nargs="+", type=int, default=[32768])
    p.add_argument("--lengths", nargs="+", type=int, default=[8, 32])
    p.add_argument("--precisions", nargs="+", default=["float32:float32", "float64:float64", "float32:float64"])
    p.add_argument("--distributions", nargs="+", default=["uniform"])
    p.add_argument("--blocks", nargs="+", default=["1x1"])
    p.add_argument("--packed", action="store_true")
    p.add_argument("--public-api", action="store_true")
    p.add_argument("--block-dims", nargs="+", type=int, default=[128])
    p.add_argument("--threads", nargs="+", type=int, default=[2, 4, 8, 16])
    p.add_argument("--modes", nargs="+", default=["graph"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--iterations", type=int, default=50)
    run(p.parse_args())
