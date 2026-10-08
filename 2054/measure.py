# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the GH-2054 reproducer without relying on Warp's test helpers."""

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

import warp as wp


def matrix_json(matrix):
    return json.dumps([[float(x) if np.isfinite(x) else str(float(x)) for x in row] for row in matrix], allow_nan=False)


def error_ratio(actual, expected, tol):
    if not np.isfinite(actual).all():
        return float("nan")
    return float(np.max(np.abs(actual - expected) / (tol + tol * np.abs(expected))))


def collect(device, dtype, n, output_arguments):
    tol = {np.float16: 1.0e-2, np.float32: 1.0e-5, np.float64: 1.0e-10}[dtype]
    scalar = wp.dtype_from_numpy(np.dtype(dtype))
    matrix = wp.types.matrix(shape=(n, n), dtype=scalar)
    vector = wp.types.vector(length=n, dtype=scalar)

    @wp.kernel(module="unique")
    def spectral(matrices: wp.array[matrix], add_trace: bool, losses: wp.array[scalar]):
        i = wp.tid()
        A = matrices[i]
        U = matrix()
        s = vector()
        V = matrix()
        if wp.static(n == 2):
            if wp.static(output_arguments):
                wp.svd2(A, U, s, V)
            else:
                U, s, V = wp.svd2(A)
        else:
            if wp.static(output_arguments):
                wp.svd3(A, U, s, V)
            else:
                U, s, V = wp.svd3(A)
        result = wp.dot(s, s)
        if add_trace:
            result = result + wp.trace(A)
        losses[i] = result

    @wp.kernel(module="unique")
    def trace(matrices: wp.array[matrix], losses: wp.array[scalar]):
        i = wp.tid()
        losses[i] = wp.trace(matrices[i])

    inputs = np.array([2.0 * np.eye(n)] * 3, dtype=dtype)
    matrices = wp.array(inputs, dtype=matrix, device=device, requires_grad=True)
    losses = wp.zeros(3, dtype=scalar, device=device, requires_grad=True)
    diagnostic = wp.zeros_like(losses, requires_grad=False)
    reference = inputs.astype(np.float64)
    rows = []
    for scenario in ("norm", "norm_trace", "masked_norm", "trace", "diagnostic_trace"):
        seed_values = np.array([0.0, 1.0, -0.5] if scenario == "masked_norm" else [1.0] * 3, dtype=dtype)
        seeds = wp.array(seed_values, dtype=scalar, device=device)
        with wp.Tape() as tape:
            if scenario in ("trace", "diagnostic_trace"):
                if scenario == "diagnostic_trace":
                    wp.launch(spectral, dim=3, inputs=[matrices, False], outputs=[diagnostic], device=device)
                wp.launch(trace, dim=3, inputs=[matrices], outputs=[losses], device=device)
            else:
                wp.launch(spectral, dim=3, inputs=[matrices, scenario == "norm_trace"], outputs=[losses], device=device)
        tape.backward(grads={losses: seeds})
        actual_losses = losses.numpy().astype(np.float64)
        actual_gradients = matrices.grad.numpy().astype(np.float64)
        for sample in range(3):
            A = reference[sample]
            if scenario in ("trace", "diagnostic_trace"):
                expected_loss = np.trace(A)
                expected_gradient = np.eye(n)
            else:
                expected_loss = np.sum(A**2) + (np.trace(A) if scenario == "norm_trace" else 0.0)
                expected_gradient = 2.0 * A + (np.eye(n) if scenario == "norm_trace" else 0.0)
            expected_gradient = expected_gradient * float(seed_values[sample])
            loss = actual_losses[sample]
            gradient = actual_gradients[sample]
            loss_error = error_ratio(loss, expected_loss, tol)
            grad_error = error_ratio(gradient, expected_gradient, tol)
            rows.append(
                {
                    "device": device,
                    "dtype": dtype.__name__,
                    "n": n,
                    "overload": "output_arguments" if output_arguments else "return_values",
                    "scenario": scenario,
                    "sample": sample,
                    "seed": float(seed_values[sample]),
                    "loss": float(loss),
                    "expected_loss": float(expected_loss),
                    "gradient": matrix_json(gradient),
                    "expected_gradient": matrix_json(expected_gradient),
                    "finite_gradient": bool(np.isfinite(gradient).all()),
                    "max_abs_gradient_error": float(np.max(np.abs(gradient - expected_gradient))),
                    "loss_error_ratio": loss_error,
                    "gradient_error_ratio": grad_error,
                    "passed": bool(loss_error <= 1.0 and grad_error <= 1.0),
                    "atol": tol,
                    "rtol": tol,
                }
            )
        tape.zero()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0", "cuda:1"])
    args = parser.parse_args()
    source = Path(wp.__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != args.expected_commit:
        raise RuntimeError("Imported Warp source does not match the requested commit")
    wp.config.kernel_cache_dir = str(args.cache)
    wp.init()
    rows = []
    for device in args.devices:
        for dtype in (np.float16, np.float32, np.float64):
            for n in (2, 3):
                for output_arguments in (False, True):
                    rows.extend(collect(device, dtype, n, output_arguments))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"{args.phase}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "phase": args.phase,
        "commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "warp": wp.__version__,
        "platform": platform.system() + " " + platform.machine(),
        "devices": [{"alias": device, "name": wp.get_device(device).name} for device in args.devices],
        "evaluations": len(rows),
        "failed": sum(not row["passed"] for row in rows),
    }
    (args.output / f"{args.phase}-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
