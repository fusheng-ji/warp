# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
from pathlib import Path

import numpy as np
import warp as wp


def collect(device, dtype, output_arguments):
    """Check SVD3 factors for degenerate inputs and different matrix scales."""
    tol = {np.float16: 5.0e-3, np.float32: 1.0e-5, np.float64: 1.0e-12}[dtype]
    wptype = wp.dtype_from_numpy(np.dtype(dtype))
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)

    @wp.kernel(module="unique", enable_backward=False)
    def decompose(
        matrices: wp.array[mat33],
        left: wp.array[mat33],
        values: wp.array[vec3],
        right: wp.array[mat33],
    ):
        i = wp.tid()
        U = mat33()
        sigma = vec3()
        V = mat33()
        if wp.static(output_arguments):
            wp.svd3(matrices[i], U, sigma, V)
        else:
            U, sigma, V = wp.svd3(matrices[i])
        left[i] = U
        values[i] = sigma
        right[i] = V

    c, s = np.cos(0.37), np.sin(0.37)
    left_rotation = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    c, s = np.cos(0.71), np.sin(0.71)
    right_rotation = np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
    cases = {
        "distinct": np.diag([1.0, 2.0, 3.0]),
        "repeated_01": np.diag([2.0, 2.0, 3.0]),
        "repeated_02": np.diag([2.0, 3.0, 2.0]),
        "repeated_12": np.diag([3.0, 2.0, 2.0]),
        "identity": np.eye(3),
        "rank_one": np.diag([0.0, 0.0, 3.0]),
        "rank_two": np.diag([0.0, 2.0, 3.0]),
        "rotated_distinct": left_rotation @ np.diag([1.0, 2.0, 3.0]) @ right_rotation.T,
        "rotated_repeated": left_rotation @ np.diag([2.0, 2.0, 3.0]) @ right_rotation.T,
        "negative_determinant": left_rotation @ np.diag([1.0, 2.0, -3.0]) @ right_rotation.T,
    }
    # The half path must also handle scales below the float32 QR threshold.
    exponents = (-20, -10, 0, 10, 14) if dtype == np.float16 else (-10, 0, 10, 14)
    names = [(name, exponent) for name in cases for exponent in exponents] + [("zero", 0)]
    matrices_np = np.array(
        [matrix * 2.0**exponent for matrix in cases.values() for exponent in exponents] + [np.zeros((3, 3))],
        dtype=dtype,
    )
    matrices = wp.array(matrices_np, dtype=mat33, device=device)
    left = wp.empty_like(matrices)
    values = wp.empty(len(names), dtype=vec3, device=device)
    right = wp.empty_like(matrices)
    wp.launch(decompose, dim=len(names), inputs=[matrices], outputs=[left, values, right], device=device)

    left_np, values_np, right_np = (a.numpy().astype(np.float64) for a in (left, values, right))
    matrices_fp64 = matrices_np.astype(np.float64)
    reference_values = np.linalg.svd(matrices_fp64, compute_uv=False)
    rounding_atol = 2.0 * float(np.finfo(dtype).smallest_subnormal)
    return names, matrices_fp64, left_np, values_np, right_np, reference_values, tol, rounding_atol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('before', 'after'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    wp.config.kernel_cache_dir = str(args.cache)
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError('CUDA validation requires an NVIDIA GPU')
    rows = []
    for device in ('cpu', 'cuda:0'):
        for dtype in (np.float16, np.float32, np.float64):
            for output_arguments in (False, True):
                names, inputs, left, values, right, refs, tol, rounding_atol = collect(
                    device, dtype, output_arguments
                )
                for i, (name, exponent) in enumerate(names):
                    U, sigma, V, A = left[i], values[i], right[i], inputs[i]
                    finite = all(np.isfinite(x).all() for x in (U, sigma, V))
                    scale = float(np.max(np.abs(A)))
                    bound = tol * scale + rounding_atol
                    row = dict(phase=args.phase, device='CPU' if device == 'cpu' else 'CUDA',
                               dtype=dtype.__name__, overload='outputs' if output_arguments else 'tuple',
                               matrix=name, exponent=exponent, finite=finite,
                               sigma0=float(sigma[0]) if np.isfinite(sigma[0]) else '',
                               sigma1=float(sigma[1]) if np.isfinite(sigma[1]) else '',
                               sigma2=float(sigma[2]) if np.isfinite(sigma[2]) else '',
                               reconstruction_abs='', reconstruction_scaled='',
                               reconstruction_budget_ratio='', orth_u='', orth_v='',
                               singular_values_abs='', passed=False)
                    if finite:
                        rec = U @ np.diag(sigma) @ V.T
                        error = float(np.max(np.abs(rec - A)))
                        orth_u = float(np.max(np.abs(U.T @ U - np.eye(3))))
                        orth_v = float(np.max(np.abs(V.T @ V - np.eye(3))))
                        serr = float(np.max(np.abs(np.abs(sigma) - refs[i])))
                        passed = bool(error <= bound and orth_u <= tol and orth_v <= tol
                                      and serr <= bound and abs(np.linalg.det(U)-1) <= tol
                                      and abs(np.linalg.det(V)-1) <= tol
                                      and sigma[0] >= 0 and sigma[1] >= 0)
                        if name == 'negative_determinant':
                            passed = passed and bool(sigma[2] < 0)
                        if name == 'zero':
                            passed = passed and bool(np.all(sigma == 0) and np.all(rec == 0))
                        row.update(reconstruction_abs=error, reconstruction_scaled=error/(scale or 1),
                                   reconstruction_budget_ratio=error/bound, orth_u=orth_u,
                                   orth_v=orth_v, singular_values_abs=serr, passed=passed)
                    rows.append(row)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f'{args.phase}: {len(rows)} samples; {sum(r["passed"] for r in rows)} passed')


if __name__ == '__main__':
    main()
