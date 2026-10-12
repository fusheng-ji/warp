# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record SVD3 invalid-input propagation and zero controls from a built checkout."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def collect(wp, dtype, device, output_arguments):
    """Measure the same 31 cases and acceptance rules as test_svd_nonfinite."""
    scalar = wp.dtype_from_numpy(np.dtype(dtype))
    matrix = wp.types.matrix(shape=(3, 3), dtype=scalar)
    vector = wp.types.vector(length=3, dtype=scalar)

    @wp.kernel(module="unique", enable_backward=False)
    def decompose(
        a: wp.array[matrix],
        u: wp.array[matrix],
        s: wp.array[vector],
        v: wp.array[matrix],
    ):
        i = wp.tid()
        U, sigma, V = matrix(), vector(), matrix()
        if wp.static(output_arguments):
            wp.svd3(a[i], U, sigma, V)
        else:
            U, sigma, V = wp.svd3(a[i])
        u[i] = U
        s[i] = sigma
        v[i] = V

    cases = {}
    for label, value in (
        ("nan", np.nan),
        ("positive_inf", np.inf),
        ("negative_inf", -np.inf),
    ):
        for i in range(3):
            for j in range(3):
                a = np.zeros((3, 3), dtype=dtype)
                a[i, j] = value
                cases[f"{label}_{i}_{j}"] = a
    cases["all_nan"] = np.full((3, 3), np.nan, dtype=dtype)
    cases["nan_with_nonzero"] = np.diag([np.nan, 1.0, 2.0]).astype(dtype)
    cases["zero"] = np.zeros((3, 3), dtype=dtype)
    cases["negative_zero"] = -np.zeros((3, 3), dtype=dtype)
    a = wp.array(np.array(list(cases.values())), dtype=matrix, device=device)
    u, v = wp.empty_like(a), wp.empty_like(a)
    s = wp.empty(len(cases), dtype=vector, device=device)
    wp.launch(decompose, dim=len(cases), inputs=[a], outputs=[u, s, v], device=device)
    un, sn, vn = (x.numpy().astype(np.float64) for x in (u, s, v))
    tol = {np.float16: 5e-3, np.float32: 1e-5, np.float64: 1e-12}[dtype]
    rows = []
    for i, name in enumerate(cases):
        U, sigma, V = un[i], sn[i], vn[i]
        with np.errstate(invalid="ignore"):
            reconstructed = U @ np.diag(sigma) @ V.T
        zero_control = name in ("zero", "negative_zero")
        sigma_finite = bool(np.isfinite(sigma).all())
        reconstruction_finite = bool(np.isfinite(reconstructed).all())
        orth_u = orth_v = ""
        detail = ""
        try:
            if zero_control:
                np.testing.assert_array_equal(sigma, np.zeros(3))
                np.testing.assert_array_equal(reconstructed, np.zeros((3, 3)))
                np.testing.assert_allclose(U.T @ U, np.eye(3), rtol=0.0, atol=tol)
                np.testing.assert_allclose(V.T @ V, np.eye(3), rtol=0.0, atol=tol)
                orth_u = float(np.max(np.abs(U.T @ U - np.eye(3))))
                orth_v = float(np.max(np.abs(V.T @ V - np.eye(3))))
            else:
                if sigma_finite or reconstruction_finite:
                    raise AssertionError(
                        "Invalid input produced finite singular values or reconstruction"
                    )
            passed = True
        except AssertionError as exc:
            passed = False
            detail = str(exc).strip()
        rows.append(
            {
                "device": str(device),
                "dtype": np.dtype(dtype).name,
                "overload": "output_arguments" if output_arguments else "return_values",
                "case": name,
                "zero_control": zero_control,
                "sigma_finite": sigma_finite,
                "reconstruction_finite": reconstruction_finite,
                "sigma_nonfinite_count": int(np.count_nonzero(~np.isfinite(sigma))),
                "reconstruction_nonfinite_count": int(
                    np.count_nonzero(~np.isfinite(reconstructed))
                ),
                "sigma": json.dumps(sigma.tolist()),
                "orth_u": orth_u,
                "orth_v": orth_v,
                "zero_orthogonality_atol": tol,
                "passed": passed,
                "detail": detail,
            }
        )
    return rows


def main():
    """Verify source identity, collect evaluations, and retain every failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if head != args.expected_commit:
        raise RuntimeError("Checkout HEAD does not match --expected-commit")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "warp"], cwd=checkout, check=True
    )
    sys.path.insert(0, str(checkout))
    import warp as wp

    if Path(wp.__file__).resolve().parent != checkout / "warp":
        raise RuntimeError("Warp was not imported from the requested checkout")
    wp.init()
    rows = []
    for device in args.devices:
        for dtype in (np.float16, np.float32, np.float64):
            for output_arguments in (False, True):
                rows.extend(collect(wp, dtype, device, output_arguments))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(not row["passed"] for row in rows)
    print(f"Source {head}: {len(rows)} evaluations; {failures} failures")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
