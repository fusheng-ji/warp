#!/usr/bin/env bash
# Run from the patched Warp checkout; both checkouts need their own native build.
set -euo pipefail
TASK_ASSETS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_BASELINE=${1:?Usage: bash path/to/reproduce.sh BASELINE_CHECKOUT [OUTPUT_DIRECTORY]}
TASK_RESULTS=${2:-_build/2069-reproduction}
mkdir -p "$TASK_RESULTS"
export OPENBLAS_NUM_THREADS=1
uv run --extra dev -m unittest warp.tests.test_sparse
uv run --extra dev -m unittest warp.tests.test_linear_solvers warp.tests.tile.test_tile_reduce
uv run "$TASK_ASSETS/numerics.py" --checkout "$TASK_BASELINE" --output "$TASK_RESULTS/numerics-before.csv"
uv run "$TASK_ASSETS/numerics.py" --checkout . --output "$TASK_RESULTS/numerics-after.csv"
uv run "$TASK_ASSETS/benchmark.py" --checkout "$TASK_BASELINE" --output "$TASK_RESULTS/final-baseline" --precisions float32:float32 float64:float64 --rows 4096 32768 --lengths 2 8 32 128 --rounds 5 --samples 6 --modes graph launch
uv run "$TASK_ASSETS/benchmark.py" --output "$TASK_RESULTS/final-uniform" --packed --public-api --rows 4096 32768 131072 --lengths 2 4 8 16 32 64 128 256 --threads 4 --rounds 5 --samples 6 --modes graph launch
uv run "$TASK_ASSETS/benchmark.py" --output "$TASK_RESULTS/final-irregular" --packed --public-api --rows 32768 131072 --lengths 8 32 128 --distributions irregular padded --threads 4 --rounds 5 --samples 6
uv run "$TASK_ASSETS/benchmark.py" --output "$TASK_RESULTS/final-blocks" --packed --public-api --rows 32768 --lengths 8 32 --distributions uniform padded --blocks 2x3 3x3 --threads 4 --rounds 5 --samples 6
uv run --with scipy "$TASK_ASSETS/fem_benchmark.py" "$TASK_RESULTS/final-fem.csv"
uv run "$TASK_ASSETS/conversion.py" "$TASK_RESULTS/conversion.json"
uv run --with matplotlib "$TASK_ASSETS/plot_results.py" "$TASK_RESULTS"
# report_results.py additionally needs a validation.json describing the rerun.
