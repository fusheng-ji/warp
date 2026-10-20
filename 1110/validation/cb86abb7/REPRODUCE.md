# Reproduce the recorded source and results

## Restore the exact source

The code branch is not published separately. `source.bundle` contains the original feature commit, public noreply attribution and DCO, with official main `0382a97e` as a prerequisite. From an ordinary clone of NVIDIA/warp containing that prerequisite:

```sh
ASSETS=/path/to/1110/validation/cb86abb7
git bundle verify "$ASSETS/source.bundle"
git fetch "$ASSETS/source.bundle" refs/heads/fusheng-ji/render-shape-segments:refs/heads/reproduce/render-shape-segments
git switch --detach cb86abb75d123876d80c6832d8d360f4cda0cc10
uv run build_lib.py
uv sync --extra dev
```

Compare source and library hashes with the archived environment records. Original measurements used Linux x86_64, Python 3.12.13, Warp 1.19.0.dev1, NumPy 2.5.3, Pyglet 2.1.16, B200 GPUs, NVIDIA driver 580.126.20 and the feature checkout's CUDA 13.0 native build. A new build or different environment must be reported as such.

## Audit saved results without GPU execution

From the restored source checkout with NumPy and Pillow available:

```sh
uv run --no-sync "$ASSETS/verify_results.py" --input "$ASSETS"
uv run --no-sync "$ASSETS/overflow/aggregate_jobs.py" --root "$ASSETS/overflow"
```

The first command recomputes all block/job summaries, P95 and paired ratios from raw frames for the shared-GPU run, the first exclusive allocation and all three primary allocations. It also checks restored PNG pixels exactly and regenerates the publication tables. The second checks the archived per-job summaries and manifests and regenerates cross-job figures. These are data-analysis commands, not new rendering or performance runs. Original reports identify separate historical measurements.

From the asset directory, verify integrity with:

```sh
sha256sum --check SHA256SUMS
```

## Run a new allocated experiment

The archived job directories contain the actual measurement and orchestration scripts used in each run. `reproduce-overflow.sbatch` is a path-parameterized submission template derived from the successful script; it was not the script submitted for the archived jobs. It requires the cluster's `overflow` partition, `wrd` account and full-GPU B200 GRES. Adapt scheduler options on another cluster and record the differences.

```sh
CHECKOUT=/path/to/restored/warp
OVERFLOW_ASSETS="$ASSETS/overflow"
sbatch --chdir="$CHECKOUT" \
  --output=/path/to/private-logs/%j.out \
  --error=/path/to/private-logs/%j.err \
  "$ASSETS/reproduce-overflow.sbatch" "$CHECKOUT" "$OVERFLOW_ASSETS" 1121
```

Create the private log directory first. Use seeds 1132 and 1143 for additional allocations, requesting distinct nodes through your scheduler if reproducing the node-level arrangement. Slurm must expose exactly one allocated CUDA device. The selector matches a CUDA-capable EGL device and verifies native interop with copying disabled; it rejects ambiguity or preexisting GPU processes. Do not copy a numeric EGL index from another machine.

New runs go to `overflow/reproduced-jobs/<job-id>` and use a new private kernel cache. They do not overwrite accepted data. The template intentionally uses the original, archived worker/helper scripts; published worker logic checks the exact source commit. All per-job commands and their seeds are recorded under `logs/`. No native compilation, GPU reruns, scheduler submissions or documentation builds were performed to publish this asset bundle.
