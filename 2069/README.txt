GH-2069 validation materials

Scope
FP32 matrix storage with FP64 vectors and arithmetic; explicit CUDA row packing.
The original automatic dispatch is retained. Transposed mixed-precision products
are supported; explicit row packing requires a non-transposed CUDA product.
No FP16/BF16 support, solver dtype-inference changes, or new autodiff support.

Reference
All numerical comparisons use the coefficients actually stored in the matrix.
Promoting FP32 values does not recover accuracy lost during their construction.

Measurement protocol
Compilation, allocation, initialization and reference computation are excluded.
Each case receives at least 50 ms of untimed graph warmup after compilation.
GPU events surround batches of 50 products. Every product overwrites its output
(beta=0), avoiding growth across repetitions. Paths are randomized within each
sample. Graph times describe device execution; ordinary-launch stream times also
include gaps caused by host submission. Raw CSV files retain every sample.
Final storage comparisons use identical matrix coefficients (FP32 values
promoted to FP64 for the converted-matrix reference) and identical FP64 vectors.
The validation sweep fixes 128 threads/block and 4 threads/row, independently of
which configuration happened to win the exploratory tuning sweep.
The final-* timing data uses the public bsr_mv API. packed_prototype.py is
included only to reproduce exploratory comparisons, not as a separate dependency
of the library implementation.
Final timing runs use CUDA_VISIBLE_DEVICES=1 to isolate the second physical
B200; its logical alias in those processes is cuda:0. The first physical GPU
had an unrelated job, which was not interrupted. Correctness checks use both
physical devices without remapping. Results establish behavior on B200, not
across GPU architectures.

Reproduce (from a Warp checkout containing the patch)
Build the patched and baseline checkouts independently with uv run build_lib.py
if their native libraries are absent. Use the base commit recorded in validation.json.
Run the included script, passing the baseline checkout and a fresh output directory:

CUDA_VISIBLE_DEVICES=1 bash path/to/2069/reproduce.sh path/to/baseline _build/2069-rerun

The supplied timing data used isolated GPU 1, while the supplied numerical data
used CPU and both GPUs. A rerun with CUDA_VISIBLE_DEVICES set covers only visible
devices, so its diagnostic case count can differ. Reproduction requires NumPy,
SciPy for the FEM reference, and Matplotlib for figures; uv supplies the extras.
report_results.py can rebuild the HTML report after a validation.json describing
the rerun has been written. Do not reuse the supplied test provenance for a new run.

Baseline comparison
Use the same scripts with --checkout pointing to an unmodified checkout of the
recorded official main commit. Baseline libraries must be built from that commit.
Omit --packed and --public-api when timing the original implementation.

Known baseline limitation
Forcing the existing tile path on CPU is not covered by the new mixed-precision
contract. On the unmodified baseline, an FP64 3x3 identity times ones returns
[0, 0, 0] with tile_size=64 and [1, 1, 1] with tile_size=-1. CPU validation uses
automatic/scalar and transpose paths; forced tiled paths are validated on CUDA.
This pre-existing CPU behavior is not changed by the patch.

Publication
These materials describe an independent correctness and performance evaluation.
No maintainer assignment or downstream production deployment is claimed.
Raw local build/test logs may contain workstation paths and are excluded from the
publication bundle. DCO sign-off is retained.
