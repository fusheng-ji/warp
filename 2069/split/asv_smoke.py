from pathlib import Path
import sys
import warp as wp

assert Path(wp.__file__).resolve().parent == Path.cwd() / 'warp'
sys.path.insert(0, str(Path('asv').resolve()))
from benchmarks.sparse.matrix_vector import BsrMvMixedPrecision
count = 0
for dtype in ('float32', 'float64'):
    for transpose in (False, True):
        bench = BsrMvMixedPrecision()
        bench.setup(dtype, 32, 64, transpose)
        bench.time_cuda(dtype, 32, 64, transpose)
        count += 1
if '--packed' in sys.argv:
    from benchmarks.sparse.matrix_vector import BsrMvRowPacked
    for rows in (1, 16, 32, 64):
        bench = BsrMvRowPacked()
        bench.setup(32, rows)
        bench.time_cuda(32, rows)
        count += 1
print(f'{count} ASV setup/time combinations passed')
