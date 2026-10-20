import hashlib
import importlib.metadata
import json
import platform
import subprocess
import unittest
from pathlib import Path

import warp as wp

wp.init()
root = Path.cwd().resolve()
runtime = wp.RegisteredGLBuffer.__init__.__globals__["runtime"]
unittest.TestCase().assertEqual(Path(wp.__file__).resolve().parent, root / "warp")
unittest.TestCase().assertEqual(Path(runtime.core._name).resolve(), root / "warp/bin/warp.so")
unittest.TestCase().assertEqual(Path(runtime.llvm._name).resolve(), root / "warp/bin/warp-clang.so")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


files = [
    "warp/native/svd.h",
    "warp/_src/context.py",
    "warp/tests/matrix/test_mat.py",
    "warp/tests/test_render_opengl.py",
    "warp/tests/unittest_suites.py",
]
result = {
    "source_commit": git("rev-parse", "HEAD"),
    "tracked_tree_clean": not bool(git("status", "--porcelain", "--untracked-files=no")),
    "loaded_libraries_from_this_checkout": True,
    "python": platform.python_version(),
    "system": platform.system(),
    "architecture": platform.machine(),
    "warp": wp.__version__,
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    "source_sha256": {f: digest(root / f) for f in files},
    "native_library_sha256": {p.name: digest(p) for p in sorted((root / "warp/bin").glob("*.so"))},
    "gpus": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"], text=True
    ).splitlines(),
    "cuda_compiler": subprocess.check_output(["/usr/local/cuda-13.0/bin/nvcc", "--version"], text=True).splitlines()[
        -2:
    ],
}
print("ENVIRONMENT_JSON=" + json.dumps(result, sort_keys=True))
