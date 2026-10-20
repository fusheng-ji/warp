"""Check that the CUDA copy-fallback test preserves its incoming warning state."""

import argparse
import json
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import warp as wp
from warp.tests.test_render_opengl import TestRegisteredGLBuffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", choices=("false", "true"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial = args.initial == "true"
    flag = "_RegisteredGLBuffer__fallback_warning_shown"
    with mock.patch.object(wp.RegisteredGLBuffer, flag, initial):
        result = unittest.TestResult()
        TestRegisteredGLBuffer("test_cuda_buffer_copy_fallback").run(result)
        after = getattr(wp.RegisteredGLBuffer, flag)
        report = {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "initial_warning_shown": initial,
            "warning_shown_after_test": after,
            "test_method_passed": result.wasSuccessful(),
            "test_methods_run": result.testsRun,
            "state_preserved": after is initial,
            "failures": [{"test": test.id(), "traceback": trace} for test, trace in result.failures],
            "errors": [{"test": test.id(), "traceback": trace} for test, trace in result.errors],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["test_method_passed"] or not report["state_preserved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
