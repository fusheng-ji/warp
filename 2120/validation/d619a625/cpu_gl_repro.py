"""Run in its own process: CUDA_VISIBLE_DEVICES=0 uv run cpu_gl_repro.py."""

import ctypes
import unittest

import numpy as np
import pyglet

import warp as wp

pyglet.options["headless"] = True
pyglet.options["headless_device"] = 0

from pyglet import gl  # noqa: E402  (headless mode must be selected first)

wp.init()
with wp.ScopedDevice("cuda:0"):
    wp.zeros(1, device="cuda:0")
    wp.synchronize_device("cuda:0")
    with wp.ScopedDevice("cpu"):
        window = pyglet.window.Window(width=16, height=16, visible=False)
        buffer_id = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(buffer_id))
        registered = None
        mapped = None
        try:
            data = np.arange(8, dtype=np.float32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_id)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            registered = wp.RegisteredGLBuffer(int(buffer_id.value), device="cpu")
            mapped = registered.map(dtype=wp.float32, shape=(len(data),))
            print("CUDA registration succeeded:", registered.resource is not None, flush=True)
            print("Mapped array device:", mapped.device, flush=True)
            # Never dereference a potentially mislabelled CUDA device pointer.
            unittest.TestCase().assertIsNone(registered.resource, "CPU buffer unexpectedly registered with CUDA")
            np.testing.assert_array_equal(mapped.numpy(), data)
        finally:
            if mapped is not None:
                registered.unmap()
            mapped = None
            registered = None
            gl.glDeleteBuffers(1, ctypes.byref(buffer_id))
            window.close()
