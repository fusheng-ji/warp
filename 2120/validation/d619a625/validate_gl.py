# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate real EGL rendering and GL-buffer transfers in one isolated process.

The caller chooses CUDA_VISIBLE_DEVICES and captures the process exit code.
Examples, run from the tested checkout with its independent environment::

    uv run --no-sync ../warp-cpu-gl-results/validate_gl.py --checkout . --output OUT --mode render --device cpu
    uv run --no-sync ../warp-cpu-gl-results/validate_gl.py --checkout . --output OUT --mode buffers --device cuda:0
    uv run --no-sync ../warp-cpu-gl-results/validate_gl.py --checkout . --output OUT --mode buffers --device cuda:0 --force-registration-failure
    uv run --no-sync ../warp-cpu-gl-results/validate_gl.py --checkout . --output OUT --mode switch --device cpu

``switch`` needs the #1110 feature. ``render`` uses only baseline geometry APIs.
Fault injection is restricted to this process and deliberately reaches the
native API through the public class's function globals; it is not product code.
CPU mode never selects or requires a CUDA device unless --ambient-cuda is set.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


class Journal:
    """Persist the current stage before each operation that could terminate the process."""

    def __init__(self, output, arguments):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()
        self.document = {
            "status": "running",
            "stage": "starting",
            "arguments": arguments,
            "environment": {},
            "events": [],
            "results": [],
        }
        self.save()

    def save(self):
        self.document["elapsed_seconds"] = time.monotonic() - self.start
        with (self.output / "result.json").open("w") as stream:
            json.dump(self.document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        rows = self.document["results"]
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (self.output / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())

    def stage(self, name, **details):
        self.document["stage"] = name
        self.document["events"].append({"stage": name, "seconds": time.monotonic() - self.start, **details})
        self.save()
        print(name, details, flush=True)

    def row(self, **details):
        self.document["results"].append(details)
        self.save()


def make_renderer(renderer_type, device):
    return renderer_type(
        screen_width=384,
        screen_height=256,
        camera_pos=(0.0, 0.5, 5.0),
        camera_front=(0.0, 0.0, -1.0),
        background_color=(0.08, 0.09, 0.12),
        headless=True,
        draw_grid=False,
        draw_sky=False,
        draw_axis=False,
        show_info=False,
        render_wireframe=True,
        vsync=False,
        device=device,
    )


def update_gl_environment(journal, gl):
    journal.document["environment"].update(
        gl_vendor=gl.gl_info.get_vendor(),
        gl_renderer=gl.gl_info.get_renderer(),
        gl_version=list(gl.gl_info.get_version()),
    )
    journal.save()


def validate_buffers(args, journal, wp, renderer_type, np, check):
    journal.stage("create_gl_context")
    renderer = make_renderer(renderer_type, args.device)
    gl = renderer_type.gl
    update_gl_environment(journal, gl)
    try:
        for flag_name in ("NONE", "READ_ONLY", "WRITE_DISCARD"):
            flag = getattr(wp.RegisteredGLBuffer, flag_name)
            journal.stage("create_gl_buffer", flag=flag_name)
            initial = np.arange(32, dtype=np.float32) * 0.25 - 3.0
            updated = np.arange(32, dtype=np.float32)[::-1].copy() * 0.5 + 5.0
            buffer_id = gl.GLuint()
            gl.glGenBuffers(1, buffer_id)
            registered = None
            mapped = None
            try:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_id)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, initial.nbytes, initial.ctypes.data, gl.GL_DYNAMIC_DRAW)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                journal.stage("register_gl_buffer", flag=flag_name)
                registered = wp.RegisteredGLBuffer(int(buffer_id.value), device=args.device, flags=flag)
                expected_copy = args.device == "cpu" or args.force_registration_failure
                check.assertEqual(registered.resource is None, expected_copy)
                journal.stage("map_gl_buffer", flag=flag_name)
                mapped = registered.map(dtype=wp.float32, shape=(len(initial),))
                check.assertEqual(mapped.device, wp.get_device(args.device))
                if expected_copy:
                    if args.device == "cpu":
                        check.assertFalse(mapped.pinned)
                        check.assertIs(registered.warp_buffer, registered.warp_buffer_cpu)
                    else:
                        check.assertTrue(registered.warp_buffer_cpu.pinned)
                if flag_name != "WRITE_DISCARD":
                    journal.stage("read_mapped_data", flag=flag_name)
                    np.testing.assert_array_equal(mapped.numpy(), initial)
                if flag_name != "READ_ONLY":
                    journal.stage("write_mapped_data", flag=flag_name)
                    wp.copy(mapped, wp.array(updated, dtype=wp.float32, device=args.device))
                journal.stage("unmap_gl_buffer", flag=flag_name)
                registered.unmap()
                mapped = None
                journal.stage("read_final_gl_data", flag=flag_name)
                actual = np.empty_like(initial)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_id)
                gl.glGetBufferSubData(gl.GL_ARRAY_BUFFER, 0, actual.nbytes, actual.ctypes.data)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                expected = initial if flag_name == "READ_ONLY" else updated
                np.testing.assert_array_equal(actual, expected)
                check.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)
                journal.row(
                    mode="buffers",
                    device=args.device,
                    flag=flag_name,
                    transfer="copy" if expected_copy else "interop",
                    count=len(actual),
                    read_checked=flag_name != "WRITE_DISCARD",
                    write_checked=flag_name != "READ_ONLY",
                    source_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                    result_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                    passed=True,
                )
            finally:
                if mapped is not None:
                    journal.stage("cleanup_unmap_gl_buffer", flag=flag_name)
                    registered.unmap()
                del registered
                gc.collect()
                gl.glDeleteBuffers(1, buffer_id)
    finally:
        journal.stage("close_gl_context")
        renderer.close()


def validate_render(args, journal, wp, renderer_type, np, image_type, check):
    switch = args.mode == "switch"
    shapes = {
        "sphere": {"radius": 0.5},
        "capsule": {"radius": 0.5, "half_height": 0.6},
        "cylinder": {"radius": 0.5, "half_height": 0.6},
        "cone": {"radius": 0.5, "half_height": 0.6},
        "arrow": {"base_radius": 0.2, "base_height": 1.0},
    }
    for kind, parameters in shapes.items():
        for cached in (False, True) if switch else (False,):
            journal.stage("create_renderer", primitive=kind, cached=cached)
            renderer = make_renderer(renderer_type, args.device)
            gl = renderer_type.gl
            update_gl_environment(journal, gl)
            try:
                render = getattr(renderer, f"render_{kind}")
                rotation = (0.0, 0.0, 0.0, 1.0)
                renderer.register_body("parent")
                body_transforms = wp.array(
                    [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], dtype=wp.transform, device=args.device
                )
                if cached:
                    render("template", (0.0, 0.0, 0.0), rotation, **parameters, segments=8, is_template=True)
                initial_segments = {"segments": 32} if switch else {}
                high = render(
                    "first", (-0.8, 0.0, 0.0), rotation, **parameters, parent_body="parent", **initial_segments
                )
                render("second", (0.8, 0.0, 0.0), rotation, **parameters, **initial_segments)
                instance_id = renderer._instances["first"][0]
                renderer.update_shape_instance("first", scale=(1.0, 0.9, 1.0))
                renderer._instance_custom_ids[instance_id] = 17
                before = renderer._instances["first"]
                if cached:
                    renderer.setup_tiled_rendering([[instance_id], [renderer._instances["second"][0]]])
                images = []
                sequence = (32, 8, 32) if switch else (32, 32)
                for frame, segments in enumerate(sequence):
                    journal.stage("begin_frame", primitive=kind, cached=cached, frame=frame, segments=segments)
                    renderer.begin_frame(frame / 60.0)
                    frame_segments = {"segments": segments} if switch else {}
                    shape = render("first", (-0.8, 0.0, 0.0), rotation, **parameters, **frame_segments)
                    if renderer._add_shape_instances:
                        renderer.allocate_shape_instances()
                    if renderer._update_shape_instances:
                        renderer.update_shape_instances()
                    renderer.update_body_transforms(body_transforms)
                    journal.stage("end_frame", primitive=kind, cached=cached, frame=frame)
                    renderer.end_frame()
                    journal.stage("get_pixels", primitive=kind, cached=cached, frame=frame)
                    image = wp.empty((256, 384, 3), dtype=wp.uint8, device=args.device)
                    check.assertTrue(renderer.get_pixels(image, split_up_tiles=False, use_uint8=True))
                    pixels = image.numpy()
                    unique_colors = np.unique(pixels.reshape(-1, 3), axis=0).shape[0]
                    check.assertGreater(unique_colors, 8, "Empty or constant render")
                    check.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)
                    current = renderer._instances["first"]
                    check.assertEqual(current[:2], before[:2])
                    check.assertEqual(current[4:], before[4:])
                    check.assertEqual(current[2], shape)
                    check.assertEqual(renderer._instance_shape[instance_id], shape)
                    check.assertEqual(renderer._instance_custom_ids[instance_id], 17)
                    check.assertEqual(renderer._instances["second"][2], high)
                    check.assertEqual(renderer._instance_count, 2)
                    check.assertEqual(len(renderer._instances), 2)
                    check.assertEqual(sorted(i for group in renderer._shape_instances.values() for i in group), [0, 1])
                    check.assertIn(instance_id, renderer._shape_instances[shape])
                    check.assertEqual(renderer._instance_ids[renderer._inverse_instance_ids[instance_id]], instance_id)
                    journal.stage("map_transforms", primitive=kind, cached=cached, frame=frame)
                    mapped = renderer._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(2,))
                    try:
                        transforms = mapped.numpy()
                    finally:
                        renderer._instance_transform_cuda_buffer.unmap()
                    expected_transform = np.array(
                        ((1.0, 0.0, 0.0, 0.0), (0.0, 0.9, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (-0.8, 0.0, 0.0, 1.0)),
                        dtype=np.float32,
                    )
                    np.testing.assert_array_equal(
                        transforms[renderer._inverse_instance_ids[instance_id]], expected_transform
                    )
                    if segments == 32:
                        check.assertEqual(shape, high)
                    vertices, indices = renderer._shapes[shape][:2]
                    check.assertTrue(np.isfinite(vertices).all())
                    check.assertLess(int(indices.max()), len(vertices))
                    restored_equal = frame == len(sequence) - 1
                    if restored_equal:
                        np.testing.assert_array_equal(pixels, images[0])
                    images.append(pixels.copy())
                    image_name = f"{kind}-cached-{int(cached)}-frame-{frame}-segments-{segments}.png"
                    image_type.fromarray(pixels[::-1]).save(journal.output / image_name)
                    journal.row(
                        mode=args.mode,
                        device=args.device,
                        primitive=kind,
                        target_pre_cached=cached,
                        tiled=cached,
                        frame=frame,
                        segments=segments,
                        vertices=len(vertices),
                        triangles=len(indices) // 3,
                        instance_id=instance_id,
                        instance_count=len(renderer._instances),
                        cached_meshes=len(renderer._shapes),
                        unique_colors=unique_colors,
                        restored_pixel_equality_checked=restored_equal,
                        image=image_name,
                        image_sha256=hashlib.sha256((journal.output / image_name).read_bytes()).hexdigest(),
                        passed=True,
                    )
            finally:
                journal.stage("close_renderer", primitive=kind, cached=cached)
                renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("render", "buffers", "switch"), required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0", "cuda:1"))
    parser.add_argument(
        "--egl-device", type=int, default=0, help="Physical EGL device index; does not change CUDA visibility"
    )
    parser.add_argument(
        "--ambient-cuda", action="store_true", help="Initialize cuda:0 and keep its context current for CPU tests"
    )
    parser.add_argument("--force-registration-failure", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    journal = Journal(
        args.output.resolve(),
        {key: value for key, value in vars(args).items() if key not in ("checkout", "output")},
    )
    check = unittest.TestCase()
    try:
        journal.stage("import_dependencies")
        sys.path.insert(0, str(checkout))
        import numpy as np  # noqa: PLC0415
        import pyglet  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        pyglet.options["headless"] = True
        pyglet.options["headless_device"] = args.egl_device
        pyglet.options["debug_gl"] = True
        import warp as wp  # noqa: PLC0415
        from warp.render import OpenGLRenderer  # noqa: PLC0415

        check.assertEqual(Path(wp.__file__).resolve().parent.parent, checkout)
        journal.stage("warp_init")
        wp.init()
        environment = journal.document["environment"]
        environment.update(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
            python=platform.python_version(),
            platform=platform.system() + " " + platform.machine(),
            warp=wp.__version__,
            numpy=np.__version__,
            pyglet=pyglet.version,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            device=args.device,
            source_sha256={
                path: hashlib.sha256((checkout / path).read_bytes()).hexdigest()
                for path in (
                    "warp/_src/context.py",
                    "warp/_src/render/render_opengl.py",
                    "warp/tests/test_render_opengl.py",
                )
            },
            library_sha256={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((checkout / "warp/bin").glob("*.so"))
            },
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        journal.save()
        with ExitStack() as stack:
            if args.ambient_cuda:
                journal.stage("create_ambient_cuda_context")
                stack.enter_context(wp.ScopedDevice("cuda:0"))
                wp.zeros(1, device="cuda:0")
                wp.synchronize_device("cuda:0")
                environment["ambient_cuda_device"] = wp.get_device("cuda:0").name
            if args.device.startswith("cuda:"):
                environment["cuda_device"] = wp.get_device(args.device).name
            stack.enter_context(wp.ScopedDevice(args.device))
            # ScopedDevice("cpu") selects CPU as the Warp default while preserving
            # an optional current CUDA context, which reproduces the unsafe case.
            runtime = wp.RegisteredGLBuffer.__init__.__globals__["runtime"]
            original_register = runtime.core.wp_cuda_graphics_register_gl_buffer
            register_calls = []

            def register(*call_args):
                register_calls.append(len(register_calls) + 1)
                journal.stage(
                    "native_registration_attempt",
                    count=len(register_calls),
                    forced_failure=args.force_registration_failure,
                )
                if args.force_registration_failure:
                    return None
                return original_register(*call_args)

            stack.enter_context(
                mock.patch.object(runtime.core, "wp_cuda_graphics_register_gl_buffer", side_effect=register)
            )
            if args.mode == "buffers":
                validate_buffers(args, journal, wp, OpenGLRenderer, np, check)
            else:
                validate_render(args, journal, wp, OpenGLRenderer, np, Image, check)
            environment["native_registration_attempts"] = len(register_calls)
            if args.device == "cpu":
                check.assertEqual(len(register_calls), 0, "CPU must never attempt CUDA/OpenGL registration")
            else:
                check.assertGreater(len(register_calls), 0)
        journal.document["status"] = "passed"
        journal.stage("complete", rows=len(journal.document["results"]))
    except BaseException as error:
        journal.document["status"] = "failed"
        journal.document["error"] = {"type": type(error).__name__, "message": str(error)}
        journal.save()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
