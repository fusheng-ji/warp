# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate primitive resolution switches with a real headless OpenGL renderer.

Run from the tested checkout with its environment:
    uv run --no-sync ../warp-1110-results/validate_render.py --checkout . --output ../warp-1110-results/gpu
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pyglet
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    sys.path.insert(0, str(checkout))

    pyglet.options["headless"] = True
    pyglet.options["headless_device"] = args.gpu_index
    pyglet.options["debug_gl"] = True

    import warp as wp  # noqa: PLC0415
    from warp.render import OpenGLRenderer  # noqa: PLC0415

    check = unittest.TestCase()
    check.assertEqual(Path(wp.__file__).resolve().parent.parent, checkout)
    wp.init()
    shapes = {
        "sphere": {"radius": 0.5},
        "capsule": {"radius": 0.5, "half_height": 0.6},
        "cylinder": {"radius": 0.5, "half_height": 0.6},
        "cone": {"radius": 0.5, "half_height": 0.6},
        "arrow": {"base_radius": 0.2, "base_height": 1.0},
    }
    rows = []
    environment = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.machine(),
        "warp": wp.__version__,
        "numpy": np.__version__,
        "pyglet": pyglet.version,
        "physical_gpu_index": args.gpu_index,
        "gpu": str(wp.get_device("cuda:0").name),
        "gpu_driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", f"--id={args.gpu_index}"],
            text=True,
        ).strip(),
        "sources_sha256": {
            path: hashlib.sha256((checkout / path).read_bytes()).hexdigest()
            for path in ("warp/_src/render/render_opengl.py", "warp/tests/test_render_opengl.py")
        },
        "libraries_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((checkout / "warp/bin").glob("*.so"))
        },
    }

    def save_results():
        if rows:
            with (output / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    save_results()
    with wp.ScopedDevice("cuda:0"):
        for kind, parameters in shapes.items():
            for cached in (False, True):
                renderer = OpenGLRenderer(
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
                    device="cuda:0",
                )
                try:
                    gl = OpenGLRenderer.gl
                    environment["gl_vendor"] = gl.gl_info.get_vendor()
                    environment["gl_renderer"] = gl.gl_info.get_renderer()
                    environment["gl_version"] = list(gl.gl_info.get_version())
                    render = getattr(renderer, f"render_{kind}")
                    rotation = (0.0, 0.0, 0.0, 1.0)
                    renderer.register_body("parent")
                    body_transforms = wp.array([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], dtype=wp.transform)
                    if cached:
                        render("template", (0.0, 0.0, 0.0), rotation, **parameters, segments=8, is_template=True)
                    high = render("first", (-0.8, 0.0, 0.0), rotation, **parameters, parent_body="parent", segments=32)
                    render("second", (0.8, 0.0, 0.0), rotation, **parameters, segments=32)
                    instance_id = renderer._instances["first"][0]
                    renderer.update_shape_instance("first", scale=(1.0, 0.9, 1.0))
                    renderer._instance_custom_ids[instance_id] = 17
                    before = renderer._instances["first"]
                    if cached:
                        renderer.setup_tiled_rendering([[instance_id], [renderer._instances["second"][0]]])
                    images = []
                    for frame, segments in enumerate((32, 8, 32)):
                        renderer.begin_frame(frame / 60.0)
                        shape = render("first", (-0.8, 0.0, 0.0), rotation, **parameters, segments=segments)
                        if renderer._add_shape_instances:
                            renderer.allocate_shape_instances()
                        if renderer._update_shape_instances:
                            renderer.update_shape_instances()
                        renderer.update_body_transforms(body_transforms)
                        renderer.end_frame()
                        image = wp.empty((256, 384, 3), dtype=wp.uint8, device="cuda:0")
                        check.assertTrue(renderer.get_pixels(image, split_up_tiles=False, use_uint8=True))
                        pixels = image.numpy()
                        check.assertGreater(
                            np.unique(pixels.reshape(-1, 3), axis=0).shape[0], 8, "Empty or constant render"
                        )
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
                        check.assertEqual(
                            sorted(i for group in renderer._shape_instances.values() for i in group), [0, 1]
                        )
                        check.assertIn(instance_id, renderer._shape_instances[shape])
                        check.assertEqual(
                            renderer._instance_ids[renderer._inverse_instance_ids[instance_id]], instance_id
                        )
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
                        if frame == 2:
                            np.testing.assert_array_equal(pixels, images[0])
                        images.append(pixels.copy())
                        image_name = f"{kind}-cached-{int(cached)}-frame-{frame}-segments-{segments}.png"
                        Image.fromarray(pixels[::-1]).save(output / image_name)
                        rows.append(
                            {
                                "primitive": kind,
                                "target_pre_cached": cached,
                                "tiled": cached,
                                "frame": frame,
                                "segments": segments,
                                "vertices": len(vertices),
                                "triangles": len(indices) // 3,
                                "instance_id": instance_id,
                                "instance_count": len(renderer._instances),
                                "cached_meshes": len(renderer._shapes),
                                "image": image_name,
                                "passed": True,
                            }
                        )
                        save_results()
                        print(kind, "cached", cached, "frame", frame, "segments", segments, "PASS", flush=True)
                finally:
                    renderer.close()
    save_results()
    print(f"Passed {len(rows)} rendered frames across {len(shapes)} primitive types.")


if __name__ == "__main__":
    main()
