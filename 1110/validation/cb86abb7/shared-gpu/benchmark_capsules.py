# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure completed filled-render frames and cached all-instance LOD switches.

Run from the frozen feature checkout using its environment. CUDA and EGL must
select the same physical device. No timing threshold is a correctness assertion.
"""

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import unittest
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def telemetry():
    fields = "index,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.gr,power.draw,temperature.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"fields": fields.split(","), "rows": result.stdout.strip().splitlines(), "exit_code": result.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--egl-device", type=int, default=1)
    parser.add_argument("--counts", type=int, nargs="+", default=[64, 512, 4096])
    parser.add_argument("--segments", type=int, nargs="+", default=[3, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--dynamic-frames", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1110)
    args = parser.parse_args()
    check = unittest.TestCase()
    checkout = args.checkout.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(checkout))
    import numpy as np  # noqa: PLC0415
    import pyglet  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    pyglet.options["headless"] = True
    pyglet.options["headless_device"] = args.egl_device
    # Correctness runs use debug_gl=True; per-call debug error checks are omitted
    # from timings, with explicit error checks at each case boundary instead.
    pyglet.options["debug_gl"] = False
    import warp as wp  # noqa: PLC0415
    from warp.render import OpenGLRenderer  # noqa: PLC0415

    check.assertTrue(Path(wp.__file__).resolve().parent.parent == checkout)
    wp.init()
    check.assertTrue(32 in args.segments and 8 in args.segments and 3 in args.segments)

    def status():
        return subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip()

    document = {
        "status": "running",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
        "source_dirty_before": status(),
        "arguments": {key: value for key, value in vars(args).items() if key not in ("checkout", "output")},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system() + " " + platform.machine(),
            "warp": wp.__version__,
            "numpy": np.__version__,
            "pyglet": pyglet.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "cuda_device": wp.get_device("cuda:0").name,
            "private_kernel_cache": True,
        },
        "source_sha256": {
            name: digest(checkout / name) for name in ("warp/_src/render/render_opengl.py", "warp/_src/context.py")
        },
        "library_sha256": {path.name: digest(path) for path in sorted((checkout / "warp/bin").glob("*.so"))},
        "script_sha256": digest(Path(__file__)),
        "telemetry_before": telemetry(),
        "cases": [],
        "controls": [],
        "method": {
            "drawing": "filled, 1280x720, no grid/sky/axis/overlay, vsync disabled",
            "steady": "begin_frame + end_frame; pre-existing static instances; no per-object API calls",
            "switch_all": "begin_frame + render_capsule for every instance + end_frame; alternating cached 32 and low resolution",
            "wall_ms": "perf_counter_ns through glFinish after each frame; includes host work and completed GL work",
            "gl_ms": "GPU timeline across frame commands; may include idle time waiting for host submission; not draw-only or CUDA kernel time",
            "excluded": "renderer/mesh creation, initial instance upload, warmup, pixel reads, telemetry and query-result retrieval",
            "order": "seeded random case order separately in each repeat; both resolutions warmed before switch timing",
            "limits": "shared GPU, no clock locking; latency reciprocal is not application or display FPS; lower LOD changes fidelity",
        },
    }
    check.assertTrue(not document["source_dirty_before"])

    def save():
        (output / "environment.json").write_text(json.dumps(document, indent=2) + "\n")

    save()
    scenes = {}
    rows = []
    gl = None
    with wp.ScopedDevice("cuda:0"):
        try:
            for count in args.counts:
                renderer = OpenGLRenderer(
                    screen_width=1280,
                    screen_height=720,
                    camera_pos=(0.0, 0.0, 5.0),
                    camera_front=(0.0, 0.0, -1.0),
                    background_color=(0.08, 0.09, 0.12),
                    headless=True,
                    draw_grid=False,
                    draw_sky=False,
                    draw_axis=False,
                    show_info=False,
                    render_wireframe=False,
                    vsync=False,
                    device="cuda:0",
                )
                gl = OpenGLRenderer.gl
                document["environment"].update(
                    gl_vendor=gl.gl_info.get_vendor(),
                    gl_renderer=gl.gl_info.get_renderer(),
                    gl_version=list(gl.gl_info.get_version()),
                )
                cols = math.ceil(math.sqrt(count * 16 / 9))
                height = math.ceil(count / cols)
                step = 5.0 / cols
                parameters = {"radius": 0.22 * step, "half_height": 0.28 * step}
                positions = [
                    ((i % cols - (cols - 1) / 2) * step, (i // cols - (height - 1) / 2) * step, 0.0)
                    for i in range(count)
                ]
                rotation = (0.0, 0.0, 0.0, 1.0)
                names = [f"capsule-{i}" for i in range(count)]
                for segments in args.segments:
                    renderer.render_capsule(
                        f"template-{segments}", (0, 0, 0), rotation, **parameters, segments=segments, is_template=True
                    )
                for name, position in zip(names, positions, strict=True):
                    renderer.render_capsule(name, position, rotation, **parameters)
                renderer.begin_frame(1.0)
                renderer.end_frame()
                gl.glFinish()
                image = wp.empty((720, 1280, 3), dtype=wp.uint8, device="cuda:0")
                check.assertTrue(renderer.get_pixels(image, split_up_tiles=False, use_uint8=True))
                default_pixels = image.numpy().copy()
                for name, position in zip(names, positions, strict=True):
                    renderer.render_capsule(name, position, rotation, **parameters, segments=32)
                renderer.begin_frame(1.0)
                renderer.end_frame()
                gl.glFinish()
                check.assertTrue(renderer.get_pixels(image, split_up_tiles=False, use_uint8=True))
                np.testing.assert_array_equal(default_pixels, image.numpy())
                check.assertTrue(len(renderer._instances) == count and len(renderer._shapes) == len(args.segments))
                check.assertTrue(gl.glGetError() == gl.GL_NO_ERROR)
                scenes[count] = (renderer, names, positions, parameters, image)
                document["controls"].append(
                    {
                        "count": count,
                        "default_equals_explicit_32_pixels": True,
                        "instance_count": count,
                        "cached_meshes": len(renderer._shapes),
                    }
                )
                save()
                print("prepared", count, flush=True)

            queries = {}
            for count, (renderer, *_) in scenes.items():
                renderer._switch_context()
                query = gl.GLuint()
                gl.glGenQueries(1, query)
                queries[count] = query
            cases = [("steady", count, segments) for count in args.counts for segments in args.segments]
            cases += [("switch_all", count, segments) for count in args.counts for segments in (3, 8)]
            generator = random.Random(args.seed)
            for repeat in range(args.repeats):
                order = cases.copy()
                generator.shuffle(order)
                for order_index, (mode, count, low) in enumerate(order):
                    renderer, names, positions, parameters, image = scenes[count]
                    renderer._switch_context()
                    query = queries[count]

                    def frame(
                        segments, change, renderer=renderer, names=names, positions=positions, parameters=parameters
                    ):
                        renderer.begin_frame(1.0)
                        if change:
                            for name, position in zip(names, positions, strict=True):
                                renderer.render_capsule(name, position, rotation, **parameters, segments=segments)
                        renderer.end_frame()

                    # Every case starts from a known cached state; both endpoints
                    # of a dynamic switch receive warmup frames before measuring.
                    frame(low, True)
                    gl.glFinish()
                    for index in range(args.warmup):
                        target = (32 if index % 2 == 0 else low) if mode == "switch_all" else low
                        frame(target, mode == "switch_all")
                        gl.glFinish()
                    count_frames = args.frames if mode == "steady" else args.dynamic_frames
                    before = telemetry()
                    case_rows = []
                    for index in range(count_frames):
                        target = (32 if index % 2 == 0 else low) if mode == "switch_all" else low
                        gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)
                        start = time.perf_counter_ns()
                        frame(target, mode == "switch_all")
                        gl.glEndQuery(gl.GL_TIME_ELAPSED)
                        gl.glFinish()
                        wall_ms = (time.perf_counter_ns() - start) / 1e6
                        elapsed = gl.GLuint64()
                        gl.glGetQueryObjectui64v(query, gl.GL_QUERY_RESULT, ctypes.byref(elapsed))
                        _vertices, indices = renderer._shapes[renderer._instances[names[0]][2]][:2]
                        row = {
                            "repeat": repeat,
                            "order": order_index,
                            "mode": mode,
                            "instances": count,
                            "low_segments": low,
                            "target_segments": target,
                            "frame": index,
                            "wall_ms": wall_ms,
                            "gl_ms": elapsed.value / 1e6,
                            "triangles_per_instance": len(indices) // 3,
                            "submitted_triangles": count * len(indices) // 3,
                        }
                        check.assertTrue(math.isfinite(wall_ms) and wall_ms > 0 and elapsed.value > 0)
                        case_rows.append(row)
                    check.assertTrue(len(renderer._instances) == count)
                    check.assertTrue(
                        sorted(i for group in renderer._shape_instances.values() for i in group) == list(range(count))
                    )
                    check.assertTrue(len(renderer._shapes) == len(args.segments))
                    check.assertTrue(gl.glGetError() == gl.GL_NO_ERROR)
                    check.assertTrue(renderer.get_pixels(image, split_up_tiles=False, use_uint8=True))
                    pixels = image.numpy()
                    foreground = int(np.any(pixels != pixels[0, 0], axis=2).sum())
                    check.assertTrue(foreground > count)
                    if repeat == 0 and mode == "steady":
                        Image.fromarray(pixels[::-1]).save(output / f"capsules-{count}-segments-{low}.png")
                    rows.extend(case_rows)
                    with (output / "frames.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                        writer.writeheader()
                        writer.writerows(rows)
                    document["cases"].append(
                        {
                            "repeat": repeat,
                            "order": order_index,
                            "mode": mode,
                            "instances": count,
                            "low_segments": low,
                            "measured_frames": len(case_rows),
                            "foreground_pixels": foreground,
                            "wall_median_ms": float(np.median([r["wall_ms"] for r in case_rows])),
                            "gl_median_ms": float(np.median([r["gl_ms"] for r in case_rows])),
                            "telemetry_before": before,
                            "telemetry_after": telemetry(),
                            "passed": True,
                        }
                    )
                    save()
                    print("case", repeat, mode, count, low, document["cases"][-1]["wall_median_ms"], flush=True)
            for count, (renderer, *_) in scenes.items():
                renderer._switch_context()
                gl.glDeleteQueries(1, queries[count])
            document["source_dirty_after"] = status()
            check.assertTrue(not document["source_dirty_after"])
            document["telemetry_after"] = telemetry()
            document["status"] = "passed"
            document["measured_frames"] = len(rows)
            save()
        except BaseException as error:
            document["status"] = "failed"
            document["error"] = {"type": type(error).__name__, "message": str(error)}
            save()
            raise
        finally:
            for renderer, *_ in scenes.values():
                renderer._switch_context()
                renderer.close()


if __name__ == "__main__":
    main()
