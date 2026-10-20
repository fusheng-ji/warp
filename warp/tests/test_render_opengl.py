# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test OpenGL renderer logic that does not require a display or OpenGL context."""

import unittest
from collections import defaultdict
from typing import ClassVar
from unittest import mock

import numpy as np

import warp as wp
from warp.render import OpenGLRenderer


class TestOpenGLRenderer(unittest.TestCase):
    primitive_parameters: ClassVar[dict[str, dict[str, float]]] = {
        "sphere": {"radius": 0.5},
        "capsule": {"radius": 0.5, "half_height": 0.8},
        "cylinder": {"radius": 0.5, "half_height": 0.8},
        "cone": {"radius": 0.5, "half_height": 0.8},
        "arrow": {"base_radius": 0.2, "base_height": 0.8},
    }

    @staticmethod
    def _make_renderer():
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)
        renderer._device = wp.get_device("cpu")
        renderer._instances = {}
        renderer._shape_geo_hash = {}
        renderer.register_shape = mock.Mock(return_value=0)
        return renderer

    @classmethod
    def _make_primitive_renderer(cls):
        renderer = cls._make_renderer()
        renderer._shapes = []
        renderer._shape_instances = defaultdict(list)
        renderer._shape_gl_buffers = {}
        renderer._instance_shape = {}
        renderer._instance_custom_ids = {}
        renderer._body_name = {"body": 0}
        renderer._instance_count = 0
        renderer._add_shape_instances = False
        renderer._update_shape_instances = False
        renderer._switch_context = mock.Mock()
        renderer.update_instance_colors = mock.Mock()

        def register_shape(geo_hash, vertices, indices, color1=None, color2=None):
            shape = len(renderer._shapes)
            renderer._shapes.append((vertices, indices, color1, color2, geo_hash))
            renderer._shape_geo_hash[geo_hash] = shape
            renderer._shape_gl_buffers[shape] = (shape, None, None, len(indices), None)
            return shape

        renderer.register_shape = mock.Mock(side_effect=register_shape)
        return renderer

    def _render_primitive(self, renderer, kind, name="primitive", pos=(0.0, 0.0, 0.0), **kwargs):
        return getattr(renderer, f"render_{kind}")(
            name, pos, (0.0, 0.0, 0.0, 1.0), **self.primitive_parameters[kind], **kwargs
        )

    def test_render_primitive_segments_geometry(self):
        """Verify default geometry and valid meshes at different resolutions."""
        for kind, parameters in self.primitive_parameters.items():
            expected_vertices, expected_indices = getattr(OpenGLRenderer, f"_create_{kind}_mesh")(**parameters)
            for segments in (None, 32, 8, 3, np.int64(8)):
                with self.subTest(kind=kind, segments=segments):
                    renderer = self._make_primitive_renderer()
                    options = {} if segments is None else {"segments": segments}
                    shape = self._render_primitive(renderer, kind, is_template=True, **options)
                    vertices, indices = renderer._shapes[shape][:2]
                    n = 32 if segments is None else int(segments)
                    if kind in ("sphere", "capsule"):
                        vertex_count = (n + 1) ** 2
                        triangle_count = 2 * (n if kind == "sphere" else n + 1) ** 2
                    else:
                        factor = 2 if kind == "arrow" else 1
                        vertex_count = factor * (4 * n + 2)
                        triangle_count = factor * 4 * n
                    self.assertEqual(vertices.shape, (vertex_count, 8))
                    self.assertEqual(indices.size, 3 * triangle_count)
                    self.assertTrue(np.isfinite(vertices).all())
                    self.assertLess(int(indices.max()), vertex_count)
                    self.assertEqual(indices.dtype, np.uint32)
                    if n == 32:
                        np.testing.assert_array_equal(vertices, expected_vertices)
                        np.testing.assert_array_equal(indices, expected_indices)

    def test_render_primitive_segments_validation(self):
        """Verify invalid resolutions do not change geometry or instance state."""
        invalid_values = (
            (TypeError, True),
            (TypeError, np.bool_(True)),
            (TypeError, 8.0),
            (TypeError, "8"),
            (TypeError, None),
            (ValueError, -1),
            (ValueError, 0),
            (ValueError, 1),
            (ValueError, np.int64(2)),
        )
        for kind in self.primitive_parameters:
            for error, segments in invalid_values:
                with self.subTest(kind=kind, segments=segments):
                    renderer = self._make_primitive_renderer()
                    with self.assertRaisesRegex(error, "segments"):
                        self._render_primitive(renderer, kind, segments=segments)
                    renderer.register_shape.assert_not_called()
                    self.assertEqual(renderer._instances, {})
                    self.assertEqual(renderer._shape_geo_hash, {})
                    self.assertFalse(renderer._add_shape_instances)

    def test_render_primitive_segments_cache_and_templates(self):
        """Verify geometry sharing, distinct resolutions, and template isolation."""
        with mock.patch.object(OpenGLRenderer, "gl", mock.Mock()):
            for kind in self.primitive_parameters:
                with self.subTest(kind=kind):
                    renderer = self._make_primitive_renderer()
                    high = self._render_primitive(renderer, kind, is_template=True)
                    self.assertEqual(self._render_primitive(renderer, kind, segments=32, is_template=True), high)
                    low = self._render_primitive(renderer, kind, segments=8, is_template=True)
                    self.assertNotEqual(low, high)
                    self.assertEqual(renderer._instances, {})
                    for name in ("first", "second"):
                        self.assertEqual(self._render_primitive(renderer, kind, name, segments=32), high)
                    self.assertEqual(self._render_primitive(renderer, kind, "third", segments=8), low)
                    self._render_primitive(renderer, kind, "first", segments=8, is_template=True)
                    self.assertEqual(renderer._instances["first"][2], high)
                    self.assertEqual(len(renderer._instances), 3)
                    self.assertEqual(renderer.register_shape.call_count, 2)

    def test_render_primitive_segments_switch(self):
        """Verify changing resolution preserves identity and shared instances."""
        with mock.patch.object(OpenGLRenderer, "gl", mock.Mock()):
            for kind in self.primitive_parameters:
                for cached in (False, True):
                    with self.subTest(kind=kind, cached=cached):
                        renderer = self._make_primitive_renderer()
                        high = self._render_primitive(renderer, kind, "first", segments=32, parent_body="body")
                        self._render_primitive(renderer, kind, "second", segments=32)
                        if cached:
                            self._render_primitive(renderer, kind, segments=8, is_template=True)
                        renderer.update_shape_instance("first", scale=(2.0, 3.0, 4.0), visible=False)
                        before = renderer._instances["first"]
                        instance_id = before[0]
                        renderer._instance_custom_ids[instance_id] = 17

                        for segments in (8, 32, 32):
                            renderer._add_shape_instances = False
                            old_shape = renderer._instances["first"][2]
                            shape = self._render_primitive(
                                renderer, kind, "first", pos=(1.0, 2.0, 3.0), segments=segments
                            )
                            after = renderer._instances["first"]
                            self.assertEqual(after[:2], before[:2])
                            self.assertEqual(after[2], shape)
                            np.testing.assert_array_equal(after[3][:3], (1.0, 2.0, 3.0))
                            np.testing.assert_array_equal(after[3][3:], before[3][3:])
                            self.assertEqual(after[4:], before[4:])
                            self.assertEqual(renderer._instance_custom_ids[instance_id], 17)
                            self.assertEqual(renderer._instance_shape[instance_id], shape)
                            self.assertEqual(renderer._instances["second"][2], high)
                            self.assertEqual(renderer._instance_count, 2)
                            self.assertEqual(len(renderer._instances), 2)
                            memberships = [i for group in renderer._shape_instances.values() for i in group]
                            self.assertCountEqual(memberships, (0, 1))
                            self.assertIn(instance_id, renderer._shape_instances[shape])
                            self.assertIn(1, renderer._shape_instances[high])
                            self.assertEqual(renderer._add_shape_instances, shape != old_shape)
                            if segments == 32:
                                self.assertEqual(shape, high)
                        self.assertEqual(renderer.register_shape.call_count, 2)

    def test_render_mesh_rejects_out_of_range_indices(self):
        """Verify that mesh rendering rejects indices outside the point array."""
        points = np.array(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            dtype=np.float32,
        )

        invalid_indices = (
            ((0, 1, -1), -1),
            ((0, 1, len(points)), len(points)),
            ((0, 1, np.iinfo(np.int32).max), np.iinfo(np.int32).max),
            (np.array((0, 1, 1 << 32), dtype=np.uint64), 1 << 32),
        )

        for smooth_shading in (False, True):
            for indices, invalid_index in invalid_indices:
                renderer = self._make_renderer()

                with self.subTest(smooth_shading=smooth_shading, invalid_index=invalid_index):
                    with mock.patch.object(wp, "launch") as launch:
                        with self.assertRaises(ValueError) as context:
                            renderer.render_mesh(
                                "mesh",
                                points,
                                indices,
                                update_topology=True,
                                is_template=True,
                                smooth_shading=smooth_shading,
                            )

                    launch.assert_not_called()
                    message = str(context.exception)
                    self.assertIn(str(invalid_index), message)
                    self.assertIn(f"[0, {len(points)})", message)

    def test_render_mesh_does_not_validate_unchanged_topology(self):
        """Verify that unchanged topology skips index validation and shading kernels."""
        for indices in ((-1, -1, -1), (0, 0, 1 << 32)):
            renderer = self._make_renderer()
            renderer._instances["mesh"] = (0, None, 7)
            renderer.update_shape_instance = mock.Mock()
            renderer.update_shape_vertices = mock.Mock()

            with self.subTest(indices=indices):
                with mock.patch.object(wp, "launch") as launch:
                    try:
                        shape = renderer.render_mesh(
                            "mesh",
                            ((0.0, 0.0, 0.0),),
                            indices,
                            update_topology=False,
                        )
                    except Exception as error:
                        self.fail(f"Expected unchanged topology to be reused, but got {type(error).__name__}: {error}")

                self.assertEqual(shape, 7)
                launch.assert_not_called()

    def test_render_mesh_reports_wide_python_indices_as_value_error(self):
        """Verify that wide Python integers raise a mesh index range ValueError."""
        renderer = self._make_renderer()

        with mock.patch.object(wp, "launch") as launch:
            try:
                renderer.render_mesh(
                    "mesh",
                    ((0.0, 0.0, 0.0),),
                    (0, 0, 1 << 32),
                    update_topology=True,
                    is_template=True,
                )
            except ValueError as error:
                message = str(error)
            except Exception as error:
                self.fail(f"Expected ValueError, but got {type(error).__name__}: {error}")
            else:
                self.fail("Expected ValueError, but no exception was raised")

        launch.assert_not_called()
        self.assertIn(str(1 << 32), message)
        self.assertIn("[0, 1)", message)

    def test_render_mesh_rejects_indices_outside_int32_range(self):
        """Verify that mesh indices must be representable by Warp's int32 index type."""
        renderer = self._make_renderer()
        points = np.array(((0.0, 0.0, 0.0),), dtype=np.float32)
        max_index = 1 << 31
        point_count = max_index + 1

        def simulated_len(value):
            if isinstance(value, np.ndarray) and value.dtype == np.float32:
                return point_count
            return len(value)

        with mock.patch.dict(OpenGLRenderer.render_mesh.__globals__, {"len": simulated_len}):
            with mock.patch.object(wp, "launch", side_effect=AssertionError("Kernel launched before validation")):
                with self.assertRaises(ValueError) as context:
                    renderer.render_mesh(
                        "mesh",
                        points,
                        np.array((0, 1, max_index), dtype=np.uint64),
                        update_topology=True,
                        is_template=True,
                        smooth_shading=False,
                    )

        message = str(context.exception)
        self.assertIn(str(max_index), message)
        self.assertIn(f"[0, {max_index})", message)

    def test_render_mesh_accepts_empty_indices(self):
        """Verify that mesh rendering accepts meshes without faces."""
        points = np.array(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            dtype=np.float32,
        )

        for smooth_shading in (False, True):
            renderer = self._make_renderer()
            with self.subTest(smooth_shading=smooth_shading):
                with mock.patch.object(wp, "launch"):
                    shape = renderer.render_mesh(
                        "mesh",
                        points,
                        (),
                        update_topology=True,
                        is_template=True,
                        smooth_shading=smooth_shading,
                    )

                self.assertEqual(shape, 0)

    def test_setup_tiled_rendering_requires_list(self):
        """Verify that tiled rendering rejects a non-list instances container."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)

        with self.assertRaisesRegex(TypeError, r"instances must be a list of lists, got tuple"):
            renderer.setup_tiled_rendering(([0],))

    def test_setup_tiled_rendering_requires_nonempty_instances(self):
        """Verify that tiled rendering rejects an empty instances list."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)

        with self.assertRaisesRegex(ValueError, r"instances must be non-empty, got \[\]"):
            renderer.setup_tiled_rendering([])

    def test_setup_tiled_rendering_requires_list_elements(self):
        """Verify that tiled rendering rejects non-list instance entries."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)

        with self.assertRaisesRegex(TypeError, r"instances elements must be lists, got tuple at index 1"):
            renderer.setup_tiled_rendering([[0], (1,)])

    def test_get_pixels_requires_tiled_rendering_setup(self):
        """Verify that tiled pixel reads require configured viewports."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)
        renderer._switch_context = mock.Mock()
        renderer._tile_instances = None
        renderer._tile_viewports = None
        renderer._tile_width = 0
        renderer._tile_height = 0

        with self.assertRaisesRegex(
            RuntimeError,
            r"get_pixels\(\) requires setup_tiled_rendering\(\) when split_up_tiles=True",
        ):
            renderer.get_pixels(object())

    def test_update_tile_reports_out_of_bounds_index(self):
        """Verify that updating an invalid tile reports an indexing error."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)
        renderer._tile_instances = [[0]]

        for tile_id in (-1, 1):
            with self.subTest(tile_id=tile_id):
                with self.assertRaisesRegex(IndexError, rf"tile_id {tile_id} is out of bounds for 1 tile"):
                    renderer.update_tile(tile_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
