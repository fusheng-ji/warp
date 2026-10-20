Rendering
=========

.. currentmodule:: warp.render

The :mod:`warp.render` module provides a set of renderers that can be used for visualizing scenes involving shapes of various types.

Standalone Renderers
--------------------

The :class:`OpenGLRenderer <warp.render.OpenGLRenderer>` provides an interactive renderer to play back animations in real time and is mostly intended for debugging,
whereas more sophisticated rendering can be achieved with the help of the :class:`UsdRenderer <warp.render.UsdRenderer>`, which allows exporting the scene to
a USD file that can then be rendered in an external 3D application or renderer of your choice.

Primitive Mesh Resolution
^^^^^^^^^^^^^^^^^^^^^^^^^

The OpenGL methods ``render_sphere()``, ``render_capsule()``, ``render_cylinder()``,
``render_cone()``, and ``render_arrow()`` accept a keyword-only ``segments`` argument
to control mesh resolution. The default is 32; lower values reduce the number of
triangles at the cost of a coarser silhouette. The value must be an integer of at
least 3. For spheres it controls both latitude and longitude subdivisions, and for
arrows it controls both the cylindrical base and the conical cap.

For example, with an existing ``OpenGLRenderer``:

.. code-block:: python

    renderer.render_capsule(
        "capsule",
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        radius=0.5,
        half_height=0.8,
        segments=8,
    )

The same name can be reused with a different resolution in subsequent frames.
The renderer switches the instance to the requested mesh while preserving its
instance ID, parent body, and scale:

.. code-block:: python

    for frame, segments in enumerate((32, 8, 32)):
        renderer.begin_frame(frame / 60.0)
        renderer.render_capsule(
            "capsule",
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            radius=0.5,
            half_height=0.8,
            segments=segments,
        )
        renderer.end_frame()

Instances with matching shape parameters and resolution share a cached mesh.
Changing resolution leaves other instances using the old mesh unchanged, and
switching back reuses the cached mesh. Each distinct mesh remains cached until
the renderer is cleared, so use a bounded set of resolutions for dynamic scenes.
Calls with ``is_template=True`` create or reuse geometry without changing an
existing instance.

CUDA Graphics Interface
-----------------------

Warp provides a CUDA graphics interface that allows you to access OpenGL buffers from CUDA kernels.
This is useful for manipulating OpenGL array buffers without having to copy them back and forth between the CPU and GPU.

See the `CUDA documentation on OpenGL Interoperability <https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GL.html>`_ for more information.

The :class:`wp.RegisteredGLBuffer <warp.RegisteredGLBuffer>` class wraps an OpenGL buffer and registers it with CUDA, allowing it to be
mapped as a Warp array for use in kernels without copying data between the CPU and GPU.
