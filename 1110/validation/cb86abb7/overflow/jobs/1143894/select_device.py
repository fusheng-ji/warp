"""Match Slurm's CUDA device to EGL without opening an unallocated GPU."""

import argparse
import ctypes
import ctypes.util
import gc
import json
import os
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


def bus_key(value):
    domain, bus, device = value.lower().split(":")
    return int(domain, 16), int(bus, 16), int(device.split(".")[0], 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check = unittest.TestCase()
    driver = ctypes.CDLL("libcuda.so.1")
    check.assertEqual(driver.cuInit(0), 0)
    count = ctypes.c_int()
    check.assertEqual(driver.cuDeviceGetCount(ctypes.byref(count)), 0)
    check.assertEqual(count.value, 1, "Slurm must expose exactly one full CUDA device")
    cuda_device = ctypes.c_int()
    check.assertEqual(driver.cuDeviceGet(ctypes.byref(cuda_device), 0), 0)
    pci = ctypes.create_string_buffer(64)
    check.assertEqual(driver.cuDeviceGetPCIBusId(pci, len(pci), cuda_device), 0)
    target_bus = bus_key(pci.value.decode())
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id,uuid,name,driver_version", "--format=csv,noheader"], text=True
    )
    matches = [row.split(", ") for row in gpu_rows.splitlines() if bus_key(row.split(", ")[1]) == target_bus]
    check.assertEqual(len(matches), 1)
    physical_index, _pci, uuid, model, version = matches[0]
    gpu_xml = ET.fromstring(subprocess.check_output(["nvidia-smi", "-i", uuid, "-q", "-x"], text=True))
    processes = gpu_xml.findall(".//processes/process_info")
    check.assertEqual(len(processes), 0, "Allocated GPU already has a compute/graphics process")
    mig = gpu_xml.findtext(".//mig_mode/current_mig")
    check.assertEqual(mig, "Disabled", "A whole GPU, not a MIG slice, is required")

    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]

    def procedure(name, result_type, *argument_types):
        address = egl.eglGetProcAddress(name.encode())
        check.assertTrue(address, name)
        return ctypes.CFUNCTYPE(result_type, *argument_types)(address)

    query_devices = procedure(
        "eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)
    )
    query_string = procedure("eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
    query_attribute = procedure(
        "eglQueryDeviceAttribEXT", ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)
    )
    devices = (ctypes.c_void_p * 64)()
    number = ctypes.c_int()
    check.assertTrue(query_devices(len(devices), devices, ctypes.byref(number)))
    pci_candidates = []
    ordinal_candidates = []
    evidence = []
    for index in range(number.value):
        device = devices[index]
        pci_match = False
        for attribute in (0x3233, 0x3377):  # EGL DRM device and render-node filenames
            path = query_string(device, attribute)
            if path:
                device_link = Path("/sys/class/drm") / Path(path.decode()).name / "device"
                try:
                    pci_match |= bus_key(device_link.resolve().name) == target_bus
                except (ValueError, FileNotFoundError):
                    pass
        native_ordinal = ctypes.c_ssize_t(-1)
        cuda_attribute = bool(query_attribute(device, 0x323A, ctypes.byref(native_ordinal)))
        ordinal_match = cuda_attribute and native_ordinal.value == int(physical_index)
        if pci_match:
            pci_candidates.append(index)
        if ordinal_match:
            ordinal_candidates.append(index)
        if pci_match or ordinal_match:
            evidence.append(
                {"egl_index": index, "drm_pci_match": pci_match, "cuda_physical_index_match": ordinal_match}
            )
    candidates = pci_candidates if pci_candidates else ordinal_candidates
    selection = {
        "allocated_physical_gpu_index": int(physical_index),
        "egl_devices_enumerated": number.value,
        "pci_candidates": pci_candidates,
        "ordinal_candidates": ordinal_candidates,
        "evidence": evidence,
    }
    args.output.with_name("selection-diagnostic.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection), flush=True)
    # A cgroup-isolated EGL implementation may enumerate only one accessible
    # device. Strict native interop below must still confirm the chosen match.
    if not candidates and number.value == 1:
        candidates = [0]
        evidence.append({"egl_index": 0, "single_accessible_egl_device": True})
    check.assertEqual(len(candidates), 1, "Cannot unambiguously map the allocated CUDA GPU to EGL")
    chosen = candidates[0]

    sys.path.insert(0, str(args.checkout.resolve()))
    import numpy as np  # noqa: PLC0415
    import pyglet  # noqa: PLC0415

    pyglet.options["headless"] = True
    pyglet.options["headless_device"] = chosen
    from pyglet import gl  # noqa: PLC0415

    import warp as wp  # noqa: PLC0415

    wp.config.quiet = True
    wp.init()
    check.assertEqual(wp.get_device("cuda:0").uuid, uuid)
    with wp.ScopedDevice("cuda:0"):
        window = pyglet.window.Window(width=16, height=16, visible=False)
        buffer_id = gl.GLuint()
        gl.glGenBuffers(1, buffer_id)
        registered = None
        try:
            expected = np.arange(8, dtype=np.float32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_id)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, expected.nbytes, None, gl.GL_DYNAMIC_DRAW)
            registered = wp.RegisteredGLBuffer(
                int(buffer_id.value), device="cuda:0", flags=wp.RegisteredGLBuffer.WRITE_DISCARD, fallback_to_copy=False
            )
            check.assertIsNotNone(registered.resource)
            mapped = registered.map(dtype=wp.float32, shape=(len(expected),))
            wp.copy(mapped, wp.array(expected, dtype=wp.float32, device="cuda:0"))
            registered.unmap()
            actual = np.empty_like(expected)
            gl.glGetBufferSubData(gl.GL_ARRAY_BUFFER, 0, actual.nbytes, actual.ctypes.data)
            np.testing.assert_array_equal(actual, expected)
            check.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)
            graphics = {"vendor": gl.gl_info.get_vendor(), "renderer": gl.gl_info.get_renderer()}
        finally:
            del registered
            gc.collect()
            gl.glDeleteBuffers(1, buffer_id)
            window.close()
    document = {
        "status": "passed",
        "visible_cuda_devices": count.value,
        "allocated_physical_gpu_index": int(physical_index),
        "egl_device_index": chosen,
        "egl_devices_enumerated": number.value,
        "selection_evidence": evidence,
        "gpu": model,
        "driver": version,
        "mig_mode": mig,
        "preexisting_compute_or_graphics_processes": len(processes),
        "native_cuda_gl_interop_verified": True,
        "graphics": graphics,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_partition": os.environ["SLURM_JOB_PARTITION"],
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document), flush=True)


if __name__ == "__main__":
    main()
