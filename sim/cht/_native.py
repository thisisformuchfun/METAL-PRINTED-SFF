"""Build and load the C kernels (gcc + OpenMP) through ctypes.

The shared library is compiled on first use into ``sim/cht/_build/`` and
rebuilt automatically whenever a C source is newer than the library.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "csrc"
_BUILD = _HERE / "_build"
_LIB = None

CFLAGS = ["-O3", "-march=native", "-ffast-math", "-fno-finite-math-only", "-fopenmp",
          "-fPIC", "-shared", "-std=gnu11", "-Wall", "-Wno-unknown-pragmas"]


def _compile() -> Path:
    srcs = sorted(_SRC.glob("*.c"))
    lib = _BUILD / "libcht.so"
    newest = max(s.stat().st_mtime for s in srcs)
    if lib.exists() and lib.stat().st_mtime >= newest:
        return lib
    _BUILD.mkdir(exist_ok=True)
    cc = os.environ.get("CC", "gcc")
    tmp = _BUILD / f"libcht.{os.getpid()}.so"
    cmd = [cc, *CFLAGS, *map(str, srcs), "-o", str(tmp), "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("C kernel build failed:\n" + " ".join(cmd) + "\n" + res.stderr)
    os.replace(tmp, lib)
    return lib


def _ptr(dtype):
    return np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")


def lib() -> ctypes.CDLL:
    """Return the loaded kernel library (compiling it if necessary)."""
    global _LIB
    if _LIB is not None:
        return _LIB
    L = ctypes.CDLL(str(_compile()))
    f32, f64, i32, u8 = _ptr(np.float32), _ptr(np.float64), _ptr(np.int32), _ptr(np.uint8)
    i64, ci, cf, cd = ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_double
    vp = ctypes.c_void_p

    L.lbm_max_threads.restype = ci
    L.lbm_step.argtypes = [f32, f32, i32, f32, f32, f32, i64, f32, ci]
    L.lbm_wale.argtypes = [f32, i32, u8, vp, f32, i64, cf, ci]
    L.lbm_accumulate.argtypes = [f32, f32, f32, f64, i64, ci]
    L.lbm_monitor.argtypes = [f32, f32, i64, f64, ci]
    for name in ("lbm_step", "lbm_wale", "lbm_accumulate", "lbm_monitor"):
        getattr(L, name).restype = None
    if hasattr(L, "fv_register"):  # finite-volume kernels declare their own prototypes
        pass
    _LIB = L
    return L
