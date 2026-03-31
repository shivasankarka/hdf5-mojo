# ===----------------------------------------------------------------------=== #
# hdf5-bindings: HDF5 runtime bindings for Mojo
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE for more information.
# ===----------------------------------------------------------------------=== #
"""
hdf5 package
============

High-level HDF5 file I/O for Mojo.

Re-exports the two primary public types from `hdf5.core`:

- `H5File`        — open, read, write, and close HDF5 files.
- `NDArray[dtype]` — heap-allocated typed array returned by read methods.

Typical usage:
    ```mojo
    from hdf5 import H5File

    var f   = H5File("data.h5")
    var arr = f.read_1d[DType.float64]("/group/dataset")
    print(arr[0])
    arr.free()
    f.close()
    ```

For direct access to the HDF5 C API import `hdf5.ffi` instead.
"""

from .core import H5File, NDArray
