# ===----------------------------------------------------------------------=== #
# hdf5-bindings: HDF5 runtime bindings for Mojo
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE for more information.
# ===----------------------------------------------------------------------=== #
"""
HDF5 v0.4.0
============
High-level HDF5 file I/O for Mojo with h5py-compatible API.

Example usage:
    ```mojo
    from hdf5 import File
    from numojo.prelude import NDArray

    with File("data.h5", "r") as f:
        var dset = f.dataset("mydataset")
        print(dset.shape(), dset.dtype())
        var row = dset[0, :].read[f64]()      # h5py-style slicing
        print(f.attrs["version"].as_scalar[DType.int32]())
    ```

For direct access to the HDF5 C API import ``hdf5.ffi`` instead.
"""
comptime __version__ = "0.4.0"

from numojo.prelude import NDArray, Item, Shape, f64, f32, i32, i64

from .core import File, AttrValue
