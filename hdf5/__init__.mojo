# ===----------------------------------------------------------------------=== #
# hdf5-bindings: HDF5 runtime bindings for Mojo
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE for more information.
# ===----------------------------------------------------------------------=== #
"""
HDF5 v0.2.0
============

High-level HDF5 file I/O for Mojo with h5py-compatible API.

Example usage:
    ```mojo
    from hdf5 import File
    from numojo.prelude import NDArray

    var f = File("data.h5", "r")
    var obj = f.get("mydataset")
    if obj.is_dataset():
        var dset = obj.dataset()
        print(dset.shape(), dset.dtype())
    for name in f.keys():
        print(name)
    f.close()
    ```

For direct access to the HDF5 C API import ``hdf5.ffi`` instead.
"""
comptime __version__ = "0.2.0"

from numojo.prelude import NDArray, Item, Shape, f64, f32, i32, i64

from .core import File
