# ===----------------------------------------------------------------------=== #
# hdf5-bindings: HDF5 runtime bindings for Mojo
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE for more information.
# ===----------------------------------------------------------------------=== #
"""
hdf5 package
============

High-level HDF5 file I/O for Mojo with h5py-compatible API.

Example usage:
    ```mojo
    from hdf5 import File

    var f = File("data.h5", "r")
    var obj = f["mydataset"]
    if obj.is_dataset():
        var dset = obj.dataset()
        print(dset.shape(), dset.dtype())
    for name in f.keys():
        print(name)
    f.close()
    ```

For direct access to the HDF5 C API import ``hdf5.ffi`` instead.
"""

from .h5py_api import File, Group, Dataset, AttributeManager, H5Object, NDArray
