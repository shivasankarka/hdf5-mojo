# ===----------------------------------------------------------------------=== #
# hdf5-bindings: HDF5 runtime bindings for Mojo
# Distributed under the Apache 2.0 License.
# See LICENSE for more information.
# ===----------------------------------------------------------------------=== #
"""
Low-level HDF5 C API bindings (`src.bindings`)
===============================================

Loads the HDF5 shared library at runtime via `OwnedDLHandle` and exposes a
thin struct (`HDF5Lib`) whose methods map one-to-one onto HDF5 C API
functions.  Every method returns the raw HDF5 return value so the caller
retains full control over error handling and handle lifetimes.

Most users should import the higher-level `src.hdf5` module instead, which
wraps these calls and manages handle lifetimes automatically.

Type aliases
------------
- `hid_t`   — HDF5 object identifier (file, group, dataset, dataspace, …)
- `herr_t`  — HDF5 error code (≥ 0 success, < 0 failure)
- `hsize_t` — HDF5 dimension size (unsigned 64-bit integer)
- `htri_t`  — HDF5 tri-state boolean (> 0 true, 0 false, < 0 error)

Constants
---------
- `H5F_ACC_TRUNC`  — create or overwrite a file
- `H5F_ACC_RDONLY` — open a file read-only
- `H5F_ACC_RDWR`   — open a file for reading and writing
- `H5P_DEFAULT`    — default property list sentinel (0)
- `H5S_ALL`        — select entire dataspace sentinel (0)

Notes
-----
- All string arguments are null-terminated C strings produced via
  `my_string.unsafe_ptr().bitcast[c_char]()`.
- Data buffer arguments are `UnsafePointer[NoneType, MutExternalOrigin]`;
  cast a typed pointer with `.bitcast[NoneType]()` before passing.
- Predefined HDF5 type constants (e.g. `H5T_NATIVE_DOUBLE_g`) are resolved
  via `get_symbol` at library load time and stored as fields on `HDF5Lib`.

Examples
--------
Low-level round-trip write then read:

    var h5  = HDF5Lib("./libhdf5.dylib")
    var fid = h5.create_file("out.h5")
    var sid = h5.create_dataspace_1d(100)
    var did = h5.create_dataset(fid, "/data", h5.native_double, sid)
    _ = h5.write_dataset(did, h5.native_double, buf.bitcast[NoneType]())
    _ = h5.close_dataset(did)
    _ = h5.close_dataspace(sid)
    _ = h5.close_file(fid)
"""

from std.ffi import (
    OwnedDLHandle,
    c_int,
    c_uint,
    c_long_long,
    c_ulong_long,
    c_char,
)
from std.memory import UnsafePointer
from std.os import getenv
from std.pathlib import Path, cwd
from std.sys.info import CompilationTarget

# TODO: Write docstrings for functions.

# ===----------------------------------------------------------------------=== #
# Type aliases
# ===----------------------------------------------------------------------=== #

comptime hid_t = c_long_long
"""HDF5 object identifier."""
comptime herr_t = c_int
"""HDF5 error/status code."""
comptime hsize_t = c_ulong_long
"""HDF5 dimension size."""
comptime htri_t = c_int  #
"""HDF5 tri-state boolean."""
comptime MutExt = MutExternalOrigin
"""Mutable external origin for pointers."""

# ===----------------------------------------------------------------------=== #
# HDF5 constants
# ===----------------------------------------------------------------------=== #

comptime H5F_ACC_TRUNC: c_uint = 0x0002
"""Create or overwrite."""
comptime H5F_ACC_RDONLY: c_uint = 0x0000
"""Open read-only."""
comptime H5F_ACC_RDWR: c_uint = 0x0001
"""Open read-write."""
comptime H5P_DEFAULT: hid_t = hid_t(0)
"""Default property list."""
comptime H5S_ALL: hid_t = hid_t(0)
"""Select entire dataspace."""

comptime H5I_FILE: c_int = c_int(1)
"""H5I_type_t: file."""
comptime H5I_GROUP: c_int = c_int(2)
"""H5I_type_t: group."""
comptime H5I_DATATYPE: c_int = c_int(3)
"""H5I_type_t: datatype."""
comptime H5I_DATASPACE: c_int = c_int(4)
"""H5I_type_t: dataspace."""
comptime H5I_DATASET: c_int = c_int(5)
"""H5I_type_t: dataset."""
comptime H5I_ATTR: c_int = c_int(6)
"""H5I_type_t: attribute."""

comptime H5T_INTEGER: c_int = c_int(0)
"""H5T_class_t: integer."""
comptime H5T_FLOAT: c_int = c_int(1)
"""H5T_class_t: floating point."""

# ===----------------------------------------------------------------------=== #
# HDF5Lib
# ===----------------------------------------------------------------------=== #


struct HDF5Lib(Movable):
    """Runtime loader and thin wrapper for the HDF5 C library.

    Loads the shared library from the given path and resolves the five most
    common predefined datatype ids from their global symbols at construction
    time.  All methods return raw HDF5 return values; the caller is
    responsible for checking return codes and closing handles in the correct
    order.

    Notes:
        Use `HDF5Lib.load()` to try several common library names automatically.

    Examples:
        ```mojo
        from hdf5.bindings import HDF5Lib
        var h5 = HDF5Lib("./libhdf5.dylib")
        var fid = h5.create_file("out.h5")
        _ = h5.close_file(fid)
        ```
    """

    var handle: OwnedDLHandle
    """The DLHandle for HDF5."""
    var native_double: hid_t
    """H5T_NATIVE_DOUBLE (Float64)."""
    var native_float: hid_t
    """H5T_NATIVE_FLOAT (Float32)."""
    var native_int: hid_t
    """H5T_NATIVE_INT (platform int)."""
    var native_int32: hid_t
    """H5T_NATIVE_INT32 (Int32)."""
    var std_i32le: hid_t
    """H5T_STD_I32LE (little-endian Int32)."""

    def __init__(out self) raises:
        """Load default HDF5 library and resolve predefined type constants.

        Raises:
            - Error: If the platform is unsupported.
            - Error: If the `$CONDA_PREFIX` is not set or the library is not available.
        """
        # pretty cool we do everything at comptime :)
        comptime libpath: String = getenv("CONDA_PREFIX", default="")
        var final_path: String = ""
        comptime if libpath != "":
            comptime if CompilationTarget.is_macos():
                comptime final_path = libpath + "/lib/libhdf5.dylib"
            elif CompilationTarget.is_linux():
                comptime final_path = libpath + "/lib/libhdf5.so"
            else:
                raise Error(
                    "Unsupported platform; cannot determine library path"
                )
            self.handle = OwnedDLHandle(libpath)
            _ = self.handle.call["H5open", herr_t]()
            self.native_double = self.handle.get_symbol[hid_t](
                "H5T_NATIVE_DOUBLE_g"
            )[]
            self.native_float = self.handle.get_symbol[hid_t](
                "H5T_NATIVE_FLOAT_g"
            )[]
            self.native_int = self.handle.get_symbol[hid_t](
                "H5T_NATIVE_INT_g"
            )[]
            self.native_int32 = self.handle.get_symbol[hid_t](
                "H5T_NATIVE_INT32_g"
            )[]
            self.std_i32le = self.handle.get_symbol[hid_t]("H5T_STD_I32LE_g")[]
        else:
            raise Error(
                "HDF5: CONDA_PREFIX not set; cannot determine library path"
            )

    def __init__(out self, libpath: String) raises:
        """Load `libpath` and resolve predefined type constants.

        Args:
            libpath: Filesystem path to the HDF5 shared library.

        Raises:
            - Error: If the library cannot be opened or a required symbol is missing.
        """
        self.handle = OwnedDLHandle(libpath)
        _ = self.handle.call["H5open", herr_t]()
        self.native_double = self.handle.get_symbol[hid_t](
            "H5T_NATIVE_DOUBLE_g"
        )[]
        self.native_float = self.handle.get_symbol[hid_t](
            "H5T_NATIVE_FLOAT_g"
        )[]
        self.native_int = self.handle.get_symbol[hid_t]("H5T_NATIVE_INT_g")[]
        self.native_int32 = self.handle.get_symbol[hid_t](
            "H5T_NATIVE_INT32_g"
        )[]
        self.std_i32le = self.handle.get_symbol[hid_t]("H5T_STD_I32LE_g")[]

    @staticmethod
    def load() raises -> HDF5Lib:
        """Try common shared-library names; raise if none found.

        Search order: ``./libhdf5.dylib``, ``./libhdf5.so``,
        ``libhdf5.dylib``, ``libhdf5.so``.

        Raises:
            - Error: If no candidate library path succeeds.

        Returns:
            HDF5Lib - A loaded library instance.
        """
        comptime candidates: List[String] = [
            "./libhdf5.dylib",
            "./libhdf5.so",
            "libhdf5.dylib",
            "libhdf5.so",
        ]
        for name in materialize[candidates]():
            try:
                return HDF5Lib(name)
            except:
                pass
        raise Error("HDF5: could not load library from any candidate path")

    # ===------------------------------------------------------------------=== #
    # File operations
    # ===------------------------------------------------------------------=== #

    def create_file(self, path: String, flags: c_uint = H5F_ACC_TRUNC) -> hid_t:
        """Call ``H5Fcreate``.

        Args:
            path: Filesystem path for the new file.
            flags: Access flags. Defaults to ``H5F_ACC_TRUNC`` (create or overwrite).

        Returns:
            File id (hid_t) on success; < 0 on failure.
        """
        return self.handle.call["H5Fcreate", hid_t](
            path.unsafe_ptr().bitcast[c_char](),
            flags,
            H5P_DEFAULT,
            H5P_DEFAULT,
        )

    def open_file(self, path: String, flags: c_uint = H5F_ACC_RDONLY) -> hid_t:
        """Call ``H5Fopen``.

        Args:
            path: Path to an existing HDF5 file.
            flags: Access flags. Defaults to ``H5F_ACC_RDONLY``.

        Returns:
            File id (hid_t) on success; < 0 on failure.
        """
        return self.handle.call["H5Fopen", hid_t](
            path.unsafe_ptr().bitcast[c_char](),
            flags,
            H5P_DEFAULT,
        )

    def close_file(self, fid: hid_t) -> herr_t:
        """Call ``H5Fclose``. Flushes pending writes and releases the file id.

        Args:
            fid: Pass.

        Returns:
           Pass.
        """
        return self.handle.call["H5Fclose", herr_t](fid)

    def flush(self, fid: hid_t) -> herr_t:
        """Call ``H5Fflush`` with ``H5F_SCOPE_GLOBAL`` to flush all file buffers.

        Args:
            fid: Pass.

        Returns:
            Pass.
        """
        return self.handle.call["H5Fflush", herr_t](fid, c_int(1))

    # ===------------------------------------------------------------------=== #
    # Group operations
    # ===------------------------------------------------------------------=== #

    def open_group(self, loc_id: hid_t, name: String) -> hid_t:
        """Call ``H5Gopen2``.

        Args:
            loc_id: Pass.
            name: Pass.

        Returns:
            Group id (hid_t) on success; < 0 if not found.
        """
        return self.handle.call["H5Gopen2", hid_t](
            loc_id,
            name.unsafe_ptr().bitcast[c_char](),
            H5P_DEFAULT,
        )

    def create_group(self, loc_id: hid_t, name: String) -> hid_t:
        """Call ``H5Gcreate2``.

        Args:
            loc_id: Pass.
            name: Pass.

        Returns:
            New group id (hid_t) on success; < 0 on error.
        """
        return self.handle.call["H5Gcreate2", hid_t](
            loc_id,
            name.unsafe_ptr().bitcast[c_char](),
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT,
        )

    def close_group(self, gid: hid_t) -> herr_t:
        """Call ``H5Gclose``.

        Args:
            gid: Pass.

        Returns:
            Pass.
        """
        return self.handle.call["H5Gclose", herr_t](gid)

    # ===------------------------------------------------------------------=== #
    # Dataspace operations
    # ===------------------------------------------------------------------=== #

    def create_dataspace_1d(self, n: Int) -> hid_t:
        """
        Create a fixed-size 1-D dataspace of length ``n`` via ``H5Screate_simple``.

        Args:
            n: Pass.

        Returns:
            Pass.
        """
        var dims = alloc[hsize_t](1)
        dims[0] = hsize_t(n)
        var sid = self.handle.call["H5Screate_simple", hid_t](
            c_int(1), dims, dims
        )
        dims.free()
        return sid

    def create_dataspace_nd(
        self, ndims: Int, dims: UnsafePointer[hsize_t, MutExt]
    ) -> hid_t:
        """Create a fixed-size N-D dataspace. ``dims`` must contain ``ndims`` values.

        Args:
            ndims: Pass.
            dims: Pass.

        Returns:
            Pass.
        """
        return self.handle.call["H5Screate_simple", hid_t](
            c_int(ndims), dims, dims
        )

    def get_dataset_space(self, did: hid_t) -> hid_t:
        """Call ``H5Dget_space``. Returns the dataspace id of an open dataset.

        Args:
            did: Pass.

        Returns:
            Pass.
        """
        return self.handle.call["H5Dget_space", hid_t](did)

    def get_space_ndims(self, sid: hid_t) -> c_int:
        """Call ``H5Sget_simple_extent_ndims``. Returns the number of dimensions.

        Args:
            sid: Pass.

        Returns:
            Pass.
        """
        return self.handle.call["H5Sget_simple_extent_ndims", c_int](sid)

    def get_space_dims(
        self, sid: hid_t, ndims: Int
    ) -> UnsafePointer[hsize_t, MutExt]:
        """Call ``H5Sget_simple_extent_dims`` and return an allocated dims array.

        Args:
            sid: Pass.
            ndims: Pass.

        Returns:
            Heap-allocated array of ``ndims`` dimension sizes. Caller must free this!
        """
        var dims = alloc[hsize_t](ndims)
        _ = self.handle.call["H5Sget_simple_extent_dims", c_int](
            sid, dims, UnsafePointer[hsize_t, MutExt]()
        )
        return dims

    def close_dataspace(self, sid: hid_t) -> herr_t:
        """Call ``H5Sclose``.

        Args:
            sid: Pass.

        Returns:
            Pass.
        """
        return self.handle.call["H5Sclose", herr_t](sid)

    # ===------------------------------------------------------------------=== #
    # Dataset operations
    # ===------------------------------------------------------------------=== #

    def create_dataset(
        self, loc_id: hid_t, name: String, type_id: hid_t, space_id: hid_t
    ) -> hid_t:
        """Call ``H5Dcreate2`` with default property lists.

        Args:
            loc_id: Pass.
            name: Pass.
            type_id: Pass.
            space_id: Pass.

        Returns:
            Dataset id on success; < 0 on failure.
        """
        return self.handle.call["H5Dcreate2", hid_t](
            loc_id,
            name.unsafe_ptr().bitcast[c_char](),
            type_id,
            space_id,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT,
        )

    def open_dataset(self, loc_id: hid_t, name: String) -> hid_t:
        """Call ``H5Dopen2``.

        Args:
            loc_id: Pass.
            name: Pass.

        Returns:
            Dataset id (hid_t) on success; < 0 if not found.
        """
        return self.handle.call["H5Dopen2", hid_t](
            loc_id,
            name.unsafe_ptr().bitcast[c_char](),
            H5P_DEFAULT,
        )

    def write_dataset(
        self,
        did: hid_t,
        mem_type_id: hid_t,
        buf: UnsafePointer[NoneType, MutExt],
    ) -> herr_t:
        """Call ``H5Dwrite`` selecting the entire dataset (``H5S_ALL``).

        Args:
            did: Open dataset id.
            mem_type_id: Memory datatype id (e.g. ``h5.native_double``).
            buf: Data buffer. Cast a typed pointer with ``.bitcast[NoneType]()``.

        Returns:
            ≥ 0 on success, < 0 on failure.
        """
        return self.handle.call["H5Dwrite", herr_t](
            did,
            mem_type_id,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            buf,
        )

    def read_dataset(
        self,
        did: hid_t,
        mem_type_id: hid_t,
        buf: UnsafePointer[NoneType, MutExt],
    ) -> herr_t:
        """Call ``H5Dread`` selecting the entire dataset (``H5S_ALL``).

        Args:
            did: Open dataset id.
            mem_type_id: Memory datatype id for the destination buffer.
            buf: Destination buffer, large enough for the full dataset.

        Returns:
            ≥ 0 on success, < 0 on failure.
        """
        return self.handle.call["H5Dread", herr_t](
            did,
            mem_type_id,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            buf,
        )

    def close_dataset(self, did: hid_t) -> herr_t:
        """Call ``H5Dclose``.

        Args:
            did: Open dataset id.

        Returns:
            Pass.
        """
        return self.handle.call["H5Dclose", herr_t](did)

    # ===------------------------------------------------------------------=== #
    # Datatype operations
    # ===------------------------------------------------------------------=== #

    def get_dataset_type(self, did: hid_t) -> hid_t:
        """Call ``H5Dget_type``. Caller must close the returned id with ``close_type``.
        """
        return self.handle.call["H5Dget_type", hid_t](did)

    def close_type(self, tid: hid_t) -> herr_t:
        """Call ``H5Tclose``."""
        return self.handle.call["H5Tclose", herr_t](tid)

    def get_type_class(self, tid: hid_t) -> c_int:
        """Call ``H5Tget_class``. Returns ``H5T_INTEGER`` (0) or ``H5T_FLOAT`` (1), etc.
        """
        return self.handle.call["H5Tget_class", c_int](tid)

    def get_type_size(self, tid: hid_t) -> c_ulong_long:
        """Call ``H5Tget_size``. Returns the size of the datatype in bytes."""
        return self.handle.call["H5Tget_size", c_ulong_long](tid)


    # ===------------------------------------------------------------------=== #
    # Attribute operations
    # ===------------------------------------------------------------------=== #

    def open_attr(self, loc_id: hid_t, name: String) -> hid_t:
        """Call ``H5Aopen``.

        Returns
        -------
        hid_t
            Attribute id on success; < 0 if not found.
        """
        return self.handle.call["H5Aopen", hid_t](
            loc_id,
            name.unsafe_ptr().bitcast[c_char](),
            H5P_DEFAULT,
        )

    def read_attr(
        self,
        attr_id: hid_t,
        mem_type_id: hid_t,
        buf: UnsafePointer[NoneType, MutExt],
    ) -> herr_t:
        """Call ``H5Aread``. ``buf`` must hold one value of type ``mem_type_id``.
        """
        return self.handle.call["H5Aread", herr_t](attr_id, mem_type_id, buf)

    def close_attr(self, attr_id: hid_t) -> herr_t:
        """Call ``H5Aclose``."""
        return self.handle.call["H5Aclose", herr_t](attr_id)

    # ===------------------------------------------------------------------=== #
    # Utility
    # ===------------------------------------------------------------------=== #

    def object_exists(self, loc_id: hid_t, name: String) -> Bool:
        """Return ``True`` if a link ``name`` exists under ``loc_id`` (``H5Lexists``).
        """
        var rc = self.handle.call["H5Lexists", htri_t](
            loc_id,
            name.unsafe_ptr().bitcast[c_char](),
            H5P_DEFAULT,
        )
        return rc > 0

    def get_object_type(self, loc_id: hid_t) -> c_int:
        """Call ``H5Iget_type``. Returns the ``H5I_type_t`` of an open object id.
        """
        return self.handle.call["H5Iget_type", c_int](loc_id)