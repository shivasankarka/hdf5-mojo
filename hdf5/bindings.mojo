# ===----------------------------------------------------------------------=== #
# hdf5-bindings: HDF5 runtime bindings for Mojo
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE for more information.
# ===----------------------------------------------------------------------=== #
"""
Low-level HDF5 C API bindings (`hdf5.bindings`)
================================================

Loads the HDF5 shared library at runtime via `OwnedDLHandle` and exposes a
thin struct (`HDF5Lib`) whose methods map one-to-one onto HDF5 C API
functions.  Every method returns the raw HDF5 return value so the caller
retains full control over error handling and handle lifetimes.

Most users should import the higher-level `hdf5.api` module instead, which
wraps these calls and manages handle lifetimes automatically.

Type aliases
------------
- `hid_t`   — HDF5 object identifier (file, group, dataset, dataspace, …)
- `herr_t`  — HDF5 error/status code (≥ 0 success, < 0 failure)
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
- All string arguments are converted to null-terminated C strings via
  `my_string.unsafe_ptr().bitcast[c_char]()`.
- Data buffer arguments are `UnsafePointer[NoneType, MutExternalOrigin]`;
  cast a typed pointer with `.bitcast[NoneType]()` before passing.
- Predefined HDF5 type constants (e.g. `H5T_NATIVE_DOUBLE_g`) are resolved
  via `get_symbol` at library load time and stored as fields on `HDF5Lib`.

Examples
--------
Low-level round-trip write then read:

    from hdf5.bindings import HDF5Lib

    var h5  = HDF5Lib()                          # auto-detect from $CONDA_PREFIX
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

# ===----------------------------------------------------------------------=== #
# Type aliases
# ===----------------------------------------------------------------------=== #

comptime hid_t = c_long_long
"""HDF5 object identifier (maps to `int64_t` on standard 64-bit builds)."""
comptime herr_t = c_int
"""HDF5 error/status code.  ≥ 0 indicates success; < 0 indicates failure."""
comptime hsize_t = c_ulong_long
"""HDF5 dimension size (unsigned 64-bit integer)."""
comptime htri_t = c_int
"""HDF5 tri-state boolean: > 0 true, 0 false, < 0 error."""
comptime MutExt = MutExternalOrigin
"""Shorthand for `MutExternalOrigin`, used for all FFI buffer pointers."""

# ===----------------------------------------------------------------------=== #
# HDF5 constants
# ===----------------------------------------------------------------------=== #

comptime H5F_ACC_TRUNC: c_uint = 0x0002
"""File-access flag: create a new file or truncate an existing one."""
comptime H5F_ACC_RDONLY: c_uint = 0x0000
"""File-access flag: open an existing file read-only."""
comptime H5F_ACC_RDWR: c_uint = 0x0001
"""File-access flag: open an existing file for reading and writing."""
comptime H5P_DEFAULT: hid_t = hid_t(0)
"""Sentinel value for "use the default property list" in HDF5 calls."""
comptime H5S_ALL: hid_t = hid_t(0)
"""Sentinel value for "select the entire dataspace" in read/write calls."""

comptime H5I_FILE: c_int = c_int(1)
"""H5I_type_t constant identifying a file object."""
comptime H5I_GROUP: c_int = c_int(2)
"""H5I_type_t constant identifying a group object."""
comptime H5I_DATATYPE: c_int = c_int(3)
"""H5I_type_t constant identifying a datatype object."""
comptime H5I_DATASPACE: c_int = c_int(4)
"""H5I_type_t constant identifying a dataspace object."""
comptime H5I_DATASET: c_int = c_int(5)
"""H5I_type_t constant identifying a dataset object."""
comptime H5I_ATTR: c_int = c_int(6)
"""H5I_type_t constant identifying an attribute object."""

comptime H5T_INTEGER: c_int = c_int(0)
"""H5T_class_t value for integer datatypes."""
comptime H5T_FLOAT: c_int = c_int(1)
"""H5T_class_t value for floating-point datatypes."""

# ===----------------------------------------------------------------------=== #
# HDF5Lib
# ===----------------------------------------------------------------------=== #


struct HDF5Lib(Movable):
    """Runtime loader and thin wrapper for the HDF5 C library.

    Loads the shared library and resolves the five most common predefined
    datatype ids from their global symbols at construction time.  All methods
    return raw HDF5 return values; the caller is responsible for checking
    return codes and closing handles in the correct order.

    Notes:
        Use `HDF5Lib()` (no arguments) to auto-detect the library from
        `$CONDA_PREFIX`, or pass an explicit path with `HDF5Lib(libpath)`.

    Examples:
        ```mojo
        from hdf5.bindings import HDF5Lib
        var h5 = HDF5Lib()                   # auto-detect
        var h5 = HDF5Lib("./libhdf5.dylib")  # explicit path
        var fid = h5.create_file("out.h5")
        _ = h5.close_file(fid)
        ```
    """

    var handle: OwnedDLHandle
    """The underlying `OwnedDLHandle` wrapping the HDF5 shared library."""
    var native_double: hid_t
    """Predefined type id for `H5T_NATIVE_DOUBLE` (64-bit IEEE float / Float64)."""
    var native_float: hid_t
    """Predefined type id for `H5T_NATIVE_FLOAT` (32-bit IEEE float / Float32)."""
    var native_int: hid_t
    """Predefined type id for `H5T_NATIVE_INT` (platform-native C int)."""
    var native_int32: hid_t
    """Predefined type id for `H5T_NATIVE_INT32` (32-bit signed integer / Int32)."""
    var std_i32le: hid_t
    """Predefined type id for `H5T_STD_I32LE` (little-endian 32-bit signed integer)."""

    def __init__(out self) raises:
        """Auto-detect and load the HDF5 library from `$CONDA_PREFIX`.

        Reads `$CONDA_PREFIX` at compile time and constructs the expected
        library path for the current platform (`libhdf5.dylib` on macOS,
        `libhdf5.so` on Linux).

        Raises:
            - Error: If the platform is not macOS or Linux.
            - Error: If `$CONDA_PREFIX` is not set at compile time.
        """
        var libpath: String = getenv("CONDA_PREFIX", default="")
        if libpath != "":
            comptime if CompilationTarget.is_macos():
                libpath += "/lib/libhdf5.dylib"
            elif CompilationTarget.is_linux():
                libpath += "/lib/libhdf5.so"
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
        """Load the HDF5 library from an explicit filesystem path.

        Args:
            libpath: Absolute or relative path to the HDF5 shared library,
                e.g. ``"./libhdf5.dylib"`` or ``"/usr/lib/libhdf5.so"``.

        Raises:
            - Error: If the library cannot be opened.
            - Error: If a required predefined type symbol is missing.
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
        """Try a list of common library names in order; raise if none succeeds.

        Search order: ``./libhdf5.dylib``, ``./libhdf5.so``,
        ``libhdf5.dylib``, ``libhdf5.so``.

        Returns:
            HDF5Lib: A successfully loaded library instance.

        Raises:
            - Error: If none of the candidate paths can be opened.
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
        """Call ``H5Fcreate`` to create a new HDF5 file.

        Args:
            path: Filesystem path for the new file.
            flags: File-creation flags. Defaults to ``H5F_ACC_TRUNC``,
                which creates the file or truncates it if it already exists.

        Returns:
            A valid file id (`hid_t`) on success; < 0 on failure.
        """
        return self.handle.call["H5Fcreate", hid_t](
            path.unsafe_ptr().bitcast[c_char](),
            flags,
            H5P_DEFAULT,
            H5P_DEFAULT,
        )

    def open_file(self, path: String, flags: c_uint = H5F_ACC_RDONLY) -> hid_t:
        """Call ``H5Fopen`` to open an existing HDF5 file.

        Args:
            path: Path to an existing HDF5 file.
            flags: File-access flags. Defaults to ``H5F_ACC_RDONLY``.
                Use ``H5F_ACC_RDWR`` to open for reading and writing.

        Returns:
            A valid file id (`hid_t`) on success; < 0 on failure.
        """
        return self.handle.call["H5Fopen", hid_t](
            path.unsafe_ptr().bitcast[c_char](),
            flags,
            H5P_DEFAULT,
        )

    def close_file(self, fid: hid_t) -> herr_t:
        """Call ``H5Fclose`` to flush pending writes and release a file id.

        Args:
            fid: An open file id returned by `create_file` or `open_file`.

        Returns:
            ≥ 0 on success; < 0 on failure.
        """
        return self.handle.call["H5Fclose", herr_t](fid)

    def flush(self, fid: hid_t) -> herr_t:
        """Call ``H5Fflush`` with ``H5F_SCOPE_GLOBAL`` to flush all buffers.

        Forces all cached data to be written to the underlying storage.
        Useful before reading back data that was just written.

        Args:
            fid: An open file id.

        Returns:
            ≥ 0 on success; < 0 on failure.
        """
        return self.handle.call["H5Fflush", herr_t](fid, c_int(1))

    # ===------------------------------------------------------------------=== #
    # Group operations
    # ===------------------------------------------------------------------=== #

    def open_group(self, loc_id: hid_t, name: String) -> hid_t:
        """Call ``H5Gopen2`` to open an existing group.

        Args:
            loc_id: File or group id under which to search.
            name: Absolute or relative HDF5 path to the group,
                e.g. ``"/cross_sections"`` or ``"results"``.

        Returns:
            A valid group id (`hid_t`) on success; < 0 if the group is not found.
        """
        return self.handle.call["H5Gopen2", hid_t](
            loc_id,
            name.unsafe_ptr().bitcast[c_char](),
            H5P_DEFAULT,
        )

    def create_group(self, loc_id: hid_t, name: String) -> hid_t:
        """Call ``H5Gcreate2`` to create a new group.

        Args:
            loc_id: File or parent group id under which to create the group.
            name: Name or path for the new group, e.g. ``"results"``.

        Returns:
            A valid group id (`hid_t`) on success; < 0 on failure.
        """
        return self.handle.call["H5Gcreate2", hid_t](
            loc_id,
            name.unsafe_ptr().bitcast[c_char](),
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT,
        )

    def close_group(self, gid: hid_t) -> herr_t:
        """Call ``H5Gclose`` to release a group id.

        Args:
            gid: An open group id returned by `open_group` or `create_group`.

        Returns:
            ≥ 0 on success; < 0 on failure.
        """
        return self.handle.call["H5Gclose", herr_t](gid)

    # ===------------------------------------------------------------------=== #
    # Dataspace operations
    # ===------------------------------------------------------------------=== #

    def create_dataspace_1d(self, n: Int) -> hid_t:
        """Call ``H5Screate_simple`` to create a fixed-size 1-D dataspace.

        Args:
            n: The number of elements in the single dimension.

        Returns:
            A valid dataspace id (`hid_t`) on success; < 0 on failure.
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
        """Call ``H5Screate_simple`` to create a fixed-size N-D dataspace.

        Args:
            ndims: Number of dimensions.
            dims: Pointer to an array of `ndims` dimension sizes. The same
                array is used for both current and maximum dimensions
                (fixed size, no chunking).

        Returns:
            A valid dataspace id (`hid_t`) on success; < 0 on failure.
        """
        return self.handle.call["H5Screate_simple", hid_t](
            c_int(ndims), dims, dims
        )

    def get_dataset_space(self, did: hid_t) -> hid_t:
        """Call ``H5Dget_space`` to retrieve the dataspace of an open dataset.

        Args:
            did: An open dataset id.

        Returns:
            A valid dataspace id (`hid_t`) on success; < 0 on failure.
            Caller must close the returned id with `close_dataspace`.
        """
        return self.handle.call["H5Dget_space", hid_t](did)

    def get_space_ndims(self, sid: hid_t) -> c_int:
        """Call ``H5Sget_simple_extent_ndims`` to query dimensionality.

        Args:
            sid: An open dataspace id.

        Returns:
            The number of dimensions (≥ 0); < 0 on failure.
        """
        return self.handle.call["H5Sget_simple_extent_ndims", c_int](sid)

    def get_space_dims(
        self, sid: hid_t, ndims: Int
    ) -> UnsafePointer[hsize_t, MutExt]:
        """Call ``H5Sget_simple_extent_dims`` and return a heap-allocated dims array.

        Args:
            sid: An open dataspace id.
            ndims: The number of dimensions (obtained from `get_space_ndims`).

        Returns:
            A heap-allocated array of `ndims` dimension sizes.
            The caller is responsible for calling `.free()` on the result.
        """
        var dims = alloc[hsize_t](ndims)
        _ = self.handle.call["H5Sget_simple_extent_dims", c_int](
            sid, dims, UnsafePointer[hsize_t, MutExt]()
        )
        return dims

    def close_dataspace(self, sid: hid_t) -> herr_t:
        """Call ``H5Sclose`` to release a dataspace id.

        Args:
            sid: An open dataspace id.

        Returns:
            ≥ 0 on success; < 0 on failure.
        """
        return self.handle.call["H5Sclose", herr_t](sid)

    # ===------------------------------------------------------------------=== #
    # Dataset operations
    # ===------------------------------------------------------------------=== #

    def create_dataset(
        self, loc_id: hid_t, name: String, type_id: hid_t, space_id: hid_t
    ) -> hid_t:
        """Call ``H5Dcreate2`` to create a new dataset with default property lists.

        Args:
            loc_id: File or group id under which to create the dataset.
            name: Name or absolute path for the new dataset,
                e.g. ``"energies"`` or ``"/results/energies"``.
            type_id: HDF5 datatype id, e.g. ``h5.native_double``.
            space_id: HDF5 dataspace id describing the dataset shape.

        Returns:
            A valid dataset id (`hid_t`) on success; < 0 on failure.
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
        """Call ``H5Dopen2`` to open an existing dataset.

        Args:
            loc_id: File or group id under which to search.
            name: Name or absolute HDF5 path to the dataset,
                e.g. ``"/cross_sections/sigma_nu"``.

        Returns:
            A valid dataset id (`hid_t`) on success; < 0 if not found.
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
        """Call ``H5Dwrite`` to write the entire dataset from a memory buffer.

        Uses ``H5S_ALL`` for both the memory and file dataspaces, which
        selects the entire dataset without any subsetting.

        Args:
            did: An open dataset id.
            mem_type_id: HDF5 type id describing the layout of `buf`,
                e.g. ``h5.native_double`` for a buffer of `Float64`.
            buf: Source data buffer. Cast a typed pointer via
                `my_ptr.bitcast[NoneType]()` before passing.

        Returns:
            ≥ 0 on success; < 0 on failure.
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
        """Call ``H5Dread`` to read the entire dataset into a memory buffer.

        Uses ``H5S_ALL`` for both the memory and file dataspaces, reading all
        elements without subsetting.

        Args:
            did: An open dataset id.
            mem_type_id: HDF5 type id for the destination buffer layout,
                e.g. ``h5.native_double`` for a `Float64` buffer.
            buf: Destination buffer, pre-allocated to hold the full dataset.
                Cast a typed pointer via `my_ptr.bitcast[NoneType]()`.

        Returns:
            ≥ 0 on success; < 0 on failure.
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
        """Call ``H5Dclose`` to release a dataset id.

        Args:
            did: An open dataset id.

        Returns:
            ≥ 0 on success; < 0 on failure.
        """
        return self.handle.call["H5Dclose", herr_t](did)

    # ===------------------------------------------------------------------=== #
    # Datatype operations
    # ===------------------------------------------------------------------=== #

    def get_dataset_type(self, did: hid_t) -> hid_t:
        """Call ``H5Dget_type`` to retrieve the datatype of an open dataset.

        Args:
            did: An open dataset id.

        Returns:
            A valid datatype id (`hid_t`). The caller must close it
            with `close_type` when done.
        """
        return self.handle.call["H5Dget_type", hid_t](did)

    def close_type(self, tid: hid_t) -> herr_t:
        """Call ``H5Tclose`` to release a datatype id.

        Args:
            tid: A datatype id returned by `get_dataset_type`.

        Returns:
            ≥ 0 on success; < 0 on failure.
        """
        return self.handle.call["H5Tclose", herr_t](tid)

    def get_type_class(self, tid: hid_t) -> c_int:
        """Call ``H5Tget_class`` to query the class of a datatype.

        Args:
            tid: A datatype id.

        Returns:
            An `H5T_class_t` value: ``H5T_INTEGER`` (0), ``H5T_FLOAT`` (1),
            etc. Returns < 0 on failure.
        """
        return self.handle.call["H5Tget_class", c_int](tid)

    def get_type_size(self, tid: hid_t) -> c_ulong_long:
        """Call ``H5Tget_size`` to query the storage size of a datatype.

        Args:
            tid: A datatype id.

        Returns:
            The size of the datatype in bytes; 0 on failure.
        """
        return self.handle.call["H5Tget_size", c_ulong_long](tid)

    # ===------------------------------------------------------------------=== #
    # Attribute operations
    # ===------------------------------------------------------------------=== #

    def open_attr(self, loc_id: hid_t, name: String) -> hid_t:
        """Call ``H5Aopen`` to open an attribute on a group or dataset.

        Args:
            loc_id: An open file, group, or dataset id that owns the attribute.
            name: Name of the attribute, e.g. ``"min_energy"``.

        Returns:
            A valid attribute id (`hid_t`) on success; < 0 if not found.
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
        """Call ``H5Aread`` to read an attribute value into a memory buffer.

        Args:
            attr_id: An open attribute id returned by `open_attr`.
            mem_type_id: HDF5 type id for the destination buffer,
                e.g. ``h5.native_double`` to read a `Float64` scalar.
            buf: Destination buffer, sized for one value of `mem_type_id`.
                Cast via `my_ptr.bitcast[NoneType]()`.

        Returns:
            ≥ 0 on success; < 0 on failure.
        """
        return self.handle.call["H5Aread", herr_t](attr_id, mem_type_id, buf)

    def close_attr(self, attr_id: hid_t) -> herr_t:
        """Call ``H5Aclose`` to release an attribute id.

        Args:
            attr_id: An open attribute id returned by `open_attr`.

        Returns:
            ≥ 0 on success; < 0 on failure.
        """
        return self.handle.call["H5Aclose", herr_t](attr_id)

    # ===------------------------------------------------------------------=== #
    # Utility
    # ===------------------------------------------------------------------=== #

    def object_exists(self, loc_id: hid_t, name: String) -> Bool:
        """Return ``True`` if a named link exists under ``loc_id``.

        Calls ``H5Lexists`` to check whether a group, dataset, or other
        named object is reachable from `loc_id`.

        Args:
            loc_id: File or group id to search under.
            name: Relative or absolute HDF5 path to check.

        Returns:
            `True` if the link exists; `False` otherwise.
        """
        var rc = self.handle.call["H5Lexists", htri_t](
            loc_id,
            name.unsafe_ptr().bitcast[c_char](),
            H5P_DEFAULT,
        )
        return rc > 0

    def get_object_type(self, loc_id: hid_t) -> c_int:
        """Call ``H5Iget_type`` to query the type of an open object id.

        Args:
            loc_id: Any open HDF5 object id (file, group, dataset, etc.).

        Returns:
            An `H5I_type_t` value such as ``H5I_FILE`` (1), ``H5I_GROUP`` (2),
            ``H5I_DATASET`` (5), etc. Returns ``H5I_BADID`` (< 0) on failure.
        """
        return self.handle.call["H5Iget_type", c_int](loc_id)
