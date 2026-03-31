from hdf5.bindings import (
    HDF5Lib,
    hid_t,
    hsize_t,
    H5F_ACC_RDONLY,
    H5F_ACC_TRUNC,
)
from std.memory import UnsafePointer

comptime MutExt = MutExternalOrigin

# ===----------------------------------------------------------------------=== #
# NDArray
# ===----------------------------------------------------------------------=== #


struct NDArray[dtype: DType]:
    """Heap-allocated N-D array with shape information, parameterised by `DType`.

    Returned by `H5File.read_1d` and `H5File.read_2d`.  The caller owns the
    memory and must call `.free()` when done to avoid leaks.

    Parameters:
        dtype: The element type, e.g. `DType.float64`.

    Notes:
        - 1-D arrays: `dim1 = 1`; index elements with `arr[i]`.
        - 2-D arrays: data is stored row-major; index with `arr[row, col]`.

    Examples:
        ```mojo
        var xs = f.read_1d[DType.float64]("/xs")
        for i in range(xs.dim0):
            print(xs[i])
        xs.free()

        var mat = f.read_2d[DType.float64]("/matrix")
        print(mat[2, 3], mat.dim0, mat.dim1)
        mat.free()
        ```
    """

    var data: UnsafePointer[Scalar[Self.dtype], MutExt]
    """Raw heap pointer to the element buffer."""
    var dim0: Int
    """Number of rows, or total length for 1-D arrays."""
    var dim1: Int
    """Number of columns; always 1 for 1-D arrays."""

    def __init__(
        out self, data: UnsafePointer[Scalar[Self.dtype], MutExt], n: Int
    ):
        """Construct a 1-D NDArray wrapping an existing heap buffer.

        Args:
            data: Heap-allocated buffer of `n` elements.
            n: Number of elements (`dim0`).
        """
        self.data = data
        self.dim0 = n
        self.dim1 = 1

    def __init__(
        out self,
        data: UnsafePointer[Scalar[Self.dtype], MutExt],
        rows: Int,
        cols: Int,
    ):
        """Construct a 2-D NDArray wrapping an existing heap buffer.

        Args:
            data: Heap-allocated row-major buffer of `rows * cols` elements.
            rows: Number of rows (`dim0`).
            cols: Number of columns (`dim1`).
        """
        self.data = data
        self.dim0 = rows
        self.dim1 = cols

    def __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        """Return the element at flat index `i`.

        Args:
            i: Zero-based flat index into the underlying buffer.

        Returns:
            The element value at position `i`.
        """
        return self.data[i]

    def __getitem__(self, row: Int, col: Int) -> Scalar[Self.dtype]:
        """Return the element at row-major position `(row, col)`.

        Args:
            row: Zero-based row index.
            col: Zero-based column index.

        Returns:
            The element value at `data[row * dim1 + col]`.
        """
        return self.data[row * self.dim1 + col]

    def size(self) -> Int:
        """Return the total number of elements (`dim0 * dim1`).

        Returns:
            Total element count.
        """
        return self.dim0 * self.dim1

    def free(mut self):
        """Release the underlying heap buffer.

        Must be called exactly once when the array is no longer needed.
        Calling `free` more than once is undefined behaviour.
        """
        self.data.free()


# ===----------------------------------------------------------------------=== #
# Internal helper
# ===----------------------------------------------------------------------=== #


def _hdf5_type_id[dtype: DType](lib: HDF5Lib) -> hid_t:
    """Map a compile-time `DType` to the matching HDF5 predefined type id.

    Parameters:
        dtype: The Mojo `DType` to map, e.g. `DType.float64`.

    Args:
        lib: A loaded `HDF5Lib` instance whose predefined type fields are used.

    Returns:
        The `hid_t` for the corresponding HDF5 native type.  Falls back to
        `lib.native_double` for unrecognised dtypes.

    Notes:
        The mapping is resolved entirely at compile time via `comptime if`,
        so there is no runtime dispatch overhead.
    """
    comptime if dtype == DType.float64:
        return lib.native_double
    elif dtype == DType.float32:
        return lib.native_float
    elif dtype == DType.int32:
        return lib.native_int32
    elif dtype == DType.int64:
        return lib.handle.get_symbol[hid_t]("H5T_NATIVE_INT64_g")[]
    else:
        return lib.native_double


# ===----------------------------------------------------------------------=== #
# H5File
# ===----------------------------------------------------------------------=== #


struct H5File:
    """Ergonomic handle to an open HDF5 file.

    Wraps `HDF5Lib` and manages all internal HDF5 handles (datasets,
    dataspaces, attributes) automatically.  The file itself stays open until
    `.close()` is called.

    Notes:
        - `HDF5Lib()` auto-detects the library from `$CONDA_PREFIX`.
          Pass an explicit `lib_path` when the library is elsewhere.
        - Datasets created by `write_1d` / `write_2d` must not already exist.
        - Always call `.close()` when done to flush pending writes.

    Examples:
        ```mojo
        var f = H5File("data.h5")
        var xs = f.read_1d[DType.float64]("/xs")
        xs.free()
        f.close()

        var f = H5File.create("out.h5")
        f.write_1d[DType.float64]("/xs", ptr, n)
        f.close()
        ```
    """

    var _lib: HDF5Lib
    var _fid: hid_t

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    def __init__(out self, path: String) raises:
        """Open an existing HDF5 file read-only, auto-detecting the library.

        Uses `HDF5Lib()` which reads `$CONDA_PREFIX` at compile time to
        locate the shared library.

        Args:
            path: Filesystem path to an existing HDF5 file.

        Raises:
            - Error: If the library cannot be loaded.
            - Error: If the file cannot be opened.
        """
        self._lib = HDF5Lib()
        self._fid = self._lib.open_file(path, H5F_ACC_RDONLY)
        if self._fid < 0:
            raise Error("H5File: cannot open '" + path + "'")

    def __init__(out self, path: String, lib_path: String) raises:
        """Open an existing HDF5 file read-only using an explicit library path.

        Args:
            path: Filesystem path to an existing HDF5 file.
            lib_path: Filesystem path to the HDF5 shared library,
                e.g. ``"/usr/lib/libhdf5.so"``.

        Raises:
            - Error: If the library at `lib_path` cannot be loaded.
            - Error: If the file cannot be opened.

        Examples:
            ```mojo
            var f = H5File("data.h5", "/usr/lib/libhdf5.so")
            ```
        """
        self._lib = HDF5Lib(lib_path)
        self._fid = self._lib.open_file(path, H5F_ACC_RDONLY)
        if self._fid < 0:
            raise Error("H5File: cannot open '" + path + "'")

    @staticmethod
    def create(path: String) raises -> H5File:
        """Create (or truncate) a file for writing, auto-detecting the library.

        Args:
            path: Filesystem path for the new HDF5 file.

        Returns:
            An `H5File` open for writing.

        Raises:
            - Error: If the library cannot be loaded.
            - Error: If the file cannot be created.
        """
        var lib = HDF5Lib()
        var fid = lib.create_file(path, H5F_ACC_TRUNC)
        if fid < 0:
            raise Error("H5File.create: cannot create '" + path + "'")
        return H5File(lib^, fid)

    @staticmethod
    def create(path: String, lib_path: String) raises -> H5File:
        """Create (or truncate) a file using an explicit library path.

        Args:
            path: Filesystem path for the new HDF5 file.
            lib_path: Filesystem path to the HDF5 shared library.

        Returns:
            An `H5File` open for writing.

        Raises:
            - Error: If the library cannot be loaded.
            - Error: If the file cannot be created.
        """
        var lib = HDF5Lib(lib_path)
        var fid = lib.create_file(path, H5F_ACC_TRUNC)
        if fid < 0:
            raise Error("H5File.create: cannot create '" + path + "'")
        return H5File(lib^, fid)

    def __init__(out self, var lib: HDF5Lib, fid: hid_t):
        # Private constructor used by the `create()` static methods.
        self._lib = lib^
        self._fid = fid

    def close(self):
        """Flush pending writes and close the file id.

        Must be called when the file is no longer needed.
        """
        _ = self._lib.close_file(self._fid)
