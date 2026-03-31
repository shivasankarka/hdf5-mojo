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
