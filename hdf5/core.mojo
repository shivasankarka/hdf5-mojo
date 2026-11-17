# ===----------------------------------------------------------------------=== #
# hdf5-bindings: HDF5 runtime bindings for Mojo
# Distributed under the Apache 2.0 License.
# See LICENSE for more information.
# ===----------------------------------------------------------------------=== #
"""
Core HDF5 API (`hdf5.core`)
=====================================
Provides an API that mirrors the Python ``h5py`` library as closely as possible.
"""

from hdf5.ffi import (
    HDF5Lib,
    hid_t,
    hsize_t,
    herr_t,
    c_int,
    c_uint,
    c_ssize_t,
    c_char,
    c_ulong_long,
    H5F_ACC_RDONLY,
    H5F_ACC_RDWR,
    H5F_ACC_TRUNC,
    H5P_DEFAULT,
    H5I_GROUP,
    H5I_DATASET,
    H5T_INTEGER,
    H5T_FLOAT,
    H5T_SGN_NONE,
)
from std.memory import ArcPointer, OwnedPointer, Pointer
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.utils import Variant
from numojo import NDArray, Item, Shape

comptime MutExt = MutUntrackedOrigin
comptime LibRef[mut: Bool, //, origin: Origin[mut=mut]] = Pointer[
    HDF5Lib, origin
]
comptime H5Z_FILTER_DEFLATE = 1
comptime H5Z_FILTER_SHUFFLE = 2
comptime H5Z_FILTER_FLETCHER32 = 3


# ===----------------------------------------------------------------------=== #
# DType <-> HDF5 type mapping
# ===----------------------------------------------------------------------=== #


comptime SupportedDType[dtype: DType] = (
    dtype == DType.float64
    or dtype == DType.float32
    or dtype == DType.float16
    or dtype == DType.int64
    or dtype == DType.int32
    or dtype == DType.int16
    or dtype == DType.int8
    or dtype == DType.uint64
    or dtype == DType.uint32
    or dtype == DType.uint16
    or dtype == DType.uint8
)
"""Whether `dtype` is a dtype this library can read/write (for `where`)."""


def _guess_chunks(
    shape: List[Int], maxshape: List[Int], itemsize: Int
) -> List[Int]:
    """Guess a chunk shape (<= 1 MiB) when the user did not give one."""
    var chunks = List[Int]()
    for i in range(len(shape)):
        var d = shape[i]
        if d <= 0:
            d = 1024 if (len(maxshape) > i and maxshape[i] < 0) else 1
        chunks.append(d)
    var limit = 1024 * 1024
    while True:
        var nbytes = itemsize
        var largest = 0
        for i in range(len(chunks)):
            nbytes *= chunks[i]
            if chunks[i] > chunks[largest]:
                largest = i
        if nbytes <= limit or chunks[largest] <= 1:
            break
        chunks[largest] = (chunks[largest] + 1) // 2
    return chunks^


def _native_type(lib: HDF5Lib, symbol: String) raises -> hid_t:
    """Resolve a predefined HDF5 type id (e.g. ``H5T_NATIVE_INT16_g``)."""
    var sym = lib.handle.get_symbol[hid_t](symbol)
    if not sym:
        raise Error("HDF5: type symbol not found: " + symbol)
    return sym.value()[]


# TODO: Use enums later in v1.3.0
def _hdf5_type_id[dtype: DType](lib: HDF5Lib) raises -> hid_t:
    """Map a Mojo DType to the matching HDF5 predefined type id."""
    comptime if dtype == DType.float64:
        return lib.native_double
    elif dtype == DType.float32:
        return lib.native_float
    elif dtype == DType.int32:
        return lib.native_int32
    elif dtype == DType.int64:
        return _native_type(lib, "H5T_NATIVE_INT64_g")
    elif dtype == DType.int8:
        return lib.native_int8
    elif dtype == DType.uint8:
        return lib.native_uint8
    elif dtype == DType.int16:
        return _native_type(lib, "H5T_NATIVE_INT16_g")
    elif dtype == DType.uint16:
        return _native_type(lib, "H5T_NATIVE_UINT16_g")
    elif dtype == DType.uint32:
        return _native_type(lib, "H5T_NATIVE_UINT32_g")
    elif dtype == DType.uint64:
        return _native_type(lib, "H5T_NATIVE_UINT64_g")
    elif dtype == DType.float16:
        return _native_type(lib, "H5T_IEEE_F16LE_g")
    else:
        raise Error("HDF5: unsupported DType")


def _dtype_code[dtype: DType]() raises -> Int:
    """Map a supported Mojo DType to this module's compact dtype code."""
    comptime if dtype == DType.float64:
        return 0
    elif dtype == DType.float32:
        return 1
    elif dtype == DType.int32:
        return 2
    elif dtype == DType.int64:
        return 3
    elif dtype == DType.int8:
        return 4
    elif dtype == DType.uint8:
        return 5
    elif dtype == DType.int16:
        return 6
    elif dtype == DType.uint16:
        return 7
    elif dtype == DType.uint32:
        return 8
    elif dtype == DType.uint64:
        return 9
    elif dtype == DType.float16:
        return 10
    else:
        raise Error("HDF5: unsupported DType")


def _infer_dtype_code(
    lib: HDF5Lib, tid: hid_t, tclass: c_int, tsize: Int
) -> Int:
    """Infer this module's compact dtype code from an on-disk HDF5 type.

    Distinguishes signed from unsigned integers via `H5Tget_sign`. Types outside the currently supported set fall back to
    the `float64` code; callers that need strict validation should check
    `Dataset.dtype()` against the expected string.
    """
    if tclass == H5T_FLOAT:
        if tsize == 8:
            return 0
        elif tsize == 4:
            return 1
        elif tsize == 2:
            return 10
    elif tclass == H5T_INTEGER:
        var unsigned = lib.get_type_sign(tid) == H5T_SGN_NONE
        if tsize == 4:
            return 8 if unsigned else 2
        elif tsize == 8:
            return 9 if unsigned else 3
        elif tsize == 2:
            return 7 if unsigned else 6
        elif tsize == 1:
            return 5 if unsigned else 4
    return 0


def _dtype_name(code: Int) -> String:
    """Map this module's compact dtype code to its string name."""
    if code == 0:
        return "float64"
    elif code == 1:
        return "float32"
    elif code == 2:
        return "int32"
    elif code == 3:
        return "int64"
    elif code == 4:
        return "int8"
    elif code == 5:
        return "uint8"
    elif code == 6:
        return "int16"
    elif code == 7:
        return "uint16"
    elif code == 8:
        return "uint32"
    elif code == 9:
        return "uint64"
    elif code == 10:
        return "float16"
    return "unknown"


def _dtype_itemsize(code: Int) -> Int:
    """Size in bytes of one element of the given compact dtype code."""
    if code == 0 or code == 3 or code == 9:
        return 8
    elif code == 1 or code == 2 or code == 8:
        return 4
    elif code == 6 or code == 7 or code == 10:
        return 2
    elif code == 4 or code == 5:
        return 1
    return 0


def _type_id_for_code(lib: HDF5Lib, code: Int) raises -> hid_t:
    """HDF5 native type id matching a compact dtype code."""
    if code == 0:
        return lib.native_double
    elif code == 1:
        return lib.native_float
    elif code == 2:
        return lib.native_int32
    elif code == 3:
        return _native_type(lib, "H5T_NATIVE_INT64_g")
    elif code == 4:
        return lib.native_int8
    elif code == 5:
        return lib.native_uint8
    elif code == 6:
        return _native_type(lib, "H5T_NATIVE_INT16_g")
    elif code == 7:
        return _native_type(lib, "H5T_NATIVE_UINT16_g")
    elif code == 8:
        return _native_type(lib, "H5T_NATIVE_UINT32_g")
    elif code == 9:
        return _native_type(lib, "H5T_NATIVE_UINT64_g")
    elif code == 10:
        return _native_type(lib, "H5T_IEEE_F16LE_g")
    raise Error("HDF5: unsupported dtype")


struct NameIterator(Iterator, Movable):
    """Iterator over member or attribute names (returned by `__iter__`)."""

    comptime Element = String

    var _names: List[String]
    var _idx: Int

    def __init__(out self, var names: List[String]):
        self._names = names^
        self._idx = 0

    def __iter__(var self) -> Self:
        return self^

    def __next__(mut self) raises StopIteration -> String:
        if self._idx >= len(self._names):
            raise StopIteration()
        self._idx += 1
        return self._names[self._idx - 1]


struct AttrValue(Copyable, Movable):
    """A type-erased attribute value returned by `AttributeManager.items()`.

    Inspect `dtype` / `shape`, then extract with `as_scalar[dtype]()` or
    `as_array[dtype]()`; extracting with the wrong dtype raises.
    """

    var dtype: String
    """Datatype name, e.g. ``"float64"``."""
    var shape: List[Int]
    """Attribute shape (`[]` for a scalar attribute)."""
    var _code: Int
    var _data: List[UInt8]

    def __init__(
        out self,
        dtype: String,
        var shape: List[Int],
        code: Int,
        var data: List[UInt8],
    ):
        self.dtype = dtype
        self.shape = shape^
        self._code = code
        self._data = data^

    def is_scalar(self) -> Bool:
        return len(self.shape) == 0

    def size(self) -> Int:
        """Number of elements (1 for a scalar attribute)."""
        var n = 1
        for d in self.shape:
            n *= d
        return n

    def as_scalar[
        dtype: DType
    ](self) raises -> Scalar[dtype] where SupportedDType[dtype]:
        """The value of a scalar attribute."""
        if self._code != _dtype_code[dtype]():
            raise Error("AttrValue: dtype is " + self.dtype)
        if not self.is_scalar():
            raise Error("AttrValue: not a scalar; use as_array()")
        return self._data.unsafe_ptr().bitcast[Scalar[dtype]]()[0]

    def as_array[
        dtype: DType
    ](self) raises -> NDArray[dtype] where SupportedDType[dtype]:
        """The value of an array attribute as an NDArray."""
        if self._code != _dtype_code[dtype]():
            raise Error("AttrValue: dtype is " + self.dtype)
        if self.is_scalar():
            raise Error("AttrValue: scalar attribute; use as_scalar()")
        var arr = NDArray[dtype](Shape(self.shape))
        var src = self._data.unsafe_ptr().bitcast[Scalar[dtype]]()
        var dst = arr.unsafe_ptr()
        for i in range(self.size()):
            dst[unsafe_offset=i] = src[i]
        return arr^


# ===----------------------------------------------------------------------=== #
# AttributeManager
# ===----------------------------------------------------------------------=== #


struct AttributeManager[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](Copyable, Movable, Sized):
    """Dict-like proxy for HDF5 attributes on a Group or Dataset.

    Attributes are user-defined metadata attached to HDF5 objects (groups or
    datasets). This class provides dict-like access to attributes.

    Access via ``.attrs()`` method on File, Group, or Dataset objects::

        var f = File("data.h5", "r")
        var version = f.attrs().get[DType.int32]("version", Int32(0))  # read
        f.attrs().set[DType.int32]("created", Int32(42))              # write
        f.attrs().delete("temp_attr")                                # delete
    """

    var _loc_id: hid_t
    var _lib: LibRef[Self.origin]

    def __init__(out self, lib: LibRef[Self.origin], loc_id: hid_t):
        self._loc_id = loc_id
        self._lib = lib

    def __contains__(self, name: String) -> Bool:
        """Check if an attribute exists.

        Args:
            name: Name of the attribute.

        Returns:
            True if the attribute exists, False otherwise.
        """
        var aid = self._lib[].open_attr(self._loc_id, name)
        if aid >= 0:
            _ = self._lib[].close_attr(aid)
            return True
        return False

    def __getitem__[dtype: DType](self, name: String) raises -> Scalar[dtype]:
        """Read a scalar attribute value.

        Parameters:
            dtype: The data type to read as (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the attribute to read.

        Returns:
            The attribute value as Scalar[dtype].

        Raises:
            Error: If the attribute does not exist or read fails.
        """
        return self.read_scalar[dtype](name)

    def __setitem__[
        dtype: DType
    ](mut self, name: String, value: Scalar[dtype]) raises:
        """Write a scalar attribute value.

        Parameters:
            dtype: The data type to read as (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the attribute to write.
            value: The value to write.

        Raises:
            Error: If writing fails.
        """
        self.write_scalar[dtype](name, value)

    def read_scalar[dtype: DType](self, name: String) raises -> Scalar[dtype]:
        """Read a scalar attribute value.

        Parameters:
            dtype: The data type to read as (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the attribute to read.

        Returns:
            The attribute value as Scalar[dtype].

        Raises:
            Error: If the attribute does not exist or read fails.
        """
        if len(self.shape(name)) != 0:
            raise Error("attrs: '" + name + "' is an array; use read_array()")
        var aid = self._lib[].open_attr(self._loc_id, name)
        if aid < 0:
            raise Error("attrs: '" + name + "' not found")
        var buf = unsafe_alloc[Scalar[dtype]](1)
        var rc = self._lib[].read_attr(
            aid,
            _hdf5_type_id[dtype](self._lib[]),
            buf.unsafe_bitcast[NoneType](),
        )
        _ = self._lib[].close_attr(aid)
        if rc < 0:
            buf.unsafe_free()
            raise Error("attrs: H5Aread failed for '" + name + "'")
        var v = buf[unsafe_offset=0]
        buf.unsafe_free()
        return v

    def write_scalar[
        dtype: DType
    ](self, name: String, value: Scalar[dtype]) raises:
        """Write a scalar attribute value.

        Parameters:
            dtype: The data type to read as (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the attribute to write.
            value: The value to write.

        Raises:
            Error: If writing fails.
        """
        if self.__contains__(name):
            self.delete(name)
        var sid = self._lib[].create_attr_dataspace_scalar()
        if sid < 0:
            raise Error("attrs: H5Screate(scalar) failed")
        var buf = unsafe_alloc[Scalar[dtype]](1)
        buf[unsafe_offset=0] = value
        var rc = self._lib[].write_attr(
            self._loc_id,
            name,
            _hdf5_type_id[dtype](self._lib[]),
            sid,
            buf.unsafe_bitcast[NoneType](),
        )
        buf.unsafe_free()
        _ = self._lib[].close_dataspace(sid)
        if rc < 0:
            raise Error("attrs: write failed for '" + name + "'")

    def shape(self, name: String) raises -> List[Int]:
        """Shape of an attribute (`[]` for a scalar attribute)."""
        var aid = self._lib[].open_attr(self._loc_id, name)
        if aid < 0:
            raise Error("attrs: '" + name + "' not found")
        var sid = self._lib[].get_attr_space(aid)
        _ = self._lib[].close_attr(aid)
        if sid < 0:
            raise Error("attrs: H5Aget_space failed for '" + name + "'")
        var ndims = Int(self._lib[].get_space_ndims(sid))
        var result = List[Int]()
        if ndims > 0:
            var dims = self._lib[].get_space_dims(sid, ndims)
            for i in range(ndims):
                result.append(Int(dims[unsafe_offset=i]))
            dims.unsafe_free()
        _ = self._lib[].close_dataspace(sid)
        return result^

    def _dtype_code_of(self, name: String) raises -> Int:
        var aid = self._lib[].open_attr(self._loc_id, name)
        if aid < 0:
            raise Error("attrs: '" + name + "' not found")
        var tid = self._lib[].get_attr_type(aid)
        _ = self._lib[].close_attr(aid)
        if tid < 0:
            raise Error("attrs: H5Aget_type failed for '" + name + "'")
        var code = _infer_dtype_code(
            self._lib[],
            tid,
            self._lib[].get_type_class(tid),
            Int(self._lib[].get_type_size(tid)),
        )
        _ = self._lib[].close_type(tid)
        return code

    def dtype(self, name: String) raises -> String:
        """Datatype name of an attribute, e.g. ``"float64"``."""
        return _dtype_name(self._dtype_code_of(name))

    def get_value(self, name: String) raises -> AttrValue:
        """Read an attribute of any supported dtype/shape as an `AttrValue`."""
        var shape = self.shape(name)
        var code = self._dtype_code_of(name)
        var n = 1
        for d in shape:
            n *= d
        var nbytes = n * _dtype_itemsize(code)
        var raw = unsafe_alloc[UInt8](nbytes + 1)
        var aid = self._lib[].open_attr(self._loc_id, name)
        if aid < 0:
            raw.unsafe_free()
            raise Error("attrs: '" + name + "' not found")
        var rc = self._lib[].read_attr(
            aid,
            _type_id_for_code(self._lib[], code),
            raw.unsafe_bitcast[NoneType](),
        )
        _ = self._lib[].close_attr(aid)
        var data = List[UInt8](capacity=nbytes)
        for i in range(nbytes):
            data.append(raw[unsafe_offset=i])
        raw.unsafe_free()
        if rc < 0:
            raise Error("attrs: H5Aread failed for '" + name + "'")
        return AttrValue(_dtype_name(code), shape^, code, data^)

    def items(self) raises -> List[Tuple[String, AttrValue]]:
        """All `(name, value)` pairs, in `keys()` order."""
        var result = List[Tuple[String, AttrValue]]()
        for name in self.keys():
            result.append((name, self.get_value(name)))
        return result^

    def values(self) raises -> List[AttrValue]:
        """All attribute values, in `keys()` order."""
        var result = List[AttrValue]()
        for name in self.keys():
            result.append(self.get_value(name))
        return result^

    def read_array[
        dtype: DType
    ](self, name: String) raises -> NDArray[dtype] where SupportedDType[dtype]:
        """Read an array attribute (rank >= 1) into an NDArray.

        Raises:
            Error: If the attribute is missing or is a scalar attribute.
        """
        var shp = self.shape(name)
        if len(shp) == 0:
            raise Error("attrs: '" + name + "' is scalar; use read_scalar()")
        var arr = NDArray[dtype](Shape(shp))
        var aid = self._lib[].open_attr(self._loc_id, name)
        if aid < 0:
            raise Error("attrs: '" + name + "' not found")
        var rc = self._lib[].read_attr(
            aid,
            _hdf5_type_id[dtype](self._lib[]),
            arr.unsafe_ptr().unsafe_bitcast[NoneType](),
        )
        _ = self._lib[].close_attr(aid)
        if rc < 0:
            raise Error("attrs: H5Aread failed for '" + name + "'")
        return arr^

    def write_array[
        dtype: DType
    ](self, name: String, value: NDArray[dtype]) raises where SupportedDType[
        dtype
    ]:
        """Write an NDArray as an array attribute; overwrites an existing one.
        """
        var ndims = value.shape.ndim
        if ndims == 0:
            raise Error("attrs: write_array needs an array of rank >= 1")
        if self.__contains__(name):
            self.delete(name)
        var dims = unsafe_alloc[hsize_t](ndims)
        for i in range(ndims):
            dims[unsafe_offset=i] = hsize_t(value.shape[i])
        var sid = self._lib[].create_dataspace_nd(ndims, dims)
        dims.unsafe_free()
        if sid < 0:
            raise Error("attrs: dataspace creation failed")
        var rc = self._lib[].write_attr(
            self._loc_id,
            name,
            _hdf5_type_id[dtype](self._lib[]),
            sid,
            value.unsafe_ptr()
            .unsafe_bitcast[NoneType]()
            .unsafe_mut_cast[True](),
        )
        _ = self._lib[].close_dataspace(sid)
        if rc < 0:
            raise Error("attrs: write failed for '" + name + "'")

    def __len__(self) -> Int:
        """Number of attributes on this object (0 if the query fails)."""
        var num = self._lib[].get_num_attrs(self._loc_id)
        return Int(num) if num > 0 else 0

    def len(self) -> Int:
        """Number of attributes on this object."""
        return self.__len__()

    def __iter__(self) raises -> NameIterator:
        """Iterate over attribute names (same order as `keys()`)."""
        return NameIterator(self.keys())

    def keys(self) raises -> List[String]:
        """Get all attribute names.

        Returns:
            A list of all attribute names on this object.

        Raises:
            Error: If listing attributes fails.
        """
        var result = List[String]()
        var num = self._lib[].get_num_attrs(self._loc_id)
        if num < 0:
            return result^
        # Creation order when the object tracks it, otherwise name order.
        var by_crt = num > 0 and self._lib[].tracks_creation_order(
            self._loc_id, True
        )
        for i in range(Int(num)):
            var attr_name = self._lib[].get_attr_name_by_idx(
                self._loc_id, i, by_crt
            )
            result.append(attr_name)
        return result^

    def contains(self, name: String) -> Bool:
        """Check if an attribute exists.

        Args:
            name: Name of the attribute.

        Returns:
            True if the attribute exists, False otherwise.
        """
        return self.__contains__(name)

    def delete(self, name: String) raises:
        """Delete an attribute.

        Args:
            name: Name of the attribute to delete.

        Raises:
            Error: If the attribute cannot be deleted.
        """
        var rc = self._lib[].delete_attr(self._loc_id, name)
        if rc < 0:
            raise Error("attrs: cannot delete '" + name + "'")

    def get[
        dtype: DType
    ](self, name: String, default: Scalar[dtype]) raises -> Scalar[dtype]:
        """Get an attribute value, returning default if not found.

        Args:
            name: Name of the attribute.
            default: Value to return if attribute doesn't exist.

        Returns:
            The attribute value, or default if not found.
        """
        var aid = self._lib[].open_attr(self._loc_id, name)
        if aid < 0:
            return default
        var buf = unsafe_alloc[Scalar[dtype]](1)
        var rc = self._lib[].read_attr(
            aid,
            _hdf5_type_id[dtype](self._lib[]),
            buf.unsafe_bitcast[NoneType](),
        )
        _ = self._lib[].close_attr(aid)
        if rc < 0:
            buf.unsafe_free()
            return default
        var v = buf[unsafe_offset=0]
        buf.unsafe_free()
        return v

    def set[dtype: DType](self, name: String, value: Scalar[dtype]) raises:
        """Write a scalar attribute value.

        Parameters:
            dtype: The data type (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the attribute to write.
            value: The value to write.

        Raises:
            Error: If writing fails.
        """
        self.write_scalar[dtype](name, value)


# ===----------------------------------------------------------------------=== #
# Dataset
# ===----------------------------------------------------------------------=== #


struct Dataset[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](Copyable, Movable, Sized):
    """Proxy for an HDF5 dataset, similar to ``h5py.Dataset``.

    Represents an HDF5 dataset containing multidimensional array data.
    Datasets can be read from or written to, and support attributes.

    Example::

        var f = File("data.h5", "r")
        var obj = f.get("mydataset")
        if obj.is_dataset():
            var dset = obj.dataset()
            print(dset.shape())   # [100, 50]
            print(dset.dtype())   # "float64"
            print(dset.ndim())    # 2
            print(dset.size())    # 5000
            var arr = dset.read[DType.float64]()
            print(arr.get(0, 0))
            arr.unsafe_free()
        f.close()
    """

    var _did: hid_t
    var _shape: List[Int]
    var _dtype_code: Int
    var _name: String
    var _filename: String
    var _lib: LibRef[Self.origin]
    var _closed_bool: Bool

    def __init__(
        out self,
        lib: LibRef[Self.origin],
        did: hid_t,
        shape: List[Int],
        dtype_code: Int,
        name: String,
        filename: String = "",
    ):
        self._did = did
        self._shape = shape.copy()
        self._dtype_code = dtype_code
        self._name = name
        self._filename = filename
        self._lib = lib
        self._closed_bool = False

    def shape(self) -> List[Int]:
        """Get the shape of the dataset.

        Returns:
            A list of dimensions, e.g., [100, 50] for a 2D array.
        """
        return self._shape.copy()

    def ndim(self) -> Int:
        """Get the number of dimensions.

        Returns:
            The number of axes in the dataset.
        """
        return len(self._shape)

    def size(self) -> Int:
        """Get the total number of elements.

        Returns:
            The product of all shape dimensions.
        """
        var s: Int = 1
        for d in self._shape:
            s *= d
        return s

    def __len__(self) -> Int:
        """Size of the first axis (0 for a scalar dataset)."""
        if len(self._shape) == 0:
            return 0
        return self._shape[0]

    def nbytes(self) -> Int:
        """Total number of bytes the dataset occupies in memory when read."""
        return self.size() * _dtype_itemsize(self._dtype_code)

    # TODO: I might be able to use comptime condition/where to
    # make sure only these 4 dtypes are used.
    def dtype(self) -> String:
        """Get the HDF5 datatype as a string.

        Returns:
            One of: "float64", "float32", "int32", "int64", "int8",
            "uint8", "unknown".
        """
        return _dtype_name(self._dtype_code)

    def name(self) -> String:
        """Get the full path name of the dataset in the HDF5 file.

        Returns:
            The dataset path, e.g., "/group/subgroup/dataset".
        """
        return self._name

    def attrs(self) -> AttributeManager[Self.origin]:
        """Get the attribute manager for this dataset.

        Returns:
            An AttributeManager for reading/writing attributes.
        """
        return AttributeManager[Self.origin](self._lib, self._did)

    def chunks(self) -> List[Int]:
        """Get the chunk shape of the dataset.

        Returns:
            A list of chunk dimensions, or empty list if not chunked.
        """
        var dcpl = self._lib[].get_dcpl(self._did)
        if dcpl < 0:
            return List[Int]()
        var chunks = self._lib[].get_chunk_dims(dcpl, len(self._shape))
        _ = self._lib[].close_dcpl(dcpl)
        return chunks^

    def filter_count(self) -> Int:
        """Get the number of filters in the dataset creation pipeline."""
        var dcpl = self._lib[].get_dcpl(self._did)
        if dcpl < 0:
            return -1
        var count = self._lib[].get_num_filters(dcpl)
        _ = self._lib[].close_dcpl(dcpl)
        return count

    def filter_ids(self) -> List[Int]:
        """Get the HDF5 filter ids in the dataset creation pipeline."""
        var result = List[Int]()
        var dcpl = self._lib[].get_dcpl(self._did)
        if dcpl < 0:
            return result^
        var count = self._lib[].get_num_filters(dcpl)
        for i in range(count):
            var filter_id = self._lib[].get_filter_id(dcpl, i)
            if filter_id >= 0:
                result.append(filter_id)
        _ = self._lib[].close_dcpl(dcpl)
        return result^

    def filter_names(self) -> List[String]:
        """Get names for known HDF5 filters in the dataset pipeline."""
        var result = List[String]()
        var ids = self.filter_ids()
        for filter_id in ids:
            if filter_id == H5Z_FILTER_DEFLATE:
                result.append("gzip")
            elif filter_id == H5Z_FILTER_SHUFFLE:
                result.append("shuffle")
            elif filter_id == H5Z_FILTER_FLETCHER32:
                result.append("fletcher32")
            else:
                result.append("filter_" + String(filter_id))
        return result^

    def compression(self) -> String:
        """Get the compression filter name, or empty string if uncompressed."""
        var ids = self.filter_ids()
        for filter_id in ids:
            if filter_id == H5Z_FILTER_DEFLATE:
                return "gzip"
        return ""

    def compression_opts(self) -> Int:
        """Get gzip compression level, or -1 if gzip is not enabled."""
        var dcpl = self._lib[].get_dcpl(self._did)
        if dcpl < 0:
            return -1
        var count = self._lib[].get_num_filters(dcpl)
        for i in range(count):
            var filter_id = self._lib[].get_filter_id(dcpl, i)
            if filter_id == H5Z_FILTER_DEFLATE:
                var cd_values = self._lib[].get_filter_cd_values(dcpl, i)
                _ = self._lib[].close_dcpl(dcpl)
                if len(cd_values) > 0:
                    return cd_values[0]
                return -1
        _ = self._lib[].close_dcpl(dcpl)
        return -1

    def shuffle(self) -> Bool:
        """Return true when the shuffle filter is enabled."""
        var ids = self.filter_ids()
        for filter_id in ids:
            if filter_id == H5Z_FILTER_SHUFFLE:
                return True
        return False

    def fletcher32(self) -> Bool:
        """Return true when the Fletcher32 checksum filter is enabled."""
        var ids = self.filter_ids()
        for filter_id in ids:
            if filter_id == H5Z_FILTER_FLETCHER32:
                return True
        return False

    def fillvalue[dtype: DType](self) raises -> Scalar[dtype]:
        """Get the dataset fill value from its creation property list."""
        var dcpl = self._lib[].get_dcpl(self._did)
        if dcpl < 0:
            raise Error("Dataset: H5Dget_create_plist failed")
        var buf = unsafe_alloc[Scalar[dtype]](1)
        var rc = self._lib[].get_fill_value(
            dcpl,
            _hdf5_type_id[dtype](self._lib[]),
            buf.unsafe_bitcast[NoneType](),
        )
        _ = self._lib[].close_dcpl(dcpl)
        if rc < 0:
            buf.unsafe_free()
            raise Error(
                "Dataset: H5Pget_fill_value failed for '" + self._name + "'"
            )
        var value = buf[unsafe_offset=0]
        buf.unsafe_free()
        return value

    def maxshape(self) -> List[Int]:
        """Get the maximum shape of the dataset.

        Returns:
            A list of maximum dimensions from the dataset dataspace.
        """
        var sid = self._lib[].get_dataset_space(self._did)
        if sid < 0:
            return List[Int]()
        var ndims = self._lib[].get_space_ndims(sid)
        if ndims < 0:
            _ = self._lib[].close_dataspace(sid)
            return List[Int]()
        var dims = self._lib[].get_space_max_dims(sid, Int(ndims))
        var result = List[Int]()
        var unlimited = hsize_t(-1)
        for i in range(Int(ndims)):
            if dims[unsafe_offset=i] == unlimited:
                result.append(-1)
            else:
                result.append(Int(dims[unsafe_offset=i]))
        dims.unsafe_free()
        _ = self._lib[].close_dataspace(sid)
        return result^

    def resize(mut self, new_size: Int) raises:
        """Resize the dataset along axis 0.

        Only works if the dataset was created with maxshape allowing expansion.

        Args:
            new_size: The new size for axis 0.

        Raises:
            Error: If the resize operation fails.
        """
        var new_shape = self._shape.copy()
        if len(new_shape) == 0:
            raise Error(
                "Dataset: cannot resize scalar dataset '" + self._name + "'"
            )
        new_shape[0] = new_size
        self.resize(new_shape)

    def resize(mut self, new_shape: List[Int]) raises:
        """Resize the dataset to a new full shape.

        Only works if the dataset was created with maxshape allowing the new
        extent. The rank must match the current dataset rank.

        Args:
            new_shape: The new full dataset shape.

        Raises:
            Error: If the rank does not match or the resize operation fails.
        """
        if len(new_shape) != len(self._shape):
            raise Error(
                "Dataset: resize rank mismatch for '" + self._name + "'"
            )
        var rc = self._lib[].resize_dataset(self._did, new_shape)
        if rc < 0:
            raise Error("Dataset: resize failed for '" + self._name + "'")
        self._shape = new_shape.copy()

    def file(self) -> String:
        """Get the filename this dataset belongs to.

        Returns:
            The path to the containing HDF5 file.
        """
        var fid = self._lib[].get_file_id(self._did)
        var name = self._lib[].get_file_name(fid)
        _ = self._lib[].close_file(fid)
        if name != "":
            return name
        return self._filename

    def parent(self) -> String:
        """Get the parent group path of this dataset.

        Returns:
            The parent path, or "/" if at root level.
        """
        var last_slash = -1
        for i in range(self._name.byte_length()):
            if self._name[byte=i] == "/":
                last_slash = i
        if last_slash <= 0:
            return "/"
        return String(self._name[byte=:last_slash])

    def read[
        dtype: DType
    ](
        self, buf: Pointer[Scalar[dtype], MutExt], n: Int
    ) raises where SupportedDType[dtype]:
        """Read dataset data into a pre-allocated buffer.

        Parameters:
            dtype: The data type to read as (e.g., DType.float64, DType.int32).

        Args:
            buf: Pre-allocated buffer to read into.
            n: Number of elements to read.

        Raises:
            Error: If the read operation fails.

        Constraints:
            `dtype` must be a supported numeric dtype (see the README).
        """
        var rc = self._lib[].read_dataset(
            self._did,
            _hdf5_type_id[dtype](self._lib[]),
            buf.unsafe_bitcast[NoneType](),
        )
        if rc < 0:
            raise Error("Dataset: H5Dread failed for '" + self._name + "'")

    def write[
        dtype: DType
    ](
        self, data: Pointer[Scalar[dtype], MutExt], n: Int
    ) raises where SupportedDType[dtype]:
        """Write data from a buffer into the dataset.

        Parameters:
            dtype: The data type to write as (e.g., DType.float64, DType.int32).

        Args:
            data: Buffer containing the data to write.
            n: Number of elements to write.

        Raises:
            Error: If the write operation fails.
        """
        var rc = self._lib[].write_dataset(
            self._did,
            _hdf5_type_id[dtype](self._lib[]),
            data.unsafe_bitcast[NoneType](),
        )
        if rc < 0:
            raise Error("Dataset: H5Dwrite failed for '" + self._name + "'")

    def read_scalar[
        dtype: DType
    ](self) raises -> Scalar[dtype] where SupportedDType[dtype]:
        """Read a scalar dataset value."""
        if len(self._shape) != 0:
            raise Error("Dataset: not a scalar dataset '" + self._name + "'")
        var buf = unsafe_alloc[Scalar[dtype]](1)
        self.read[dtype](buf, 1)
        var value = buf[unsafe_offset=0]
        buf.unsafe_free()
        return value

    def write_scalar[
        dtype: DType
    ](self, value: Scalar[dtype]) raises where SupportedDType[dtype]:
        """Write a scalar dataset value."""
        if len(self._shape) != 0:
            raise Error("Dataset: not a scalar dataset '" + self._name + "'")
        var buf = unsafe_alloc[Scalar[dtype]](1)
        buf[unsafe_offset=0] = value
        self.write[dtype](buf, 1)
        buf.unsafe_free()

    def __getitem__[
        dtype: DType
    ](self, index: Tuple[]) raises -> Scalar[dtype] where SupportedDType[dtype]:
        """Read a scalar dataset value via h5py-style empty-tuple indexing.

        Mirrors ``h5py``'s ``dset[()]`` for scalar datasets.

        Args:
            index: Must be the empty tuple `()`.
        """
        return self.read_scalar[dtype]()

    def __setitem__[
        dtype: DType
    ](
        mut self, index: Tuple[], value: Scalar[dtype]
    ) raises where SupportedDType[dtype]:
        """Write a scalar dataset value via h5py-style empty-tuple indexing.

        Mirrors ``h5py``'s ``dset[()] = value`` for scalar datasets.

        Args:
            index: Must be the empty tuple `()`.
            value: The scalar value to write.
        """
        self.write_scalar[dtype](value)

    def read_hyperslab[
        dtype: DType
    ](
        self,
        start: List[Int],
        count: List[Int],
        buf: Pointer[Scalar[dtype], MutExt],
    ) raises where SupportedDType[dtype]:
        """Read a contiguous hyperslab into a compact memory buffer.

        ``start`` and ``count`` must have one entry per dataset dimension.
        """
        var ndims = len(self._shape)
        if len(start) != ndims or len(count) != ndims:
            raise Error(
                "Dataset: hyperslab rank mismatch for '" + self._name + "'"
            )
        var start_dims = unsafe_alloc[hsize_t](ndims)
        var count_dims = unsafe_alloc[hsize_t](ndims)
        for i in range(ndims):
            if (
                start[i] < 0
                or count[i] < 0
                or start[i] + count[i] > self._shape[i]
            ):
                start_dims.unsafe_free()
                count_dims.unsafe_free()
                raise Error(
                    "Dataset: hyperslab out of bounds for '" + self._name + "'"
                )
            start_dims[unsafe_offset=i] = hsize_t(start[i])
            count_dims[unsafe_offset=i] = hsize_t(count[i])

        var file_space = self._lib[].get_dataset_space(self._did)
        if file_space < 0:
            start_dims.unsafe_free()
            count_dims.unsafe_free()
            raise Error("Dataset: H5Dget_space failed for '" + self._name + "'")
        var rc = self._lib[].select_hyperslab(
            file_space, start_dims, count_dims, ndims
        )
        start_dims.unsafe_free()
        if rc < 0:
            count_dims.unsafe_free()
            _ = self._lib[].close_dataspace(file_space)
            raise Error(
                "Dataset: H5Sselect_hyperslab failed for '" + self._name + "'"
            )

        var mem_space = self._lib[].create_dataspace_nd(ndims, count_dims)
        count_dims.unsafe_free()
        if mem_space < 0:
            _ = self._lib[].close_dataspace(file_space)
            raise Error("Dataset: memory dataspace creation failed")

        rc = self._lib[].read_dataset_selection(
            self._did,
            _hdf5_type_id[dtype](self._lib[]),
            mem_space,
            file_space,
            buf.unsafe_bitcast[NoneType](),
        )
        _ = self._lib[].close_dataspace(mem_space)
        _ = self._lib[].close_dataspace(file_space)
        if rc < 0:
            raise Error(
                "Dataset: H5Dread hyperslab failed for '" + self._name + "'"
            )

    def write_hyperslab[
        dtype: DType
    ](
        self,
        start: List[Int],
        count: List[Int],
        data: Pointer[Scalar[dtype], MutExt],
    ) raises where SupportedDType[dtype]:
        """Write a compact memory buffer into a contiguous hyperslab."""
        var ndims = len(self._shape)
        if len(start) != ndims or len(count) != ndims:
            raise Error(
                "Dataset: hyperslab rank mismatch for '" + self._name + "'"
            )
        var start_dims = unsafe_alloc[hsize_t](ndims)
        var count_dims = unsafe_alloc[hsize_t](ndims)
        for i in range(ndims):
            if (
                start[i] < 0
                or count[i] < 0
                or start[i] + count[i] > self._shape[i]
            ):
                start_dims.unsafe_free()
                count_dims.unsafe_free()
                raise Error(
                    "Dataset: hyperslab out of bounds for '" + self._name + "'"
                )
            start_dims[unsafe_offset=i] = hsize_t(start[i])
            count_dims[unsafe_offset=i] = hsize_t(count[i])

        var file_space = self._lib[].get_dataset_space(self._did)
        if file_space < 0:
            start_dims.unsafe_free()
            count_dims.unsafe_free()
            raise Error("Dataset: H5Dget_space failed for '" + self._name + "'")
        var rc = self._lib[].select_hyperslab(
            file_space, start_dims, count_dims, ndims
        )
        start_dims.unsafe_free()
        if rc < 0:
            count_dims.unsafe_free()
            _ = self._lib[].close_dataspace(file_space)
            raise Error(
                "Dataset: H5Sselect_hyperslab failed for '" + self._name + "'"
            )

        var mem_space = self._lib[].create_dataspace_nd(ndims, count_dims)
        count_dims.unsafe_free()
        if mem_space < 0:
            _ = self._lib[].close_dataspace(file_space)
            raise Error("Dataset: memory dataspace creation failed")

        rc = self._lib[].write_dataset_selection(
            self._did,
            _hdf5_type_id[dtype](self._lib[]),
            mem_space,
            file_space,
            data.unsafe_bitcast[NoneType](),
        )
        _ = self._lib[].close_dataspace(mem_space)
        _ = self._lib[].close_dataspace(file_space)
        if rc < 0:
            raise Error(
                "Dataset: H5Dwrite hyperslab failed for '" + self._name + "'"
            )

    def _slice_count(
        self, start: List[Int], stop: List[Int]
    ) raises -> List[Int]:
        var ndims = len(self._shape)
        if len(start) != ndims or len(stop) != ndims:
            raise Error("Dataset: slice rank mismatch for '" + self._name + "'")
        var count = List[Int]()
        for i in range(ndims):
            if start[i] < 0 or stop[i] < start[i] or stop[i] > self._shape[i]:
                raise Error(
                    "Dataset: slice out of bounds for '" + self._name + "'"
                )
            count.append(stop[i] - start[i])
        return count^

    def _slice_starts(self, slices: List[Slice]) raises -> List[Int]:
        var ndims = len(self._shape)
        if len(slices) != ndims:
            raise Error("Dataset: slice rank mismatch for '" + self._name + "'")
        var start = List[Int]()
        for i in range(ndims):
            var axis_slice = slices[i]
            if axis_slice.step and axis_slice.step.value() != 1:
                raise Error("Dataset: slice step is not supported")
            if axis_slice.start:
                start.append(axis_slice.start.value())
            else:
                start.append(0)
        return start^

    def _slice_stops(self, slices: List[Slice]) raises -> List[Int]:
        var ndims = len(self._shape)
        if len(slices) != ndims:
            raise Error("Dataset: slice rank mismatch for '" + self._name + "'")
        var stop = List[Int]()
        for i in range(ndims):
            var axis_slice = slices[i]
            if axis_slice.step and axis_slice.step.value() != 1:
                raise Error("Dataset: slice step is not supported")
            if axis_slice.end:
                stop.append(axis_slice.end.value())
            else:
                stop.append(self._shape[i])
        return stop^

    def read_slice[
        dtype: DType
    ](self, start: List[Int], stop: List[Int]) raises -> NDArray[
        dtype
    ] where SupportedDType[dtype]:
        """Read a simple rectangular slice into a compact NDArray."""
        var count = self._slice_count(start, stop)
        if len(count) == 0:
            raise Error("Dataset: cannot slice scalar '" + self._name + "'")
        var arr = NDArray[dtype](Shape(count))
        self.read_hyperslab[dtype](
            start, count, arr.unsafe_ptr().unsafe_origin_cast[MutExt]()
        )
        return arr^

    def read_slice[
        dtype: DType
    ](self, slices: List[Slice]) raises -> NDArray[dtype] where SupportedDType[
        dtype
    ]:
        """Read a simple rectangular slice described by Mojo Slice values."""
        var start = self._slice_starts(slices)
        var stop = self._slice_stops(slices)
        return self.read_slice[dtype](start, stop)

    def write_slice[
        dtype: DType
    ](
        self,
        start: List[Int],
        stop: List[Int],
        data: Pointer[Scalar[dtype], MutExt],
    ) raises where SupportedDType[dtype]:
        """Write a compact buffer into a simple rectangular slice."""
        var count = self._slice_count(start, stop)
        self.write_hyperslab[dtype](start, count, data)

    def write_slice[
        dtype: DType
    ](
        self,
        slices: List[Slice],
        data: Pointer[Scalar[dtype], MutExt],
    ) raises where SupportedDType[dtype]:
        """Write a compact buffer into a slice described by Mojo Slice values.
        """
        var start = self._slice_starts(slices)
        var stop = self._slice_stops(slices)
        self.write_slice[dtype](start, stop, data)

    def read_point[
        dtype: DType
    ](self, indices: List[Int]) raises -> Scalar[dtype] where SupportedDType[
        dtype
    ]:
        """Read a single scalar value at integer indices."""
        var ndims = len(self._shape)
        if len(indices) != ndims:
            raise Error("Dataset: point rank mismatch for '" + self._name + "'")
        var count = List[Int]()
        for _ in range(ndims):
            count.append(1)
        var buf = unsafe_alloc[Scalar[dtype]](1)
        self.read_hyperslab[dtype](indices, count, buf)
        var value = buf[unsafe_offset=0]
        buf.unsafe_free()
        return value

    def write_point[
        dtype: DType
    ](
        self, indices: List[Int], value: Scalar[dtype]
    ) raises where SupportedDType[dtype]:
        """Write a single scalar value at integer indices."""
        var ndims = len(self._shape)
        if len(indices) != ndims:
            raise Error("Dataset: point rank mismatch for '" + self._name + "'")
        var count = List[Int]()
        for _ in range(ndims):
            count.append(1)
        var buf = unsafe_alloc[Scalar[dtype]](1)
        buf[unsafe_offset=0] = value
        self.write_hyperslab[dtype](indices, count, buf)
        buf.unsafe_free()

    def read[
        dtype: DType
    ](self) raises -> NDArray[dtype] where SupportedDType[dtype]:
        """Read the entire dataset into an NDArray of any rank (>= 1).

        Parameters:
            dtype: The data type to read as (e.g., DType.float64, DType.int32).

        Returns:
            An NDArray containing the dataset data.

        Raises:
            Error: If the dataset is scalar (use `read_scalar`) or the read
                fails.
        """
        if len(self._shape) == 0:
            raise Error(
                "Dataset: '" + self._name + "' is scalar; use read_scalar()"
            )
        var arr = NDArray[dtype](Shape(self._shape))
        self.read_direct[dtype](arr)
        return arr^

    def read_direct[
        dtype: DType
    ](self, mut dest: NDArray[dtype]) raises where SupportedDType[dtype]:
        """Read the whole dataset into an existing NDArray (no new allocation).

        `dest` must hold exactly as many elements as the dataset.
        """
        if dest.size != self.size():
            raise Error(
                "Dataset: read_direct destination has "
                + String(dest.size)
                + " elements, dataset '"
                + self._name
                + "' has "
                + String(self.size())
            )
        var rc = self._lib[].read_dataset(
            self._did,
            _hdf5_type_id[dtype](self._lib[]),
            dest.unsafe_ptr().unsafe_bitcast[NoneType](),
        )
        if rc < 0:
            raise Error("Dataset: H5Dread failed for '" + self._name + "'")

    def read_direct[
        dtype: DType
    ](
        self, mut dest: NDArray[dtype], source_sel: List[Slice]
    ) raises where SupportedDType[dtype]:
        """Read a rectangular selection straight into an existing NDArray.

        `dest` must hold exactly as many elements as the selection.
        """
        var start = self._slice_starts(source_sel)
        var count = self._slice_count(start, self._slice_stops(source_sel))
        var n = 1
        for c in count:
            n *= c
        if dest.size != n:
            raise Error(
                "Dataset: read_direct destination has "
                + String(dest.size)
                + " elements, selection has "
                + String(n)
            )
        self.read_hyperslab[dtype](
            start, count, dest.unsafe_ptr().unsafe_origin_cast[MutExt]()
        )

    def write_direct[
        dtype: DType
    ](self, source: NDArray[dtype]) raises where SupportedDType[dtype]:
        """Write an NDArray to the dataset; sizes must match."""
        if source.size != self.size():
            raise Error(
                "Dataset: write_direct source has "
                + String(source.size)
                + " elements, dataset '"
                + self._name
                + "' has "
                + String(self.size())
            )
        self.write[dtype](source)

    def write_direct[
        dtype: DType
    ](
        self, source: NDArray[dtype], dest_sel: List[Slice]
    ) raises where SupportedDType[dtype]:
        """Write an NDArray into a rectangular selection of the dataset."""
        var start = self._slice_starts(dest_sel)
        var count = self._slice_count(start, self._slice_stops(dest_sel))
        var n = 1
        for c in count:
            n *= c
        if source.size != n:
            raise Error(
                "Dataset: write_direct source has "
                + String(source.size)
                + " elements, selection has "
                + String(n)
            )
        self.write_hyperslab[dtype](
            start,
            count,
            source.unsafe_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutExt](),
        )

    def write[
        dtype: DType
    ](self, data: NDArray[dtype]) raises where SupportedDType[dtype]:
        """Write entire NDArray contents to the dataset.

        Parameters:
            dtype: The data type to write as (e.g., DType.float64, DType.int32).

        Args:
            data: NuMojo NDArray containing the data to write.

        Raises:
            Error: If the write operation fails.
        """
        # var total = data.size
        var ptr = data.unsafe_ptr()
        var rc = self._lib[].write_dataset(
            self._did,
            _hdf5_type_id[dtype](self._lib[]),
            ptr.unsafe_bitcast[NoneType]().unsafe_mut_cast[True](),
        )
        # TODO: check if this mut cast is correct solution or not for mojo v1.0 error.
        if rc < 0:
            raise Error("Dataset: H5Dwrite failed for '" + self._name + "'")

    def close(mut self):
        if not self._closed_bool and self._did >= 0:
            _ = self._lib[].close_dataset(self._did)
            self._did = -1
            self._closed_bool = True


# ===----------------------------------------------------------------------=== #
# Group
# ===----------------------------------------------------------------------=== #


struct Group[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](Copyable, Movable):
    """HDF5 group with dict-like interface, similar to ``h5py.Group``.

    Groups are container structures that can hold datasets and other groups,
    forming a hierarchical filesystem-like structure within an HDF5 file.

    Example::

        var f = File("data.h5", "r")
        var grp = f.get("mygroup")
        if grp.is_group():
            var g = grp.group()
            for name in g.keys():
                print(name)
            # Read a dataset within the group
            var obj = g.get("data")
            if obj.is_dataset():
                var dset = obj.dataset()
                var arr = dset.read[DType.float64]()
                arr.unsafe_free()
        f.close()
    """

    var _gid: hid_t
    var _name: String
    var _filename: String
    var _lib: LibRef[Self.origin]
    var _is_file: Bool
    var _closed_bool: Bool

    def __init__(
        out self,
        lib: LibRef[Self.origin],
        gid: hid_t,
        name: String,
        is_file: Bool = False,
        filename: String = "",
    ):
        self._gid = gid
        self._name = name
        self._filename = filename
        self._lib = lib
        self._is_file = is_file
        self._closed_bool = False

    def name(self) -> String:
        """Get the full path name of the group in the HDF5 file.

        Returns:
            The group path, e.g., "/group/subgroup".
        """
        return self._name

    def attrs(self) -> AttributeManager[Self.origin]:
        """Get the attribute manager for this group.

        Returns:
            An AttributeManager for reading/writing attributes.
        """
        return AttributeManager[Self.origin](self._lib, self._gid)

    def file(self) -> String:
        """Get the filename this group belongs to.

        Returns:
            The path to the containing HDF5 file.
        """
        var fid = self._lib[].get_file_id(self._gid)
        var name = self._lib[].get_file_name(fid)
        _ = self._lib[].close_file(fid)
        if name != "":
            return name
        return self._filename

    def parent(self) -> String:
        """Get the parent group path.

        Returns:
            The parent path, or "/" if this is the root group.
        """
        if self._name == "/":
            return "/"
        var last_slash = -1
        for i in range(self._name.byte_length()):
            if self._name[byte=i] == "/":
                last_slash = i
        if last_slash <= 0:
            return "/"
        return String(self._name[byte=:last_slash])

    def close(mut self):
        """Close the group and release resources.

        For File objects, prefer calling File.close() instead.
        """
        if self._closed_bool:
            return
        if not self._is_file and self._gid >= 0:
            _ = self._lib[].close_group(self._gid)
        self._gid = -1
        self._closed_bool = True

    def __contains__(self, member_name: String) -> Bool:
        """Check if a member (group or dataset) exists.

        Args:
            member_name: Name of the member to check.

        Returns:
            True if the member exists, False otherwise.
        """
        return self._lib[].object_exists(self._gid, member_name)

    def contains(self, member_name: String) -> Bool:
        """Check if a member exists.

        Args:
            member_name: Name of the member to check.

        Returns:
            True if the member exists, False otherwise.
        """
        return self.__contains__(member_name)

    def get(self, member_name: String) raises -> H5Object[Self.origin]:
        """Get a member by name.

        Args:
            member_name: Name of the member to retrieve.

        Returns:
            An H5Object wrapping the group or dataset.

        Raises:
            Error: If the member does not exist or cannot be opened.
        """
        return self.__getitem__(member_name)

    def get_opt(
        self, member_name: String
    ) raises -> Optional[H5Object[Self.origin]]:
        """Get a member by name without raising if it is absent.

        Mirrors the fallback behavior of ``h5py.Group.get(name,
        default=...)``: unlike `get`/`__getitem__`, this returns `None`
        instead of raising when `member_name` does not exist.

        Args:
            member_name: Name of the member to retrieve.

        Returns:
            `Some(H5Object)` if the member exists, `None` otherwise.

        Raises:
            Error: If the member exists but cannot be opened.
        """
        if not self.__contains__(member_name):
            var empty: Optional[H5Object[Self.origin]] = None
            return empty^
        return self.__getitem__(member_name)

    def delete(self, member_name: String) raises:
        """Delete a member (group or dataset).

        Args:
            member_name: Name of the member to delete.

        Raises:
            Error: If deletion fails.
        """
        var rc = self._lib[].delete_object(self._gid, member_name)
        if rc < 0:
            raise Error("Group: cannot delete '" + member_name + "'")

    def len(self) raises -> Int:
        """Get the number of members in this group.

        Returns:
            The number of direct children.

        Raises:
            Error: If counting fails.
        """
        return self.__len__()

    def keys(self) raises -> List[String]:
        """Get names of all members in this group.

        Returns:
            A list of member names.

        Raises:
            Error: If listing fails.
        """
        return self._get_member_names().copy()

    def __len__(self) raises -> Int:
        """Get the number of members in this group.

        Returns:
            The number of direct children.

        Raises:
            Error: If counting fails.
        """
        return len(self._get_member_names())

    def __iter__(self) raises -> NameIterator:
        """Iterate over member names (same order as `keys()`)."""
        return NameIterator(self.keys())

    def items(self) raises -> List[String]:
        """Get names of all members (alias for keys()).

        Returns:
            A list of member names.

        Raises:
            Error: If listing fails.
        """
        return self._get_member_names()

    def values(self) raises -> List[H5Object[Self.origin]]:
        """Get all member objects (groups and datasets).

        Returns:
            A list of H5Object wrappers.

        Raises:
            Error: If listing fails.
        """
        var result = List[H5Object[Self.origin]]()
        var names = self._get_member_names()
        for name in names:
            result.append(self[name])
        return result^

    def _get_member_names(self) raises -> List[String]:
        var result = List[String]()
        var idx = 0
        # Creation order when the group tracks it, otherwise name order.
        var by_crt = self._lib[].tracks_creation_order(self._gid, False)
        while True:
            var member_name = self._lib[].get_link_name_by_idx(
                self._gid, idx, by_crt
            )
            if member_name == "":
                break
            result.append(member_name)
            idx += 1
        return result^

    def __getitem__(self, member_name: String) raises -> H5Object[Self.origin]:
        """Get a member (group or dataset) by name.

        Args:
            member_name: Name of the member to retrieve.

        Returns:
            An H5Object wrapping the group or dataset.

        Raises:
            Error: If the member does not exist or cannot be opened.
        """
        if not self._lib[].object_exists(self._gid, member_name):
            raise Error("Group: '" + member_name + "' not found")
        var oid = self._lib[].open_object(self._gid, member_name)
        if oid < 0:
            raise Error("Group: '" + member_name + "' not found")
        var otype = self._lib[].get_object_type(oid)
        _ = self._lib[].close_object(oid)

        if otype == H5I_GROUP:
            var gid = self._lib[].open_group(self._gid, member_name)
            if gid < 0:
                raise Error("Group: cannot open '" + member_name + "'")
            var full_name = self._name
            if full_name == "/":
                full_name = "/" + member_name
            else:
                full_name = full_name + "/" + member_name
            var g = Group(self._lib, gid, full_name, filename=self._filename)
            return H5Object[Self.origin](g^)

        if otype == H5I_DATASET:
            var dset = self._open_dataset(member_name)
            return H5Object[Self.origin](dset^)

        raise Error("Group: '" + member_name + "' not found")

    def _open_dataset(self, member_name: String) raises -> Dataset[Self.origin]:
        """Open and return a dataset by name."""
        var did = self._lib[].open_dataset(self._gid, member_name)
        if did < 0:
            raise Error("Dataset: cannot open '" + member_name + "'")

        var sid = self._lib[].get_dataset_space(did)
        var ndims = self._lib[].get_space_ndims(sid)
        var dims = self._lib[].get_space_dims(sid, Int(ndims))

        var shape = List[Int]()
        for i in range(Int(ndims)):
            shape.append(Int(dims[unsafe_offset=i]))
        dims.unsafe_free()
        _ = self._lib[].close_dataspace(sid)

        var tid = self._lib[].get_dataset_type(did)
        var tclass = self._lib[].get_type_class(tid)
        var tsize = Int(self._lib[].get_type_size(tid))
        var dtype_code = _infer_dtype_code(self._lib[], tid, tclass, tsize)
        _ = self._lib[].close_type(tid)

        var full_name = self._name
        if full_name == "/":
            full_name = "/" + member_name
        else:
            full_name = full_name + "/" + member_name

        return Dataset[Self.origin](
            self._lib, did, shape, dtype_code, full_name, self._filename
        )

    def create_group(
        self, name: String, track_order: Bool = False
    ) raises -> Group[Self.origin]:
        """Create a group, including any intermediate groups in the path.

        Args:
            name: Name or path of the group to create.
                Can be a simple name ("mygroup") or nested path ("a/b/c").
            track_order: Remember creation order of members and attributes,
                so `keys()` lists them in the order they were created.

        Returns:
            The created Group object.

        Raises:
            Error: If creation fails.
        """
        var parts = name.split("/")
        var current_gid = self._gid
        var current_path = self._name

        var i_start: Int = 0
        if len(parts) > 0 and parts[0] == "":
            i_start = 1
            current_path = "/"

        if i_start >= len(parts):
            return Group[Self.origin](
                self._lib,
                current_gid,
                current_path,
                self._is_file,
                self._filename,
            )

        var prev_gid: hid_t = -1
        for i in range(i_start, len(parts)):
            var part = String(parts[i])
            if part == "":
                continue
            var child_gid = self._lib[].open_group(current_gid, part)
            if child_gid < 0:
                child_gid = self._lib[].create_group(
                    current_gid, part, track_order
                )
                if child_gid < 0:
                    raise Error(
                        "Group.create_group: cannot create '" + part + "'"
                    )
            if prev_gid >= 0 and prev_gid != self._gid:
                _ = self._lib[].close_group(prev_gid)
            prev_gid = current_gid
            current_gid = child_gid
            if current_path == "/":
                current_path = "/" + part
            else:
                current_path = current_path + "/" + part

        return Group[Self.origin](
            self._lib, current_gid, current_path, filename=self._filename
        )

    def require_group(self, name: String) raises -> Group[Self.origin]:
        """Open an existing group or create it if it doesn't exist.

        Args:
            name: Name or path of the group.

        Returns:
            The Group object (existing or newly created).

        Raises:
            Error: If the path exists but is not a group, or creation fails.
        """
        if self._lib[].object_exists(self._gid, name):
            var gid = self._lib[].open_group(self._gid, name)
            if gid < 0:
                raise Error("Group: cannot open '" + name + "'")
            var full_name = self._name
            if full_name == "/":
                full_name = "/" + name
            else:
                full_name = full_name + "/" + name
            return Group[Self.origin](
                self._lib, gid, full_name, filename=self._filename
            )
        return self.create_group(name)

    def _create_dataspace_for_shape(self, shape: List[Int]) -> hid_t:
        var ndims = len(shape)
        if ndims == 0:
            return self._lib[].create_dataspace_scalar()
        var dims = unsafe_alloc[hsize_t](ndims)
        for i in range(ndims):
            dims[unsafe_offset=i] = hsize_t(shape[i])
        var sid = self._lib[].create_dataspace_nd(ndims, dims)
        dims.unsafe_free()
        return sid

    def create_dataset[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        fillvalue: Optional[Scalar[dtype]] = None,
        chunks: List[Int] = [],
        maxshape: List[Int] = [],
        compression: String = "",
        compression_opts: Int = 4,
        shuffle: Bool = False,
        fletcher32: Bool = False,
        track_order: Bool = False,
    ) raises -> Dataset[Self.origin] where SupportedDType[dtype]:
        """Create a new dataset (h5py-style ``create_dataset``).

        Chunking is enabled when any of ``chunks``, ``maxshape``,
        ``compression``, ``shuffle`` or ``fletcher32`` is given; if ``chunks``
        is omitted a chunk shape is guessed from the dataset shape.

        Parameters:
            dtype: The data type for the dataset (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the dataset to create.
            shape: List of dimensions, e.g., [100] or [10, 20]; `[]` creates
                a scalar dataset.
            fillvalue: Optional fill value for unwritten elements.
            chunks: Chunk dimensions (empty = automatic when chunking).
            maxshape: Maximum dimensions, ``-1`` = unlimited (empty = fixed).
            compression: ``"gzip"`` or ``""`` (no compression).
            compression_opts: Level of gzip, 0..9 (default 4).
            shuffle: Enable the shuffle filter.
            fletcher32: Enable the fletcher32 checksum filter.
            track_order: Remember the creation order of the dataset's
                attributes (so `attrs().keys()` lists them in that order).

        Returns:
            The created Dataset object.

        Raises:
            Error: If arguments are inconsistent or dataset creation fails.
        """
        var ndims = len(shape)
        if compression != "" and compression != "gzip":
            raise Error(
                "create_dataset: unsupported compression '" + compression + "'"
            )
        if compression_opts < 0 or compression_opts > 9:
            raise Error("create_dataset: gzip level must be 0..9")
        var filtered = compression != "" or shuffle or fletcher32
        var has_chunks = len(chunks) > 0
        var has_maxshape = len(maxshape) > 0
        var chunked = has_chunks or has_maxshape or filtered
        if chunked and ndims == 0:
            raise Error(
                "create_dataset: scalar datasets cannot be chunked or filtered"
            )
        if has_chunks and len(chunks) != ndims:
            raise Error(
                "create_dataset: chunks rank mismatch for '" + name + "'"
            )
        if has_maxshape and len(maxshape) != ndims:
            raise Error(
                "create_dataset: maxshape rank mismatch for '" + name + "'"
            )

        var sid: hid_t
        if has_maxshape:
            var dims = unsafe_alloc[hsize_t](ndims)
            var maxdims = unsafe_alloc[hsize_t](ndims)
            for i in range(ndims):
                dims[unsafe_offset=i] = hsize_t(shape[i])
                if maxshape[i] < 0:
                    maxdims[unsafe_offset=i] = hsize_t(-1)
                else:
                    maxdims[unsafe_offset=i] = hsize_t(maxshape[i])
            sid = self._lib[].create_dataspace_nd_max(ndims, dims, maxdims)
            dims.unsafe_free()
            maxdims.unsafe_free()
        else:
            sid = self._create_dataspace_for_shape(shape)
        if sid < 0:
            raise Error("create_dataset: dataspace creation failed")

        var tid = _hdf5_type_id[dtype](self._lib[])
        var did: hid_t
        if not chunked and not fillvalue and not track_order:
            did = self._lib[].create_dataset(self._gid, name, tid, sid)
        else:
            var dcpl = self._lib[].create_dcpl()
            if dcpl < 0:
                _ = self._lib[].close_dataspace(sid)
                raise Error("create_dataset: H5Pcreate failed")
            var err = String()
            if chunked:
                var chunk_shape = List[Int]()
                if has_chunks:
                    chunk_shape = chunks.copy()
                else:
                    chunk_shape = _guess_chunks(
                        shape, maxshape, size_of[Scalar[dtype]]()
                    )
                var chunk_dims = unsafe_alloc[hsize_t](ndims)
                for i in range(ndims):
                    chunk_dims[unsafe_offset=i] = hsize_t(chunk_shape[i])
                if self._lib[].set_chunk(dcpl, chunk_dims, ndims) < 0:
                    err = "H5Pset_chunk failed"
                chunk_dims.unsafe_free()
            if err == "" and shuffle:
                if self._lib[].set_shuffle(dcpl) < 0:
                    err = "H5Pset_shuffle failed"
            if err == "" and compression == "gzip":
                if self._lib[].set_deflate(dcpl, compression_opts) < 0:
                    err = "H5Pset_deflate failed"
            if err == "" and fletcher32:
                if self._lib[].set_fletcher32(dcpl) < 0:
                    err = "H5Pset_fletcher32 failed"
            if err == "" and track_order:
                if not self._lib[].set_track_order(dcpl):
                    err = "H5Pset_attr_creation_order failed"
            if err == "" and fillvalue:
                var buf = unsafe_alloc[Scalar[dtype]](1)
                buf[unsafe_offset=0] = fillvalue.value()
                if (
                    self._lib[].set_fill_value(
                        dcpl, tid, buf.unsafe_bitcast[NoneType]()
                    )
                    < 0
                ):
                    err = "H5Pset_fill_value failed"
                buf.unsafe_free()
            if err != "":
                _ = self._lib[].close_dcpl(dcpl)
                _ = self._lib[].close_dataspace(sid)
                raise Error("create_dataset: " + err)
            did = self._lib[].create_dataset_with_dcpl(
                self._gid, name, tid, sid, dcpl
            )
            _ = self._lib[].close_dcpl(dcpl)
        _ = self._lib[].close_dataspace(sid)
        if did < 0:
            raise Error("create_dataset: H5Dcreate2 failed for '" + name + "'")

        var full_name = self._name
        if full_name == "/":
            full_name = "/" + name
        else:
            full_name = full_name + "/" + name

        return Dataset[Self.origin](
            self._lib,
            did,
            shape.copy(),
            _dtype_code[dtype](),
            full_name,
            self._filename,
        )

    def create_scalar_dataset[
        dtype: DType
    ](
        self,
        name: String,
        value: Scalar[dtype],
    ) raises -> Dataset[
        Self.origin
    ] where SupportedDType[dtype]:
        """Create a scalar dataset and write its single value."""
        var shape = List[Int]()
        var dset = self.create_dataset[dtype](name, shape)
        dset.write_scalar[dtype](value)
        return dset^

    def create_dataset_chunked[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        maxshape: List[Int],
        chunks: List[Int],
    ) raises -> Dataset[Self.origin] where SupportedDType[dtype]:
        """Create a chunked dataset; prefer ``create_dataset(chunks=, maxshape=)``.

        Use ``-1`` in ``maxshape`` for an unlimited dimension.
        """
        return self.create_dataset[dtype](
            name, shape, chunks=chunks.copy(), maxshape=maxshape.copy()
        )

    def create_dataset_chunked[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        maxshape: List[Int],
        chunks: List[Int],
        fillvalue: Scalar[dtype],
    ) raises -> Dataset[Self.origin] where SupportedDType[dtype]:
        """Chunked dataset with a fill value; prefer ``create_dataset``."""
        return self.create_dataset[dtype](
            name,
            shape,
            fillvalue=fillvalue,
            chunks=chunks.copy(),
            maxshape=maxshape.copy(),
        )

    def create_dataset_filtered[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        maxshape: List[Int],
        chunks: List[Int],
        compression: String,
        compression_opts: Int,
        shuffle: Bool,
        fletcher32: Bool,
    ) raises -> Dataset[Self.origin] where SupportedDType[dtype]:
        """Chunked dataset with built-in filters; prefer ``create_dataset``."""
        return self.create_dataset[dtype](
            name,
            shape,
            chunks=chunks.copy(),
            maxshape=maxshape.copy(),
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            fletcher32=fletcher32,
        )

    def create_dataset_with_data[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        data: Pointer[Scalar[dtype], MutExt],
    ) raises -> Dataset[Self.origin] where SupportedDType[dtype]:
        """Create a dataset and write data to it.

        Parameters:
            dtype: The data type for the dataset (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the dataset to create.
            shape: List of dimensions, e.g., [100] or [10, 20].
            data: Buffer containing the data to write.

        Returns:
            The created Dataset object with data written.

        Raises:
            Error: If creation or writing fails.
        """
        var dset = self.create_dataset[dtype](name, shape)
        var total: Int = 1
        var shp = dset.shape()
        for d in shp:
            total *= d
        dset.write[dtype](data, total)
        return dset^

    def create_dataset_with_data[
        dtype: DType
    ](
        self,
        name: String,
        data: NDArray[dtype],
    ) raises -> Dataset[
        Self.origin
    ] where SupportedDType[dtype]:
        """Create a dataset and write data from NuMojo NDArray.

        Parameters:
            dtype: The data type for the dataset (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the dataset to create.
            data: NuMojo NDArray containing the data.

        Returns:
            The created Dataset object with data written.

        Raises:
            Error: If creation or writing fails.
        """
        var shape = List[Int]()
        var shp = data.shape
        for i in range(shp.ndim):
            shape.append(shp[i])
        var dset = self.create_dataset[dtype](name, shape)
        dset.write[dtype](data)
        return dset^

    def require_dataset[
        dtype: DType
    ](self, name: String, shape: List[Int],) raises -> Dataset[Self.origin]:
        """Open an existing dataset or create a new one if it doesn't exist.

        If the dataset exists, returns it. If not, creates a new dataset with
        the given shape and dtype.

        Parameters:
            dtype: The datatype for the dataset (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the dataset.
            shape: Shape for the new dataset if created.

        Returns:
            The Dataset object (existing or newly created).

        Raises:
            Error: If the name exists but is not a dataset, or creation fails.
        """
        var did = self._lib[].require_dataset(
            self._gid, name, shape, _hdf5_type_id[dtype](self._lib[])
        )
        if did < 0:
            raise Error("require_dataset: failed for '" + name + "'")

        var full_name = self._name
        if full_name == "/":
            full_name = "/" + name
        else:
            full_name = full_name + "/" + name

        var sid = self._lib[].get_dataset_space(did)
        var ndims = self._lib[].get_space_ndims(sid)
        var dims = self._lib[].get_space_dims(sid, Int(ndims))
        var read_shape = List[Int]()
        for i in range(Int(ndims)):
            read_shape.append(Int(dims[unsafe_offset=i]))
        dims.unsafe_free()
        _ = self._lib[].close_dataspace(sid)

        var tid = self._lib[].get_dataset_type(did)
        var tclass = self._lib[].get_type_class(tid)
        var tsize = Int(self._lib[].get_type_size(tid))
        var dtype_code = _infer_dtype_code(self._lib[], tid, tclass, tsize)
        _ = self._lib[].close_type(tid)

        return Dataset[Self.origin](
            self._lib, did, read_shape, dtype_code, full_name, self._filename
        )


# ===----------------------------------------------------------------------=== #
# H5Object
# ===----------------------------------------------------------------------=== #


struct H5Object[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](Copyable, Movable):
    """Polymorphic wrapper for HDF5 groups or datasets.

    When accessing items from a Group or File using get(),
    the result is an H5Object that can represent either a Group or Dataset.
    Use is_group() / is_dataset() to check the type, then unwrap with
    group() / dataset().

    Example::

        var f = File("data.h5", "r")
        var obj = f.get("path/to/item")
        if obj.is_dataset():
            var dset = obj.dataset()
            var arr = dset.read[DType.float64]()
            arr.unsafe_free()
        elif obj.is_group():
            var grp = obj.group()
            print(grp.name())
        f.close()
    """

    var _value: Variant[Group[Self.origin], Dataset[Self.origin]]

    def __init__(out self, var group: Group[Self.origin]):
        self._value = group^

    def __init__(out self, var dataset: Dataset[Self.origin]):
        self._value = dataset^

    def is_group(self) -> Bool:
        """Check if this object is a Group.

        Returns:
            True if this is a Group, False if it's a Dataset.
        """
        return self._value.isa[Group[Self.origin]]()

    def is_dataset(self) -> Bool:
        """Check if this object is a Dataset.

        Returns:
            True if this is a Dataset, False if it's a Group.
        """
        return self._value.isa[Dataset[Self.origin]]()

    def group(self) raises -> Group[Self.origin]:
        """Unwrap this object as a Group.

        Returns:
            The underlying Group.

        Raises:
            Error: If this object is not a Group.
        """
        if not self._value.isa[Group[Self.origin]]():
            raise Error("H5Object: not a group")
        return self._value[Group[Self.origin]].copy()

    def dataset(self) raises -> Dataset[Self.origin]:
        """Unwrap this object as a Dataset.

        Returns:
            The underlying Dataset.

        Raises:
            Error: If this object is not a Dataset.
        """
        if not self._value.isa[Dataset[Self.origin]]():
            raise Error("H5Object: not a dataset")
        return self._value[Dataset[Self.origin]].copy()


# ===----------------------------------------------------------------------=== #
# File
# ===----------------------------------------------------------------------=== #


# TODO: Add libver support.
struct _FileHandle(Movable):
    """Shared owner of an open HDF5 file id; closes it when the last `File`
    copy goes away (or earlier via `File.close()`)."""

    var lib: ArcPointer[HDF5Lib]
    var fid: hid_t
    var is_open: Bool

    def __init__(out self, lib: ArcPointer[HDF5Lib], fid: hid_t):
        self.lib = lib
        self.fid = fid
        self.is_open = fid >= 0

    def close(mut self):
        if self.is_open:
            _ = self.lib[].close_file(self.fid)
            self.is_open = False

    def __deinit__(deinit self):
        if self.is_open:
            _ = self.lib[].close_file(self.fid)


struct File(Copyable, Movable):
    """HDF5 file object, similar to ``h5py.File``.

    Opens an HDF5 file and provides dict-like access to its contents.
    The File acts as the root Group of the HDF5 hierarchy.

    Supported modes:
    - "r": Read-only, file must exist.
    - "r+": Read/write, file must exist.
    - "w": Create/truncate, always creates new file.
    - "w-" or "x": Create, fails if file exists.
    - "a": Append (read/write), creates if doesn't exist.

    Example::

        # Reading
        var f = File("data.h5", "r")
        var obj = f.get("mydataset")
        if obj.is_dataset():
            var dset = obj.dataset()
            print(dset.shape())
        f.close()

        # Writing
        var f = File("output.h5", "w")
        f.create_dataset[DType.float64]("data", [100, 100])
        f.create_group("nested/group")
        f.close()
    """

    var _lib: ArcPointer[HDF5Lib]
    var _handle: ArcPointer[_FileHandle]
    var _fid: hid_t
    var _filename: String
    var _mode: String

    def __init__(
        out self, path: String, mode: String = "r", track_order: Bool = False
    ) raises:
        """Open or create an HDF5 file.

        Args:
            path: Filesystem path to the HDF5 file.
            mode: Access mode - "r", "r+", "w", "w-", "x", or "a".
            track_order: When creating a file, remember the creation order of
                root-level members and file attributes.

        Raises:
            Error: If the file cannot be opened/created with the given mode.
        """
        self._lib = ArcPointer(HDF5Lib())
        self._filename = path
        self._mode = mode

        var fid: hid_t
        if mode == "r":
            fid = self._lib[].open_file(path, H5F_ACC_RDONLY)
        elif mode == "r+":
            fid = self._lib[].open_file(path, H5F_ACC_RDWR)
        elif mode == "w":
            fid = self._lib[].create_file(path, H5F_ACC_TRUNC, track_order)
        elif mode == "w-" or mode == "x":
            var existing = self._lib[].open_file(path, H5F_ACC_RDONLY)
            if existing >= 0:
                _ = self._lib[].close_file(existing)
                raise Error("File: file exists '" + path + "'")
            fid = self._lib[].create_file(path, H5F_ACC_TRUNC, track_order)
        elif mode == "a":
            fid = self._lib[].open_file(path, H5F_ACC_RDWR)
            if fid < 0:
                fid = self._lib[].create_file(path, H5F_ACC_TRUNC, track_order)
        else:
            raise Error("File: invalid mode '" + mode + "'")

        if fid < 0:
            raise Error(
                "File: cannot open/create '"
                + path
                + "' with mode '"
                + mode
                + "'"
            )

        self._fid = fid
        self._handle = ArcPointer(_FileHandle(self._lib, fid))

    def __enter__(self) -> Self:
        """Enter a ``with`` block; the file is closed on exit."""
        return self.copy()

    def __exit__(self):
        """Close the file (all copies share it) when leaving a ``with`` block.
        """
        self._handle[].close()

    def close(mut self):
        """Close the file and release resources.

        Safe to call more than once. The file is also closed automatically
        when the last copy of this `File` is destroyed.
        """
        self._handle[].close()
        self._fid = -1

    def flush(self):
        """Flush pending writes to disk.

        Does not close the file.
        """
        if self._handle[].is_open:
            _ = self._lib[].flush(self._fid)

    def filename(self) -> String:
        """Get the filename.

        Returns:
            The path used to open the file.
        """
        return self._filename

    def mode(self) -> String:
        """Get the access mode.

        Returns:
            The mode string used to open the file.
        """
        return self._mode

    def _lib_ptr(self) -> LibRef[origin_of(self._lib[])]:
        return Pointer(to=self._lib[])

    def attrs(self) -> AttributeManager[origin_of(self._lib[])]:
        """Get the attribute manager for the file root.

        Returns:
            An AttributeManager for reading/writing file-level attributes.
        """
        return AttributeManager[origin_of(self._lib[])](
            self._lib_ptr(), self._fid
        )

    def name(self) -> String:
        """Get the name of the root group.

        Returns:
            Always returns "/".
        """
        return "/"

    def __contains__(self, member_name: String) -> Bool:
        """Check if a member exists at the root level.

        Args:
            member_name: Name of the member to check.

        Returns:
            True if the member exists, False otherwise.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.__contains__(member_name)

    def contains(self, member_name: String) -> Bool:
        """Check if a member exists at the root level.

        Args:
            member_name: Name of the member to check.

        Returns:
            True if the member exists, False otherwise.
        """
        return self.__contains__(member_name)

    def get(
        self, member_name: String
    ) raises -> H5Object[origin_of(self._lib[])]:
        """Get a member by name at the root level.

        Args:
            member_name: Name of the member to retrieve.

        Returns:
            An H5Object wrapping the group or dataset.

        Raises:
            Error: If the member does not exist or cannot be opened.
        """
        return self.__getitem__(member_name)

    def get_opt(
        self, member_name: String
    ) raises -> Optional[H5Object[origin_of(self._lib[])]]:
        """Get a member by name at the root level without raising if absent.

        Mirrors the fallback behavior of ``h5py.File.get(name,
        default=...)``: unlike `get`/`__getitem__`, this returns `None`
        instead of raising when `member_name` does not exist.

        Args:
            member_name: Name of the member to retrieve.

        Returns:
            `Some(H5Object)` if the member exists, `None` otherwise.

        Raises:
            Error: If the member exists but cannot be opened.
        """
        var lib_ref = self._lib_ptr()
        if not self.__contains__(member_name):
            var empty: Optional[H5Object[origin_of(lib_ref[])]] = None
            return empty^
        _ = lib_ref
        return self.__getitem__(member_name)

    def delete(self, member_name: String) raises:
        """Delete a member at the root level.

        Args:
            member_name: Name of the member to delete.

        Raises:
            Error: If deletion fails.
        """
        var rc = self._lib[].delete_object(self._fid, member_name)
        if rc < 0:
            raise Error("File: cannot delete '" + member_name + "'")

    def len(self) raises -> Int:
        """Get the number of members at the root level.

        Returns:
            The number of root-level members.

        Raises:
            Error: If counting fails.
        """
        return len(self.keys())

    def __iter__(self) raises -> NameIterator:
        """Iterate over root-level member names (same order as `keys()`)."""
        return NameIterator(self.keys())

    def keys(self) raises -> List[String]:
        """Get names of all members at the root level.

        Returns:
            A list of root-level member names.

        Raises:
            Error: If listing fails.
        """
        var root = Group(self._lib_ptr(), self._fid, "/", is_file=True)
        return root.keys()

    def values(self) raises -> List[H5Object[origin_of(self._lib[])]]:
        """Get all member objects at the root level.

        Returns:
            A list of H5Object wrappers.

        Raises:
            Error: If listing fails.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.values()

    def __getitem__(
        self, member_name: String
    ) raises -> H5Object[origin_of(self._lib[])]:
        """Get a member (group or dataset) at the root level.

        Args:
            member_name: Name of the member to retrieve.

        Returns:
            An H5Object wrapping the group or dataset.

        Raises:
            Error: If the member does not exist or cannot be opened.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.__getitem__(member_name)

    def _get_dataset(
        self, member_name: String
    ) raises -> Dataset[origin_of(self._lib[])]:
        """Get a dataset directly by name.

        Args:
            member_name: Name of the dataset.

        Returns:
            The Dataset object.

        Raises:
            Error: If not a dataset or cannot be opened.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root._open_dataset(member_name)

    def create_group(
        self, name: String, track_order: Bool = False
    ) raises -> Group[origin_of(self._lib[])]:
        """Create a group at the root level.

        Args:
            name: Name of the group to create.
            track_order: Remember creation order of the group's members.

        Returns:
            The created Group object.

        Raises:
            Error: If creation fails.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_group(name, track_order)

    def require_group(
        self, name: String
    ) raises -> Group[origin_of(self._lib[])]:
        """Open an existing group or create it if it doesn't exist.

        Args:
            name: Name of the group.

        Returns:
            The Group object (existing or newly created).

        Raises:
            Error: If path exists but is not a group.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.require_group(name)

    def require_dataset[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
    ) raises -> Dataset[
        origin_of(self._lib[])
    ]:
        """Open an existing dataset or create a new one if it doesn't exist.

        Parameters:
            dtype: The datatype for the dataset (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the dataset.
            shape: Shape for the new dataset if created.

        Returns:
            The Dataset object (existing or newly created).

        Raises:
            Error: If the name exists but is not a dataset, or creation fails.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.require_dataset[dtype](name, shape)

    def create_dataset[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        fillvalue: Optional[Scalar[dtype]] = None,
        chunks: List[Int] = [],
        maxshape: List[Int] = [],
        compression: String = "",
        compression_opts: Int = 4,
        shuffle: Bool = False,
        fletcher32: Bool = False,
        track_order: Bool = False,
    ) raises -> Dataset[origin_of(self._lib[])] where SupportedDType[dtype]:
        """Create a root-level dataset; see ``Group.create_dataset``."""
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_dataset[dtype](
            name,
            shape,
            fillvalue=fillvalue,
            chunks=chunks.copy(),
            maxshape=maxshape.copy(),
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            fletcher32=fletcher32,
            track_order=track_order,
        )

    def create_scalar_dataset[
        dtype: DType
    ](
        self,
        name: String,
        value: Scalar[dtype],
    ) raises -> Dataset[
        origin_of(self._lib[])
    ] where SupportedDType[dtype]:
        """Create a scalar dataset at the root level."""
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_scalar_dataset[dtype](name, value)

    def create_dataset_chunked[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        maxshape: List[Int],
        chunks: List[Int],
    ) raises -> Dataset[origin_of(self._lib[])] where SupportedDType[dtype]:
        """Create a chunked dataset; prefer ``create_dataset(chunks=, maxshape=)``.

        Use ``-1`` in ``maxshape`` for an unlimited dimension.
        """
        return self.create_dataset[dtype](
            name, shape, chunks=chunks.copy(), maxshape=maxshape.copy()
        )

    def create_dataset_chunked[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        maxshape: List[Int],
        chunks: List[Int],
        fillvalue: Scalar[dtype],
    ) raises -> Dataset[origin_of(self._lib[])] where SupportedDType[dtype]:
        """Chunked dataset with a fill value; prefer ``create_dataset``."""
        return self.create_dataset[dtype](
            name,
            shape,
            fillvalue=fillvalue,
            chunks=chunks.copy(),
            maxshape=maxshape.copy(),
        )

    def create_dataset_filtered[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        maxshape: List[Int],
        chunks: List[Int],
        compression: String,
        compression_opts: Int,
        shuffle: Bool,
        fletcher32: Bool,
    ) raises -> Dataset[origin_of(self._lib[])] where SupportedDType[dtype]:
        """Chunked dataset with built-in filters; prefer ``create_dataset``."""
        return self.create_dataset[dtype](
            name,
            shape,
            chunks=chunks.copy(),
            maxshape=maxshape.copy(),
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            fletcher32=fletcher32,
        )

    def create_dataset_with_data[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        data: Pointer[Scalar[dtype], MutExt],
    ) raises -> Dataset[origin_of(self._lib[])] where SupportedDType[dtype]:
        """Create a dataset at the root level and write data.

        Parameters:
            dtype: The data type for the dataset (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the dataset to create.
            shape: List of dimensions, e.g., [100] or [10, 20].
            data: Buffer containing the data to write.

        Returns:
            The created Dataset object with data written.

        Raises:
            Error: If creation or writing fails.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_dataset_with_data[dtype](name, shape, data)

    def create_dataset_with_data[
        dtype: DType
    ](self, name: String, data: NDArray[dtype]) raises -> Dataset[
        origin_of(self._lib[])
    ] where SupportedDType[dtype]:
        """Create a root-level dataset and write a NuMojo NDArray to it."""
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_dataset_with_data[dtype](name, data)

    def __bool__(self) -> Bool:
        """Check if the file is open.

        Returns:
            True if the file is open, False if closed.
        """
        return self._handle[].is_open
