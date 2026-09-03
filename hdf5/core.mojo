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
)
from std.memory import OwnedPointer, Pointer
from std.memory.alloc import unsafe_alloc
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


# NOTE: Perhaps we should raise here, otherwise there'll be UB.
def _hdf5_type_id[dtype: DType](lib: HDF5Lib) raises -> hid_t:
    """Map a Mojo DType to the matching HDF5 predefined type id."""
    comptime if dtype == DType.float64:
        return lib.native_double
    elif dtype == DType.float32:
        return lib.native_float
    elif dtype == DType.int32:
        return lib.native_int32
    elif dtype == DType.int64:
        return lib.handle.get_symbol[hid_t]("H5T_NATIVE_INT64_g").value()[]
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
    else:
        raise Error("HDF5: unsupported DType")


# ===----------------------------------------------------------------------=== #
# AttributeManager
# ===----------------------------------------------------------------------=== #


struct AttributeManager[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](Copyable, Movable):
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
        var aid = self._lib[].open_attr(self._loc_id, name)
        if aid < 0:
            raise Error("attrs: '" + name + "' not found")
        var buf = unsafe_alloc[Scalar[dtype]](1)
        var rc = self._lib[].read_attr(
            aid, _hdf5_type_id[dtype](self._lib[]), buf.unsafe_bitcast[NoneType]()
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
        for i in range(Int(num)):
            var attr_name = self._lib[].get_attr_name_by_idx(self._loc_id, i)
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
            aid, _hdf5_type_id[dtype](self._lib[]), buf.unsafe_bitcast[NoneType]()
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
](Copyable, Movable):
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

    # TODO: I might be able to use comptime condition/where to
    # make sure only these 4 dtypes are used.
    def dtype(self) -> String:
        """Get the HDF5 datatype as a string.

        Returns:
            One of: "float64", "float32", "int32", "int64", "unknown".
        """
        if self._dtype_code == 0:
            return "float64"
        elif self._dtype_code == 1:
            return "float32"
        elif self._dtype_code == 2:
            return "int32"
        elif self._dtype_code == 3:
            return "int64"
        else:
            return "unknown"

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
    ](self, buf: Pointer[Scalar[dtype], MutExt], n: Int) raises where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
        """Read dataset data into a pre-allocated buffer.

        Parameters:
            dtype: The data type to read as (e.g., DType.float64, DType.int32).

        Args:
            buf: Pre-allocated buffer to read into.
            n: Number of elements to read.

        Raises:
            Error: If the read operation fails.

        Constraints:
            `dtype` must be one of DType.float64, DType.float32, DType.int32, or DType.int64.
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
    ](self, data: Pointer[Scalar[dtype], MutExt], n: Int) raises where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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

    def read_scalar[dtype: DType](self) raises -> Scalar[dtype] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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
    ](self, value: Scalar[dtype]) raises where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
        """Write a scalar dataset value."""
        if len(self._shape) != 0:
            raise Error("Dataset: not a scalar dataset '" + self._name + "'")
        var buf = unsafe_alloc[Scalar[dtype]](1)
        buf[unsafe_offset=0] = value
        self.write[dtype](buf, 1)
        buf.unsafe_free()

    def read_hyperslab[
        dtype: DType
    ](
        self,
        start: List[Int],
        count: List[Int],
        buf: Pointer[Scalar[dtype], MutExt],
    ) raises where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
        """Read a contiguous hyperslab into a compact memory buffer.

        ``start`` and ``count`` must have one entry per dataset dimension.
        """
        var ndims = len(self._shape)
        if len(start) != ndims or len(count) != ndims:
            raise Error("Dataset: hyperslab rank mismatch for '" + self._name + "'")
        var start_dims = unsafe_alloc[hsize_t](ndims)
        var count_dims = unsafe_alloc[hsize_t](ndims)
        for i in range(ndims):
            if start[i] < 0 or count[i] < 0 or start[i] + count[i] > self._shape[i]:
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
            raise Error("Dataset: H5Dread hyperslab failed for '" + self._name + "'")

    def write_hyperslab[
        dtype: DType
    ](
        self,
        start: List[Int],
        count: List[Int],
        data: Pointer[Scalar[dtype], MutExt],
    ) raises where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
        """Write a compact memory buffer into a contiguous hyperslab."""
        var ndims = len(self._shape)
        if len(start) != ndims or len(count) != ndims:
            raise Error("Dataset: hyperslab rank mismatch for '" + self._name + "'")
        var start_dims = unsafe_alloc[hsize_t](ndims)
        var count_dims = unsafe_alloc[hsize_t](ndims)
        for i in range(ndims):
            if start[i] < 0 or count[i] < 0 or start[i] + count[i] > self._shape[i]:
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
            raise Error("Dataset: H5Dwrite hyperslab failed for '" + self._name + "'")

    def _slice_count(self, start: List[Int], stop: List[Int]) raises -> List[Int]:
        var ndims = len(self._shape)
        if len(start) != ndims or len(stop) != ndims:
            raise Error("Dataset: slice rank mismatch for '" + self._name + "'")
        var count = List[Int]()
        for i in range(ndims):
            if start[i] < 0 or stop[i] < start[i] or stop[i] > self._shape[i]:
                raise Error("Dataset: slice out of bounds for '" + self._name + "'")
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
    ](self, start: List[Int], stop: List[Int]) raises -> NDArray[dtype] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
        """Read a simple rectangular slice into a compact NDArray."""
        var count = self._slice_count(start, stop)
        if len(count) == 1:
            var arr = NDArray[dtype](Shape(count[0]))
            var total = count[0]
            var buf = unsafe_alloc[Scalar[dtype]](total)
            self.read_hyperslab[dtype](start, count, buf)
            var ptr = arr.unsafe_ptr()
            for i in range(total):
                ptr[unsafe_offset=i] = buf[unsafe_offset=i]
            buf.unsafe_free()
            return arr^
        elif len(count) == 2:
            var arr = NDArray[dtype](Shape(count[0], count[1]))
            var total = count[0] * count[1]
            var buf = unsafe_alloc[Scalar[dtype]](total)
            self.read_hyperslab[dtype](start, count, buf)
            var ptr = arr.unsafe_ptr()
            for i in range(total):
                ptr[unsafe_offset=i] = buf[unsafe_offset=i]
            buf.unsafe_free()
            return arr^
        else:
            raise Error("Dataset: unsupported slice rank for '" + self._name + "'")

    def read_slice[
        dtype: DType
    ](self, slices: List[Slice]) raises -> NDArray[dtype] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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
    ) raises where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
        """Write a compact buffer into a simple rectangular slice."""
        var count = self._slice_count(start, stop)
        self.write_hyperslab[dtype](start, count, data)

    def write_slice[
        dtype: DType
    ](
        self,
        slices: List[Slice],
        data: Pointer[Scalar[dtype], MutExt],
    ) raises where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
        """Write a compact buffer into a slice described by Mojo Slice values."""
        var start = self._slice_starts(slices)
        var stop = self._slice_stops(slices)
        self.write_slice[dtype](start, stop, data)

    def read_point[
        dtype: DType
    ](self, indices: List[Int]) raises -> Scalar[dtype] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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
    ](self, indices: List[Int], value: Scalar[dtype]) raises where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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

    # TODO: Still contains old 1D, 2D code, clean this up.
    def read[
        dtype: DType
    ](self) raises -> NDArray[dtype] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
        """Read the entire dataset into an NDArray.
        Uses NuMojo NDArray for heap-allocated array storage.

        Parameters:
            dtype: The data type to read as (e.g., DType.float64, DType.int32).

        Returns:
            An NDArray containing the dataset data.

        Raises:
            Error: If the dataset has more than 2 dimensions or read fails.

        Constraints:
            `dtype` must be one of DType.float64, DType.float32, DType.int32, or DType.int64.
        """
        var nd = len(self._shape)
        if nd == 1:
            var n = self._shape[0]
            var arr = NDArray[dtype](Shape(n))
            var ptr = arr.unsafe_ptr()
            var rc = self._lib[].read_dataset(
                self._did,
                _hdf5_type_id[dtype](self._lib[]),
                ptr.unsafe_bitcast[NoneType](),
            )
            if rc < 0:
                raise Error("Dataset: H5Dread failed for '" + self._name + "'")
            return arr^
        elif nd == 2:
            var rows = self._shape[0]
            var cols = self._shape[1]
            var arr = NDArray[dtype](Shape(rows, cols))
            var ptr = arr.unsafe_ptr()
            var rc = self._lib[].read_dataset(
                self._did,
                _hdf5_type_id[dtype](self._lib[]),
                ptr.unsafe_bitcast[NoneType](),
            )
            if rc < 0:
                raise Error("Dataset: H5Dread failed for '" + self._name + "'")
            return arr^
        else:
            raise Error("Dataset: unsupported rank for '" + self._name + "'")

    def write[
        dtype: DType
    ](self, data: NDArray[dtype]) raises where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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

    def __iter__(self) raises -> List[String]:
        """Iterate over member names (returns list for Mojo compatibility).

        Returns:
            A list of member names.

        Raises:
            Error: If iteration fails.
        """
        return self._get_member_names()

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
        while True:
            var member_name = self._lib[].get_link_name_by_idx(self._gid, idx)
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
        _ = self._lib[].close_type(tid)

        var dtype_code: Int = 0
        if tclass == H5T_FLOAT:
            if tsize == 8:
                dtype_code = 0
            elif tsize == 4:
                dtype_code = 1
        elif tclass == H5T_INTEGER:
            if tsize == 4:
                dtype_code = 2
            elif tsize == 8:
                dtype_code = 3

        var full_name = self._name
        if full_name == "/":
            full_name = "/" + member_name
        else:
            full_name = full_name + "/" + member_name

        return Dataset[Self.origin](
            self._lib, did, shape, dtype_code, full_name, self._filename
        )

    def create_group(self, name: String) raises -> Group[Self.origin]:
        """Create a group, including any intermediate groups in the path.

        Args:
            name: Name or path of the group to create.
                Can be a simple name ("mygroup") or nested path ("a/b/c").

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
                child_gid = self._lib[].create_group(current_gid, part)
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
    ](self, name: String, shape: List[Int],) raises -> Dataset[Self.origin]:
        """Create a new empty dataset.

        Parameters:
            dtype: The data type for the dataset (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the dataset to create.
            shape: List of dimensions, e.g., [100] or [10, 20].

        Returns:
            The created Dataset object.

        Raises:
            Error: If dataset creation fails.
        """
        var sid = self._create_dataspace_for_shape(shape)
        if sid < 0:
            raise Error("create_dataset: dataspace creation failed")
        var tid = _hdf5_type_id[dtype](self._lib[])
        var did = self._lib[].create_dataset(self._gid, name, tid, sid)
        _ = self._lib[].close_dataspace(sid)
        if did < 0:
            raise Error("create_dataset: H5Dcreate2 failed for '" + name + "'")

        var full_name = self._name
        if full_name == "/":
            full_name = "/" + name
        else:
            full_name = full_name + "/" + name

        var dtype_code = _dtype_code[dtype]()

        return Dataset[Self.origin](
            self._lib, did, shape.copy(), dtype_code, full_name, self._filename
        )

    def create_dataset[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        fillvalue: Scalar[dtype],
    ) raises -> Dataset[Self.origin]:
        """Create a dataset with a scalar fill value."""
        var sid = self._create_dataspace_for_shape(shape)
        if sid < 0:
            raise Error("create_dataset: dataspace creation failed")

        var dcpl = self._lib[].create_dcpl()
        if dcpl < 0:
            _ = self._lib[].close_dataspace(sid)
            raise Error("create_dataset: H5Pcreate failed")

        var tid = _hdf5_type_id[dtype](self._lib[])
        var buf = unsafe_alloc[Scalar[dtype]](1)
        buf[unsafe_offset=0] = fillvalue
        var rc = self._lib[].set_fill_value(
            dcpl, tid, buf.unsafe_bitcast[NoneType]()
        )
        buf.unsafe_free()
        if rc < 0:
            _ = self._lib[].close_dcpl(dcpl)
            _ = self._lib[].close_dataspace(sid)
            raise Error("create_dataset: H5Pset_fill_value failed")

        var did = self._lib[].create_dataset_with_dcpl(
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
    ) raises -> Dataset[Self.origin] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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
    ) raises -> Dataset[Self.origin]:
        """Create a chunked dataset with explicit maxshape.

        Use ``-1`` in ``maxshape`` for an unlimited dimension.

        Parameters:
            dtype: The data type for the dataset (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the dataset to create.
            shape: Current dimensions.
            maxshape: Maximum dimensions; -1 means unlimited.
            chunks: Chunk dimensions.

        Returns:
            The created Dataset object.

        Raises:
            Error: If ranks mismatch or HDF5 creation fails.
        """
        var ndims = len(shape)
        if len(maxshape) != ndims or len(chunks) != ndims:
            raise Error(
                "create_dataset_chunked: rank mismatch for '" + name + "'"
            )

        var dims = unsafe_alloc[hsize_t](ndims)
        var maxdims = unsafe_alloc[hsize_t](ndims)
        var chunk_dims = unsafe_alloc[hsize_t](ndims)
        var unlimited = hsize_t(-1)
        for i in range(ndims):
            dims[unsafe_offset=i] = hsize_t(shape[i])
            if maxshape[i] < 0:
                maxdims[unsafe_offset=i] = unlimited
            else:
                maxdims[unsafe_offset=i] = hsize_t(maxshape[i])
            chunk_dims[unsafe_offset=i] = hsize_t(chunks[i])

        var sid = self._lib[].create_dataspace_nd_max(ndims, dims, maxdims)
        dims.unsafe_free()
        maxdims.unsafe_free()
        if sid < 0:
            chunk_dims.unsafe_free()
            raise Error("create_dataset_chunked: H5Screate_simple failed")

        var dcpl = self._lib[].create_dcpl()
        if dcpl < 0:
            chunk_dims.unsafe_free()
            _ = self._lib[].close_dataspace(sid)
            raise Error("create_dataset_chunked: H5Pcreate failed")

        var rc = self._lib[].set_chunk(dcpl, chunk_dims, ndims)
        chunk_dims.unsafe_free()
        if rc < 0:
            _ = self._lib[].close_dcpl(dcpl)
            _ = self._lib[].close_dataspace(sid)
            raise Error("create_dataset_chunked: H5Pset_chunk failed")

        var tid = _hdf5_type_id[dtype](self._lib[])
        var did = self._lib[].create_dataset_with_dcpl(
            self._gid, name, tid, sid, dcpl
        )
        _ = self._lib[].close_dcpl(dcpl)
        _ = self._lib[].close_dataspace(sid)
        if did < 0:
            raise Error(
                "create_dataset_chunked: H5Dcreate2 failed for '" + name + "'"
            )

        var full_name = self._name
        if full_name == "/":
            full_name = "/" + name
        else:
            full_name = full_name + "/" + name

        var dtype_code = _dtype_code[dtype]()

        return Dataset[Self.origin](
            self._lib, did, shape.copy(), dtype_code, full_name, self._filename
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
    ) raises -> Dataset[Self.origin]:
        """Create a chunked dataset with explicit maxshape and fill value."""
        var ndims = len(shape)
        if len(maxshape) != ndims or len(chunks) != ndims:
            raise Error(
                "create_dataset_chunked: rank mismatch for '" + name + "'"
            )

        var dims = unsafe_alloc[hsize_t](ndims)
        var maxdims = unsafe_alloc[hsize_t](ndims)
        var chunk_dims = unsafe_alloc[hsize_t](ndims)
        var unlimited = hsize_t(-1)
        for i in range(ndims):
            dims[unsafe_offset=i] = hsize_t(shape[i])
            if maxshape[i] < 0:
                maxdims[unsafe_offset=i] = unlimited
            else:
                maxdims[unsafe_offset=i] = hsize_t(maxshape[i])
            chunk_dims[unsafe_offset=i] = hsize_t(chunks[i])

        var sid = self._lib[].create_dataspace_nd_max(ndims, dims, maxdims)
        dims.unsafe_free()
        maxdims.unsafe_free()
        if sid < 0:
            chunk_dims.unsafe_free()
            raise Error("create_dataset_chunked: H5Screate_simple failed")

        var dcpl = self._lib[].create_dcpl()
        if dcpl < 0:
            chunk_dims.unsafe_free()
            _ = self._lib[].close_dataspace(sid)
            raise Error("create_dataset_chunked: H5Pcreate failed")

        var rc = self._lib[].set_chunk(dcpl, chunk_dims, ndims)
        chunk_dims.unsafe_free()
        if rc < 0:
            _ = self._lib[].close_dcpl(dcpl)
            _ = self._lib[].close_dataspace(sid)
            raise Error("create_dataset_chunked: H5Pset_chunk failed")

        var tid = _hdf5_type_id[dtype](self._lib[])
        var buf = unsafe_alloc[Scalar[dtype]](1)
        buf[unsafe_offset=0] = fillvalue
        rc = self._lib[].set_fill_value(dcpl, tid, buf.unsafe_bitcast[NoneType]())
        buf.unsafe_free()
        if rc < 0:
            _ = self._lib[].close_dcpl(dcpl)
            _ = self._lib[].close_dataspace(sid)
            raise Error("create_dataset_chunked: H5Pset_fill_value failed")

        var did = self._lib[].create_dataset_with_dcpl(
            self._gid, name, tid, sid, dcpl
        )
        _ = self._lib[].close_dcpl(dcpl)
        _ = self._lib[].close_dataspace(sid)
        if did < 0:
            raise Error(
                "create_dataset_chunked: H5Dcreate2 failed for '" + name + "'"
            )

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
    ) raises -> Dataset[Self.origin]:
        """Create a chunked dataset with built-in HDF5 filters.

        Supported compression values are ``"gzip"`` and ``""``.
        """
        var ndims = len(shape)
        if len(maxshape) != ndims or len(chunks) != ndims:
            raise Error(
                "create_dataset_filtered: rank mismatch for '" + name + "'"
            )
        if compression != "" and compression != "gzip":
            raise Error(
                "create_dataset_filtered: unsupported compression '"
                + compression
                + "'"
            )
        if compression_opts < 0 or compression_opts > 9:
            raise Error("create_dataset_filtered: gzip level must be 0..9")

        var dims = unsafe_alloc[hsize_t](ndims)
        var maxdims = unsafe_alloc[hsize_t](ndims)
        var chunk_dims = unsafe_alloc[hsize_t](ndims)
        var unlimited = hsize_t(-1)
        for i in range(ndims):
            dims[unsafe_offset=i] = hsize_t(shape[i])
            if maxshape[i] < 0:
                maxdims[unsafe_offset=i] = unlimited
            else:
                maxdims[unsafe_offset=i] = hsize_t(maxshape[i])
            chunk_dims[unsafe_offset=i] = hsize_t(chunks[i])

        var sid = self._lib[].create_dataspace_nd_max(ndims, dims, maxdims)
        dims.unsafe_free()
        maxdims.unsafe_free()
        if sid < 0:
            chunk_dims.unsafe_free()
            raise Error("create_dataset_filtered: H5Screate_simple failed")

        var dcpl = self._lib[].create_dcpl()
        if dcpl < 0:
            chunk_dims.unsafe_free()
            _ = self._lib[].close_dataspace(sid)
            raise Error("create_dataset_filtered: H5Pcreate failed")

        var rc = self._lib[].set_chunk(dcpl, chunk_dims, ndims)
        chunk_dims.unsafe_free()
        if rc < 0:
            _ = self._lib[].close_dcpl(dcpl)
            _ = self._lib[].close_dataspace(sid)
            raise Error("create_dataset_filtered: H5Pset_chunk failed")

        if shuffle:
            rc = self._lib[].set_shuffle(dcpl)
            if rc < 0:
                _ = self._lib[].close_dcpl(dcpl)
                _ = self._lib[].close_dataspace(sid)
                raise Error("create_dataset_filtered: H5Pset_shuffle failed")

        if compression == "gzip":
            rc = self._lib[].set_deflate(dcpl, compression_opts)
            if rc < 0:
                _ = self._lib[].close_dcpl(dcpl)
                _ = self._lib[].close_dataspace(sid)
                raise Error("create_dataset_filtered: H5Pset_deflate failed")

        if fletcher32:
            rc = self._lib[].set_fletcher32(dcpl)
            if rc < 0:
                _ = self._lib[].close_dcpl(dcpl)
                _ = self._lib[].close_dataspace(sid)
                raise Error(
                    "create_dataset_filtered: H5Pset_fletcher32 failed"
                )

        var tid = _hdf5_type_id[dtype](self._lib[])
        var did = self._lib[].create_dataset_with_dcpl(
            self._gid, name, tid, sid, dcpl
        )
        _ = self._lib[].close_dcpl(dcpl)
        _ = self._lib[].close_dataspace(sid)
        if did < 0:
            raise Error(
                "create_dataset_filtered: H5Dcreate2 failed for '" + name + "'"
            )

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

    def create_dataset_with_data[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        data: Pointer[Scalar[dtype], MutExt],
    ) raises -> Dataset[Self.origin] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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
    ) raises -> Dataset[Self.origin] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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
        if shp.ndim == 1:
            shape.append(shp[0])
        elif shp.ndim == 2:
            shape.append(shp[0])
            shape.append(shp[1])
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
        _ = self._lib[].close_type(tid)

        var dtype_code: Int = 0
        if tclass == H5T_FLOAT:
            if tsize == 8:
                dtype_code = 0
            elif tsize == 4:
                dtype_code = 1
        elif tclass == H5T_INTEGER:
            if tsize == 4:
                dtype_code = 2
            elif tsize == 8:
                dtype_code = 3

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

    def group(mut self) raises -> Group[Self.origin]:
        """Unwrap this object as a Group.

        Returns:
            The underlying Group.

        Raises:
            Error: If this object is not a Group.
        """
        if not self._value.isa[Group[Self.origin]]():
            raise Error("H5Object: not a group")
        return self._value[Group[Self.origin]].copy()

    def dataset(mut self) raises -> Dataset[Self.origin]:
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
struct File(Movable):
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

    var _lib: OwnedPointer[HDF5Lib]
    var _fid: hid_t
    var _filename: String
    var _mode: String
    var _closed: Bool

    def __init__(out self, path: String, mode: String = "r") raises:
        """Open or create an HDF5 file.

        Args:
            path: Filesystem path to the HDF5 file.
            mode: Access mode - "r", "r+", "w", "w-", "x", or "a".

        Raises:
            Error: If the file cannot be opened/created with the given mode.
        """
        self._closed = False
        self._lib = OwnedPointer(HDF5Lib())
        self._filename = path
        self._mode = mode

        var fid: hid_t
        if mode == "r":
            fid = self._lib[].open_file(path, H5F_ACC_RDONLY)
        elif mode == "r+":
            fid = self._lib[].open_file(path, H5F_ACC_RDWR)
        elif mode == "w":
            fid = self._lib[].create_file(path, H5F_ACC_TRUNC)
        elif mode == "w-" or mode == "x":
            var existing = self._lib[].open_file(path, H5F_ACC_RDONLY)
            if existing >= 0:
                _ = self._lib[].close_file(existing)
                raise Error("File: file exists '" + path + "'")
            fid = self._lib[].create_file(path, H5F_ACC_TRUNC)
        elif mode == "a":
            fid = self._lib[].open_file(path, H5F_ACC_RDWR)
            if fid < 0:
                fid = self._lib[].create_file(path, H5F_ACC_TRUNC)
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

    def close(mut self):
        """Close the file and release resources.

        Always call this when done with the file to flush writes.
        """
        if not self._closed and self._fid >= 0:
            _ = self._lib[].close_file(self._fid)
            self._fid = -1
            self._closed = True

    def flush(self):
        """Flush pending writes to disk.

        Does not close the file.
        """
        if not self._closed and self._fid >= 0:
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
        self, name: String
    ) raises -> Group[origin_of(self._lib[])]:
        """Create a group at the root level.

        Args:
            name: Name of the group to create.

        Returns:
            The created Group object.

        Raises:
            Error: If creation fails.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_group(name)

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
    ) raises -> Dataset[
        origin_of(self._lib[])
    ]:
        """Create a dataset at the root level.

        Parameters:
            dtype: The data type for the dataset (e.g., DType.float64, DType.int32).

        Args:
            name: Name of the dataset to create.
            shape: List of dimensions, e.g., [100] or [10, 20].

        Returns:
            The created Dataset object.

        Raises:
            Error: If dataset creation fails.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_dataset[dtype](name, shape)

    def create_dataset[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        fillvalue: Scalar[dtype],
    ) raises -> Dataset[
        origin_of(self._lib[])
    ]:
        """Create a root-level dataset with a scalar fill value."""
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_dataset[dtype](name, shape, fillvalue)

    def create_scalar_dataset[
        dtype: DType
    ](
        self,
        name: String,
        value: Scalar[dtype],
    ) raises -> Dataset[origin_of(self._lib[])] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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
    ) raises -> Dataset[origin_of(self._lib[])]:
        """Create a chunked dataset at the root level.

        Use ``-1`` in ``maxshape`` for an unlimited dimension.
        """
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_dataset_chunked[dtype](name, shape, maxshape, chunks)

    def create_dataset_chunked[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        maxshape: List[Int],
        chunks: List[Int],
        fillvalue: Scalar[dtype],
    ) raises -> Dataset[origin_of(self._lib[])]:
        """Create a chunked root-level dataset with a scalar fill value."""
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_dataset_chunked[
            dtype
        ](name, shape, maxshape, chunks, fillvalue)

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
    ) raises -> Dataset[origin_of(self._lib[])]:
        """Create a filtered chunked dataset at the root level."""
        var root = Group[origin_of(self._lib[])](
            self._lib_ptr(), self._fid, "/", is_file=True
        )
        return root.create_dataset_filtered[
            dtype
        ](
            name,
            shape,
            maxshape,
            chunks,
            compression,
            compression_opts,
            shuffle,
            fletcher32,
        )

    def create_dataset_with_data[
        dtype: DType
    ](
        self,
        name: String,
        shape: List[Int],
        data: Pointer[Scalar[dtype], MutExt],
    ) raises -> Dataset[origin_of(self._lib[])] where (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.int64
        or dtype == DType.int32
    ):
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

    def __bool__(self) -> Bool:
        """Check if the file is open.

        Returns:
            True if the file is open, False if closed.
        """
        return not self._closed and self._fid >= 0
