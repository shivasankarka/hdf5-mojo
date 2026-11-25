# API Reference

Supported dtypes across this API: `float64`, `float32`, `float16`, `int64`,
`int32`, `int16`, `int8`, `uint64`, `uint32`, `uint16`, `uint8` (as `DType.*`). See the
[README](../README.md#known-limitations) for what's not yet supported.

## File

The main entry point for HDF5 file operations. Acts as the root `Group` of
the HDF5 hierarchy.

**Constructor**

| Method | Description |
|--------|-------------|
| `File(path, mode="r", track_order=False)` | Open/create a file. Mode: `"r"` (read), `"r+"` (read/write), `"w"` (truncate), `"w-"`/`"x"` (create/fail if exists), `"a"` (append). `track_order=True` (on creation) remembers creation order of root members and attributes. |

**Properties**

| Method | Description |
|--------|-------------|
| `filename()` | Return the filename path. |
| `mode()` | Return the access mode string. |
| `name()` | Return `"/"` (root group name). |
| `attrs` | `AttributeManager` field for file-level attributes (`f.attrs["k"] = v`; `attrs()` still works). |
| `__bool__()` | `True` if the file is open. |

**Access**

| Method | Description |
|--------|-------------|
| `__contains__(name)` / `contains(name)` | Check if a root-level member exists. |
| `keys()` | List of root-level member names (creation order if tracked, else name order). |
| `for name in f` | Iterate over root-level member names. |
| `len()` / `__len__()` | Number of root-level members. |
| `values()` | List of `H5Object` wrappers for all root-level members. |
| `get(name)` / `__getitem__(name)` | Get an `H5Object` by name; raises if missing. |
| `get_opt(name)` | Get an `H5Object` by name, returning `Optional[H5Object]`; `None` instead of raising when absent. |

**Creation**

| Method | Description |
|--------|-------------|
| `create_group(name, track_order=False)` | Create a group at root level (nested paths auto-created). With `track_order=True`, `keys()` lists its members in creation order. |
| `require_group(name)` | Open existing group or create it. |
| `create_dataset[dtype](name, shape, fillvalue=, chunks=, maxshape=, compression=, compression_opts=, shuffle=, fletcher32=)` | Create a dataset. All keywords optional; chunking is enabled by any of `chunks`/`maxshape`/filters (chunk shape is guessed if omitted). `-1` in `maxshape` means unlimited; `compression` is `"gzip"` or `""`. |
| `create_scalar_dataset[dtype](name, value)` | Create and initialize a scalar (rank-0) dataset. |
| `create_dataset_chunked[dtype](name, shape, maxshape, chunks)` | Create a chunked, resizable dataset (`-1` in `maxshape` means unlimited). |
| `create_dataset_chunked[dtype](name, shape, maxshape, chunks, fillvalue)` | Same, with a fill value. |
| `create_dataset_filtered[dtype](name, shape, maxshape, chunks, compression, compression_opts, shuffle, fletcher32)` | Create a chunked dataset with built-in filters (`compression` is `"gzip"` or `""`). |
| `create_dataset_with_data[dtype](name, shape, data)` | Create a dataset and write a buffer to it. |
| `create_dataset_with_data[dtype](name, data: NDArray[dtype])` | Create a dataset and write an `NDArray` to it. |
| `create_string_dataset(name, shape, fixed_length=0)` | Create an empty UTF-8 string dataset (variable-length by default, as h5py does for `str`; `fixed_length=N` for fixed-length). |
| `create_dataset_with_strings(name, values, fixed_length=0)` | Create a 1-D string dataset holding a `List[String]`. |
| `require_dataset[dtype](name, shape)` | Open an existing dataset (shape unchanged) or create a new one. |

**Deletion**

| Method | Description |
|--------|-------------|
| `delete(name)` | Delete a root-level member. |
| `move(source, dest)` | Move/rename an object. |
| `copy_object(source, dest)` | Deep-copy an object (and attributes) within the file. (`File.copy()` duplicates the handle.) |
| `create_soft_link(name, target)` | Create a soft (path) link. |
| `create_hard_link(name, target)` | Add another name for an existing object. |
| `create_external_link(name, filename, path)` | Link to an object in another file. |
| `get_link(name) -> LinkInfo` | Describe a link without following it. |
| `create_dataset_like(name, other_path)` | Empty dataset with the dtype/shape/chunks/filters/fill value of the dataset at `other_path`. |
| `visit_names()` | Paths of every object below, via hard links (each object once; soft/external links not followed). |
| `visit[func]() -> Optional[String]` | Call `func(name)` for each object; stops and returns the name when `func` returns `True`. `func` must be a plain `def(String) raises -> Bool`. |

**Operations**

| Method | Description |
|--------|-------------|
| `close()` | Close the file and release resources (idempotent; copies share state). |
| `with File(...) as f:` | Context manager; closes the file on exit. |
| `dataset(path)` / `group(path)` | Open a dataset/group directly (raises if missing or of the other kind). |
| `flush()` | Flush pending writes to disk. |

---

## Group

Represents an HDF5 group with dict-like access to members. Obtained via
`File.create_group()`/`require_group()`, or by unwrapping an `H5Object`.

**Properties**

| Method | Description |
|--------|-------------|
| `name()` | Full path of the group. |
| `attrs` | `AttributeManager` field for group attributes (`attrs()` still works). |
| `file()` | Path of the file this group belongs to. |
| `dataset(name)` / `group(name)` | Open a member dataset/group directly. |
| `with g:` / `close()` | Closes on exit; also closes automatically when the last copy is destroyed. |
| `parent()` | Path of the parent group. |

**Access**

| Method | Description |
|--------|-------------|
| `__contains__(name)` / `contains(name)` | Check if a member exists. |
| `keys()` | List of member names. |
| `len()` / `__len__()` | Number of members. |
| `for name in group` | Iterate over member names (`__iter__` returns a `NameIterator`). |
| `items()` | List of member names (alias for `keys()`). |
| `values()` | List of `H5Object` wrappers for all members. |
| `get(name)` / `__getitem__(name)` | Get an `H5Object` by name; raises if missing. |
| `get_opt(name)` | Get an `H5Object` by name, returning `Optional[H5Object]`; `None` instead of raising when absent. |

**Creation**

| Method | Description |
|--------|-------------|
| `create_group(name, track_order=False)` | Create a group (nested paths auto-created). With `track_order=True`, `keys()` lists its members in creation order. |
| `require_group(name)` | Open existing group or create it. |
| `create_dataset[dtype](name, shape, fillvalue=, chunks=, maxshape=, compression=, compression_opts=, shuffle=, fletcher32=)` | Create a dataset. All keywords optional; chunking is enabled by any of `chunks`/`maxshape`/filters (chunk shape is guessed if omitted). `-1` in `maxshape` means unlimited; `compression` is `"gzip"` or `""`. |
| `create_scalar_dataset[dtype](name, value)` | Create and initialize a scalar (rank-0) dataset. |
| `create_dataset_chunked[dtype](name, shape, maxshape, chunks)` | Create a chunked, resizable dataset (`-1` in `maxshape` means unlimited). |
| `create_dataset_chunked[dtype](name, shape, maxshape, chunks, fillvalue)` | Same, with a fill value. |
| `create_dataset_filtered[dtype](name, shape, maxshape, chunks, compression, compression_opts, shuffle, fletcher32)` | Create a chunked dataset with built-in filters (`compression` is `"gzip"` or `""`). |
| `create_dataset_with_data[dtype](name, shape, data)` | Create a dataset and write a buffer to it. |
| `create_dataset_with_data[dtype](name, data: NDArray[dtype])` | Create a dataset and write an `NDArray` to it. |
| `create_string_dataset(name, shape, fixed_length=0)` | Create an empty UTF-8 string dataset (variable-length by default, as h5py does for `str`; `fixed_length=N` for fixed-length). |
| `create_dataset_with_strings(name, values, fixed_length=0)` | Create a 1-D string dataset holding a `List[String]`. |
| `require_dataset[dtype](name, shape)` | Open an existing dataset (shape unchanged) or create a new one. |

**Deletion**

| Method | Description |
|--------|-------------|
| `delete(name)` | Delete a member. |
| `move(source, dest)` | Move/rename an object. |
| `copy(source, dest)` | Deep-copy an object (and attributes) within the file. |
| `create_soft_link(name, target)` | Create a soft (path) link. |
| `create_hard_link(name, target)` | Add another name for an existing object. |
| `create_external_link(name, filename, path)` | Link to an object in another file. |
| `get_link(name) -> LinkInfo` | Describe a link without following it. |
| `create_dataset_like(name, other_path)` | Empty dataset with the dtype/shape/chunks/filters/fill value of the dataset at `other_path`. |
| `visit_names()` | Paths of every object below, via hard links (each object once; soft/external links not followed). |
| `visit[func]() -> Optional[String]` | Call `func(name)` for each object; stops and returns the name when `func` returns `True`. `func` must be a plain `def(String) raises -> Bool`. |

**Operations**

| Method | Description |
|--------|-------------|
| `close()` | Close the group. |

---

## Dataset

Represents an HDF5 dataset containing array data.

**Properties**

| Method | Description |
|--------|-------------|
| `shape()` | List of dimensions, e.g., `[100, 50]` (`[]` for a scalar dataset). |
| `ndim()` | Number of dimensions (`0` for a scalar dataset). |
| `size()` | Total number of elements. |
| `dtype()` | Datatype string, e.g. `"float64"`, `"uint16"`, `"string"`, or `"unknown"`. |
| `len(dset)` | Size of the first axis (0 for a scalar dataset). |
| `nbytes()` | Total size in bytes (`size() * itemsize`). |
| `name()` | Full path of the dataset. |
| `attrs` | `AttributeManager` field for dataset attributes (`attrs()` still works). |
| `file()` | Path of the file this dataset belongs to. |
| `parent()` | Path of the parent group. |

**Chunking, filters, and fill values**

| Method | Description |
|--------|-------------|
| `chunks()` | Chunk dimensions, or `[]` for a contiguous dataset. |
| `maxshape()` | Dataspace max dimensions; unlimited dimensions read as `-1`. |
| `resize(new_size)` / `resize(new_shape)` | Resize a chunked dataset (rank-correct; requires `maxshape` to allow it). |
| `fillvalue[dtype]()` | The dataset's fill value. |
| `filter_count()` | Number of filters applied. |
| `filter_ids()` | Raw HDF5 filter IDs. |
| `filter_names()` | Filter names (`"gzip"`, `"shuffle"`, `"fletcher32"`, or `"filter_<id>"` for unrecognized ones). |
| `compression()` | `"gzip"` if gzip is applied, else `""`. |
| `compression_opts()` | gzip level, or `-1` if gzip is absent. |
| `shuffle()` | `True` if the shuffle filter is applied. |
| `fletcher32()` | `True` if the fletcher32 checksum filter is applied. |

**Whole-buffer reading and writing**

| Method | Description |
|--------|-------------|
| `read[dtype](buf, n)` | Read into a pre-allocated buffer. |
| `write[dtype](data, n)` | Write from a buffer to the dataset. |
| `read[dtype]()` | Read the entire dataset into an `NDArray[dtype]` (any rank >= 1). |
| `read_direct[dtype](mut dest)` / `read_direct[dtype](mut dest, source_sel: List[Slice])` | Read the whole dataset, or a rectangular selection, into an existing `NDArray` (sizes must match). |
| `write_direct[dtype](source)` / `write_direct[dtype](source, dest_sel: List[Slice])` | Write an `NDArray` to the whole dataset or a rectangular selection. |
| `write[dtype](array: NDArray[dtype])` | Write an entire `NDArray` to the dataset. |

**Scalar (rank-0) datasets**

| Method | Description |
|--------|-------------|
| `read_scalar[dtype]()` | Read a scalar dataset's value. |
| `read_strings() -> List[String]` | Read a string dataset (variable- or fixed-length) as a flat row-major list. |
| `write_strings(values)` | Write a flat list of strings (fixed-length datasets reject strings that do not fit). |
| `write_scalar[dtype](value)` | Write a scalar dataset's value. |
| `dset[()].read_scalar[dtype]()` | h5py-style read of a scalar dataset. |
| `dset[()] = value` | h5py-style write of a scalar dataset. |

**Indexing (`Selection`)**

| Syntax | Description |
|--------|-------------|
| `dset[1:3, 2]`, `dset[..., 0]`, `dset[-1, 2:]` | Lazy `Selection`; ints drop their axis. |
| `sel.read[dtype]()` / `sel.read_scalar[dtype]()` | Read into an `NDArray` / a single value. |
| `dset[0:3, 1] = arr` / `= scalar` | Write; a one-element value is broadcast. |
| `obj.attrs["k"] = v`, `obj.attrs["k"].as_scalar[dtype]()` | Attribute item access on `File`/`Group`/`Dataset`. |

**Hyperslab, slice, and point I/O**

Slices and points support simple rectangular, unit-stride regions; see
[Known limitations](../README.md#known-limitations) for what's not yet
supported (fancy indexing, multi-block selections).

| Method | Description |
|--------|-------------|
| `read_hyperslab[dtype](start, count, buf)` | Read a contiguous hyperslab into a compact buffer. |
| `write_hyperslab[dtype](start, count, data)` | Write a compact buffer into a contiguous hyperslab. |
| `read_slice[dtype](start, stop) -> NDArray[dtype]` | Read a rectangular region (any rank). |
| `read_slice[dtype](slices: List[Slice]) -> NDArray[dtype]` | Same, described with Mojo `Slice` values (unit step only). |
| `write_slice[dtype](start, stop, data)` | Write a compact buffer into a rectangular region. |
| `write_slice[dtype](slices: List[Slice], data)` | Same, described with Mojo `Slice` values. |
| `read_point[dtype](indices) -> Scalar[dtype]` | Read a single element by per-dimension index. |
| `write_point[dtype](indices, value)` | Write a single element by per-dimension index. |

**Operations**

| Method | Description |
|--------|-------------|
| `close()` | Close the dataset. |

---

## H5Object

Polymorphic wrapper for `Group` or `Dataset`, returned by `get()` /
`get_opt()` / `__getitem__()` / `values()`.

**Type checking**

| Method | Description |
|--------|-------------|
| `is_group()` | `True` if this is a `Group`. |
| `is_dataset()` | `True` if this is a `Dataset`. |

**Unwrapping**

| Method | Description |
|--------|-------------|
| `group()` | Unwrap as `Group` (raises if not a group). |
| `dataset()` | Unwrap as `Dataset` (raises if not a dataset). |

---

## AttributeManager

Dict-like proxy for HDF5 attributes on `File`, `Group`, or `Dataset`
objects. Access via the `.attrs` field:

```mojo
f.attrs["created"] = Int32(42)                  # write (overwrites existing)
f.attrs["units"] = "m/s"                        # String, scalars, NDArray convert implicitly
var v = f.attrs["created"].as_scalar[DType.int32]()   # read via AttrValue
var d = f.attrs.get[DType.int32]("version", Int32(0)) # read with default
del f.attrs["temp_attr"]                        # delete
```

**Access**

| Method | Description |
|--------|-------------|
| `__contains__(name)` / `contains(name)` | Check if an attribute exists. |
| `__getitem__(name)` | Read an attribute as an `AttrValue` (raises if missing). |
| `__setitem__(name, value)` | Write an attribute; `value` converts implicitly to `AttrValue` (scalars, `Int`, `Float64`, `String`, `NDArray`). |
| `read_scalar[dtype](name)` | Read an attribute value (raises if missing). |
| `write_scalar[dtype](name, value)` | Write an attribute value; overwrites an existing attribute of the same name. |
| `get[dtype](name, default)` | Read an attribute value, or `default` if absent. |
| `set[dtype](name, value)` | Write an attribute value (alias for `write_scalar`). |
| `set_string(name, value)` | Write a scalar variable-length UTF-8 string attribute (h5py layout; overwrites). |
| `get_string(name)` | Read a scalar string attribute (variable- or fixed-length). |
| `read_array[dtype](name)` | Read an array attribute (rank >= 1) as an `NDArray`. |
| `write_array[dtype](name, array)` | Write an `NDArray` as an array attribute (overwrites). |
| `shape(name)` | Attribute shape (`[]` for scalar attributes). |
| `dtype(name)` | Attribute datatype string. |
| `len(attrs)` / `len()` | Number of attributes. |
| `keys()` | List of attribute names (creation order if tracked, else name order). |
| `for name in attrs` | Iterate over attribute names. |
| `get_value(name)` | Read an attribute of any dtype/shape as an `AttrValue`. |
| `items()` | List of `(name, AttrValue)` pairs. |
| `values()` | List of `AttrValue`. |

**Operations**

| Method | Description |
|--------|-------------|
| `delete(name)` / `__delitem__(name)` | Delete an attribute. |

---

## LinkInfo

Returned by `get_link(name)`. Fields: `kind` (`"hard"`, `"soft"` or
`"external"`), `path` (target path for soft/external links) and `filename`
(target file for external links).

---

## AttrValue

Type-erased attribute value returned by `attrs[name]`, `attrs.get_value()`, `items()` and
`values()`. Fields: `dtype` (e.g. `"float64"`) and `shape` (`[]` for scalars).

| Method | Description |
|--------|-------------|
| `is_scalar()` / `size()` | Scalar check / number of elements. |
| `as_string()` | Extract a string attribute's value (raises if the attribute is not a string). |
| `as_scalar[dtype]()` | Extract a scalar value (raises on dtype mismatch or arrays). |
| `as_array[dtype]()` | Extract an array value as an `NDArray[dtype]` (raises on mismatch or scalars). |

---

## NDArray[dtype]

Re-exported from [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo);
used as the return type of whole-dataset `read[dtype]()` and slice reads
(`read_slice`), and accepted directly as a write buffer.
See NuMojo's own documentation for its full API.
