# API Reference

## File

The main entry point for HDF5 file operations.

**Constructor**

| Method | Description |
|--------|-------------|
| `File(path, mode)` | Open/create a file. Mode: "r" (read), "r+" (read/write), "w" (truncate), "w-" (create/fail if exists), "x" (same as "w-"), "a" (append). |

**Properties**

| Method | Description |
|--------|-------------|
| `filename()` | Return the filename path. |
| `mode()` | Return the access mode string. |
| `name()` | Return "/" (root group name). |
| `attrs()` | Return AttributeManager for file-level attributes. |

**Access**

| Method | Description |
|--------|-------------|
| `contains(name)` | Check if a root-level member exists. |
| `keys()` | Get list of root-level member names. |
| `get(name)` | Get an H5Object by name. |

**Creation**

| Method | Description |
|--------|-------------|
| `create_group(name)` | Create a group at root level. |
| `require_group(name)` | Open existing or create new group. |
| `create_dataset[dtype](name, shape)` | Create an empty dataset. |
| `create_dataset_with_data[dtype](name, shape, data)` | Create and write dataset. |

**Operations**

| Method | Description |
|--------|-------------|
| `close()` | Close the file and release resources. |
| `flush()` | Flush pending writes to disk. |
| `__bool__()` | Check if file is open. |

---

## Group

Represents an HDF5 group with dict-like access to members.

**Properties**

| Method | Description |
|--------|-------------|
| `name()` | Return full path of the group. |
| `attrs()` | Return AttributeManager for group attributes. |

**Access**

| Method | Description |
|--------|-------------|
| `contains(name)` | Check if member exists. |
| `keys()` | Get list of member names. |
| `len()` | Get number of members. |
| `__iter__()` | Iterate over member names. |
| `items()` | Get list of member names. |
| `get(name)` | Get H5Object by name. |

**Creation**

| Method | Description |
|--------|-------------|
| `create_group(name)` | Create a group (supports nested paths). |
| `require_group(name)` | Open existing or create new group. |
| `create_dataset[dtype](name, shape)` | Create an empty dataset. |
| `create_dataset_with_data[dtype](name, shape, data)` | Create and write dataset. |

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
| `shape()` | Return list of dimensions, e.g., [100, 50]. |
| `ndim()` | Return number of dimensions. |
| `size()` | Return total number of elements. |
| `dtype()` | Return datatype string: "float64", "float32", "int32", "int64". |
| `name()` | Return full path of the dataset. |
| `attrs()` | Return AttributeManager for dataset attributes. |

**Reading**

| Method | Description |
|--------|-------------|
| `read[dtype](buf, n)` | Read into pre-allocated buffer. |
| `read_all[dtype]()` | Read entire dataset into NDArray. |

**Writing**

| Method | Description |
|--------|-------------|
| `write[dtype](data, n)` | Write from buffer to dataset. |
| `write_all[dtype](data)` | Write entire buffer to dataset. |

**Operations**

| Method | Description |
|--------|-------------|
| `close()` | Close the dataset. |

---

## H5Object

Polymorphic wrapper for Groups or Datasets returned by `__getitem__`.

**Type Checking**

| Method | Description |
|--------|-------------|
| `is_group()` | Return True if this is a Group. |
| `is_dataset()` | Return True if this is a Dataset. |

**Unwrapping**

| Method | Description |
|--------|-------------|
| `group()` | Unwrap as Group (raises if not a group). |
| `dataset()` | Unwrap as Dataset (raises if not a dataset). |

---

## AttributeManager

Dict-like proxy for HDF5 attributes on Groups or Datasets.

Access via `.attrs()` method:

```mojo
var version = f.attrs().get[DType.int32]("version", Int32(0))    # read
f.attrs().set[DType.int32]("created", Int32(42))            # write
f.attrs().delete("temp_attr")                            # delete
```

**Access**

| Method | Description |
|--------|-------------|
| `__contains__(name)` | Check if attribute exists. |
| `__getitem__[dtype](name)` | Read attribute value. |
| `__setitem__[dtype](name, value)` | Write attribute value. |

**Operations**

| Method | Description |
|--------|-------------|
| `read_scalar[dtype](name)` | Read attribute value. |
| `write_scalar[dtype](name, value)` | Write attribute value. |
| `delete(name)` | Delete attribute. |
| `keys()` | Get list of attribute names. |
| `__delitem__(name)` | Delete attribute (via `del`). |

---

## NDArray[dtype]

Heap-allocated shaped array returned by `Dataset.read_all()`.

Call `.free()` when done to release memory.

**Properties**

| Field | Description |
|-------|-------------|
| `data` | Raw pointer to heap buffer. |
| `dim0` | Size of first dimension (0 for 2D). |
| `dim1` | Size of second dimension (0 for 1D). |

**Methods**

| Method | Description |
|--------|-------------|
| `__getitem__(i)` | Index 1-D array. |
| `__getitem__(row, col)` | Index 2-D array (row-major). |
| `size()` | Total number of elements. |
| `free()` | Release the heap buffer. |

> **Note:** `NDArray` will be replaced in a future release by the `NDArray` type from [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo).
