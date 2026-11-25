# Changelog

## Unreleased (UX)

- h5py-style indexing: `dset[1:3, 2]`, `dset[..., 0]`, `dset[-1, 2:]`,
  `dset[()]` return a lazy `Selection`; finish with `.read[f64]()` or
  `.read_scalar[f64]()`. Integers drop their axis, `...` expands. Assign with
  `dset[0:3, 1] = arr` or a scalar (a one-element value is broadcast).
  Strided slices are not supported yet. A `Selection` does not own the
  dataset; do not keep it past the file's lifetime.
- Attribute item syntax: `f.attrs["n"] = 3`, `g.attrs["units"] = "m"`,
  `d.attrs["scale"] = arr`, read with `f.attrs["n"].as_scalar[i64]()`.
  `attrs` is now a field on `File`/`Group`/`Dataset`; `attrs()` still works.
  `Int` is stored as int64, `Float64` as float64, etc. This replaces
  `attrs[name]`/`attrs[name] = v` with an explicit dtype parameter.
- `f.dataset("a/b")`, `f.group("a")`, and chaining `f["a"]["b"]`.
- `Group` and `Dataset` close automatically when their last copy is destroyed,
  and support `with`. `close()` still works (copies share the handle).
  TODO: revisit with Mojo 1.3, where the `Copyable` requirement on `with` is
  lifted.
- Removed: `dset.__getitem__[dtype](())`; use `dset[()].read_scalar[dtype]()`.

## v0.4.0

- Strings: scalar string attributes (`attrs().set_string` / `get_string`),
  string datasets of any shape (`create_string_dataset`,
  `create_dataset_with_strings`, `read_strings`, `write_strings`), variable-length
  UTF-8 by default or fixed-length with `fixed_length=`. Reads variable- and
  fixed-length strings written by h5py (null-terminated, null- or space-padded).
  String attributes also appear in `items()`/`values()`/`get_value()`
  (`AttrValue.as_string()`); `Dataset.dtype()` reports `"string"`.

- Links: `create_soft_link`, `create_hard_link`, `create_external_link` and
  `get_link(name) -> LinkInfo` on `File` and `Group`.
- `move(source, dest)` and `copy(source, dest)` (`File.copy_object`) for
  objects within a file.
- `create_dataset_like(name, other_path)`: empty dataset with the same dtype,
  shape, maxshape, chunking, filters and fill value.
- Traversal: `visit_names()` (recursive object paths; each object once; soft
  and external links not followed) and `visit[func]()` with early stop.

## v0.3.0

- New dtypes: `int16`, `uint16`, `uint32`, `uint64`, `float16`. Unsigned 32/64-bit
  datasets are now identified correctly when reopened from disk.
- `create_dataset` is now h5py-style and takes `fillvalue`, `chunks`, `maxshape`,
  `compression`, `compression_opts`, `shuffle` and `fletcher32`. Chunk shape is
  guessed when chunking is implied but `chunks` is omitted.
  `create_dataset_chunked` / `create_dataset_filtered` remain as thin wrappers.
- `Dataset.read[dtype]()` and `read_slice` now work for any rank; added
  `read_direct` / `write_direct` (whole dataset or a `List[Slice]` selection).
- Array attributes: `read_array`, `write_array`, `shape(name)`, `dtype(name)`,
  `len(attrs)`. `read_scalar` now raises on array attributes instead of
  reading past a one-element buffer.
- `track_order=True` on `File(...)`, `create_group` and `create_dataset`: members and
  attributes are listed in creation order (default stays name order, like h5py).
- `for name in file/group/attrs` now works (`__iter__` returns a `NameIterator`;
  it previously returned a `List`, which Mojo cannot iterate).
- `AttributeManager.items()`, `values()`, `get_value()` and the type-erased
  `AttrValue` (`as_scalar[dtype]`, `as_array[dtype]`).
- `H5Object.group()` / `dataset()` no longer need a mutable receiver.
- `len(dset)`, `Dataset.nbytes()`, `File.create_dataset_with_data(name, NDArray)`.
- `File` is now `Copyable` (copies share one open file) and supports
  `with File(path, mode) as f:`. It closes when the last copy is destroyed, so a
  forgotten `close()` no longer leaks the handle. `close()` is idempotent.

## v0.2.0

- Updated methods to be more `h5py` compatible. 
- Added NuMojo NDArray as the backend for array operations!
- Refactored internal HDF5 library handle ownership:
  - `File` now owns `HDF5Lib` with `OwnedPointer`.
  - Group/dataset/attribute wrappers now carry typed pointer references with
    origin-aware generic parameters.
- Preserved `UnsafePointer` usage only for raw FFI data buffers.
- Updated high-level examples and docs to use `read[dtype]()` consistently.
- Corrected several docstring typos and dtype constraint wording.
- Added `get()` method instead of `__getitem__` temporarily.
- Added dataset creation properties:
  - Explicit chunked dataset creation (`create_dataset_chunked`).
  - `maxshape`, including unlimited dimensions via `-1`, and rank-correct
    `resize()`.
  - `fillvalue` support for contiguous, chunked, and scalar datasets.
  - Built-in filters: gzip compression (`compression="gzip"`,
    `compression_opts`), `shuffle`, and `fletcher32`
    (`create_dataset_filtered`).
  - Filter metadata accessors: `compression()`, `compression_opts()`,
    `shuffle()`, `fletcher32()`, `filter_ids()`, `filter_names()`.
- Added array-like dataset I/O:
  - Hyperslab read/write (`read_hyperslab`/`write_hyperslab`).
  - Simple rectangular slice read/write (`read_slice`/`write_slice`),
    accepting either explicit `start`/`stop` lists or `List[Slice]`.
  - Point read/write (`read_point`/`write_point`).
  - Explicit scalar dataset support (`create_scalar_dataset`,
    `read_scalar`/`write_scalar`), plus h5py-style empty-tuple indexing
    (`dset[()]` / `dset[()] = value`).
- Added type coverage for `int8` and `uint8` datasets and attributes,
  including correct signed/unsigned disambiguation when reopening a
  dataset from disk.
- Added `Group.get_opt(name)` / `File.get_opt(name)`: a non-raising lookup
  returning `Optional[H5Object]`, for h5py's `get(name, default=...)`
  fallback behavior.
- Added `Group.values()` for listing member objects (not just names).
- Added CI (GitHub Actions) running the test suite on Linux and macOS, plus
  a formatting check.
