# Changelog

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
