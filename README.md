# hdf5-mojo

High-level HDF5 bindings for Mojo: Read and write HDF5 files using a clean, ergonomic API that wraps the HDF5 C library.

## Motivation
I'm currently porting several scientific-computation libraries to Mojo for my own research. HDF5 is widely used in my field and is a big dependency of one of the projects I'm porting (look out for that one :)). So I decided to write bindings for it.

They work for now, but are still pretty rough. I have lots of ideas to improve the API and make it feel more like Python's `h5py` so that it's more polished and user-friendly.

## Overview

The library is split into two layers:

| Layer | File | Purpose |
|-------|------|---------|
| Low-level | `hdf5/bindings.mojo` | Thin FFI wrapper around the HDF5 C library (`HDF5Lib`) |
| High-level | `hdf5/api.mojo` | Ergonomic Mojo-friendly API (`H5File`, `NDArray`) |

For most use cases, only the high-level API is needed.

## Features

- Read 1-D and 2-D datasets without knowing sizes ahead of time — shapes are discovered automatically.
- Read scalar attributes from groups or datasets.
- Create files and write 1-D and 2-D datasets (row-major).
- `require_group` helper to create nested groups safely.
- Automatic HDF5 library discovery via `$CONDA_PREFIX` (set by pixi), with optional explicit path override.

## Ongoing work
- Align the current API with `h5py` library and make it more user friendly. 
- Add `NuMojo` backend for all array operations so that users can manipulate data with access to much richer features. 

## Supported DType mappings

| Mojo `DType` | HDF5 C type |
|---|---|
| `DType.float64` | `H5T_NATIVE_DOUBLE` |
| `DType.float32` | `H5T_NATIVE_FLOAT` |
| `DType.int32` | `H5T_NATIVE_INT32` |
| `DType.int64` | `H5T_NATIVE_INT64` |

## Installation

Add the following to your `pixi.toml`:

```toml
[workspace]
preview = ["pixi-build"]

[package]
name = "your_project_name"
version = "0.1.0"

[package.build]
backend = {name = "pixi-build-mojo", version = "0.*"}

[package.build.config.pkg]
name = "your_package_name"

[package.host-dependencies]
mojo = = "26.2.*"

[package.build-dependencies]
mojo = = "26.2.*"
hdf5-mojo = { git = "https://github.com/shivasankarka/hdf5-mojo.git", branch = "main"}

[package.run-dependencies]
mojo = = "26.2.*"
hdf5-mojo = { git = "https://github.com/shivasankarka/hdf5-mojo.git", branch = "main"}

[dependencies]
mojo = ">=0.26.2.0,<0.27"
hdf5-mojo = { git = "https://github.com/shivasankarka/hdf5-mojo.git", branch = "main"}
```

Then run:

```bash
pixi install
```

## Quickstart

### Reading a file

```mojo
from hdf5 import H5File

def main() raises:
    var f = H5File("data.h5")

    # Read scalar attributes
    var emin   = f.read_scalar_attr[DType.float64]("/group", "min_energy")
    var nnodes = f.read_scalar_attr[DType.int32]("/group", "number_energy_nodes")

    # Read 1-D and 2-D datasets (sizes discovered automatically)
    var xs  = f.read_1d[DType.float64]("/group/dataset")
    var mat = f.read_2d[DType.float64]("/group/matrix")

    print(xs[0], mat[0, 0])

    xs.free()
    mat.free()
    f.close()
```

### Writing a file

```mojo
from hdf5 import H5File

def main() raises:
    var f = H5File.create("out.h5")

    f.require_group("/results")
    f.write_1d[DType.float64]("/results/energies", ptr, n)
    f.write_2d[DType.float64]("/results/matrix",   ptr, rows, cols)

    f.close()
```

## API Reference

### `H5File`

**Constructors**

| Method | Description |
|--------|-------------|
| `H5File(path)` | Open an existing file read-only. Library auto-detected from `$CONDA_PREFIX`. |
| `H5File(path, lib_path)` | Open read-only with an explicit HDF5 library path. |
| `H5File.create(path)` | Create a new file for writing. |
| `H5File.create(path, lib_path)` | Create with an explicit library path. |

**Reading**

| Method | Description |
|--------|-------------|
| `read_1d[dtype](path) -> NDArray[dtype]` | Read a 1-D dataset. Shape is discovered automatically. |
| `read_2d[dtype](path) -> NDArray[dtype]` | Read a 2-D dataset. Shape is discovered automatically. |
| `read_scalar_attr[dtype](loc_path, attr_name) -> Scalar[dtype]` | Read a scalar attribute from a group or dataset. |

**Writing**

| Method | Description |
|--------|-------------|
| `write_1d[dtype](path, data_ptr, n)` | Create and write a 1-D dataset. |
| `write_2d[dtype](path, data_ptr, rows, cols)` | Create and write a 2-D dataset (row-major). |
| `require_group(name)` | Create a group if it does not already exist. |

**Closing**

| Method | Description |
|--------|-------------|
| `close()` | Flush pending writes and close the file. |

---

### `NDArray[dtype: DType]`

A heap-allocated, shaped array returned by read methods.

| Field / Method | Description |
|----------------|-------------|
| `data` | Raw pointer to the heap buffer. |
| `dim0` | Size of the first dimension. |
| `dim1` | Size of the second dimension (1-D arrays: `0`). |
| `arr[i]` | Index into a 1-D array. |
| `arr[row, col]` | Index into a 2-D array (row-major). |
| `.size()` | Total number of elements (`dim0 * max(dim1, 1)`). |
| `.free()` | Release the underlying heap buffer. |

> **Note:** `NDArray` will be replaced in a future release by the `NDArray` type from [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo), which provides a richer numerical array interface.

## Important notes

- Always call `.free()` on any `NDArray` returned by read methods to avoid memory leaks.
- Always call `.close()` on `H5File` to flush writes and close the file handle.
- Datasets must not already exist when calling `write_1d` / `write_2d`. Use `require_group` to create parent groups first.
- The library defaults to `$CONDA_PREFIX/lib/libhdf5.dylib` (set by pixi). Pass an explicit `lib_path` when the library is located elsewhere.

## Examples

The `examples/` directory contains two programs that read `sample_data.h5`, a small bundled HDF5 file with a structure similar to real neutrino cross-section data:

| File | Description |
|------|-------------|
| `examples/read_sample_api.mojo` | High-level `H5File` API — the recommended approach. |
| `examples/read_sample_bindings.mojo` | Low-level `HDF5Lib` bindings — for cases needing direct C-API control. |

## Project structure

```
hdf5/
  __init__.mojo       # Package entry point
  bindings.mojo       # Low-level HDF5 C FFI wrapper (HDF5Lib)
  api.mojo            # High-level API (H5File, NDArray)
examples/
  sample_data.h5              # Bundled sample HDF5 file
  read_sample_api.mojo        # High-level API example
  read_sample_bindings.mojo   # Low-level bindings example
```

## Troubleshooting

**File fails to open** — Ensure `libhdf5.dylib` is accessible. If using pixi, run `pixi install` first so `$CONDA_PREFIX` is set. Otherwise pass an explicit `lib_path` to the constructor.

**"dataset already exists" on write** — The write helpers refuse to overwrite existing datasets. Choose a new path or recreate the file.

**Memory leak warnings** — Call `.free()` on every `NDArray` returned by read methods and `.close()` on `H5File` when done.

## License

Distributed under the Apache 2.0 License. See `LICENSE` for details.

## Acknowledgement
Huge thanks to the HDF5 maintainers, this cool library exists thanks to their work :)

## Contributing

Contributions are always welcome! Please follow the repository contribution guidelines and include tests or examples where appropriate.

## Contact

For questions or bug reports, open an issue in the repository.
