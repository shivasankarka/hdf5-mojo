# hdf5-mojo

High-level HDF5 bindings for Mojo: Read and write HDF5 files using an h5py-compatible API that wraps the HDF5 C library.

## Overview

I'm working on porting some particle physics simulation libraries to Mojo. Since HDF5 is widely used there, I wrote these bindings to make it easier to use HDF5 datasets directly from Mojo!

It has most of the basic features needed for working with datasets (and for my current projects :) ). Full HDF5 feature parity might come later if I get more free time.

## Features

- Read/write HDF5 files with h5py-style API
- Create groups, datasets, and attributes
- Support for `float64`, `float32`, `int32`, `int64` dtypes
- 1-D and 2-D dataset reading with `read_all[dtype]()`
- `require_group` / `require_dataset` helpers
- Automatic library discovery via pixi

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
from hdf5 import File

def main() raises:
    var f = File("data.h5", "r")

    # Access datasets and groups with .get() method
    var obj = f.get("group/dataset")
    if obj.is_dataset():
        var dset = obj.dataset()
        print(dset.shape())   # [100, 50]
        print(dset.dtype())   # "float64"
        
        # Read all data into an NDArray
        var arr = dset.read_all[DType.float64]()
        print(arr[0, 0])
        arr.free()

    # Iterate over root-level items
    for name in f.keys():
        print(name)

    # Read attributes
    var version = f.attrs().get[DType.int32]("version", Int32(0))

    f.close()
```

### Writing a file

```mojo
from hdf5 import File
from std.memory import UnsafePointer, alloc

def main() raises:
    var f = File("output.h5", "w")

    # Create groups (nested paths are automatically created)
    f.create_group("results/nested")

    # Create datasets
    var shape = List[Int](100, 50)
    var dset = f.create_dataset[DType.float64]("results/data", shape)

    # Write data
    var n = 100 * 50
    var buf = alloc[Scalar[DType.float64]](n)
    # ... fill buf with data ...
    dset.write[DType.float64](buf, n)
    buf.free()

    # Write attributes
    f.attrs().set[DType.int32]("version", Int32(1))

    f.close()
```

## API Reference

See [docs/api_reference.md](docs/api_reference.md) for the complete API documentation.

---

## Examples

See `examples/demo_api.mojo` for a complete example demonstrating the h5py-style API with an 
example dataset. 

## Project structure

```
hdf5/
  __init__.mojo       # Package entry point
  ffi.mojo            # Low-level HDF5 C FFI wrapper (HDF5Lib)
  core.mojo       # High-level h5py-compatible API
examples/
  demo_api.mojo       # h5py-style API demonstration
  sample_data.h5      # Sample HDF5 file
```

## Troubleshooting

**File fails to open** — Ensure `libhdf5.dylib` is accessible. If using pixi, run `pixi install` first so `$CONDA_PREFIX` is set.

**Memory leak warnings** — Call `.free()` on every `NDArray` returned by read methods and `.close()` on `File` when done.

**"dataset already exists" on write** — The write helpers refuse to overwrite existing datasets. Choose a new path or recreate the file.

## License

Distributed under the Apache 2.0 License. See `LICENSE` for details.

## Acknowledgement

Huge thanks to the HDF5 maintainers, this cool library exists thanks to their work :)

## Contributing

Contributions are always welcome! If you think there's a feature missing, make an issue or give it a try and make a PR! 

## Contact

For questions or bug reports, please open an issue in the repository.
