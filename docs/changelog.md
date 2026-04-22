# Changelog

## v0.2.0

- Refactored internal HDF5 library handle ownership:
  - `File` now owns `HDF5Lib` with `OwnedPointer`.
  - Group/dataset/attribute wrappers now carry typed pointer references with
    origin-aware generic parameters.
- Preserved `UnsafePointer` usage only for raw FFI data buffers.
- Updated high-level examples and docs to use `read[dtype]()` consistently.
- Corrected several docstring typos and dtype constraint wording.
