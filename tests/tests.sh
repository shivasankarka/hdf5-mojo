#!/bin/bash
# Run all HDF5 binding tests

set -e

cd "$(dirname "$0")/.."

echo "Running HDF5 binding tests..."

for test_file in tests/test_*.mojo; do
    test_name=$(basename "$test_file" .mojo)
    echo "Running $test_name..."
    pixi run mojo run -I . "$test_file"
done

echo "All tests complete!"
