from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from numojo.prelude import NDArray, Shape

from hdf5 import File


def test_read_3d_roundtrip() raises:
    var f = File("tests/test_nd.h5", "w")
    var arr = NDArray[DType.float64](Shape(2, 3, 4))
    for i in range(24):
        arr.unsafe_ptr()[unsafe_offset=i] = Float64(i)
    var d = f.create_dataset_with_data("x", arr)
    assert_equal(len(d.shape()), 3)
    var back = d.read[DType.float64]()
    assert_equal(back.size, 24)
    for i in range(24):
        assert_true(back.unsafe_ptr()[unsafe_offset=i] == Float64(i))
    d.close()
    f.close()


def test_read_4d_and_slice() raises:
    var f = File("tests/test_nd4.h5", "w")
    var d = f.create_dataset[DType.int32]("x", [2, 3, 4, 5])
    var arr = NDArray[DType.int32](Shape(2, 3, 4, 5))
    for i in range(120):
        arr.unsafe_ptr()[unsafe_offset=i] = Int32(i)
    d.write[DType.int32](arr)
    var sub = d.read_slice[DType.int32]([1, 0, 0, 0], [2, 2, 4, 5])
    assert_equal(sub.size, 40)
    assert_true(sub.unsafe_ptr()[unsafe_offset=0] == Int32(60))
    d.close()
    f.close()


def test_read_direct_write_direct() raises:
    var f = File("tests/test_direct.h5", "w")
    var d = f.create_dataset[DType.float32]("x", [4, 4], Float32(0))
    var src = NDArray[DType.float32](Shape(2, 2))
    for i in range(4):
        src.unsafe_ptr()[unsafe_offset=i] = Float32(i + 1)
    d.write_direct[DType.float32](src, [Slice(1, 3), Slice(1, 3)])
    var out = NDArray[DType.float32](Shape(2, 2))
    d.read_direct[DType.float32](out, [Slice(1, 3), Slice(1, 3)])
    for i in range(4):
        assert_true(out.unsafe_ptr()[unsafe_offset=i] == Float32(i + 1))
    var whole = NDArray[DType.float32](Shape(4, 4))
    d.read_direct[DType.float32](whole)
    assert_true(whole.unsafe_ptr()[unsafe_offset=0] == Float32(0))
    assert_true(whole.unsafe_ptr()[unsafe_offset=5] == Float32(1))
    var bad = NDArray[DType.float32](Shape(3))
    with assert_raises():
        d.read_direct[DType.float32](bad)
    d.close()
    f.close()


def test_scalar_read_raises() raises:
    var f = File("tests/test_nd_scalar.h5", "w")
    var d = f.create_scalar_dataset[DType.int32]("s", Int32(3))
    with assert_raises():
        _ = d.read[DType.int32]()
    d.close()
    f.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
