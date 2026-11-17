from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from numojo.prelude import NDArray, Shape

from hdf5 import File


def test_array_attr_roundtrip() raises:
    var f = File("tests/test_attr_arr.h5", "w")
    var a = NDArray[DType.float64](Shape(2, 3))
    for i in range(6):
        a.unsafe_ptr()[unsafe_offset=i] = Float64(i) * 0.5
    f.attrs().write_array[DType.float64]("m", a)
    var g = f.create_group("g")
    g.attrs().write_array[DType.float64]("m", a)
    f.close()

    var f2 = File("tests/test_attr_arr.h5", "r")
    var at = f2.attrs()
    assert_equal(at.shape("m")[0], 2)
    assert_equal(at.shape("m")[1], 3)
    assert_equal(at.dtype("m"), "float64")
    var b = at.read_array[DType.float64]("m")
    assert_equal(b.size, 6)
    assert_true(b.unsafe_ptr()[unsafe_offset=5] == 2.5)
    f2.close()


def test_scalar_and_array_guards() raises:
    var f = File("tests/test_attr_guard.h5", "w")
    var a = NDArray[DType.int32](Shape(3))
    f.attrs().write_array[DType.int32]("arr", a)
    f.attrs().set[DType.int32]("s", Int32(4))
    assert_equal(len(f.attrs().shape("s")), 0)
    with assert_raises():
        _ = f.attrs().read_scalar[DType.int32]("arr")
    with assert_raises():
        _ = f.attrs().read_array[DType.int32]("s")
    with assert_raises():
        _ = f.attrs().read_array[DType.int32]("missing")
    f.close()


def test_len_and_iter() raises:
    var f = File("tests/test_attr_iter.h5", "w")
    f.attrs().set[DType.int32]("a", Int32(1))
    f.attrs().set[DType.float32]("b", Float32(2))
    assert_equal(len(f.attrs()), 2)
    assert_equal(f.attrs().dtype("b"), "float32")
    var n = 0
    for _ in f.attrs().keys():
        n += 1
    assert_equal(n, 2)
    # overwriting keeps the count
    f.attrs().set[DType.int32]("a", Int32(5))
    assert_equal(len(f.attrs()), 2)
    f.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
