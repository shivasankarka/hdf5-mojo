from std.memory.alloc import unsafe_alloc
from std.testing import TestSuite, assert_equal, assert_true

from hdf5 import File
from hdf5.core import SupportedDType


def _roundtrip[
    dtype: DType
](name: String, expected: String) raises where SupportedDType[dtype]:
    var path = "tests/test_dtype_" + name + ".h5"
    var f = File(path, "w")
    var shape: List[Int] = [4]
    var dset = f.create_dataset[dtype](name, shape)
    var buf = unsafe_alloc[Scalar[dtype]](4)
    for i in range(4):
        buf[unsafe_offset=i] = Scalar[dtype](i + 1)
    dset.write[dtype](buf, 4)
    buf.unsafe_free()
    dset.attrs().set[dtype]("a", Scalar[dtype](7))
    dset.close()
    f.close()

    var f2 = File(path, "r")
    var obj = f2.get(name)
    var d = obj.dataset()
    assert_equal(d.dtype(), expected)
    var out = unsafe_alloc[Scalar[dtype]](4)
    d.read[dtype](out, 4)
    for i in range(4):
        assert_true(
            out[unsafe_offset=i] == Scalar[dtype](i + 1), "value mismatch"
        )
    out.unsafe_free()
    assert_true(d.attrs().read_scalar[dtype]("a") == Scalar[dtype](7))
    d.close()
    f2.close()


def test_int16() raises:
    _roundtrip[DType.int16]("int16", "int16")


def test_uint16() raises:
    _roundtrip[DType.uint16]("uint16", "uint16")


def test_uint32() raises:
    _roundtrip[DType.uint32]("uint32", "uint32")


def test_uint64() raises:
    _roundtrip[DType.uint64]("uint64", "uint64")


def test_float16() raises:
    _roundtrip[DType.float16]("float16", "float16")


def test_existing_dtypes_still_distinguished() raises:
    _roundtrip[DType.int32]("i32", "int32")
    _roundtrip[DType.int64]("i64", "int64")
    _roundtrip[DType.uint8]("u8", "uint8")
    _roundtrip[DType.int8]("i8", "int8")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
