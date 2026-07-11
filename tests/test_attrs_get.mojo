from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal

from hdf5 import File


def test_attrs_get_existing() raises:
    print("Testing AttributeManager.get (existing)...")
    var f = File("tests/test_attrs_get.h5", "w")

    f.attrs().set[DType.int32]("version", Int32(42))
    f.attrs().set[DType.float64]("scale", Float64(3.14))

    f.close()

    var f2 = File("tests/test_attrs_get.h5", "r")
    var version = f2.attrs().get[DType.int32]("version", Scalar[DType.int32](0))
    assert_equal(version, Int32(42), "version should be 42")

    var scale = f2.attrs().get[DType.float64](
        "scale", Scalar[DType.float64](0.0)
    )
    assert_equal(scale, Float64(3.14), "scale should be 3.14")
    f2.close()


def test_attrs_get_default() raises:
    print("\nTesting AttributeManager.get (default)...")
    var f = File("tests/test_attrs_get2.h5", "w")

    f.attrs().set[DType.int32]("existing", Scalar[DType.int32](100))

    f.close()

    var f2 = File("tests/test_attrs_get2.h5", "r")
    var missing = f2.attrs().get[DType.int32](
        "missing", Scalar[DType.int32](999)
    )
    assert_equal(missing, Int32(999), "missing attr should return default 999")

    var existing = f2.attrs().get[DType.int32](
        "existing", Scalar[DType.int32](0)
    )
    assert_equal(existing, Int32(100), "existing attr should be 100")
    f2.close()


def test_attrs_set_overwrites_existing() raises:
    print("\nTesting AttributeManager.set overwrites existing attr...")
    var f = File("tests/test_attrs_overwrite.h5", "w")

    f.attrs().set[DType.int32]("version", Scalar[DType.int32](1))
    f.attrs().set[DType.int32]("version", Scalar[DType.int32](2))
    var version = f.attrs().get[DType.int32]("version", Scalar[DType.int32](0))
    assert_equal(version, Int32(2), "version should be overwritten to 2")

    f.close()


def test_attrs_long_name_listing() raises:
    print("\nTesting long attribute name listing...")
    var f = File("tests/test_attrs_long_name.h5", "w")

    var long_name = String("attr_")
    for _ in range(320):
        long_name += "x"
    f.attrs().set[DType.int32](long_name, Scalar[DType.int32](7))

    var keys = f.attrs().keys()
    var found = False
    for key in keys:
        if key == long_name:
            found = True
    assert_equal(found, True, "long attr name should round-trip in keys()")

    f.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
