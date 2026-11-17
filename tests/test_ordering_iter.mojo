from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from numojo.prelude import NDArray, Shape

from hdf5 import File


def test_track_order_groups_and_attrs() raises:
    var f = File("tests/test_order.h5", "w", track_order=True)
    for n in ["zeta", "alpha", "mid"]:
        _ = f.create_group(n)
    f.attrs().set[DType.int32]("z", Int32(1))
    f.attrs().set[DType.int32]("a", Int32(2))
    var g = f.create_group("sub", track_order=True)
    _ = g.create_group("y")
    _ = g.create_group("b")
    var d = f.create_dataset[DType.int32]("d", [2], track_order=True)
    d.attrs().set[DType.int32]("two", Int32(2))
    d.attrs().set[DType.int32]("one", Int32(1))
    f.close()

    var r = File("tests/test_order.h5", "r")
    var keys = r.keys()
    assert_equal(keys[0], "zeta")
    assert_equal(keys[1], "alpha")
    assert_equal(keys[2], "mid")
    var ak = r.attrs().keys()
    assert_equal(ak[0], "z")
    assert_equal(ak[1], "a")
    var sub = r.get("sub").group()
    assert_equal(sub.keys()[0], "y")
    assert_equal(sub.keys()[1], "b")
    var dd = r.get("d").dataset()
    assert_equal(dd.attrs().keys()[0], "two")
    r.close()


def test_default_is_name_order() raises:
    var f = File("tests/test_noorder.h5", "w")
    for n in ["zeta", "alpha"]:
        _ = f.create_group(n)
    f.attrs().set[DType.int32]("z", Int32(1))
    f.attrs().set[DType.int32]("a", Int32(2))
    assert_equal(f.keys()[0], "alpha")
    assert_equal(f.attrs().keys()[0], "a")
    f.close()


def test_for_loops() raises:
    var f = File("tests/test_iter2.h5", "w")
    _ = f.create_group("a")
    var g = f.create_group("g")
    _ = g.create_group("x")
    _ = g.create_group("y")
    f.attrs().set[DType.int32]("p", Int32(1))
    var n = 0
    for _ in f:
        n += 1
    assert_equal(n, 2)
    var m = 0
    for name in g:
        m += 1
        assert_true(name == "x" or name == "y")
    assert_equal(m, 2)
    var k = 0
    for _ in f.attrs():
        k += 1
    assert_equal(k, 1)
    f.close()


def test_attr_items_values() raises:
    var f = File("tests/test_items.h5", "w")
    f.attrs().set[DType.float64]("pi", 3.5)
    f.attrs().set[DType.uint16]("u", UInt16(9))
    var a = NDArray[DType.int32](Shape(2, 2))
    for i in range(4):
        a.unsafe_ptr()[unsafe_offset=i] = Int32(i * 10)
    f.attrs().write_array[DType.int32]("arr", a)
    f.close()

    var r = File("tests/test_items.h5", "r")
    var items = r.attrs().items()
    assert_equal(len(items), 3)
    var seen = 0
    for item in items:
        var name = item[0]
        ref v = item[1]
        if name == "pi":
            assert_equal(v.dtype, "float64")
            assert_true(v.as_scalar[DType.float64]() == 3.5)
            seen += 1
        elif name == "u":
            assert_equal(v.dtype, "uint16")
            assert_true(v.as_scalar[DType.uint16]() == 9)
            seen += 1
        elif name == "arr":
            assert_equal(v.shape[0], 2)
            var back = v.as_array[DType.int32]()
            assert_true(back.unsafe_ptr()[unsafe_offset=3] == 30)
            seen += 1
    assert_equal(seen, 3)
    var vals = r.attrs().values()
    assert_equal(len(vals), 3)
    with assert_raises():
        _ = vals[0].as_scalar[DType.int64]()
    r.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
