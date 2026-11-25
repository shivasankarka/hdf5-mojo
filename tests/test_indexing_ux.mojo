from std.testing import TestSuite, assert_equal, assert_true, assert_raises

from hdf5 import File, f64, i32
from numojo.prelude import NDArray, Shape


def _grid() raises -> NDArray[DType.float64]:
    var a = NDArray[DType.float64](Shape(4, 5))
    for i in range(20):
        a.unsafe_ptr()[unsafe_offset=i] = Float64(i)
    return a^


def test_selection_read() raises:
    var f = File("tests/test_ux_sel.h5", "w")
    var d = f.create_dataset[f64]("m", [4, 5])
    d[...] = _grid()

    var col = d[1:3, 2].read[f64]()
    assert_equal(col.size, 2)
    assert_equal(col.unsafe_ptr()[unsafe_offset=0], 7.0)
    assert_equal(col.unsafe_ptr()[unsafe_offset=1], 12.0)

    var first = d[..., 0].read[f64]()
    assert_equal(first.size, 4)
    assert_equal(first.unsafe_ptr()[unsafe_offset=3], 15.0)

    var neg = d[-1, 2:].read[f64]()
    assert_equal(neg.size, 3)
    assert_equal(neg.unsafe_ptr()[unsafe_offset=0], 17.0)

    assert_equal(d[2, 3].read_scalar[f64](), 13.0)
    assert_equal(d[()].read[f64]().size, 20)
    assert_equal(len(d[1:3, 2:5].shape()), 2)

    with assert_raises():
        _ = d[4, 0].read_scalar[f64]()
    with assert_raises():
        _ = d[0, 0, 0].read_scalar[f64]()
    with assert_raises():
        _ = d[::2, 0].read[f64]()
    d.close()
    f.close()


def test_selection_write() raises:
    var f = File("tests/test_ux_write.h5", "w")
    var d = f.create_dataset[f64]("m", [4, 5])
    d[...] = Float64(0)
    d[0, 0:2] = Float64(-1)
    d[3, ...] = 9
    var patch = NDArray[DType.float64](Shape(2, 2))
    for i in range(4):
        patch.unsafe_ptr()[unsafe_offset=i] = Float64(100 + i)
    d[1:3, 1:3] = patch

    var all = d[()].read[f64]()
    var p = all.unsafe_ptr()
    assert_equal(p[unsafe_offset=0], -1.0)
    assert_equal(p[unsafe_offset=2], 0.0)
    assert_equal(p[unsafe_offset=6], 100.0)
    assert_equal(p[unsafe_offset=12], 103.0)
    assert_equal(p[unsafe_offset=19], 9.0)
    with assert_raises():
        d[0, 0:3] = patch
    d.close()
    f.close()


def test_scalar_dataset_indexing() raises:
    var f = File("tests/test_ux_scalar.h5", "w")
    var s = f.create_scalar_dataset[i32]("s", Int32(0))
    s[()] = Int32(7)
    assert_equal(s[()].read_scalar[i32](), 7)
    s.close()
    f.close()


def test_attrs_item_syntax() raises:
    var f = File("tests/test_ux_attrs.h5", "w")
    f.attrs["n"] = 3
    f.attrs["x"] = Float64(2.5)
    f.attrs["i"] = Int32(7)
    f.attrs["s"] = "hello"
    var arr = NDArray[DType.float64](Shape(3))
    arr.unsafe_ptr()[unsafe_offset=2] = 4.0
    f.attrs["v"] = arr

    var g = f.create_group("g")
    g.attrs["version"] = 2
    var d = f.create_dataset[f64]("d", [3])
    d.attrs["units"] = "m"
    g.close()
    d.close()
    f.close()

    var r = File("tests/test_ux_attrs.h5", "r")
    assert_equal(r.attrs["n"].as_scalar[DType.int64](), 3)
    assert_equal(r.attrs["x"].as_scalar[f64](), 2.5)
    assert_equal(r.attrs["i"].as_scalar[i32](), 7)
    assert_equal(r.attrs["s"].as_string(), "hello")
    assert_equal(
        r.attrs["v"].as_array[f64]().unsafe_ptr()[unsafe_offset=2], 4.0
    )
    assert_equal(r.attrs()["n"].as_scalar[DType.int64](), 3)
    assert_equal(r["g"].group().attrs["version"].as_scalar[DType.int64](), 2)
    assert_equal(r.dataset("d").attrs["units"].as_string(), "m")
    assert_true("n" in r.attrs)
    r.close()


def test_chaining_and_shortcuts() raises:
    var f = File("tests/test_ux_chain.h5", "w")
    _ = f.create_group("a/b")
    var d = f.create_dataset[f64]("a/b/d", [2])
    d.close()
    assert_equal(f["a"]["b"]["d"].dataset().shape()[0], 2)
    assert_equal(f.group("a").dataset("b/d").shape()[0], 2)
    f.close()


def test_auto_close_and_with() raises:
    var f = File("tests/test_ux_close.h5", "w")
    with f.create_group("g") as g:
        g.attrs["k"] = 1
    # handles close when their last copy is destroyed; reopening must work
    f.close()
    var r = File("tests/test_ux_close.h5", "r")
    assert_equal(r.group("g").attrs["k"].as_scalar[DType.int64](), 1)
    r.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
