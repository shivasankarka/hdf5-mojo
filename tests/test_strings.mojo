from std.testing import TestSuite, assert_equal, assert_true, assert_raises

from hdf5 import File


def test_string_attr_roundtrip() raises:
    var f = File("tests/test_str_attr.h5", "w")
    f.attrs().set_string("title", "hello héllo")
    f.attrs().set_string("empty", "")
    f.attrs().set[DType.int32]("n", Int32(3))
    f.attrs().set_string("title", "overwritten")
    f.close()

    var r = File("tests/test_str_attr.h5", "r")
    assert_equal(r.attrs().get_string("title"), "overwritten")
    assert_equal(r.attrs().get_string("empty"), "")
    assert_equal(r.attrs().dtype("title"), "string")
    assert_equal(len(r.attrs().shape("title")), 0)
    with assert_raises():
        _ = r.attrs().get_string("n")
    with assert_raises():
        _ = r.attrs().read_scalar[DType.int32]("title")
    var v = r.attrs().get_value("title")
    assert_equal(v.dtype, "string")
    assert_equal(v.as_string(), "overwritten")
    with assert_raises():
        _ = v.as_scalar[DType.int32]()
    assert_equal(len(r.attrs().items()), 3)
    r.close()


def test_vlen_string_dataset() raises:
    var f = File("tests/test_vlen.h5", "w")
    var vals: List[String] = ["alpha", "", "gamma ünï", "d"]
    var d = f.create_dataset_with_strings("names", vals)
    assert_equal(d.dtype(), "string")
    assert_equal(len(d), 4)
    d.close()
    f.close()

    var r = File("tests/test_vlen.h5", "r")
    var back = r.get("names").dataset().read_strings()
    assert_equal(len(back), 4)
    assert_equal(back[0], "alpha")
    assert_equal(back[1], "")
    assert_equal(back[2], "gamma ünï")
    assert_equal(back[3], "d")
    r.close()


def test_fixed_string_dataset() raises:
    var f = File("tests/test_fixed.h5", "w")
    var vals: List[String] = ["ab", "abcdefgh", ""]
    var d = f.create_dataset_with_strings("fx", vals, fixed_length=8)
    assert_equal(d.dtype(), "string")
    with assert_raises():
        var too_long: List[String] = ["ab", "abcdefghi", ""]
        d.write_strings(too_long)
    d.close()
    f.close()

    var r = File("tests/test_fixed.h5", "r")
    var back = r.get("fx").dataset().read_strings()
    assert_equal(back[0], "ab")
    assert_equal(back[1], "abcdefgh")
    assert_equal(back[2], "")
    r.close()


def test_nd_and_scalar_strings() raises:
    var f = File("tests/test_str_nd.h5", "w")
    var d = f.create_string_dataset("grid", [2, 2])
    var vals: List[String] = ["a", "b", "c", "d"]
    d.write_strings(vals)
    var s = f.create_string_dataset("one", [])
    var one: List[String] = ["solo"]
    s.write_strings(one)
    with assert_raises():
        var wrong: List[String] = ["x"]
        d.write_strings(wrong)
    var back = d.read_strings()
    assert_equal(back[3], "d")
    assert_equal(s.read_strings()[0], "solo")
    d.close()
    s.close()
    f.close()


def test_numeric_dataset_rejects_strings() raises:
    var f = File("tests/test_str_guard.h5", "w")
    var d = f.create_dataset[DType.int32]("n", [2])
    with assert_raises():
        _ = d.read_strings()
    d.close()
    f.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
