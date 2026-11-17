from std.testing import TestSuite, assert_equal, assert_true, assert_raises

from hdf5 import File


def test_move_and_copy() raises:
    var f = File("tests/test_mc.h5", "w")
    var g = f.create_group("g")
    var d = g.create_dataset[DType.int32]("a", [3])
    d.attrs().set[DType.int32]("tag", Int32(7))
    d.close()
    g.move("a", "b")
    assert_true("b" in g)
    assert_true("a" not in g)
    f.copy_object("g/b", "copied")
    assert_true("copied" in f)
    assert_true("b" in g)
    var c = f.get("copied").dataset()
    assert_equal(Int(c.attrs().read_scalar[DType.int32]("tag")), 7)
    with assert_raises():
        f.move("missing", "x")
    with assert_raises():
        f.copy_object("g/b", "copied")
    c.close()
    f.close()


def test_soft_and_hard_links() raises:
    var f = File("tests/test_links.h5", "w")
    var d = f.create_dataset[DType.float64]("real", [4])
    d.close()
    f.create_soft_link("soft", "/real")
    f.create_soft_link("soft2", "/real")
    f.create_hard_link("hard", "real")
    var ds = f.get("soft").dataset()
    assert_equal(ds.shape()[0], 4)
    ds.close()
    assert_equal(f.get_link("soft").kind, "soft")
    assert_equal(f.get_link("soft").path, "/real")
    assert_equal(f.get_link("hard").kind, "hard")
    assert_equal(f.get_link("real").kind, "hard")
    with assert_raises():
        _ = f.get_link("nope")
    with assert_raises():
        f.create_hard_link("x", "nope")
    f.close()


def test_external_link() raises:
    var other = File("tests/test_ext_target.h5", "w")
    var d = other.create_dataset[DType.int32]("data", [5])
    d.close()
    other.close()
    var f = File("tests/test_ext.h5", "w")
    f.create_external_link("ext", "tests/test_ext_target.h5", "/data")
    var info = f.get_link("ext")
    assert_equal(info.kind, "external")
    assert_equal(info.filename, "tests/test_ext_target.h5")
    assert_equal(info.path, "/data")
    var ds = f.get("ext").dataset()
    assert_equal(ds.shape()[0], 5)
    ds.close()
    f.close()


def test_visit_names_and_dedupe() raises:
    var f = File("tests/test_visit.h5", "w")
    var a = f.create_group("a")
    _ = a.create_group("deep")
    var d = a.create_dataset[DType.int32]("x", [2])
    d.close()
    var e = f.create_dataset[DType.int32]("top", [1])
    e.close()
    f.create_soft_link("soft", "/a")  # not followed
    f.create_hard_link("hard", "top")  # same object as "top": listed once
    var names = f.visit_names()
    assert_equal(len(names), 4)
    assert_true("a" in names)
    assert_true("a/deep" in names)
    assert_true("a/x" in names)
    assert_true(("top" in names) != ("hard" in names))
    assert_true("soft" not in names)
    var sub = a.visit_names()
    assert_equal(len(sub), 2)
    f.close()


def stop_at_x(name: String) raises -> Bool:
    return name.endswith("/x")


def never(name: String) raises -> Bool:
    return False


def test_visit_callback_stops() raises:
    var f = File("tests/test_visitcb.h5", "w")
    var a = f.create_group("a")
    var d = a.create_dataset[DType.int32]("x", [2])
    d.close()
    var hit = f.visit[stop_at_x]()
    assert_true(Bool(hit))
    assert_equal(hit.value(), "a/x")
    assert_true(not Bool(f.visit[never]()))
    f.close()


def test_create_dataset_like() raises:
    var f = File("tests/test_like.h5", "w")
    var src = f.create_dataset[DType.int16](
        "src",
        [10, 4],
        fillvalue=Int16(-2),
        chunks=[5, 4],
        maxshape=[-1, 4],
        compression="gzip",
        shuffle=True,
    )
    var dup = f.create_dataset_like("dup", "src")
    assert_equal(dup.dtype(), "int16")
    assert_equal(dup.shape()[0], 10)
    assert_equal(dup.chunks()[0], 5)
    assert_equal(dup.maxshape()[0], -1)
    assert_equal(dup.compression(), "gzip")
    assert_true(dup.shuffle())
    assert_equal(Int(dup.fillvalue[DType.int16]()), -2)
    src.close()
    dup.close()
    f.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
