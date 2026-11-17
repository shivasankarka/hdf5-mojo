from std.testing import TestSuite, assert_equal, assert_true

from hdf5 import File


def _write_without_close() raises:
    var f = File("tests/test_autoclose.h5", "w")
    _ = f.create_dataset[DType.int32]("a", [3])


def test_autoclose_on_destruction() raises:
    _write_without_close()
    var g = File("tests/test_autoclose.h5", "r")
    assert_true("a" in g)
    g.close()


def test_with_block() raises:
    with File("tests/test_with.h5", "w") as f:
        _ = f.create_dataset[DType.int32]("a", [3])
    var g = File("tests/test_with.h5", "r")
    assert_true("a" in g)
    g.close()


def test_copies_share_state() raises:
    var f = File("tests/test_copy.h5", "w")
    var g = f.copy()
    _ = g.create_dataset[DType.int32]("a", [2])
    assert_true(f.__bool__())
    g.close()
    assert_true(not f.__bool__())
    f.close()  # idempotent


def test_len_nbytes() raises:
    var f = File("tests/test_len.h5", "w")
    var d = f.create_dataset[DType.int16]("a", [5, 3])
    assert_equal(len(d), 5)
    assert_equal(d.nbytes(), 30)
    var s = f.create_scalar_dataset[DType.float64]("s", 1.0)
    assert_equal(len(s), 0)
    assert_equal(s.nbytes(), 8)
    d.close()
    s.close()
    f.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
