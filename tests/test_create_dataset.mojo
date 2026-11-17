from std.testing import TestSuite, assert_equal, assert_true, assert_raises

from hdf5 import File


def test_plain() raises:
    var f = File("tests/test_cd_plain.h5", "w")
    var d = f.create_dataset[DType.float64]("a", [3, 4])
    assert_equal(len(d.chunks()), 0)
    assert_equal(d.shape()[1], 4)
    d.close()
    f.close()


def test_fillvalue_positional() raises:
    var f = File("tests/test_cd_fill.h5", "w")
    var d = f.create_dataset[DType.int32]("a", [5], Int32(-3))
    assert_equal(Int(d.fillvalue[DType.int32]()), -3)
    d.close()
    f.close()


def test_chunks_maxshape_resize() raises:
    var f = File("tests/test_cd_chunk.h5", "w")
    var d = f.create_dataset[DType.float32](
        "a", [10], chunks=[5], maxshape=[-1], fillvalue=Float32(1.5)
    )
    assert_equal(d.chunks()[0], 5)
    assert_equal(d.maxshape()[0], -1)
    d.resize(20)
    assert_equal(d.shape()[0], 20)
    d.close()
    f.close()


def test_filters_with_auto_chunks() raises:
    var f = File("tests/test_cd_filt.h5", "w")
    var g = f.create_group("g")
    var d = g.create_dataset[DType.float64](
        "a",
        [100, 50],
        compression="gzip",
        compression_opts=6,
        shuffle=True,
        fletcher32=True,
    )
    assert_equal(len(d.chunks()), 2)
    assert_equal(d.compression(), "gzip")
    assert_equal(d.compression_opts(), 6)
    assert_true(d.shuffle())
    assert_true(d.fletcher32())
    d.close()
    f.close()


def test_unlimited_auto_chunks() raises:
    var f = File("tests/test_cd_unl.h5", "w")
    var d = f.create_dataset[DType.int64]("a", [0], maxshape=[-1])
    assert_equal(len(d.chunks()), 1)
    assert_true(d.chunks()[0] >= 1)
    d.resize(7)
    assert_equal(d.shape()[0], 7)
    d.close()
    f.close()


def test_legacy_wrappers() raises:
    var f = File("tests/test_cd_legacy.h5", "w")
    var d = f.create_dataset_filtered[DType.int32](
        "a", [8], [-1], [4], "gzip", 3, True, False
    )
    assert_equal(d.compression_opts(), 3)
    var e = f.create_dataset_chunked[DType.int32]("b", [8], [8], [4])
    assert_equal(e.chunks()[0], 4)
    d.close()
    e.close()
    f.close()


def test_errors() raises:
    var f = File("tests/test_cd_err.h5", "w")
    with assert_raises():
        _ = f.create_dataset[DType.int32]("a", [4], compression="lzf")
    with assert_raises():
        _ = f.create_dataset[DType.int32]("b", [4], chunks=[2, 2])
    with assert_raises():
        _ = f.create_dataset[DType.int32]("c", [], shuffle=True)
    f.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
