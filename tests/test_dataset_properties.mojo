from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal, assert_true

from hdf5 import File


def test_dataset_chunks() raises:
    print("Testing Dataset.chunks on contiguous dataset...")
    var f = File("tests/test_chunks.h5", "w")

    var shape = List[Int]()
    shape.append(10)
    shape.append(20)
    _ = f.create_dataset[DType.float64]("chunked", shape)

    f.close()

    var f2 = File("tests/test_chunks.h5", "r")
    var obj = f2.get("chunked")
    assert_true(obj.is_dataset(), "should be a dataset")
    var dset = obj.dataset()
    var chunks = dset.chunks()
    assert_equal(len(chunks), 0, "contiguous datasets should not report chunks")
    dset.close()
    f2.close()


def test_dataset_maxshape() raises:
    print("\nTesting Dataset.maxshape...")
    var f = File("tests/test_maxshape.h5", "w")

    var shape = List[Int]()
    shape.append(10)
    var dset = f.create_dataset[DType.float64]("data", shape)
    dset.close()

    f.close()

    var f2 = File("tests/test_maxshape.h5", "r")
    var obj = f2.get("data")
    assert_true(obj.is_dataset(), "should be a dataset")
    var ds = obj.dataset()
    var maxshape = ds.maxshape()
    assert_equal(len(maxshape), 1, "maxshape should have 1 dimension")
    assert_equal(maxshape[0], 10, "non-chunked maxshape equals shape")
    ds.close()
    f2.close()


def test_dataset_chunked_resize() raises:
    print("\nTesting chunked dataset metadata and resize...")
    var f = File("tests/test_chunked_resize.h5", "w")

    var shape = List[Int]()
    shape.append(10)
    shape.append(20)
    var maxshape = List[Int]()
    maxshape.append(30)
    maxshape.append(-1)
    var chunks = List[Int]()
    chunks.append(5)
    chunks.append(10)

    var dset = f.create_dataset_chunked[DType.float64](
        "data", shape, maxshape, chunks
    )
    var read_chunks = dset.chunks()
    assert_equal(len(read_chunks), 2, "chunk rank should be 2")
    assert_equal(read_chunks[0], 5, "first chunk dim should be 5")
    assert_equal(read_chunks[1], 10, "second chunk dim should be 10")

    var read_maxshape = dset.maxshape()
    assert_equal(read_maxshape[0], 30, "first max dim should be 30")
    assert_equal(read_maxshape[1], -1, "second max dim should be unlimited")

    var new_shape = List[Int]()
    new_shape.append(12)
    new_shape.append(25)
    dset.resize(new_shape)
    assert_equal(dset.shape()[0], 12, "first resized dim should be 12")
    assert_equal(dset.shape()[1], 25, "second resized dim should be 25")

    dset.close()
    f.close()


def test_dataset_file_property() raises:
    print("\nTesting Dataset.file property...")
    var f = File("tests/test_file_prop.h5", "w")

    var shape = List[Int]()
    shape.append(10)
    _ = f.create_dataset[DType.float64]("data", shape)

    f.close()

    var f2 = File("tests/test_file_prop.h5", "r")
    var obj = f2.get("data")
    assert_true(obj.is_dataset(), "should be a dataset")
    var dset = obj.dataset()
    var filename = dset.file()
    assert_true(len(filename) > 0, "filename should not be empty")
    dset.close()
    f2.close()


def test_dataset_parent_property() raises:
    print("\nTesting Dataset.parent property...")
    var f = File("tests/test_parent.h5", "w")

    _ = f.create_group("group1")
    var grp = f.get("group1")
    assert_true(grp.is_group())
    var g = grp.group()
    var shape = List[Int]()
    shape.append(10)
    _ = g.create_dataset[DType.float64]("data", shape)
    g.close()
    f.close()

    var f2 = File("tests/test_parent.h5", "r")
    var obj = f2.get("group1/data")
    assert_true(obj.is_dataset(), "should be a dataset")
    var dset = obj.dataset()
    var parent = dset.parent()
    assert_equal(parent, "/group1", "parent path should be /group1")
    dset.close()
    f2.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
