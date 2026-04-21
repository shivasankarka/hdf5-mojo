from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal, assert_true

from hdf5 import File


def test_dataset_chunks() raises:
    print("Testing Dataset.chunks...")
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
    assert_equal(len(chunks), 2, "chunks returns shape copy")
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
