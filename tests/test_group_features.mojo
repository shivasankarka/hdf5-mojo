from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_equal, assert_true

from hdf5 import File


def test_group_file_property() raises:
    print("Testing Group.file property...")
    var f = File("tests/test_group_file.h5", "w")

    _ = f.create_group("mygroup")

    f.close()

    var f2 = File("tests/test_group_file.h5", "r")
    var obj = f2.get("mygroup")
    assert_true(obj.is_group(), "should be a group")
    var grp = obj.group()
    var filename = grp.file()
    assert_true(len(filename) > 0, "filename should not be empty")
    grp.close()
    f2.close()


def test_group_parent_property() raises:
    print("\nTesting Group.parent property...")
    var f = File("tests/test_group_parent.h5", "w")

    _ = f.create_group("parent_group")
    var grp = f.get("parent_group")
    assert_true(grp.is_group())
    var g = grp.group()
    _ = g.create_group("child_group")
    g.close()
    f.close()

    var f2 = File("tests/test_group_parent.h5", "r")
    var obj = f2.get("parent_group/child_group")
    assert_true(obj.is_group(), "should be a group")
    var child = obj.group()
    var parent = child.parent()
    assert_equal(parent, "/parent_group", "parent path should be /parent_group")
    child.close()
    f2.close()


def test_group_values() raises:
    print("\nTesting Group.values()...")
    var f = File("tests/test_values.h5", "w")

    _ = f.create_group("group1")
    var shape = List[Int]()
    shape.append(10)
    _ = f.create_dataset[DType.float64]("data1", shape)

    f.close()

    var f2 = File("tests/test_values.h5", "r")
    var names = f2.keys()
    print("root keys:", len(names))
    for n in names:
        print("  -", n)
    f2.close()


def test_require_dataset_existing() raises:
    print("\nTesting Group.require_dataset (existing)...")
    var f = File("tests/test_require_ds.h5", "w")

    var shape = List[Int]()
    shape.append(10)
    _ = f.create_dataset[DType.float64]("existing", shape)

    f.close()

    var f2 = File("tests/test_require_ds.h5", "r+")
    var shape2 = List[Int]()
    shape2.append(20)
    var dset = f2.require_dataset[DType.float64]("existing", shape2)
    assert_equal(dset.shape()[0], 10, "shape should remain 10 (existing)")
    dset.close()
    f2.close()


def test_require_dataset_new() raises:
    print("\nTesting Group.require_dataset (new)...")
    var f = File("tests/test_require_ds2.h5", "w")

    var shape = List[Int]()
    shape.append(15)
    var dset = f.require_dataset[DType.float64]("new_ds", shape)
    assert_equal(dset.shape()[0], 15, "shape should be 15")
    dset.close()

    f.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
