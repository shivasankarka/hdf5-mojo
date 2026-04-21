from std.memory import UnsafePointer, alloc
from std.testing import TestSuite, assert_true, assert_equal
from numojo import Item

from hdf5 import File


def test_file_modes() raises:
    print("Testing File modes...")

    var f = File("tests/test_file_modes.h5", "w")
    f.close()

    var f2 = File("tests/test_file_modes.h5", "r")
    print("opened in r mode")
    f2.close()

    var f3 = File("tests/test_file_modes.h5", "r+")
    print("opened in r+ mode")
    f3.close()

    assert_true(True)


def test_file_properties() raises:
    print("\nTesting File properties...")
    var f = File("tests/test_file_props.h5", "w")

    var shape = List[Int]()
    shape.append(10)
    _ = f.create_dataset[DType.float64]("data", shape)

    print("filename:", f.filename())
    print("mode:", f.mode())
    print("name:", f.name())

    f.close()


def test_group_create() raises:
    print("\nTesting Group.create_group...")
    var f = File("tests/test_group_create.h5", "w")

    var grp = f.create_group("mygroup")
    print("created group:", grp.name())
    grp.close()

    var nested = f.create_group("a/b/c")
    print("created nested:", nested.name())
    nested.close()

    f.close()


def test_group_require() raises:
    print("\nTesting Group.require_group...")
    var f = File("tests/test_group_require.h5", "w")

    var grp1 = f.require_group("existing")
    print("require existing:", grp1.name())
    grp1.close()

    var grp2 = f.require_group("new_group")
    print("require new:", grp2.name())
    grp2.close()

    f.close()


def test_group_contains() raises:
    print("\nTesting Group.contains...")
    var f = File("tests/test_group_contains.h5", "w")

    _ = f.create_group("group1")
    var shape = List[Int]()
    shape.append(10)
    _ = f.create_dataset[DType.float64]("data1", shape)

    assert_true(f.contains("group1"), "should contain group1")
    assert_true(f.contains("data1"), "should contain data1")
    assert_true(not f.contains("missing"), "should not contain missing")

    f.close()


def test_group_keys() raises:
    print("\nTesting Group.keys...")
    var f = File("tests/test_group_keys.h5", "w")

    _ = f.create_group("g1")
    _ = f.create_group("g2")
    var shape = List[Int]()
    shape.append(5)
    _ = f.create_dataset[DType.float64]("d1", shape)

    var names = f.keys()
    assert_equal(len(names), 3, "should have 3 items")

    f.close()


def test_group_len() raises:
    print("\nTesting Group.len...")
    var f = File("tests/test_group_len.h5", "w")

    _ = f.create_group("g1")
    _ = f.create_group("g2")
    var shape = List[Int]()
    shape.append(5)
    _ = f.create_dataset[DType.float64]("d1", shape)

    assert_equal(f.len(), 3, "len should be 3")

    f.close()


def test_group_delete() raises:
    print("\nTesting Group.delete...")
    var f = File("tests/test_group_delete.h5", "w")

    _ = f.create_group("to_delete")
    assert_true(f.contains("to_delete"), "should contain before delete")

    f.delete("to_delete")
    assert_true(not f.contains("to_delete"), "should not contain after delete")

    f.close()


def test_group_get() raises:
    print("\nTesting Group.get...")
    var f = File("tests/test_group_get.h5", "w")

    _ = f.create_group("mygroup")
    var shape = List[Int]()
    shape.append(10)
    _ = f.create_dataset[DType.float64]("mydata", shape)

    var obj = f.get("mygroup")
    print("is_group:", obj.is_group())

    var obj2 = f.get("mydata")
    print("is_dataset:", obj2.is_dataset())

    f.close()


def test_dataset_create() raises:
    print("\nTesting Dataset.create...")
    var f = File("tests/test_dataset_create.h5", "w")

    var shape1 = List[Int]()
    shape1.append(100)
    var d1 = f.create_dataset[DType.float64]("data1", shape1)
    print("created 1D:", d1.shape()[0])
    d1.close()

    var shape2 = List[Int]()
    shape2.append(10)
    shape2.append(20)
    var d2 = f.create_dataset[DType.int32]("data2", shape2)
    print("created 2D:", d2.shape()[0], "x", d2.shape()[1])
    d2.close()

    f.close()


def test_dataset_properties() raises:
    print("\nTesting Dataset properties...")
    var f = File("tests/test_dataset_props.h5", "w")

    var shape = List[Int]()
    shape.append(10)
    shape.append(20)
    var dset = f.create_dataset[DType.float64]("data", shape)
    dset.close()

    f.close()

    var f2 = File("tests/test_dataset_props.h5", "r")
    var obj = f2.get("data")
    var ds = obj.dataset()

    assert_equal(ds.shape()[0], 10, "first dim should be 10")
    assert_equal(ds.shape()[1], 20, "second dim should be 20")
    assert_equal(ds.ndim(), 2, "ndim should be 2")
    assert_equal(ds.size(), 200, "size should be 200")
    assert_equal(ds.name(), "/data", "name should be /data (full path)")

    ds.close()
    f2.close()


def test_dataset_chunks() raises:
    print("\nTesting Dataset.chunks...")
    var f = File("tests/test_dataset_chunks.h5", "w")

    var shape = List[Int]()
    shape.append(100)
    var dset = f.create_dataset[DType.float64]("data", shape)
    dset.close()

    f.close()

    var f2 = File("tests/test_dataset_chunks.h5", "r")
    var obj = f2.get("data")
    var ds = obj.dataset()
    var chunks = ds.chunks()
    print("chunks (empty for non-chunked):", len(chunks))
    ds.close()
    f2.close()


def test_dataset_maxshape() raises:
    print("\nTesting Dataset.maxshape...")
    var f = File("tests/test_dataset_maxshape.h5", "w")

    var shape = List[Int]()
    shape.append(10)
    var dset = f.create_dataset[DType.float64]("data", shape)
    dset.close()

    f.close()

    var f2 = File("tests/test_dataset_maxshape.h5", "r")
    var obj = f2.get("data")
    var ds = obj.dataset()
    var maxshape = ds.maxshape()
    print("maxshape:", len(maxshape))
    ds.close()
    f2.close()


def test_dataset_read_write() raises:
    print("\nTesting Dataset read/write...")
    var f = File("tests/test_dataset_rw.h5", "w")

    var n = 5
    var buf = alloc[Scalar[DType.float64]](n)
    buf[0] = 1.0
    buf[1] = 2.0
    buf[2] = 3.0
    buf[3] = 4.0
    buf[4] = 5.0

    var shape = List[Int]()
    shape.append(n)
    var dset = f.create_dataset_with_data[DType.float64]("data", shape, buf)
    dset.close()
    buf.free()

    f.close()

    var f2 = File("tests/test_dataset_rw.h5", "r")
    var obj = f2.get("data")
    var ds = obj.dataset()
    var arr = ds.read_all[DType.float64]()
    assert_equal(arr[Item(0)], 1.0, "first value should be 1.0")
    assert_equal(arr[Item(1)], 2.0, "second value should be 2.0")
    assert_equal(arr[Item(2)], 3.0, "third value should be 3.0")
    assert_equal(arr[Item(3)], 4.0, "fourth value should be 4.0")
    assert_equal(arr[Item(4)], 5.0, "fifth value should be 5.0")
    ds.close()
    f2.close()


def test_dataset_attrs() raises:
    print("\nTesting Dataset.attrs...")
    var f = File("tests/test_dataset_attrs.h5", "w")

    var shape = List[Int]()
    shape.append(10)
    var dset = f.create_dataset[DType.float64]("data", shape)
    dset.attrs().set[DType.float64]("scale", Float64(1.5))
    dset.close()

    f.close()

    var f2 = File("tests/test_dataset_attrs.h5", "r")
    var obj = f2.get("data")
    var ds = obj.dataset()
    var scale = ds.attrs().get[DType.float64]("scale", Float64(0.0))
    assert_equal(scale, 1.5, "attr scale should be 1.5")
    ds.close()
    f2.close()


def test_dataset_file_parent() raises:
    print("\nTesting Dataset.file and .parent...")
    var f = File("tests/test_dataset_fp.h5", "w")

    _ = f.create_group("mygroup")
    var shape = List[Int]()
    shape.append(10)
    var dset = f.create_dataset[DType.float64]("mygroup/data", shape)
    dset.close()

    f.close()

    var f2 = File("tests/test_dataset_fp.h5", "r")
    var obj = f2.get("mygroup/data")
    var ds = obj.dataset()
    print("file:", ds.file())
    print("parent:", ds.parent())
    ds.close()
    f2.close()


def test_attrs_operations() raises:
    print("\nTesting AttributeManager operations...")
    var f = File("tests/test_attrs_ops.h5", "w")

    f.attrs().set[DType.int32]("count", Int32(42))
    f.attrs().set[DType.float64]("scale", Float64(3.14))

    assert_true(f.attrs().contains("count"), "should contain count")
    assert_true(not f.attrs().contains("missing"), "should not contain missing")

    var count = f.attrs().get[DType.int32]("count", Int32(0))
    assert_equal(count, Int32(42), "count should be 42")

    var keys = f.attrs().keys()
    assert_equal(len(keys), 2, "should have 2 attr keys")

    f.close()


def test_h5object_type_check() raises:
    print("\nTesting H5Object type checking...")
    var f = File("tests/test_h5object.h5", "w")

    _ = f.create_group("mygroup")
    var shape = List[Int]()
    shape.append(10)
    _ = f.create_dataset[DType.float64]("mydata", shape)

    f.close()

    var f2 = File("tests/test_h5object.h5", "r")

    var obj1 = f2.get("mygroup")
    assert_true(obj1.is_group(), "mygroup should be a group")
    assert_true(not obj1.is_dataset(), "mygroup should not be a dataset")

    var obj2 = f2.get("mydata")
    assert_true(not obj2.is_group(), "mydata should not be a group")
    assert_true(obj2.is_dataset(), "mydata should be a dataset")

    f2.close()


def test_h5object_unwrap() raises:
    print("\nTesting H5Object unwrap...")
    var f = File("tests/test_h5object2.h5", "w")

    _ = f.create_group("mygroup")
    var shape = List[Int]()
    shape.append(10)
    _ = f.create_dataset[DType.float64]("mydata", shape)

    f.close()

    var f2 = File("tests/test_h5object2.h5", "r")

    var obj1 = f2.get("mygroup")
    if obj1.is_group():
        var grp = obj1.group()
        print("unwrapped group:", grp.name())
        grp.close()

    var obj2 = f2.get("mydata")
    if obj2.is_dataset():
        var ds = obj2.dataset()
        print("unwrapped dataset:", ds.name())
        ds.close()

    f2.close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
