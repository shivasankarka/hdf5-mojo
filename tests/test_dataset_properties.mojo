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


def test_dataset_fillvalue() raises:
    print("\nTesting dataset fillvalue...")
    var f = File("tests/test_fillvalue.h5", "w")

    var shape = List[Int]()
    shape.append(4)
    var dset = f.create_dataset[DType.int32]("data", shape, Int32(7))

    assert_equal(
        dset.fillvalue[DType.int32](), Int32(7), "fillvalue should be 7"
    )

    var buf = alloc[Int32](4)
    dset.read[DType.int32](buf, 4)
    for i in range(4):
        assert_equal(buf[i], Int32(7), "unwritten data should use fillvalue")
    buf.free()

    dset.close()
    f.close()


def test_dataset_chunked_fillvalue() raises:
    print("\nTesting chunked dataset fillvalue...")
    var f = File("tests/test_chunked_fillvalue.h5", "w")

    var shape = List[Int]()
    shape.append(3)
    shape.append(2)
    var maxshape = List[Int]()
    maxshape.append(-1)
    maxshape.append(2)
    var chunks = List[Int]()
    chunks.append(2)
    chunks.append(2)

    var dset = f.create_dataset_chunked[DType.float64](
        "data", shape, maxshape, chunks, Float64(1.25)
    )

    assert_equal(
        dset.fillvalue[DType.float64](),
        Float64(1.25),
        "fillvalue should be 1.25",
    )

    var buf = alloc[Float64](6)
    dset.read[DType.float64](buf, 6)
    for i in range(6):
        assert_equal(
            buf[i], Float64(1.25), "chunked unwritten data should use fillvalue"
        )
    buf.free()

    dset.close()
    f.close()


def test_dataset_filtered_gzip_shuffle_fletcher32() raises:
    print("\nTesting filtered dataset creation...")
    var f = File("tests/test_filtered_dataset.h5", "w")

    var shape = List[Int]()
    shape.append(8)
    var maxshape = List[Int]()
    maxshape.append(-1)
    var chunks = List[Int]()
    chunks.append(4)

    var dset = f.create_dataset_filtered[DType.int32](
        "data", shape, maxshape, chunks, "gzip", 4, True, True
    )
    assert_equal(dset.filter_count(), 3, "should record three filters")
    assert_equal(dset.compression(), "gzip", "compression should be gzip")
    assert_equal(dset.compression_opts(), 4, "gzip level should be 4")
    assert_equal(dset.shuffle(), True, "shuffle should be enabled")
    assert_equal(dset.fletcher32(), True, "fletcher32 should be enabled")

    var filter_ids = dset.filter_ids()
    assert_equal(len(filter_ids), 3, "should return three filter ids")
    assert_equal(filter_ids[0], 2, "shuffle filter should be first")
    assert_equal(filter_ids[1], 1, "gzip filter should be second")
    assert_equal(filter_ids[2], 3, "fletcher32 filter should be third")

    var filter_names = dset.filter_names()
    assert_equal(len(filter_names), 3, "should return three filter names")
    assert_equal(filter_names[0], "shuffle", "first filter should be shuffle")
    assert_equal(filter_names[1], "gzip", "second filter should be gzip")
    assert_equal(
        filter_names[2], "fletcher32", "third filter should be fletcher32"
    )

    var data = alloc[Int32](8)
    for i in range(8):
        data[i] = Int32(i * 3)
    dset.write[DType.int32](data, 8)
    data.free()

    var readback = alloc[Int32](8)
    dset.read[DType.int32](readback, 8)
    for i in range(8):
        assert_equal(
            readback[i],
            Int32(i * 3),
            "filtered dataset should round-trip data",
        )
    readback.free()

    dset.close()
    f.close()


def test_dataset_filter_metadata_empty() raises:
    print("\nTesting empty filter metadata...")
    var f = File("tests/test_empty_filter_metadata.h5", "w")

    var shape = List[Int]()
    shape.append(4)
    var dset = f.create_dataset[DType.int32]("data", shape)

    assert_equal(dset.filter_count(), 0, "contiguous dataset has no filters")
    assert_equal(len(dset.filter_ids()), 0, "filter_ids should be empty")
    assert_equal(len(dset.filter_names()), 0, "filter_names should be empty")
    assert_equal(dset.compression(), "", "compression should be empty")
    assert_equal(dset.compression_opts(), -1, "compression opts should be -1")
    assert_equal(dset.shuffle(), False, "shuffle should be false")
    assert_equal(dset.fletcher32(), False, "fletcher32 should be false")

    dset.close()
    f.close()


def test_dataset_read_hyperslab() raises:
    print("\nTesting Dataset.read_hyperslab...")
    var f = File("tests/test_read_hyperslab.h5", "w")

    var shape = List[Int]()
    shape.append(4)
    shape.append(5)
    var data = alloc[Int32](20)
    for i in range(20):
        data[i] = Int32(i)

    var dset = f.create_dataset_with_data[DType.int32]("data", shape, data)
    data.free()

    var start = List[Int]()
    start.append(1)
    start.append(2)
    var count = List[Int]()
    count.append(2)
    count.append(2)
    var buf = alloc[Int32](4)

    dset.read_hyperslab[DType.int32](start, count, buf)
    assert_equal(buf[0], Int32(7), "first hyperslab value")
    assert_equal(buf[1], Int32(8), "second hyperslab value")
    assert_equal(buf[2], Int32(12), "third hyperslab value")
    assert_equal(buf[3], Int32(13), "fourth hyperslab value")
    buf.free()

    dset.close()
    f.close()


def test_dataset_write_hyperslab() raises:
    print("\nTesting Dataset.write_hyperslab...")
    var f = File("tests/test_write_hyperslab.h5", "w")

    var shape = List[Int]()
    shape.append(4)
    shape.append(5)
    var dset = f.create_dataset[DType.int32]("data", shape, Int32(0))

    var start = List[Int]()
    start.append(1)
    start.append(1)
    var count = List[Int]()
    count.append(2)
    count.append(2)
    var patch = alloc[Int32](4)
    patch[0] = Int32(11)
    patch[1] = Int32(12)
    patch[2] = Int32(21)
    patch[3] = Int32(22)

    dset.write_hyperslab[DType.int32](start, count, patch)
    patch.free()

    var readback = alloc[Int32](20)
    dset.read[DType.int32](readback, 20)
    assert_equal(readback[6], Int32(11), "written row 1 col 1")
    assert_equal(readback[7], Int32(12), "written row 1 col 2")
    assert_equal(readback[11], Int32(21), "written row 2 col 1")
    assert_equal(readback[12], Int32(22), "written row 2 col 2")
    assert_equal(readback[0], Int32(0), "unselected values stay unchanged")
    readback.free()

    dset.close()
    f.close()


def test_dataset_read_write_point() raises:
    print("\nTesting Dataset point reads and writes...")
    var f = File("tests/test_dataset_points.h5", "w")

    var shape1 = List[Int]()
    shape1.append(5)
    var data1 = alloc[Int32](5)
    for i in range(5):
        data1[i] = Int32(i + 10)
    var dset1 = f.create_dataset_with_data[DType.int32]("one_d", shape1, data1)
    data1.free()

    var idx1 = List[Int]()
    idx1.append(2)
    assert_equal(
        dset1.read_point[DType.int32](idx1),
        Int32(12),
        "1D point read should return selected value",
    )
    dset1.write_point[DType.int32](idx1, Int32(99))
    assert_equal(
        dset1.read_point[DType.int32](idx1),
        Int32(99),
        "1D point write should update selected value",
    )
    dset1.close()

    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    var data2 = alloc[Int32](12)
    for i in range(12):
        data2[i] = Int32(i)
    var dset2 = f.create_dataset_with_data[DType.int32]("two_d", shape2, data2)
    data2.free()

    var idx2 = List[Int]()
    idx2.append(1)
    idx2.append(2)
    assert_equal(
        dset2.read_point[DType.int32](idx2),
        Int32(6),
        "2D point read should return selected value",
    )
    dset2.write_point[DType.int32](idx2, Int32(77))
    assert_equal(
        dset2.read_point[DType.int32](idx2),
        Int32(77),
        "2D point write should update selected value",
    )
    dset2.close()
    f.close()


def test_dataset_read_write_slice() raises:
    print("\nTesting Dataset slice reads and writes...")
    var f = File("tests/test_dataset_slices.h5", "w")

    var shape1 = List[Int]()
    shape1.append(6)
    var data1 = alloc[Int32](6)
    for i in range(6):
        data1[i] = Int32(i)
    var dset1 = f.create_dataset_with_data[DType.int32]("one_d", shape1, data1)
    data1.free()

    var none = Optional[Int]()
    var slices1 = List[Slice]()
    slices1.append(Slice(2, 5))
    var arr1 = dset1.read_slice[DType.int32](slices1)
    var arr1_ptr = arr1.unsafe_ptr()
    assert_equal(arr1_ptr[0], Int32(2), "1D slice first value")
    assert_equal(arr1_ptr[1], Int32(3), "1D slice second value")
    assert_equal(arr1_ptr[2], Int32(4), "1D slice third value")

    var patch1 = alloc[Int32](3)
    patch1[0] = Int32(20)
    patch1[1] = Int32(30)
    patch1[2] = Int32(40)
    dset1.write_slice[DType.int32](slices1, patch1)
    patch1.free()
    var start1 = List[Int]()
    start1.append(2)
    assert_equal(
        dset1.read_point[DType.int32](start1),
        Int32(20),
        "1D slice write should update first selected value",
    )
    dset1.close()

    var shape2 = List[Int]()
    shape2.append(4)
    shape2.append(5)
    var data2 = alloc[Int32](20)
    for i in range(20):
        data2[i] = Int32(i)
    var dset2 = f.create_dataset_with_data[DType.int32]("two_d", shape2, data2)
    data2.free()

    var slices2 = List[Slice]()
    slices2.append(Slice(1, 3))
    slices2.append(Slice(Optional[Int](2), none, none))
    var arr2 = dset2.read_slice[DType.int32](slices2)
    var arr2_ptr = arr2.unsafe_ptr()
    assert_equal(arr2_ptr[0], Int32(7), "2D slice first value")
    assert_equal(arr2_ptr[1], Int32(8), "2D slice second value")
    assert_equal(arr2_ptr[2], Int32(9), "2D slice third value")
    assert_equal(arr2_ptr[3], Int32(12), "2D slice fourth value")
    assert_equal(arr2_ptr[4], Int32(13), "2D slice fifth value")
    assert_equal(arr2_ptr[5], Int32(14), "2D slice sixth value")

    var patch2 = alloc[Int32](6)
    patch2[0] = Int32(101)
    patch2[1] = Int32(102)
    patch2[2] = Int32(103)
    patch2[3] = Int32(201)
    patch2[4] = Int32(202)
    patch2[5] = Int32(203)
    dset2.write_slice[DType.int32](slices2, patch2)
    patch2.free()
    var start2 = List[Int]()
    start2.append(1)
    start2.append(2)
    assert_equal(
        dset2.read_point[DType.int32](start2),
        Int32(101),
        "2D slice write should update first selected value",
    )
    var idx2 = List[Int]()
    idx2.append(2)
    idx2.append(4)
    assert_equal(
        dset2.read_point[DType.int32](idx2),
        Int32(203),
        "2D slice write should update last selected value",
    )

    dset2.close()
    f.close()


def test_scalar_dataset_read_write() raises:
    print("\nTesting scalar dataset reads and writes...")
    var f = File("tests/test_scalar_dataset.h5", "w")

    var dset = f.create_scalar_dataset[DType.int32]("scalar", Int32(42))
    assert_equal(len(dset.shape()), 0, "scalar dataset shape should be empty")
    assert_equal(dset.ndim(), 0, "scalar dataset rank should be zero")
    assert_equal(dset.size(), 1, "scalar dataset size should be one")
    assert_equal(
        dset.read_scalar[DType.int32](),
        Int32(42),
        "scalar dataset should read initial value",
    )
    dset.write_scalar[DType.int32](Int32(99))
    assert_equal(
        dset.read_scalar[DType.int32](),
        Int32(99),
        "scalar dataset should read updated value",
    )
    dset.close()
    f.close()

    var f2 = File("tests/test_scalar_dataset.h5", "r")
    var obj = f2.get("scalar")
    assert_true(obj.is_dataset(), "scalar should reopen as a dataset")
    var reopened = obj.dataset()
    assert_equal(
        len(reopened.shape()), 0, "reopened scalar shape should be empty"
    )
    assert_equal(
        reopened.read_scalar[DType.int32](),
        Int32(99),
        "reopened scalar dataset should keep updated value",
    )
    reopened.close()
    f2.close()


def test_scalar_dataset_fillvalue() raises:
    print("\nTesting scalar dataset fillvalue...")
    var f = File("tests/test_scalar_fillvalue.h5", "w")

    var shape = List[Int]()
    var dset = f.create_dataset[DType.float64]("scalar", shape, Float64(1.5))
    assert_equal(len(dset.shape()), 0, "scalar fillvalue shape should be empty")
    assert_equal(
        dset.fillvalue[DType.float64](),
        Float64(1.5),
        "scalar fillvalue metadata should round-trip",
    )
    assert_equal(
        dset.read_scalar[DType.float64](),
        Float64(1.5),
        "scalar fillvalue should be used for unwritten data",
    )

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
    assert_true(filename.byte_length() > 0, "filename should not be empty")
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
