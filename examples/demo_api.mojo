# demo_api.mojo
#
# Comprehensive tutorial of the File API.
#
# Covers every available method using a simple weather-station dataset
# (demo_data.h5) that is bundled with the repository.
#
# The file layout:
#   /metadata                  — group
#       attrs: num_days (Int32), num_stations (Int32),
#              base_year (Int32), lat_origin (Float64)
#   /sensors                   — group
#       elevation_m            — 1-D Float64 (one value per station)
#       station_id             — 1-D Int32   (integer ids)
#   /observations              — group
#       temperature_c          — 2-D Float64 (num_days × num_stations)
#       humidity_pct           — 2-D Float32 (num_days × num_stations)
#       wind_speed_ms          — 2-D Float64 (num_days × num_stations)
#
# Topics demonstrated:
#   - Opening a file for reading:  File(path, mode)
#   - Reading scalar attributes:   group.attrs().read_scalar[dtype](name)
#   - Reading datasets:            dset.read_all[dtype]()
#   - NDArray shape and indexing:  .dim0, .dim1, .size(), arr.get(i), arr.get(row, col)
#   - Freeing heap buffers:        .free()
#   - Creating a file for writing: File(path, "w")
#   - Creating groups:             require_group(name)
#   - Writing datasets:           create_dataset_with_data[dtype](name, shape, ptr)
#   - Closing handles:            f.close()

from hdf5 import File
from numojo.prelude import NDArray


def main() raises:
    # ===----------------------------------------------------------------------=== #
    # 1. Open a file for reading
    # ===----------------------------------------------------------------------=== #
    var f = File("./examples/demo_data.h5", "r")

    # ===----------------------------------------------------------------------=== #
    # 2. Read scalar attributes
    # ===----------------------------------------------------------------------=== #
    # Attributes can be Float64 or Int32 (pass the matching dtype).
    var meta_obj = f.get("/metadata")
    var meta = meta_obj.group()
    var num_days = meta.attrs().read_scalar[DType.int32]("num_days")
    var num_stations = meta.attrs().read_scalar[DType.int32]("num_stations")
    var base_year = meta.attrs().read_scalar[DType.int32]("base_year")
    var lat_origin = meta.attrs().read_scalar[DType.float64]("lat_origin")
    meta.close()

    print("=== Metadata attributes ===")
    print("  num_days     =", num_days)
    print("  num_stations =", num_stations)
    print("  base_year    =", base_year)
    print("  lat_origin   =", lat_origin, "°N")

    # ===----------------------------------------------------------------------=== #
    # 3. Read a 1-D Float64 dataset
    # Shape is discovered automatically — no need to pass a size.
    # ===----------------------------------------------------------------------=== #
    var elevation_obj = f.get("/sensors/elevation_m")
    var elevation_dset = elevation_obj.dataset()
    var elevation = elevation_dset.read_all[DType.float64]()
    elevation_dset.close()

    print("\n=== 1-D Float64: /sensors/elevation_m ===")
    print("  length (dim0) =", elevation.shape)
    print("  total elements:", elevation.size)
    print("  values:", end="")
    for i in range(elevation.shape[0]):
        print("", elevation.item(i), end="")
    print()

    # ===----------------------------------------------------------------------=== #
    # 4. Read a 1-D Int32 dataset
    # ===----------------------------------------------------------------------=== #
    var station_id_obj = f.get("/sensors/station_id")
    var station_id_dset = station_id_obj.dataset()
    var station_id = station_id_dset.read_all[DType.int32]()
    station_id_dset.close()

    print("\n=== 1-D Int32: /sensors/station_id ===")
    print("  values:", end="")
    for i in range(station_id.shape[0]):
        print("", station_id.item(i), end="")
    print()

    # ===----------------------------------------------------------------------=== #
    # 5. Read a 2-D Float64 dataset
    # arr[row, col] — row-major indexing.
    # ===----------------------------------------------------------------------=== #
    var temp_obj = f.get("/observations/temperature_c")
    var temp_dset = temp_obj.dataset()
    var temp = temp_dset.read_all[DType.float64]()
    temp_dset.close()

    print("\n=== 2-D Float64: /observations/temperature_c ===")
    print("  shape:", temp.shape[0], "×", temp.shape[1], " (days × stations)")
    print("  total elements:", temp.size)
    print("  row 0 (day 1):", end="")
    for s in range(Int(num_stations)):
        print("", temp.item(0, s), end="")
    # or use ndarray syntax to get the first row with temp[0, :]!
    print()
    print("  col 0 (station 1, first 5 days):", end="")
    for d in range(5):
        print("", temp.get(d, 0), end="")
    # or use temp[:, 0] to get the first column (all days for station 1).
    print()


    # ===----------------------------------------------------------------------=== #
    # 6. Read a 2-D Float32 dataset
    # ===----------------------------------------------------------------------=== #
    var humidity_obj = f.get("/observations/humidity_pct")
    var humidity_dset = humidity_obj.dataset()
    var humidity = humidity_dset.read_all[DType.float32]()
    humidity_dset.close()

    print("\n=== 2-D Float32: /observations/humidity_pct ===")
    print("  shape:", humidity.shape[0], "×", humidity.shape[1])
    print("  humidity[0, 0] =", humidity.item(0, 0), "%")
    print("  humidity[14,2] =", humidity.item(14, 2), "%")

    # ===----------------------------------------------------------------------=== #
    # 7. Free file buffer
    # ===----------------------------------------------------------------------=== #
    f.close()

    # ===----------------------------------------------------------------------=== #
    # 8. Write a new file
    # ===----------------------------------------------------------------------=== #
    # File(..., "w") truncates any existing file at the given path.
    var out = File("./examples/demo_output.h5", "w")

    # require_group creates the named group; safe to call if it already exists.
    var summary = out.require_group("/summary")
    var daily = summary.require_group("daily")

    # Write a 1-D Float64 dataset.
    var means = alloc[Float64](Int(num_stations))
    means[0] = 22.1
    means[1] = 19.8
    means[2] = 25.3
    means[3] = 17.6
    means[4] = 21.0
    var means_shape = List[Int]()
    means_shape.append(Int(num_stations))
    var means_dset = summary.create_dataset_with_data[DType.float64](
        "mean_temp_c", means_shape, means
    )
    means_dset.close()
    means.free()

    # Write a 2-D Float64 dataset (3 days × 5 stations, row-major).
    var n_days_out = 3
    var n_stat_out = Int(num_stations)
    var excerpt = alloc[Float64](n_days_out * n_stat_out)
    for d in range(n_days_out):
        for s in range(n_stat_out):
            excerpt[d * n_stat_out + s] = Float64(d * 10 + s)
    var excerpt_shape = List[Int]()
    excerpt_shape.append(n_days_out)
    excerpt_shape.append(n_stat_out)
    var excerpt_dset = daily.create_dataset_with_data[DType.float64](
        "temp_excerpt", excerpt_shape, excerpt
    )
    excerpt_dset.close()
    excerpt.free()

    daily.close()
    summary.close()

    out.close()
    print("\n=== Wrote demo_output.h5 ===")
    print("  /summary/mean_temp_c          — 1-D Float64, length", num_stations)
    print(
        "  /summary/daily/temp_excerpt   — 2-D Float64,",
        n_days_out,
        "×",
        num_stations,
    )

    # ===----------------------------------------------------------------------=== #
    # 9. Round-trip: read back what we wrote
    # ===----------------------------------------------------------------------=== #
    var r = File("./examples/demo_output.h5", "r")

    var summary_obj = r.get("/summary")
    var summary_r = summary_obj.group()

    var mean_obj = summary_r.get("mean_temp_c")
    var mean_dset = mean_obj.dataset()
    var mean_back = mean_dset.read_all[DType.float64]()
    mean_dset.close()
    print("\n=== Round-trip read ===")
    print("  mean_temp_c:", end="")
    for i in range(mean_back.shape[0]):
        print("", mean_back.item(i), end="")
    print()

    var daily_obj = summary_r.get("daily")
    var daily_r = daily_obj.group()

    var exc_obj = daily_r.get("temp_excerpt")
    var exc_dset = exc_obj.dataset()
    var exc_back = exc_dset.read_all[DType.float64]()
    exc_dset.close()
    print("  temp_excerpt shape:", exc_back.shape[0], "×", exc_back.shape[1])
    print("  exc_back[1, 2] =", exc_back.item(1, 2), " (expected 12.0)")

    daily_r.close()
    summary_r.close()
    r.close()

    print("\nDone.")
