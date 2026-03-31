# demo_api.mojo
#
# Comprehensive tutorial of the H5File API.
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
#   - Opening a file for reading:  H5File(path)
#   - Reading scalar attributes:   read_scalar_attr[dtype](group, attr_name)
#   - Reading 1-D datasets:        read_1d[dtype](path)  — Int32 and Float64
#   - Reading 2-D datasets:        read_2d[dtype](path)  — Float64 and Float32
#   - NDArray shape and indexing:  .dim0, .dim1, .size(), arr[i], arr[row, col]
#   - Freeing heap buffers:        .free()
#   - Creating a file for writing: H5File.create(path)
#   - Creating groups:             require_group(name)
#   - Writing 1-D datasets:        write_1d[dtype](path, ptr, n)
#   - Writing 2-D datasets:        write_2d[dtype](path, ptr, rows, cols)
#   - Closing handles:             f.close()

from hdf5 import H5File


def main() raises:

    # ===----------------------------------------------------------------------=== #
    # 1. Open a file for reading
    # ===----------------------------------------------------------------------=== #
    var f = H5File("./examples/demo_data.h5")

    # ===----------------------------------------------------------------------=== #
    # 2. Read scalar attributes
    # ===----------------------------------------------------------------------=== #
    # Attributes can be Float64 or Int32 (pass the matching dtype).
    var num_days     = f.read_scalar_attr[DType.int32]("/metadata", "num_days")
    var num_stations = f.read_scalar_attr[DType.int32]("/metadata", "num_stations")
    var base_year    = f.read_scalar_attr[DType.int32]("/metadata", "base_year")
    var lat_origin   = f.read_scalar_attr[DType.float64]("/metadata", "lat_origin")

    print("=== Metadata attributes ===")
    print("  num_days     =", num_days)
    print("  num_stations =", num_stations)
    print("  base_year    =", base_year)
    print("  lat_origin   =", lat_origin, "°N")

    # ===----------------------------------------------------------------------=== #
    # 3. Read a 1-D Float64 dataset
    # Shape is discovered automatically — no need to pass a size.
    # ===----------------------------------------------------------------------=== #
    var elevation = f.read_1d[DType.float64]("/sensors/elevation_m")

    print("\n=== 1-D Float64: /sensors/elevation_m ===")
    print("  length (dim0) =", elevation.dim0)
    print("  dim1          =", elevation.dim1, "  (always 1 for 1-D arrays)")
    print("  total elements:", elevation.size())
    print("  values:", end="")
    for i in range(elevation.dim0):
        print("", elevation[i], end="")
    print()

    # ===----------------------------------------------------------------------=== #
    # 4. Read a 1-D Int32 dataset
    # ===----------------------------------------------------------------------=== #
    var station_id = f.read_1d[DType.int32]("/sensors/station_id")

    print("\n=== 1-D Int32: /sensors/station_id ===")
    print("  values:", end="")
    for i in range(station_id.dim0):
        print("", station_id[i], end="")
    print()

    # ===----------------------------------------------------------------------=== #
    # 5. Read a 2-D Float64 dataset
    # arr[row, col] — row-major indexing.
    # ===----------------------------------------------------------------------=== #
    var temp = f.read_2d[DType.float64]("/observations/temperature_c")

    print("\n=== 2-D Float64: /observations/temperature_c ===")
    print("  shape:", temp.dim0, "×", temp.dim1, " (days × stations)")
    print("  total elements:", temp.size())
    print("  row 0 (day 1):", end="")
    for s in range(Int(num_stations)):
        print("", temp[0, s], end="")
    print()
    print("  col 0 (station 1, first 5 days):", end="")
    for d in range(5):
        print("", temp[d, 0], end="")
    print()

    # ===----------------------------------------------------------------------=== #
    # 6. Read a 2-D Float32 dataset
    # ===----------------------------------------------------------------------=== #
    var humidity = f.read_2d[DType.float32]("/observations/humidity_pct")

    print("\n=== 2-D Float32: /observations/humidity_pct ===")
    print("  shape:", humidity.dim0, "×", humidity.dim1)
    print("  humidity[0, 0] =", humidity[0, 0], "%")
    print("  humidity[14,2] =", humidity[14, 2], "%")

    # ===----------------------------------------------------------------------=== #
    # 7. Free all read buffers
    # ===----------------------------------------------------------------------=== #
    elevation.free()
    station_id.free()
    temp.free()
    humidity.free()

    f.close()

    # ===----------------------------------------------------------------------=== #
    # 8. Write a new file
    # ===----------------------------------------------------------------------=== #
    # H5File.create truncates any existing file at the given path.
    var out = H5File.create("./examples/demo_output.h5")

    # require_group creates the named group; safe to call if it already exists.
    out.require_group("/summary")
    out.require_group("/summary/daily")

    # Write a 1-D Float64 dataset.
    var means = alloc[Float64](Int(num_stations))
    means[0] = 22.1
    means[1] = 19.8
    means[2] = 25.3
    means[3] = 17.6
    means[4] = 21.0
    out.write_1d[DType.float64]("/summary/mean_temp_c", means, Int(num_stations))
    means.free()

    # Write a 2-D Float64 dataset (3 days × 5 stations, row-major).
    var n_days_out = 3
    var n_stat_out = Int(num_stations)
    var excerpt = alloc[Float64](n_days_out * n_stat_out)
    for d in range(n_days_out):
        for s in range(n_stat_out):
            excerpt[d * n_stat_out + s] = Float64(d * 10 + s)
    out.write_2d[DType.float64](
        "/summary/daily/temp_excerpt", excerpt, n_days_out, n_stat_out
    )
    excerpt.free()

    out.close()
    print("\n=== Wrote demo_output.h5 ===")
    print("  /summary/mean_temp_c          — 1-D Float64, length", num_stations)
    print("  /summary/daily/temp_excerpt   — 2-D Float64,", n_days_out, "×", num_stations)

    # ===----------------------------------------------------------------------=== #
    # 9. Round-trip: read back what we wrote
    # ===----------------------------------------------------------------------=== #
    var r = H5File("./examples/demo_output.h5")

    var mean_back = r.read_1d[DType.float64]("/summary/mean_temp_c")
    print("\n=== Round-trip read ===")
    print("  mean_temp_c:", end="")
    for i in range(mean_back.dim0):
        print("", mean_back[i], end="")
    print()

    var exc_back = r.read_2d[DType.float64]("/summary/daily/temp_excerpt")
    print("  temp_excerpt shape:", exc_back.dim0, "×", exc_back.dim1)
    print("  exc_back[1, 2] =", exc_back[1, 2], " (expected 12.0)")

    mean_back.free()
    exc_back.free()
    r.close()

    print("\nDone.")
