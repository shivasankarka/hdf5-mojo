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
#   - Reading scalar attributes:   group.attrs["name"].as_scalar[dtype]()
#   - Reading datasets:            dset.read[dtype](), dset[rows, col].read[dtype]()
#   - NDArray shape and indexing:  .dim0, .dim1, .size(), arr.get(i), arr.get(row, col)
#   - Writing with indexing:       dset[...] = arr, dset[0, 0:2] = value
#   - Creating a file for writing: File(path, "w")
#   - Creating groups:             require_group(name)
#   - Creating datasets:           create_dataset[dtype](name, shape)
#   - Closing handles:            automatic when the last copy goes away; f.close() is optional

from hdf5 import File, f64, f32, i32
from numojo.prelude import NDArray, Shape


def main() raises:
    # ===----------------------------------------------------------------------=== #
    # 1. Open a file for reading
    # ===----------------------------------------------------------------------=== #
    var f = File("./examples/demo_data.h5", "r")

    # ===----------------------------------------------------------------------=== #
    # 2. Read attributes with h5py-style item access
    # ===----------------------------------------------------------------------=== #
    # f["/metadata"] returns an H5Object; f.group(path) / f.dataset(path) open
    # the member directly. Attributes are read back with the dtype you expect.
    var meta = f.group("/metadata")
    var num_days = meta.attrs["num_days"].as_scalar[i32]()
    var num_stations = meta.attrs["num_stations"].as_scalar[i32]()
    var base_year = meta.attrs["base_year"].as_scalar[i32]()
    var lat_origin = meta.attrs["lat_origin"].as_scalar[f64]()

    print("=== Metadata attributes ===")
    print("  num_days     =", num_days)
    print("  num_stations =", num_stations)
    print("  base_year    =", base_year)
    print("  lat_origin   =", lat_origin, "°N")

    # ===----------------------------------------------------------------------=== #
    # 3. Read whole datasets
    # ===----------------------------------------------------------------------=== #
    # Handles close themselves; no explicit close() needed.
    var elevation = f.dataset("/sensors/elevation_m").read[f64]()
    print("\n=== 1-D Float64: /sensors/elevation_m ===")
    print("  shape:", elevation.shape, " size:", elevation.size)
    print("  values:", end="")
    for i in range(elevation.shape[0]):
        print("", elevation.item(i), end="")
    print()

    var station_id = f.dataset("/sensors/station_id").read[i32]()
    print("\n=== 1-D Int32: /sensors/station_id ===")
    print("  values:", end="")
    for i in range(station_id.shape[0]):
        print("", station_id.item(i), end="")
    print()

    # ===----------------------------------------------------------------------=== #
    # 4. Index datasets like h5py: ints drop an axis, slices keep it, `...`
    #    expands. Only the selected region is read from disk.
    # ===----------------------------------------------------------------------=== #
    var temp_dset = f.dataset("/observations/temperature_c")
    print("\n=== 2-D Float64: /observations/temperature_c ===")
    print("  shape:", temp_dset.shape()[0], "×", temp_dset.shape()[1])

    var row0 = temp_dset[0, :].read[f64]()  # day 1, all stations
    print("  row 0 (day 1):", end="")
    for s in range(row0.shape[0]):
        print("", row0.item(s), end="")
    print()

    var col0 = temp_dset[0:5, 0].read[f64]()  # station 1, first 5 days
    print("  col 0 (station 1, first 5 days):", end="")
    for d in range(col0.shape[0]):
        print("", col0.item(d), end="")
    print()

    print("  temp[3, 2] =", temp_dset[3, 2].read_scalar[f64]())

    # ===----------------------------------------------------------------------=== #
    # 5. Read a Float32 dataset
    # ===----------------------------------------------------------------------=== #
    var humidity_dset = f.dataset("/observations/humidity_pct")
    print("\n=== 2-D Float32: /observations/humidity_pct ===")
    print("  humidity[0, 0] =", humidity_dset[0, 0].read_scalar[f32](), "%")
    print("  humidity[14,2] =", humidity_dset[14, 2].read_scalar[f32](), "%")

    # ===----------------------------------------------------------------------=== #
    # 6. Write a new file
    # ===----------------------------------------------------------------------=== #
    # File(..., "w") truncates any existing file at the given path.
    var out = File("./examples/demo_output.h5", "w")
    var summary = out.require_group("/summary")
    var daily = summary.require_group("daily")
    summary.attrs["source"] = "demo_data.h5"  # strings, ints, floats, arrays

    # Create a dataset, then fill it with `dset[...] = array`.
    var n_stat = Int(num_stations)
    var means = NDArray[f64](Shape(n_stat))
    means.unsafe_ptr()[unsafe_offset=0] = 22.1
    means.unsafe_ptr()[unsafe_offset=1] = 19.8
    means.unsafe_ptr()[unsafe_offset=2] = 25.3
    means.unsafe_ptr()[unsafe_offset=3] = 17.6
    means.unsafe_ptr()[unsafe_offset=4] = 21.0
    var means_dset = summary.create_dataset[f64]("mean_temp_c", [n_stat])
    means_dset[...] = means
    means_dset.attrs["units"] = "degC"

    # A 3 × 5 dataset: fill with zeros (a scalar broadcasts), then overwrite
    # one row with an array.
    var excerpt = daily.create_dataset[f64]("temp_excerpt", [3, n_stat])
    excerpt[...] = Float64(0)
    var row1 = NDArray[f64](Shape(n_stat))
    for s in range(n_stat):
        row1.unsafe_ptr()[unsafe_offset=s] = Float64(10 + s)
    excerpt[1, :] = row1
    excerpt[2, 0:2] = Float64(-1)

    out.close()
    print("\n=== Wrote demo_output.h5 ===")
    print("  /summary/mean_temp_c          — 1-D Float64, length", num_stations)
    print("  /summary/daily/temp_excerpt   — 2-D Float64, 3 ×", num_stations)

    # ===----------------------------------------------------------------------=== #
    # 7. Round-trip: read back what we wrote
    # ===----------------------------------------------------------------------=== #
    var r = File("./examples/demo_output.h5", "r")
    var mean_back = r["summary"]["mean_temp_c"].dataset().read[f64]()
    print("\n=== Round-trip read ===")
    print("  mean_temp_c:", end="")
    for i in range(mean_back.shape[0]):
        print("", mean_back.item(i), end="")
    print()

    var exc = r.dataset("/summary/daily/temp_excerpt")
    print("  exc[1, 2]  =", exc[1, 2].read_scalar[f64](), " (expected 12.0)")
    print("  exc[2, 1]  =", exc[2, 1].read_scalar[f64](), " (expected -1.0)")
    print("  source attr:", r.group("summary").attrs["source"].as_string())
    print(
        "  units attr :",
        r.dataset("/summary/mean_temp_c").attrs["units"].as_string(),
    )
    r.close()

    print("\nDone.")
