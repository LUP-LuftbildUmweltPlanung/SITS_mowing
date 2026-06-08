from pathlib import Path
import cProfile
import pstats
import time

from utils.force_class_utils import force_class_udf
from utils.utils import create_folder_structure, execute_cmd, export_selected_mowing_bands


BASE_PATH = Path("/rvt_mount")
PROJECT_NAME = "mowing_2021_germany_1tile_"
FORCE_DIR = "/force:/force"
LOCAL_DIR = f"{BASE_PATH}:{BASE_PATH}"
HOLD = False
CLEAN_RERUN = True
RUN_FORCE = True
ENABLE_PROFILING = True
PROFILE_OUTPUT = "sits_mowing_profile.prof"

DATE_RANGE = "2021-01-01 2021-12-31"
AOIS = sorted(BASE_PATH.glob("3DTests/data/xml/1_2021_2024_v1_0.shp"))


def process_aoi(aoi_path):
    basename = aoi_path.name

    if RUN_FORCE:
        force_class_udf(
            project_name=PROJECT_NAME,
            force_dir=FORCE_DIR,
            local_dir=LOCAL_DIR,
            base_path=str(BASE_PATH),
            aois=[str(aoi_path)],
            hold=HOLD,
            date_range=DATE_RANGE,
            clean=CLEAN_RERUN,
        )
        execute_cmd(HOLD, LOCAL_DIR, FORCE_DIR, str(BASE_PATH), PROJECT_NAME, basename)

    output_path = export_selected_mowing_bands(
        base_path=str(BASE_PATH),
        project_name=PROJECT_NAME,
        basename=basename,
        aoi_path=str(aoi_path),
    )
    print(f"Finished AOI {basename}: {output_path}")


def main():
    if not AOIS:
        raise FileNotFoundError("No AOI shapefiles matched the configured AOI pattern.")

    create_folder_structure(str(BASE_PATH), PROJECT_NAME)

    for aoi_path in AOIS:
        process_aoi(aoi_path)


if __name__ == "__main__":
    if ENABLE_PROFILING:
        profiler = cProfile.Profile()
        start_time = time.time()

        profiler.enable()
        main()
        profiler.disable()

        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f}s")

        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(30)
        stats.dump_stats(PROFILE_OUTPUT)
        print(f"cProfile data written to {PROFILE_OUTPUT}")
    else:
        main()
