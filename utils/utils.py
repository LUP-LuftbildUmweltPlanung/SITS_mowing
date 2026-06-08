from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


DEFAULT_BANDS = [1, 5, 6, 7, 8, 9, 10, 11]
REQUIRED_GDAL_TOOLS = ["gdalbuildvrt", "gdal_translate", "gdalwarp"]
DEFAULT_COMPRESSION = "DEFLATE"
DEFAULT_ZLEVEL = "9"
DEFAULT_BIGTIFF = "YES"
OUTPUT_NODATA = -9999
DEFAULT_BLOCKSIZE = 512
DEFAULT_OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64]
DEFAULT_OVERVIEW_RESAMPLING = "nearest"


@dataclass(frozen=True)
class TileCollection:
    tiles_root: Path
    tile_paths: list[Path]


def create_folder_structure(base_path, project_name):
    folder_structure = [
        "process",
        "process/data",
        "process/results",
        f"process/results/{project_name}",
        "process/temp",
        "process/temp/_mask",
    ]

    for folder in folder_structure:
        path = os.path.join(base_path, folder)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")
        else:
            print(f"Folder already exists: {path}")


def run_shell_command(cmd, hold=False):
    print(f"Running command:\n{cmd}\n")
    if hold:
        print("`hold=True` requested. Command output will stay in the current terminal.")
    result = subprocess.run(cmd, shell=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def execute_cmd(hold, local_dir, force_dir, base_path, project_name, basename):
    cmd = (
        f'sudo docker run -v {local_dir} -v {force_dir} -u "$(id -u):$(id -g)" davidfrantz/force '
        "force-higher-level "
        f"{base_path}/process/temp/{project_name}/FORCE/{basename}/tsa_UDF.prm"
    )
    run_shell_command(cmd, hold=hold)


def make_output_tile_name(tile_path: Path) -> str:
    return f"{tile_path.parent.name}_{tile_path.name}"


def resolve_tiles_root(base_path: Path, project_name: str, basename: str) -> Path:
    return base_path / "process" / "temp" / project_name / "FORCE" / basename / "tiles_tss"


def find_tile_paths(tiles_root: Path) -> list[Path]:
    tile_paths = sorted(tiles_root.glob("X*/*.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No tile TIFFs found below: {tiles_root}")
    return tile_paths


def load_tiles(base_path: Path, project_name: str, basename: str) -> TileCollection:
    tiles_root = resolve_tiles_root(base_path, project_name, basename)
    if not tiles_root.exists():
        raise FileNotFoundError(f"Tiles root does not exist: {tiles_root}")
    return TileCollection(tiles_root=tiles_root, tile_paths=find_tile_paths(tiles_root))


def ensure_gdal_tools() -> None:
    missing = [tool for tool in REQUIRED_GDAL_TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "Missing required GDAL tool(s): "
            + ", ".join(missing)
            + ". Install gdal-bin on the target machine."
        )


def get_band_descriptions(src, band_indexes: Iterable[int]) -> list[str | None]:
    descriptions = list(src.descriptions)
    return [descriptions[index - 1] for index in band_indexes]


def derive_nodata_value(src, fallback_nodata: int | float | None = None) -> int | float:
    src_nodata = src.nodata
    if src_nodata is not None:
        return src_nodata

    if fallback_nodata is not None:
        return fallback_nodata

    src_dtype = src.dtypes[0]
    if src_dtype in ["int8", "byte"]:
        return -128
    if src_dtype == "uint8":
        return 255
    if src_dtype == "int16":
        return -32768
    if src_dtype == "uint16":
        return 65535
    if src_dtype == "int32":
        return -2147483648
    if src_dtype == "uint32":
        return 4294967295
    if src_dtype in ["float32", "float64"]:
        return -9999.0

    raise ValueError(f"Unsupported dtype for nodata derivation: {src_dtype}")


def validate_compression_method(compression_method: str) -> str:
    supported = {"DEFLATE", "LZW", "ZSTD", "PACKBITS", "NONE"}
    value = compression_method.upper()
    if value not in supported:
        raise ValueError(
            f"Unsupported compression method '{compression_method}'. "
            f"Supported values: {', '.join(sorted(supported))}."
        )
    return value


def normalize_bigtiff(value: str) -> str:
    supported = {"YES", "NO", "IF_NEEDED", "IF_SAFER"}
    normalized = value.upper()
    if normalized not in supported:
        raise ValueError(
            f"Unsupported BIGTIFF option '{value}'. "
            f"Supported values: {', '.join(sorted(supported))}."
        )
    return normalized


def infer_predictor(dtype: str, compression_method: str) -> str | None:
    if compression_method not in {"DEFLATE", "LZW", "ZSTD"}:
        return None
    if "float" in dtype.lower():
        return "3"
    return "2"


def build_creation_options(
    compression_method: str,
    predictor: str | None,
    zlevel: str,
    bigtiff: str,
    blocksize: int,
) -> list[str]:
    options = [
        "-co",
        f"COMPRESS={compression_method}",
        "-co",
        f"BIGTIFF={bigtiff}",
        "-co",
        "TILED=YES",
        "-co",
        f"BLOCKXSIZE={blocksize}",
        "-co",
        f"BLOCKYSIZE={blocksize}",
    ]
    if predictor is not None:
        options.extend(["-co", f"PREDICTOR={predictor}"])
    if compression_method == "DEFLATE":
        options.extend(["-co", f"ZLEVEL={zlevel}"])
    elif compression_method == "ZSTD":
        options.extend(["-co", f"ZSTD_LEVEL={zlevel}"])
    return options


def run_command(cmd: list[str]) -> None:
    print("Running command:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def write_tile_list(tile_paths: list[Path], temp_dir: Path) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", dir=temp_dir, delete=False) as tmp_file:
        for tile_path in tile_paths:
            tmp_file.write(f"{tile_path}\n")
        return Path(tmp_file.name)


def write_report(report_path: Path, payload: dict) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))
    return report_path


def summarize_raster_values(raster_path: Path) -> dict:
    import numpy as np
    import rasterio

    summary = {
        "path": str(raster_path),
        "bands": [],
        "all_nodata": True,
    }

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for band_index in range(1, src.count + 1):
            nan_count = 0
            nodata_count = 0
            zero_count = 0
            valid_count = 0

            for _, window in src.block_windows(band_index):
                data = src.read(band_index, window=window)
                invalid_mask = np.zeros(data.shape, dtype=bool)
                if np.issubdtype(data.dtype, np.floating):
                    invalid_mask |= ~np.isfinite(data)
                    nan_count += int(np.count_nonzero(~np.isfinite(data)))
                if nodata is not None:
                    nodata_count += int(np.count_nonzero(data == nodata))
                    invalid_mask |= data == nodata

                valid_mask = ~invalid_mask
                valid_values = data[valid_mask]
                zero_count += int(np.count_nonzero(valid_values == 0))
                valid_count += int(valid_values.size)

            summary["bands"].append(
                {
                    "band": band_index,
                    "nan_count": nan_count,
                    "nodata_count": nodata_count,
                    "zero_count": zero_count,
                    "valid_count": valid_count,
                }
            )
            if valid_count > 0:
                summary["all_nodata"] = False

    return summary


def normalize_output_tile(output_tile_path: Path, output_nodata: int = OUTPUT_NODATA) -> dict:
    import numpy as np
    import rasterio

    with rasterio.open(output_tile_path, "r+") as dest:
        existing_nodata = dest.nodata
        for band_index in range(1, dest.count + 1):
            for _, window in dest.block_windows(band_index):
                data = dest.read(band_index, window=window)
                invalid_mask = np.zeros(data.shape, dtype=bool)
                if np.issubdtype(data.dtype, np.floating):
                    invalid_mask |= ~np.isfinite(data)
                if existing_nodata is not None:
                    invalid_mask |= data == existing_nodata
                if invalid_mask.any():
                    data = data.copy()
                    data[invalid_mask] = output_nodata
                    dest.write(data, band_index, window=window)
        dest.nodata = output_nodata

    return summarize_raster_values(output_tile_path)


def build_overviews_inplace(
    raster_path: Path,
    levels: list[int] | None = None,
    resampling: str = DEFAULT_OVERVIEW_RESAMPLING,
) -> None:
    import rasterio
    from rasterio.enums import Resampling

    if levels is None:
        levels = DEFAULT_OVERVIEW_LEVELS

    with rasterio.open(raster_path, "r+") as src:
        if src.overviews(1):
            return
        src.build_overviews(levels, Resampling[resampling])
        src.update_tags(ns="rio_overview", resampling=resampling)


def prepare_aoi_for_gdal(aoi_path: Path, target_crs, temp_dir: Path):
    import geopandas as gpd

    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI file contains no features: {aoi_path}")
    if target_crs and aoi.crs != target_crs:
        aoi = aoi.to_crs(target_crs)

    if hasattr(aoi.geometry, "make_valid"):
        aoi["geometry"] = aoi.geometry.make_valid()
    else:
        aoi["geometry"] = aoi.buffer(0)

    aoi = aoi[~aoi.geometry.is_empty & aoi.geometry.notnull()].copy()
    if aoi.empty:
        raise ValueError(f"AOI file has no valid geometries after repair: {aoi_path}")

    union_geom = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    aoi_out = gpd.GeoDataFrame({"id": [1]}, geometry=[union_geom], crs=aoi.crs)

    fd, prepared_path = tempfile.mkstemp(suffix=".gpkg", dir=temp_dir)
    prepared_aoi_path = Path(prepared_path)
    prepared_aoi_path.unlink(missing_ok=True)
    try:
        os.close(fd)
    except OSError:
        pass

    aoi_out.to_file(prepared_aoi_path, driver="GPKG")
    return prepared_aoi_path, union_geom


def write_geometry_cutline(geometry, crs, temp_dir: Path) -> Path:
    import geopandas as gpd

    cutline = gpd.GeoDataFrame({"id": [1]}, geometry=[geometry], crs=crs)
    fd, prepared_path = tempfile.mkstemp(suffix=".gpkg", dir=temp_dir)
    prepared_cutline_path = Path(prepared_path)
    prepared_cutline_path.unlink(missing_ok=True)
    try:
        os.close(fd)
    except OSError:
        pass

    cutline.to_file(prepared_cutline_path, driver="GPKG")
    return prepared_cutline_path


def clip_tile(
    tile_path: Path,
    output_tile_path: Path,
    aoi_union,
    aoi_crs,
    temp_dir: Path,
    band_indexes: list[int],
    band_descriptions: list[str | None],
    dtype: str,
    num_threads: str,
    cachemax_mb: int,
    overwrite_tiles: bool,
    compression_method: str,
    predictor: str | None,
    zlevel: str,
    bigtiff: str,
    blocksize: int,
    output_nodata: int | float,
    fallback_nodata: int | float | None,
) -> dict:
    import rasterio
    from shapely.geometry import box

    if output_tile_path.exists() and not overwrite_tiles:
        diagnostics = summarize_raster_values(output_tile_path)
        diagnostics["tile"] = tile_path.name
        diagnostics["tile_folder"] = tile_path.parent.name
        diagnostics["source_path"] = str(tile_path)
        diagnostics["status"] = "reused"
        return diagnostics

    with rasterio.open(tile_path) as src:
        tile_bounds = box(*src.bounds)
        src_nodata = derive_nodata_value(src, fallback_nodata=fallback_nodata)
        source_dtype = src.dtypes[0]

    tile_cutline_path = None
    if aoi_union is not None:
        if not tile_bounds.intersects(aoi_union):
            return {
                "tile": tile_path.name,
                "tile_folder": tile_path.parent.name,
                "source_path": str(tile_path),
                "status": "skipped_non_intersecting",
                "source_dtype": source_dtype,
                "source_nodata": src_nodata,
            }
        tile_cutline_geom = tile_bounds.intersection(aoi_union)
        if tile_cutline_geom.is_empty:
            return {
                "tile": tile_path.name,
                "tile_folder": tile_path.parent.name,
                "source_path": str(tile_path),
                "status": "skipped_empty_intersection",
                "source_dtype": source_dtype,
                "source_nodata": src_nodata,
            }
        tile_cutline_path = write_geometry_cutline(tile_cutline_geom, aoi_crs, temp_dir)

    output_tile_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".vrt", dir=output_tile_path.parent, delete=False) as selected_vrt_file:
        selected_vrt_path = Path(selected_vrt_file.name)

    try:
        select_bands_cmd = ["gdal_translate", "-of", "VRT"]
        for band in band_indexes:
            select_bands_cmd.extend(["-b", str(band)])
        select_bands_cmd.extend([str(tile_path), str(selected_vrt_path)])
        run_command(select_bands_cmd)

        if tile_cutline_path:
            export_cmd = [
                "gdalwarp",
                "-overwrite",
                "-cutline",
                str(tile_cutline_path),
                "-crop_to_cutline",
                "-srcnodata",
                str(src_nodata),
                "-dstnodata",
                str(output_nodata),
                "-multi",
                "-wo",
                f"NUM_THREADS={num_threads}",
                "-wm",
                str(cachemax_mb),
                "-ot",
                dtype.upper(),
            ]
            export_cmd.extend(
                build_creation_options(compression_method, predictor, zlevel, bigtiff, blocksize)
            )
            export_cmd.extend([str(selected_vrt_path), str(output_tile_path)])
        else:
            export_cmd = [
                "gdal_translate",
                "-of",
                "GTiff",
                "-a_nodata",
                str(output_nodata),
            ]
            export_cmd.extend(
                build_creation_options(compression_method, predictor, zlevel, bigtiff, blocksize)
            )
            export_cmd.extend(["-ot", dtype.upper(), str(selected_vrt_path), str(output_tile_path)])
        run_command(export_cmd)

        with rasterio.open(output_tile_path, "r+") as dest:
            for band_number, description in enumerate(band_descriptions, start=1):
                if description:
                    dest.set_band_description(band_number, description)

        diagnostics = normalize_output_tile(output_tile_path, output_nodata=output_nodata)
        diagnostics["tile"] = tile_path.name
        diagnostics["tile_folder"] = tile_path.parent.name
        diagnostics["source_path"] = str(tile_path)
        diagnostics["status"] = "written"
        diagnostics["source_dtype"] = source_dtype
        diagnostics["source_nodata"] = src_nodata
        if diagnostics["all_nodata"]:
            print(f"Keeping empty clipped tile {tile_path.name}: all output pixels are nodata.")
            diagnostics["status"] = "written_all_nodata"
            return diagnostics

        band_debug = [
            f"B{band['band']}: valid={band['valid_count']} zero={band['zero_count']} nodata={band['nodata_count']} nan={band['nan_count']}"
            for band in diagnostics["bands"]
        ]
        print(f"Tile diagnostics for {tile_path.name}: " + "; ".join(band_debug))
        return diagnostics
    finally:
        if selected_vrt_path.exists():
            selected_vrt_path.unlink()
        if tile_cutline_path is not None and tile_cutline_path.exists():
            tile_cutline_path.unlink()


def build_vrt(clipped_tile_paths: list[Path], vrt_output_path: Path, temp_dir: Path) -> Path:
    tile_list_path = write_tile_list(clipped_tile_paths, temp_dir)
    try:
        run_command(["gdalbuildvrt", "-input_file_list", str(tile_list_path), str(vrt_output_path)])
        return vrt_output_path
    finally:
        if tile_list_path.exists():
            tile_list_path.unlink()


def build_final_raster(
    vrt_path: Path,
    final_output_path: Path,
    dtype: str,
    build_overviews: bool,
    overview_resampling: str,
    compression_method: str,
    predictor: str | None,
    zlevel: str,
    bigtiff: str,
    blocksize: int,
    output_nodata: int | float,
) -> Path:
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal_translate",
        "-of",
        "GTiff",
        "-a_nodata",
        str(output_nodata),
    ]
    cmd.extend(build_creation_options(compression_method, predictor, zlevel, bigtiff, blocksize))
    cmd.extend(["-ot", dtype.upper(), str(vrt_path), str(final_output_path)])
    run_command(cmd)
    if build_overviews:
        build_overviews_inplace(final_output_path, resampling=overview_resampling)
    return final_output_path


def export_tiles(
    base_path: Path,
    project_name: str,
    basename: str,
    output_dir: Path,
    vrt_output_path: Path | None,
    final_output_path: Path | None,
    band_indexes: list[int],
    aoi_path: Path | None,
    dtype: str,
    num_threads: str,
    cachemax_mb: int,
    skip_vrt: bool,
    skip_final_raster: bool,
    overwrite_tiles: bool,
    build_overviews: bool,
    overview_resampling: str,
    report_path: Path | None,
    compression_method: str,
    bigtiff: str,
    zlevel: str,
    blocksize: int,
    output_nodata: int | float,
    fallback_nodata: int | float | None,
) -> tuple[Path, Path | None, Path | None]:
    import rasterio

    ensure_gdal_tools()
    compression_method = validate_compression_method(compression_method)
    bigtiff = normalize_bigtiff(bigtiff)
    predictor = infer_predictor(dtype, compression_method)
    start_total = time.time()
    tiles = load_tiles(base_path, project_name, basename)
    print(f"Found {len(tiles.tile_paths)} tile(s) in {tiles.tiles_root}")

    temp_dir = base_path / "process" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if aoi_path and not aoi_path.exists():
        raise FileNotFoundError(f"AOI path does not exist: {aoi_path}")

    aoi_union = None
    aoi_crs = None

    with rasterio.open(tiles.tile_paths[0]) as src:
        band_descriptions = get_band_descriptions(src, band_indexes)
        if aoi_path:
            prepared_aoi_path, aoi_union = prepare_aoi_for_gdal(aoi_path, src.crs, temp_dir)
            aoi_crs = src.crs
            prepared_aoi_path.unlink(missing_ok=True)

    clipped_tile_paths: list[Path] = []
    skipped_tiles = 0
    reused_tiles = 0
    tile_reports: list[dict] = []
    tile_progress = tqdm(
        tiles.tile_paths,
        desc="Clipping tiles",
        unit="tile",
        dynamic_ncols=True,
    )
    tile_stage_start = time.time()

    for tile_path in tile_progress:
        output_tile_path = output_dir / make_output_tile_name(tile_path)
        if output_tile_path.exists() and not overwrite_tiles:
            reused_tiles += 1
            clipped_tile_paths.append(output_tile_path)
            tile_progress.set_postfix_str(f"reused={tile_path.name}")
            tile_reports.append(
                summarize_raster_values(output_tile_path)
                | {
                    "tile": tile_path.name,
                    "tile_folder": tile_path.parent.name,
                    "source_path": str(tile_path),
                    "status": "reused",
                }
            )
            continue

        tile_progress.set_postfix_str(f"processing={tile_path.name}")
        tile_report = clip_tile(
            tile_path=tile_path,
            output_tile_path=output_tile_path,
            aoi_union=aoi_union,
            aoi_crs=aoi_crs,
            temp_dir=temp_dir,
            band_indexes=band_indexes,
            band_descriptions=band_descriptions,
            dtype=dtype,
            num_threads=num_threads,
            cachemax_mb=cachemax_mb,
            overwrite_tiles=overwrite_tiles,
            compression_method=compression_method,
            predictor=predictor,
            zlevel=zlevel,
            bigtiff=bigtiff,
            blocksize=blocksize,
            output_nodata=output_nodata,
            fallback_nodata=fallback_nodata,
        )
        tile_reports.append(tile_report)
        if tile_report["status"] in {"written", "written_all_nodata", "reused"}:
            clipped_tile_paths.append(output_tile_path)
            if build_overviews and tile_report["status"] in {"written", "written_all_nodata"}:
                tile_progress.set_postfix_str(f"overviews={tile_path.name}")
                build_overviews_inplace(output_tile_path, resampling=overview_resampling)
        else:
            skipped_tiles += 1
            tile_progress.set_postfix_str(f"skipped={tile_report['status']}")

    tile_progress.close()
    tile_stage_elapsed = time.time() - tile_stage_start

    if not clipped_tile_paths:
        raise RuntimeError("No clipped tiles were produced. Check AOI path and tile coverage.")

    print(
        f"Wrote or reused {len(clipped_tile_paths)} clipped tile(s); "
        f"reused {reused_tiles}; skipped {skipped_tiles}."
    )
    print(f"Tile clipping stage finished in {format_duration(tile_stage_elapsed)}.")

    final_vrt = None
    final_raster = None
    vrt_stage_elapsed = None
    final_stage_elapsed = None
    if not skip_vrt:
        if vrt_output_path is None:
            vrt_output_path = output_dir.parent / f"mowing-events_{Path(basename).stem}.vrt"
        print("Building VRT from clipped tiles...")
        vrt_stage_start = time.time()
        final_vrt = build_vrt(clipped_tile_paths, vrt_output_path, temp_dir)
        vrt_stage_elapsed = time.time() - vrt_stage_start
        print(f"VRT written to {final_vrt}")
        print(f"VRT stage finished in {format_duration(vrt_stage_elapsed)}.")

        if not skip_final_raster:
            if final_output_path is None:
                final_output_path = output_dir.parent / f"mowing-events_{Path(basename).stem}.tif"
            print("Building final merged raster from VRT...")
            final_stage_start = time.time()
            final_raster = build_final_raster(
                final_vrt,
                final_output_path,
                dtype,
                build_overviews,
                overview_resampling,
                compression_method,
                predictor,
                zlevel,
                bigtiff,
                blocksize,
                output_nodata,
            )
            final_stage_elapsed = time.time() - final_stage_start
            print(f"Final merged raster written to {final_raster}")
            print(f"Final raster stage finished in {format_duration(final_stage_elapsed)}.")
    elif not skip_final_raster:
        raise RuntimeError("A final raster requires a VRT. Remove skip_vrt or add skip_final_raster.")

    runtime_seconds = time.time() - start_total
    if report_path is None:
        report_path = output_dir.parent / f"mowing-events_{Path(basename).stem}_report.json"

    report_payload = {
        "project_name": project_name,
        "basename": basename,
        "source_tiles_root": str(tiles.tiles_root),
        "tile_output_dir": str(output_dir),
        "vrt_output": str(final_vrt) if final_vrt else None,
        "final_raster_output": str(final_raster) if final_raster else None,
        "band_indexes": band_indexes,
        "output_dtype": dtype,
        "output_nodata": output_nodata,
        "compression": {
            "compress": compression_method,
            "predictor": predictor,
            "zlevel": zlevel,
            "bigtiff": bigtiff,
            "tiled": True,
            "blocksize": blocksize,
        },
        "overviews": {
            "enabled": build_overviews,
            "resampling": overview_resampling,
            "levels": DEFAULT_OVERVIEW_LEVELS,
        },
        "runtime_seconds": runtime_seconds,
        "runtime_human": format_duration(runtime_seconds),
        "stage_timings": {
            "tile_clipping_seconds": tile_stage_elapsed,
            "tile_clipping_human": format_duration(tile_stage_elapsed),
            "vrt_seconds": vrt_stage_elapsed,
            "vrt_human": format_duration(vrt_stage_elapsed) if vrt_stage_elapsed is not None else None,
            "final_raster_seconds": final_stage_elapsed,
            "final_raster_human": format_duration(final_stage_elapsed) if final_stage_elapsed is not None else None,
        },
        "reused_tiles": reused_tiles,
        "skipped_tiles": skipped_tiles,
        "written_or_reused_tiles": len(clipped_tile_paths),
        "tiles": tile_reports,
    }
    report_written = write_report(report_path, report_payload)
    print(f"Validation report written to {report_written}")
    print(f"Finished in {format_duration(runtime_seconds)}")
    return output_dir, final_vrt, final_raster


def mosaic_rasters(
    base_path,
    project_name,
    basename,
    aoi_path=None,
    dtype="uint16",
    band_indexes=None,
    output_filename=None,
    num_threads="ALL_CPUS",
    cachemax_mb=512,
    skip_vrt=False,
    skip_final_raster=False,
    overwrite_tiles=False,
    build_overviews=True,
    overview_resampling=DEFAULT_OVERVIEW_RESAMPLING,
    compression_method=DEFAULT_COMPRESSION,
    bigtiff=DEFAULT_BIGTIFF,
    zlevel=DEFAULT_ZLEVEL,
    blocksize=DEFAULT_BLOCKSIZE,
    output_nodata=OUTPUT_NODATA,
    fallback_nodata=None,
    report_path=None,
):
    if band_indexes is None:
        band_indexes = DEFAULT_BANDS.copy()

    base_path_obj = Path(base_path)
    basename_stem = Path(basename).stem
    results_dir = base_path_obj / "process" / "results" / project_name

    if output_filename is None:
        final_output_path = results_dir / f"mowing-events_{basename_stem}.tif"
    else:
        final_output_path = Path(output_filename)

    output_dir = final_output_path.parent / f"{final_output_path.stem}_tiles"
    vrt_output_path = final_output_path.with_suffix(".vrt")
    report_path_obj = Path(report_path) if report_path else final_output_path.with_name(
        f"{final_output_path.stem}_report.json"
    )

    _, final_vrt, final_raster = export_tiles(
        base_path=base_path_obj,
        project_name=project_name,
        basename=basename,
        output_dir=output_dir,
        vrt_output_path=None if skip_vrt else vrt_output_path,
        final_output_path=None if skip_final_raster else final_output_path,
        band_indexes=band_indexes,
        aoi_path=Path(aoi_path) if aoi_path else None,
        dtype=dtype,
        num_threads=num_threads,
        cachemax_mb=cachemax_mb,
        skip_vrt=skip_vrt,
        skip_final_raster=skip_final_raster,
        overwrite_tiles=overwrite_tiles,
        build_overviews=build_overviews,
        overview_resampling=overview_resampling,
        report_path=report_path_obj,
        compression_method=compression_method,
        bigtiff=bigtiff,
        zlevel=zlevel,
        blocksize=blocksize,
        output_nodata=output_nodata,
        fallback_nodata=fallback_nodata,
    )

    if final_raster is not None:
        return str(final_raster)
    if final_vrt is not None:
        return str(final_vrt)
    return str(output_dir)


def export_selected_mowing_bands(base_path, project_name, basename, aoi_path=None):
    output_filename = os.path.join(
        base_path,
        "process",
        "results",
        project_name,
        f"mowing-events_{os.path.splitext(basename)[0]}.tif",
    )

    return mosaic_rasters(
        base_path=base_path,
        project_name=project_name,
        basename=basename,
        aoi_path=aoi_path,
        dtype="int16",
        band_indexes=DEFAULT_BANDS,
        output_filename=output_filename,
    )
