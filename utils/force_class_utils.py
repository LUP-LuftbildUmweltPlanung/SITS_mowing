import os
import time
import shutil
import geopandas as gpd
from utils.utils import run_shell_command

def generate_input_feature_line(tif_path, num_layers):
    sequence = ' '.join(str(i) for i in range(1, num_layers + 1))
    return f"INPUT_FEATURE = {tif_path} {sequence}"


def replace_parameters(filename, replacements):
    with open(filename, 'r') as f:
        content = f.read()
        for key, value in replacements.items():
            content = content.replace(key, value)
    with open(filename, 'w') as f:
        f.write(content)

def extract_coordinates(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    #Skip the first line
    lines = lines[1:]
    #Extract X and Y values
    x_values = [int(line.split('_')[0][1:]) for line in lines]
    y_values = [int(line.split('_')[1][1:]) for line in lines]
    #Extract the desired values
    x_str = f"{min(x_values)} {max(x_values)}"
    y_str = f"{min(y_values)} {max(y_values)}"

    return x_str, y_str

def check_and_reproject_shapefile(shapefile_path, target_epsg=3035):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    # Check the current CRS of the shapefile
    if gdf.crs.to_epsg() != target_epsg:
        print("Reprojecting shapefile to EPSG: 3035")
        # Reproject the shapefile
        gdf = gdf.to_crs(epsg=target_epsg)
        # Define the new file path
        new_shapefile_path = shapefile_path.replace(".shp", "_3035.shp")
        # Save the reprojected shapefile
        gdf.to_file(new_shapefile_path, driver='ESRI Shapefile')
        print(f"Shapefile reprojected and saved to {new_shapefile_path}")
        return new_shapefile_path
    else:
        print("Shapefile is already in EPSG: 3035")
        return shapefile_path

from pathlib import Path

def tile_has_complete_output(tile_dir):
    if not tile_dir.is_dir():
        return False

    # A completed FORCE tile should have a main GeoTIFF written directly into
    # the tile directory. Ignore nested output folders from later processing.
    tif_files = sorted(tile_dir.glob("*.tif"))
    if not tif_files:
        return False

    main_tif = None
    for tif_file in tif_files:
        if tif_file.stat().st_size <= 0:
            continue
        main_tif = tif_file
        break

    return main_tif is not None


def generate_tiles_to_process(base_path, project_name, basename):
    # Define the new output root path using the structure {base_path}/process/temp/{project_name}/FORCE/{basename}/tiles_tss
    output_root = Path(base_path) / 'process' / 'temp' / project_name / 'FORCE' / basename / 'tiles_tss'
    tile_extent_file = Path(base_path) / 'process' / 'temp' / project_name / 'FORCE' / basename / 'tile_extent.txt'
    output_tile_list = Path(base_path) / 'process' / 'temp' / project_name / 'FORCE' / basename / 'provenance' / 'resume_tiles.txt'

    # Check if the tile extent file exists
    if not tile_extent_file.is_file():
        raise FileNotFoundError(f"Tile extent file not found: {tile_extent_file}")

    # Read the tile extent file
    with open(tile_extent_file, 'r') as file:
        lines = file.readlines()
        all_tiles = [line.strip() for line in lines if line.strip()]
        all_tiles = all_tiles[1:]  # Skip first line (which contains the count)

    # Remove duplicate tiles if any
    all_tiles = list(set(all_tiles))

    # List to store tiles that still need processing. A tile is only treated as
    # complete if the directory exists and the main output GeoTIFF is present.
    tiles_to_process = []
    completed_tiles = 0

    # Loop over each tile and requeue incomplete/partial outputs.
    for tile in all_tiles:
        tile_dir = output_root / tile

        if tile_has_complete_output(tile_dir):
            completed_tiles += 1
        else:
            tiles_to_process.append(tile)

    # Ensure the output directory for the tile list exists
    output_tile_list.parent.mkdir(parents=True, exist_ok=True)

    # Write the tiles to the output list file
    with open(output_tile_list, 'w') as file:
        # First, write the count of tiles
        file.write(f"{len(tiles_to_process)}\n")

        # Then, write each tile name
        for tile in tiles_to_process:
            file.write(f"{tile}\n")

    print(
        f"Tiles complete: {completed_tiles} | "
        f"tiles remaining: {len(tiles_to_process)} | "
        f"resume list written to: {output_tile_list}"
    )
    return output_tile_list


def reset_aoi_workspace(base_path, project_name, basename):
    force_root = os.path.join(base_path, "process", "temp", project_name, "FORCE", basename)
    mask_root = os.path.join(base_path, "process", "temp", "_mask", project_name, basename)

    for path in (force_root, mask_root):
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Removed previous run state: {path}")


def force_class_udf(project_name, force_dir, local_dir, base_path, aois, hold, date_range, clean=True):
    # defining parameters outsourced from main script

    # subprocess.run(['sudo', 'chmod', '-R', '777', f"{Path(temp_folder).parent}"])
    # subprocess.run(['sudo', 'chmod', '-R', '777', f"{Path(scripts_skel).parent}"])
    base_path_script = os.getcwd()
    startzeit = time.time()
    force_dir_parts = force_dir.split(":")
    if len(force_dir_parts) != 2:
        raise ValueError(f"force_dir must be given as 'host_path:container_path', got: {force_dir}")
    _, force_container_dir = force_dir_parts

    for aoi in aois:
        print(f"FORCE PROCESSING FOR {aoi}")

        basename = os.path.basename(aoi)
        if clean:
            reset_aoi_workspace(base_path, project_name, basename)
        print(f"Checking AOI path: {aoi}")
        if not os.path.exists(aoi):
            print(f"Error: AOI path does not exist -> {aoi}")
        aoi = check_and_reproject_shapefile(aoi)
        print(f"Reprojected AOI path: {aoi}")



        ### get force extend
        os.makedirs(f'{base_path}/process/temp/{project_name}/FORCE/{basename}', exist_ok=True)

        # subprocess.run(['sudo', 'chmod', '-R', '777', f"{temp_folder}/{project_name}/FORCE/{basename}"])

        shutil.copy(f"{base_path_script}/utils/skel/force_cube_sceleton/datacube-definition.prj",
                    f"{base_path}/process/temp/{project_name}/FORCE/{basename}/datacube-definition.prj")

        print(f"Checking AOI path: {aoi} -> Exists: {os.path.exists(aoi)}")

        cmd = (
            f'sudo docker run -v {local_dir} -v {force_dir} -u "$(id -u):$(id -g)" davidfrantz/force '
            f'force-tile-extent {aoi} -d {base_path_script}/utils/skel/force_cube_sceleton '
            f'-a {base_path}/process/temp/{project_name}/FORCE/{basename}/tile_extent.txt'
        )
        run_shell_command(cmd, hold=hold)

        # subprocess.run(['sudo','chmod','-R','777',f"{temp_folder}/{project_name}/FORCE/{basename}"])

        generate_tiles_to_process(base_path, project_name, basename)

        ### mask
        os.makedirs(f"{base_path}/process/temp/_mask/{project_name}/{basename}", exist_ok=True)

        # subprocess.run(['sudo', 'chmod', '-R', '777', f"{mask_folder}"])

        shutil.copy(f"{base_path_script}/utils/skel/force_cube_sceleton/datacube-definition.prj",
                    f"{base_path}/process/temp/_mask/{project_name}/{basename}/datacube-definition.prj")
        cmd = (
            f'sudo docker run -v {local_dir} -u "$(id -u):$(id -g)" davidfrantz/force '
            f"force-cube -o {base_path}/process/temp/_mask/{project_name}/{basename} {aoi}"
        )
        run_shell_command(cmd, hold=hold)
        # subprocess.run(['sudo','chmod','-R','777',f"{mask_folder}/{project_name}/{basename}"])

        ###mask mosaic
        cmd = (
            f'sudo docker run -v {local_dir} -u "$(id -u):$(id -g)" davidfrantz/force '
            f"force-mosaic {base_path}/process/temp/_mask/{project_name}/{basename}"
        )
        run_shell_command(cmd, hold=hold)

        # subprocess.run(['sudo','chmod','-R','777',f"{temp_folder}/{project_name}/FORCE/{basename}"])

        ###force param

        os.makedirs(f"{base_path}/process/temp/{project_name}/FORCE/{basename}/provenance", exist_ok=True)
        os.makedirs(f"{base_path}/process/temp/{project_name}/FORCE/{basename}/tiles_tss", exist_ok=True)

        shutil.copy(f"{base_path_script}/utils/skel/force_cube_sceleton/datacube-definition.prj",
                    f"{base_path}/process/temp/{project_name}/FORCE/{basename}/datacube-definition.prj")
        shutil.copy(f"{base_path_script}/utils/skel/force_cube_sceleton/datacube-definition.prj",
                    f"{base_path}/process/temp/{project_name}/FORCE/{basename}/tiles_tss/datacube-definition.prj")
        shutil.copy(f"{base_path_script}/utils/skel/UDF_NoCom.prm",
                    f"{base_path}/process/temp/{project_name}/FORCE/{basename}/tsa_UDF.prm")
        shutil.copy(f"{base_path_script}/utils/skel/udf_pixel.py",
                    f"{base_path}/process/temp/{project_name}/FORCE/{basename}/UDF_pixel.py")

        X_TILE_RANGE, Y_TILE_RANGE = extract_coordinates(f"{base_path}/process/temp/{project_name}/FORCE/{basename}/tile_extent.txt")
        DATE_RANGE = date_range

        # Define replacements
        replacements = {
            # INPUT/OUTPUT DIRECTORIES
            f'DIR_LOWER = NULL': f'DIR_LOWER = {force_container_dir}/FORCE/C1/L2/ard',
            f'DIR_HIGHER = NULL': f'DIR_HIGHER = {base_path}/process/temp/{project_name}/FORCE/{basename}/tiles_tss',
            f'DIR_PROVENANCE = NULL': f'DIR_PROVENANCE = {base_path}/process/temp/{project_name}/FORCE/{basename}/provenance',
            # MASKING
            f'DIR_MASK = NULL': f'DIR_MASK = {base_path}/process/temp/_mask/{project_name}/{basename}',
            f'BASE_MASK = NULL': f'BASE_MASK = {os.path.basename(aoi).replace(".shp", ".tif")}',
            # PROCESSING EXTENT AND RESOLUTION
            f'X_TILE_RANGE = 0 0': f'X_TILE_RANGE = {X_TILE_RANGE}',
            f'Y_TILE_RANGE = 0 0': f'Y_TILE_RANGE = {Y_TILE_RANGE}',
            f'FILE_TILE = NULL': f'FILE_TILE = {base_path}/process/temp/{project_name}/FORCE/{basename}/provenance/resume_tiles.txt',
            f'DATE_RANGE = YYYY-MM-DD YYYY-MM-DD': f'DATE_RANGE = {DATE_RANGE}',
            f'FILE_PYTHON = NULL': f'FILE_PYTHON = {base_path}/process/temp/{project_name}/FORCE/{basename}/UDF_pixel.py',
            f'STREAMING = TRUE': 'STREAMING = FALSE',
            f'PRETTY_PROGRESS = TRUE': 'PRETTY_PROGRESS = FALSE',
        }
        # Replace parameters in the file
        replace_parameters(f"{base_path}/process/temp/{project_name}/FORCE/{basename}/tsa_UDF.prm", replacements)

    endzeit = time.time()
    print("FORCE-Processing beendet nach " + str((endzeit - startzeit) / 60) + " Minuten")
