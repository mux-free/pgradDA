import os
import yaml
import time
import f90nml
import random
import subprocess
import numpy as np
import xarray as xr
import pandas as pd

from shutil import copy2
from pathlib import Path
from netCDF4 import Dataset
from datetime import datetime, timedelta

# *********************************************************************************************
#             FUNCTIONS TO RUN AND MODIFY ASPIRE TASKS:
#                   - ASPFORGE.YML RUN
#                   - Change namelist timing and restart-file handling
#                   - Run ASPIRE
# *********************************************************************************************

def get_aspire_image(version="newest", verbose=0):
    # Construct command e_path = "/home/maxf/singularity/images/"
    singularity_image_path = "/home/maxf/singularity/images/"
    model_version_18_0_0 = "aspkit-devel_18.0.0.sif"
    

    if version=="newest":
        model_image = model_version_18_0_0
    else:
        raise ValueError(version)
    
    if verbose:
        print(f"Aspire version: {model_image}")
    return singularity_image_path + model_image


def aspire(
    namelist_file: str | Path,
    aspire_version: str = "newest",
    singularity_image=None,
    verbose: int = 0,
    retries: int = 6,
    base_delay: float = 10.0,
) -> None:
    """
    Runs aspire within a Singularity container with retry on transient failures.

    Retries are intended to handle flaky LicenseSpring network timeouts.
    """

    if verbose > 0:
        print(f"Running aspire: {namelist_file}")

    if singularity_image is None:
        singularity_image = get_aspire_image(version=aspire_version, verbose=verbose)

    
    # --- Construct command to 
        # Why not "singularity shell --nv" -->  NOTE that "exec" runs a command in the contained then exits,     
        #                                       whereas "shell" starts an interactive shell inside the container (e.g terminal)
        # THUS, use exec for autmoation and shell for interactive sessions
    if str(namelist_file).endswith(".nml"):
        nml_file = str(namelist_file)
        workdir = os.path.dirname(nml_file)
    else:
        workdir = str(namelist_file)
        nml_file = f"{workdir}/graspIn.000.nml"

    cmd = [
        "singularity", "exec", "--nv",
        singularity_image,
        "aspire",
        nml_file,
    ]

    for attempt in range(1, retries + 1):
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=workdir,
                capture_output=True,
            )
            return  # success

        except subprocess.CalledProcessError as e:
            stdout = e.stdout.decode("utf-8", errors="ignore") if e.stdout else ""
            stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""

            is_retryable = (
                "LicenseSpring" in stderr
                or "Network operation timed out" in stderr
                or e.returncode < 0  # SIGABRT etc.
            )

            if verbose > 0:
                print(
                    f"\nAspire failed (attempt {attempt}/{retries})\n"
                    f"Return code: {e.returncode}\n"
                    f"--- stdout ---\n{stdout}\n"
                    f"--- stderr ---\n{stderr}\n"
                    "Retry now ..."
                )

            if not is_retryable or attempt == retries:
                raise

            # Exponential backoff + jitter
            delay = base_delay * (2 ** (attempt - 1))
            delay += random.uniform(0, base_delay)

            if verbose > 0:
                print(f"Retrying in {delay:.1f} s...\n")

            time.sleep(delay)
    




def aspforge_task_run(aspforge_file, 
                      overwrite=False, 
                      whiffle_env="~/whiffle_env/whiffle_env25/bin/activate",
                      verbose=1, 
                      ):
    """
    This function processes all aspforge_<type>.yml files for base and sim/seq in  the assimilation 
    and prediciton directory. It moves and unzips the files in the run_<type> directories.

    Parameters:
    -----------
        - aspforge_file (str): Path to the directory containing the aspforge<>.yml file.
        - overwrite (bool): If true, aspforge files are being run again   

    """

    if not str(aspforge_file).endswith(".yml"):
        raise ValueError(f"The input file must be the absolute directory of the yml file (ending with .yml)")
    
    # Extract absolute path and filename
    path, filename = os.path.split(aspforge_file)

    # Ensure traget dir (where zip file if sopied to) exists
    if not os.path.isfile(aspforge_file):
        raise ValueError(f"File at {aspforge_file} does not exist.")

    # --- Check if target_dir is empty
        # Logic: Checks if the run_dir (which should be empty) contains folders/files. In case file has been unzipped there are many (~8 to  10 files) 
    if (len(os.listdir(path)) > 3) and not overwrite: 
        if verbose > 0:
            print(f"aspforge already processed, skipping task for {filename}")    
    else:
        # Run the `whiffle` command in the target directory
        run_command = f"""
        source {whiffle_env} && \
        cd {path} && \
        whiffle task run {filename} && \
        unzip -o "01*.zip" && \
        rm -f "01*.zip"
        """

        try:
            if verbose > 0: print(f"Process task {filename}")
            subprocess.run(
                run_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable="/bin/bash")
        except subprocess.CalledProcessError as e:
            print(f"Error while running whiffle task for {aspforge_file}:\n{e.stderr.decode()}")




def modify_namelist_file(
        namelist_file:Path,
        tstart:datetime|None, 
        tend:datetime|None, 
        dtwrite_restart:str, 
        read_restart_active:bool=False, 
        write_restart_active:bool=False,
        restart_input:str|None=None,
        spinup_phase:bool=False,
        add_statsimdata:bool=False,
        only_modify_simdata:bool=False,
        ):
    """ 
    This function modifies graspIn.000.nml files in the following ways:
        - Change start datetime of simulation (dofdif)      --> t_start
        - Change end datetime of simulation (doldif)        --> t_start + dt_iter
        - Change name of read-restart file  --> graspInRestart_<YYYYMMDDHHmm>.000.nc according to dofdif
    """
    def _get_datelist(t:datetime):
        """ Return list of date constituents; from year to seconds based on DATETIME """
        return [t.year, t.month, t.day, t.hour, t.minute, t.second, 0]
    


    # Ensure that timing exists
    if not only_modify_simdata:
        if not isinstance(tstart, datetime) or not isinstance(tend, datetime):
            raise ValueError(f"Time of start/end (tstart_window/tend_window) simulation must be of type datetime!!")
        
        # --- 1) Handle timing
        dtwrite    = pd.Timedelta(dtwrite_restart).total_seconds()
        t_previous = tstart - timedelta(seconds=dtwrite)

        timetag_now  = tstart.strftime('%Y%m%d%H%M')
        
        if spinup_phase:
            timetag_prev = "original"
        else:
            timetag_prev = t_previous.strftime('%Y%m%d%H%M')
        
        # Set default
        if restart_input is None and read_restart_active==1:
            restart_input = f"graspInRestart_{timetag_now}.meso.nc"

        # --- 2) Apply changes to namelist file
        restart_vars = (
            "u,v,w,qt,Thl,qr,t_soil,q_soil,"
            "up,vp,wp,qtp,Thlp,qrp,t_soilp,q_soilp,nutm,nutb"
        )
    
        if read_restart_active:
            patch["READRESTART"] = {
                "file": restart_input,
                "lactive": True,
                "var": restart_vars,
            }
    
        patch = {
            "RUN": {
                "dofdif": _get_datelist(tstart),
                "doldif": _get_datelist(tend),
            },
            "WRITERESTART": {
                "dtwrite": int(dtwrite),
                "lactive": write_restart_active,
                "var": restart_vars,
            },
        }
    
    else:
        print("Only Simdata relevant stuff is considered")
        patch = {}

    if add_statsimdata:
        patch["STATSIMDATA"] = {
            "dtav": 30,
            "dtwrite": 600,
            "lactive": True,
            "var": 'u[:,:,:], v[:,:,:], M[:,:,:]'
        }


    # Patch only specified fields; leave all other sections untouched
    tmp_file = namelist_file.parent / "graspIn_TMP.000.nml"
    f90nml.patch(namelist_file, patch, tmp_file)

    import re
    text = tmp_file.read_text()

    # Force only these section headers to uppercase
    for section in ("RUN", "WRITERESTART", "READRESTART", "STATSIMDATA"):
        text = re.sub(
            rf"(?im)^&{section}\b",
            f"&{section}",
            text,
        )

    # --- If READRESTART is inactive --> remove section from nml-file
    if not read_restart_active:
        text = re.sub(r"(?ims)^\s*&READRESTART\b.*?^\s*/\s*\n?", "", text)
    tmp_file.write_text(text)

    # --- 3) Rename new nml-file  and save old namelist files for debugging purposes
    
    if not only_modify_simdata:
        archive_name = namelist_file.parent / f"graspIn_{timetag_prev}.meso.nml"
        namelist_file.rename(archive_name)
        tmp_file.rename(namelist_file)




def transfer_vars_to_graspInNWP(
    base_file: str | Path,
    member_file: str | Path,
    variables: list[str],
    output_file: str | Path,
) -> Path:
    """
    Create output_file as an exact copy of base_file, then replace selected variable 
    values with values from member_file.

    Only the numerical data of "variables" are changed, but NetCDF structure 
    and metadata come entirely from base_file.
    """

    base_file = Path(base_file)
    member_file = Path(member_file)
    output_file = Path(output_file)

    replacement_data = {}
    target_dims = {}

    # --- Read member data and validate against base
    with xr.open_dataset(base_file) as ds_base, xr.open_dataset(member_file) as ds_member:

        for name in variables:
            if name not in ds_base:
                raise KeyError(f"{name!r} missing from base file")
            if name not in ds_member:
                raise KeyError(f"{name!r} missing from member file")

            base_var   = ds_base[name]
            member_var = ds_member[name]
            member_var = member_var.transpose(*base_var.dims)
            replacement_data[name] = member_var.load().values.copy()
            target_dims[name] = tuple(base_var.dims)
    # All xarray file handles are closed here.

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Work on a temporary copy of the base
    tmp_file = output_file.with_suffix(".tmp"+ base_file.suffix)
    copy2(base_file, tmp_file)

    # --- Replace selected data only
    with Dataset(tmp_file, "r+") as nc:

        for name, values in replacement_data.items():

            dst = nc.variables[name]

            if tuple(dst.dimensions) != target_dims[name]:
                raise ValueError(
                    f"{name}: target dimensions differ\n"
                    f"expected: {target_dims[name]}\n"
                    f"found:    {dst.dimensions}"
                )

            if dst.shape != values.shape:
                raise ValueError(
                    f"{name}: target shape differs\n"
                    f"expected: {values.shape}\n"
                    f"found:    {dst.shape}"
                )

            dst[:] = values

        # Atomically place completed file at output path
        os.replace(tmp_file, output_file)

    return output_file




# *********************************************************************************************
# *********************************************************************************************


def load_yml(yml_file:Path|str):
    """ Load ayml file """
    if isinstance(yml_file, Path):
        with yml_file.open("r") as file:
            return yaml.safe_load(file)
    elif isinstance(yml_file, str):
        with open(yml_file, 'r') as file:
            return yaml.safe_load(file)
    else:
        raise ValueError(f"yml_file must be directory to file provided as string or Path!!")


def path_to_str(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: path_to_str(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(path_to_str(v) for v in obj)
    
    # If none of these cases, just return input
    return obj

def save_yml(data, save_path: Path, fname: str):
    if not (fname.endswith(".yml") or fname.endswith(".yaml")):
        raise ValueError(f"Yaml file name must end with .yml or .yaml!! Fname provided was: {fname}")
    if not save_path.is_dir():
        raise ValueError(f"Save-path does not exist: {save_path}")

    yml_file = save_path / fname

    clean_data = path_to_str(data)
    with yml_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean_data, f, sort_keys=False)






# *********************************************************************************************
#             Windfield manipulations
#                   - unstagger u and v wind components
#                   - calculate wind magnitude
#                   - calculate wind components from magnitude and direction
#                   - calculate wind speed and direction from components
# *********************************************************************************************

# --- Unstagger u and v wind components
def unstagger_u_v(ds, xfoffset, yfoffset):
    """
    Unstagger C-grid to full-levels (xf, yf) by interpolating
    to the axis with full grid (xh --> xf  /  yh --> yf)
    """
    ds_new = ds.copy()
    # --- 1) interpolate from staggered xh -> xf (u-wind) AND yh -> yf (v-wind)
    u_destag = ds_new["u"].interp(xh=ds_new["xf"], kwargs={"fill_value":"extrapolate"}).drop_vars("xh")
    v_destag = ds_new["v"].interp(yh=ds_new["yf"], kwargs={"fill_value":"extrapolate"}).drop_vars("yh")
    # --- 2) Replace DataArray in Dataset with destaggered arrays    
    ds_new["u"] = u_destag
    ds_new["v"] = v_destag
    
    # --- 2) Add x-/y-offsets if specified
    if xfoffset:
        ds_new = ds_new.assign_coords({"xf": ds_new["xf"] + float(xfoffset)})
    if yfoffset:
        ds_new = ds_new.assign_coords({"yf": ds_new["yf"] + float(yfoffset)})

    # Return dataset without any xh or yh coords/dims
    return ds_new.drop_vars("xh").drop_vars("yh")

# --- Calculate Wind components from Wind-speed and direction
def restagger_u_v(ds, xh, yh, xf_offset:int, yf_offset:int):
    """
    Re-stagger u and v back to C-grid (xh, yf) and (xf, yh)
    Must supply xh and yh coordinates (saved from original staggered dataset)
    """
    # --- 1) Subtract offset 
    ds = ds.assign_coords({"xf": ds["xf"] - float(xf_offset)})
    ds = ds.assign_coords({"yf": ds["yf"] - float(yf_offset)})
    # --- 2) Interpolate back from full-level to staggered coordinates
    ds["u"] = ds["u"].interp(xf=xh, kwargs={"fill_value":"extrapolate"})
    ds["v"] = ds["v"].interp(yf=yh, kwargs={"fill_value":"extrapolate"})
    return ds


# --- Calculate Wind components from Wind-speed and direction
def calc_wind_components(mag, angle_radians:xr.DataArray):
    """
    Calcualte u and v component of wind
        WindDeg gives angle w.r.t wind origin --> norhterly(î):0°, easterly(<--):90°, westerly(-->):270°
    """
    if angle_radians.max() > 6.29:
        raise ValueError(f"Direction_angle has to be passed in radians!!!! (Error is raised because amxvlaue is above 6.29 ~ 360°!)")
    u = xr.DataArray(-mag * np.sin(angle_radians), dims=mag.dims, coords=mag.coords)
    v = xr.DataArray(-mag * np.cos(angle_radians), dims=mag.dims, coords=mag.coords)
    return u, v


def calc_wind_speed(da_u, da_v):
    M_vals = np.hypot(da_u.values, da_v.values)
    da_M = xr.DataArray(data=M_vals, coords=da_u.coords, dims=da_u.dims)
    return da_M

# --- Calculate wind direction from wind components
def calc_wind_direction(u, v):
    # Calculate wind direction in degrees
    d = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    direction = xr.DataArray(d, dims=u.dims, coords=u.coords)
    return direction
# *********************************************************************************************
# *********************************************************************************************

