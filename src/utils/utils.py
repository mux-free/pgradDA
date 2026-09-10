import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import glob
import shutil



# *********************************************************************************************
#             METRICS FOR MODEL EVALUATION:
#                   - rmse, mbe, mae, rmsd
# *********************************************************************************************
def calc_rmse(pred, true):
    """Calc rmse, over given dim if specified"""
    return np.sqrt(np.nanmean((pred-true)**2))

def calc_mbe(pred, true):
    """Calc rmse, over given dim if specified"""
    return np.nanmean(pred-true)

def calc_mae(pred, true):
    """Calc rmse, over given dim if specified"""
    return np.nanmean(np.abs(pred-true))

def calc_rmsd(pred, true):
    true_avg = np.nanmean(true)
    pred_avg = np.nanmean(pred)
    return np.sqrt(np.mean( ((pred - pred_avg) - (true - true_avg)) **2 ))
# *********************************************************************************************
# *********************************************************************************************




# *********************************************************************************************
#             FILE loading and manipulations
#                   - recursive file finder
#                   - get horizontal dim names
#                   - last n folders of path
#                   - add timedelta to date string
#                   - save dataset safely
# *********************************************************************************************

def find_files_recursive(path_to_folder, file_pattern):
    """Find files matching a pattern recursively."""
    pattern = os.path.join(path_to_folder, '**', file_pattern)
    return sorted(glob.glob(pattern, recursive=True))

def get_horizontal_dim_names(da:xr.DataArray) -> list:
    """ Function reurns dimension names for x and y coordinates (e.g. for u:["xh","yf"], v:["xf","yh"] and Thl:["xf","yf"]). """
    return [dim for dim in da.dims if dim.startswith("x") or dim.startswith("y")]

def last_n_folders(path: str, n: int = 6) -> str:
    """Return the last n folder components of a given path."""
    parts = path.split(os.sep)
    return os.sep.join(parts[-n:])

def add_timedelta(date, seconds, return_string=True, date_format='%Y/%m/%d/%H'):
    """ 
    Adds timedelta to string-date and outputs in same format as input.
    Parameters:
    -----------
        date (str) OR (datetime): If string, must be of format  "YYYY/MM/DD/HH"!!!!   TODO: write a regex test that checks this
        
        seconds (float): Timedelta that is added to date (in seconds)
    """
    # --- Get datetime object from date
    if not isinstance(date, datetime): date = datetime.strptime(date, date_format)
    # --- Convert ∂time to int if it is passed as str. And check that no other types are passed for delta_hours.
    assert isinstance(seconds, int) or isinstance(seconds, float), f"seconds must be either of type int or float not: {type(seconds)}"
    newdate = date + timedelta(seconds=seconds) # Add time-delta to datetime-object
    # --- Either return datetime or string of date
    return newdate.strftime(date_format) if return_string else newdate

def save_ds(ds, path):
    """ Save dataset (netCDF) in a save way to avoid PermissionDenied errors """
    try:
        ds.to_netcdf(path)
    except PermissionError as e: 
        tmp_path = os.path.splitext(path)[0] + "_tmp.nc"
        # Save to the temporary file
        ds.to_netcdf(tmp_path)
        # Replace the original file with the temporary file
        os.replace(tmp_path, path)


def copy_folder(infolder: str|Path, outfolder: str|Path, 
                verbose: int = 1, overwrite: bool = False
                ) -> None:
    """
    Copies an entire folder recursively.

    Args:
        infolder (str): Source folder path.
        outfolder (str): Destination folder path.
        overwrite (bool): If True, existing files are overwritten. Otherwise skip.
    """
    try:
        shutil.copytree(src=infolder, dst=outfolder, dirs_exist_ok=True)
    except FileExistsError:
        if overwrite:
            # print("\tFolder already exists, But folder overwritten!")
            shutil.copytree(src=infolder, dst=outfolder, dirs_exist_ok=True)
        else:
            if verbose > 0: print("Folder already exists, no folders were copied!")
            return
    if verbose > 0:
        short_in, short_out = last_n_folders(infolder, 8),  last_n_folders(outfolder, 8)
        print(f"Copied folder: .../{short_in} --> .../{short_out}")
# *********************************************************************************************
# *********************************************************************************************






