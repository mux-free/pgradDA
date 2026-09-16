import pandas as pd
import numpy as np
import xarray as xr
import glob
import re

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Sequence, ClassVar


import dask
dask.config.set({
    "array.slicing.split_large_chunks": False,
    "optimization.fuse.active": True,
    "scheduler": "threads",      # <- use local threaded scheduler
})

import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "utils"))
from utils_aspire import unstagger_u_v

"""--------------------------------------------------------------------------------------------------
ASSUMPTIONS OF DATA LOADER:
---------------------------
1) File-names are of form: graspOut<Restart>_YYYYMMDDHH(MM).000.nc.  --> Date is contiend in filename
2) Ensemble ID is contained in filepath as ensemble_XX, only for control this is NOT present
---------------------------------------------------------------------------------------------------"""



@dataclass
class BaseLoader:
    """ This class provides basic function needed to load GraspInRestart and GraspOutSimdata files """

    ## Inits
    exp_dir: Path
    load_cotrol:bool = True

    # --- Define common settings how to load data
    round_freq: str = "min"
    z_range: tuple[int, int]  | None = None
    xy_subset: int | Sequence[int] | None = None
    
    # --- Dask
    memory: str = "30GB"
    n_workers: int = 4
    
    chunks: ClassVar[dict] = {"time": 8, "yf": 128, "xf": 128}
    common_open_kwargs: ClassVar[dict] = dict(
        engine="h5netcdf",
        chunks=chunks,
        parallel=True,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )




    def handle_paths(self, fname_pattern: str):
        """ 
        Handle paths and retrieve list of files to load
        """
        # --- Generate list of dates to load  ---  either with control or without
        if self.load_cotrol:
            date_dirs = sorted(glob.glob(f"{self.exp_dir}/**/20*/**/{fname_pattern}*.nc", recursive=True))
        else:
            date_dirs = sorted(glob.glob(f"{self.exp_dir}/**/20*/**/ensemble_*/{fname_pattern}*.nc", recursive=True))

        return date_dirs
    

    def _preprocess(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Preprocess Simdata files:

        1.  Round time coordinate to nearest minute (to avoid floating point issues).
        2.  Subset spatially in x/y and z if requested.
        """
        # --- 1) round time coordinate
        if "time" in ds.dims:  
            ds = ds.assign_coords(time=ds["time"].dt.round(self.round_freq))

        # --- 2) Get a ensemble coordinate from the filename, e.g., ...ensemble_10/
        src   = ds.encoding["source"]
        m_mem = re.search(r'ensemble_(\d+)', src)
        ens_id = int(m_mem.group(1)) if m_mem else 0

        ds = ds.expand_dims("ensemble").assign_coords(ensemble=("ensemble", [ens_id]))
        return ds


    def postprocess(self, ds: xr.Dataset, calculate_wind_magnitude:bool) -> xr.Dataset:
        """ Postprocess loaded dataset: """
        # --- 1) Unstagger u/v to full levels + calculate wind magnitude (optionally)
        ds = unstagger_u_v(ds, xfoffset=None, yfoffset=None)
        if calculate_wind_magnitude:
            ds["M"] = np.hypot(ds["u"], ds["v"])

        # --- 2) spatial subselection
        if self.xy_subset is not None:
            if isinstance(self.xy_subset, int):
                ds = ds.isel(xf=slice(self.xy_subset, -self.xy_subset), yf=slice(self.xy_subset, -self.xy_subset))
            elif isinstance(self.xy_subset, (list, tuple)) and len(self.xy_subset) == 2:
                ds = ds.isel(xf=slice(self.xy_subset[0], self.xy_subset[1]), yf=slice(self.xy_subset[0], self.xy_subset[1]))
            else:
                raise ValueError("xy_subset must be an int or a list/tuple of two ints.")
        if self.z_range is not None:
            zmin, zmax = self.z_range
            ds = ds.sel(zf=slice(zmin, zmax))
        return ds


    def load_ensemble(self, files, f_preprocess):
        ds = xr.open_mfdataset(
            files, 
            combine="by_coords",
            preprocess=f_preprocess,
            **self.common_open_kwargs,
        )
        ds = self.postprocess(ds, calculate_wind_magnitude=True)
        return ds



class RestartLoader(BaseLoader):
    """ 
    DataLoader for GraspInRestart files
    """

    def _preprocess_restartfile(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Extract time from the filename stored in ds.encoding['source'] and
        attach it as a proper time coordinate.

        Expects patterns like: ..._YYYYMMDDHHMM.sss.nc
        e.g. graspOutSimdata_202504142200.000.nc -> 2025-04-14 22:00
        """
        
        # --- 1) Extract time from filename 
        # Grab the 12-digit datetime block before the dot (YYYYMMDDHHMM)
        src = ds.encoding["source"]
        m = re.search(r'(\d{12})\.\d+', src)
        if m:
            tstr = m.group(1)
        else:
            m = re.search(r'(\d{10})\.\d+', src)
            if not m:
                tstr=""
            tstr=""
        
        if len(tstr) == 12:
            time = pd.to_datetime(tstr, format="%Y%m%d%H%M")
        elif len(tstr) == 10:
            time = pd.to_datetime(tstr, format="%Y%m%d%H")
        else:
            time=None

        if time is not None:
            # --- 2) Attach as time coordinate.
            if "time" in ds.dims:         # If a time dimension already exists with length 1, fill that.
                if ds.dims["time"] != 1:
                    raise ValueError(f"Expected time dim of size 1 in file {src}, found {ds.dims['time']}")
                ds = ds.assign_coords(time=("time", [time]))
            else:
                ds = ds.expand_dims("time").assign_coords(time=("time", [time]))
        
        # --- 3) spatial trimming and diagnostics copied verbatim 
        ds = self._preprocess(ds)
        return ds

    def __call__(self, fname_pattern):
        # --- 1) Handle paths
        f_rs = self.handle_paths(fname_pattern)
        if len(f_rs) == 0:
            raise ValueError(f"No files found in {self.exp_dir} matching pattern {fname_pattern}")
        # --- 2) Load data
        return self.load_ensemble(files=f_rs, f_preprocess=self._preprocess_restartfile)



class SimdataLoader(BaseLoader):
    """ 
    DataLoader for GraspOutSimdata files
    """
    def __call__(self):
         # --- 1) Handle paths
        f_simdata = self.handle_paths("graspOutSimdata")
        if len(f_simdata) == 0:
            raise ValueError(f"No files found in {self.exp_dir} matching pattern graspOutSimdata")
        # --- 2) Load data
        return self.load_ensemble(files=f_simdata, f_preprocess=self._preprocess)



def load_data(
        exp_dir: Path,
        load_rsi: bool = True,
        load_rso: bool = True,
        load_simdata: bool = True,
        load_control: bool = True,
        round_freq: str = "5min",
        domain_subset:dict = dict(xy_iboundary = None, z_range = None),
        memory_handling:dict = dict(n_workers=5, memory="40GB"),
        other_data=None,
        fpattern=None
        ):
    """ function to load data using the DataLoader classes """
    
    # --- Define parameters
    xy_iboundary = domain_subset["xy_iboundary"]
    z_range      = domain_subset["z_range"]
    nworkers = memory_handling["n_workers"]
    memory   = memory_handling["memory"]

    # --- Initialize loaders
    restart_loader = RestartLoader(
        exp_dir=exp_dir,
        xy_subset=xy_iboundary,
        z_range=z_range,
        round_freq=round_freq,
        n_workers=nworkers,
        memory=memory,
        load_cotrol=load_control,
    )
    simdata_loader = SimdataLoader(
        exp_dir=exp_dir,
        xy_subset=xy_iboundary,
        z_range=z_range,
        round_freq=round_freq,
        n_workers=nworkers,
        memory=memory,
        load_cotrol=load_control,
    )

    outdict = {}
    # --- Load data
    if load_rsi:
        rsi_data = restart_loader(fname_pattern="graspInRestart")
        outdict["restart_in"] = rsi_data
    if load_rso:
        rso_data = restart_loader(fname_pattern="graspOutRestart")
        outdict["restart_out"] = rso_data
    if load_simdata:
        simdata = simdata_loader()
        outdict["simdata"] = simdata
    if other_data is not None:
        data = restart_loader(fname_pattern=fpattern)
        outdict[other_data] = data

    return outdict

