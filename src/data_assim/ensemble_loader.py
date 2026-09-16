
import numpy as np
import xarray as xr
import glob
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(1, '../utils')
import utils_aspire

@dataclass
class EnsembleLoader:
    nmembers: int
    data_folder: Path
    timestamp: datetime
    
    # Hanlde staggered-grid
    xh = None
    yh = None
    xf_offset = None
    yf_offset = None

    date_string: str = field(init=False)

    def __post_init__(self):
        self.date_string = self.timestamp.strftime("%Y%m%d%H%M")


    def restagger_u_v(self, ds):
        ds = utils_aspire.restagger_u_v(
            ds, xh=self.xh, yh=self.yh, 
            xf_offset=self.xf_offset, yf_offset=self.yf_offset #type: ignore
            )
        return ds


    def get_coordinate_offset(self):
        """ 
            Restart files are not location sensitive (start at 0 meter North and 0meter West), thus offset needs to be 
            calculated based on Simdata file
        """
        # --- 1) Load Dummy simdata and restart file (all files have same grid by design, thus it doesn't matter which is chosen)
        dummy_simdata_dir = sorted(glob.glob(f"{self.data_folder}/ensemble_01/graspOutSimdata*.meso.nc"))
        if len(dummy_simdata_dir)>0:
            dummy_simdata_dir = dummy_simdata_dir[0]  
        else: 
            raise ValueError(f"No simdata files found in {self.data_folder}/ensemble_01/")
    
        dummy_restart_dir = sorted(glob.glob(f"{self.data_folder}/ensemble_01/graspOutRestart*.meso.nc"))
        if len(dummy_restart_dir)>0:
            dummy_restart_dir = dummy_restart_dir[0]  
        else: 
            raise ValueError(f"No restart files found in {self.data_folder}/ensemble_01/")

        if len(dummy_simdata_dir) == 0:
            print("Warning! Coordinate offset of restart file couldn't be calculated, as no graspOutSimdata file is present!")

        ds_sd = xr.open_dataset(dummy_simdata_dir)
        ds_rs = xr.open_dataset(dummy_restart_dir)

        # --- 2) Store xh/yh values of rs file for later re-interpoaltion to original grid
        self.xh, self.yh = ds_rs["xh"], ds_rs["yh"]

        # --- 3) Calculate offset value for x and y direction (1 value, as offset is the same for all points along x/y direction)
        xf_offset = np.unique((ds_sd["xf"].values - ds_rs["xf"].values))
        yf_offset = np.unique((ds_sd["yf"].values - ds_rs["yf"].values))
        self.xf_offset, self.yf_offset = xf_offset[0], yf_offset[0]
        

    def load_ensemble(self, fname):
        """ Load graspOutRestart.meso.nc files """
        # --- 1) LOAD RESTART DATA 
        all_members:  list[xr.Dataset] = []
        for i in range(1,self.nmembers+1):
            fdir = f"{self.data_folder}/ensemble_{str(i).zfill(2)}/{fname}"
            ds = xr.open_dataset(fdir, engine="h5netcdf")
            
            # If time-dimension is present, round it to next minute! --> S.t. times can be concatenated
            if "time" in ds.sizes:
                ds["time"] = ds["time"].dt.round("5min")

            all_members.append(ds)
        ds_rs = xr.concat(all_members, dim="ensemble", join="override")
        ds_rs["ensemble"] = ds_rs["ensemble"] + 1

        # Store ds_prior as attribute
        self.ds_prior = ds_rs
    


    def save_posterior_ensemble(self, ds_post): 
        """ Save Restart Files of each member (dimension is called 'ensemble') """
        fname  = f"graspInRestart_{self.date_string}.meso.nc" 
        print(f"Save files as {fname}")
        for i in range(1,self.nmembers+1):
            fdir = f"{self.data_folder}/ensemble_{str(i).zfill(2)}/{fname}"
            ds_member = ds_post.sel(ensemble=i)
            ds_member.to_netcdf(fdir)




    def __call__(self, fname_base:str):

        if not (
            fname_base == "graspOutRestart" 
            or 
            fname_base == "graspOutSimdata"
            or
            fname_base == "graspInRestart" 
            ):
            raise ValueError(
                f"Base name of file has to be either 'graspOutRestart' or 'graspOutSimdata'. \n"
                f"But fname-base name passed: {fname_base} is NOT specified"
                )
        
        # --- 1) Load ensemble
        fname = f"{fname_base}_{self.date_string}.meso.nc"
        self.load_ensemble(fname)
        ds = self.ds_prior

        # --- 2) Process ensemble
        # - 2.1) Caluclate x-/y offsets for graspOut.RESTART files
        if fname_base == "graspOutRestart":
            self.get_coordinate_offset()
            # Add ensemble timestamp for Restart files, as they don't already have one (Simdata already has time dimension)
            ds = ds.expand_dims(time=[self.timestamp])
        
        ds = utils_aspire.unstagger_u_v(ds, xfoffset=self.xf_offset, yfoffset=self.yf_offset)
        
        return ds


    ######## LOAD CONTROL SIMDATA ###################################
    def load_control_simdata(self):
        """ Load graspOutRestart.meso.nc files """
        # --- 1) Define Control Data and find all files in ctrl directory
        fname  = f"graspOutSimdata_*.meso.nc" 
        fdir = sorted(glob.glob(f"{self.data_folder}/{fname}"))
        if len(fdir) == 0:
            raise ValueError(f"No graspOutSimdata files found in {self.data_folder}")
        # --- 2) Load control data
        ds = xr.open_mfdataset(fdir, engine="h5netcdf")
        # --- 3) Unstagger dataset
        ds = utils_aspire.unstagger_u_v(ds=ds, xfoffset=None, yfoffset=None)
        return ds

        