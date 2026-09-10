import xarray as xr
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path

# Add .../REFORM/src to sys.path, s.t. i can improt my own modelus!!


## --- OWN MODULES
from .ensemble_loader import EnsembleLoader
from .member_nanny import MemberNanny



# sys.path.insert(1, '../../')
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(PROJECT_ROOT))
from run_experiments.experiment_context import ExperimentContext # type: ignore

sys.path.insert(1, str(PROJECT_ROOT / "utils"))
# sys.path.insert(1, '../../utils')
import utils_aspire

# LETKF MODULES 
from .letkf_interface import run_letkf


class ForwardOperator(MemberNanny):
    def __init__(self, 
                 context: ExperimentContext, 
                 simdir_ctrl: Path,
                 tstart_window: datetime,
                 dt_window: int, 
                 ensemble_nr: int, 
                 verbose: int = 1,
                 ):
        
        super().__init__(context, simdir_ctrl, ensemble_nr, verbose)
        self.tstart_window: datetime = tstart_window
        self.dt_window: float = dt_window

    def __call__(
            self, 
            f_namelist: Path,
            read_restart: bool,
            write_restart: bool,
            add_statsimdata: bool = True,
            ):
        
        # --- 1) Update timing variables according to new cycle (using tstart_window and dt_window)
        self.update_timing(tstart_window=self.tstart_window, dt_run=self.dt_window)
        
        # --- 2) Modify and save new namelist file according to timing vaariables  
        utils_aspire.modify_namelist_file( # type: ignore
            namelist_file=f_namelist,
            tstart=self.tstart_window,
            tend=self.tend_window,
            dtwrite_restart=f"{self.dt_da}s",
            read_restart_active=read_restart,
            write_restart_active=write_restart,
            spinup_phase=True,
            add_statsimdata=add_statsimdata
        )

        # --- 3) Run Aspire    
        utils_aspire.aspire(namelist_file=f_namelist) # type: ignore

        # --- 4) Rename output graspOut-files --> Add a time-tag to the files 
        self.rename_output_files(restart_present=write_restart)





class AssimilationOperator:
    def __init__(self, 
                 context: ExperimentContext,
                 t_assim: datetime, 
                 gpu: bool,
                 ):
        
        self.ctx = context
        self.gpu = gpu
        self.tda = t_assim


        self.rttp_factor = self.ctx.rtpp 

        if isinstance(self.rttp_factor, float):
            if not (0 < self.rttp_factor < 0.95):
                raise ValueError(f"RTTP factor is out of bounds [0,0.95]. RTTP specified: {self.rttp_factor}")
            self.apply_rtpp = True
        else:
            self.apply_rtpp = False

        
        # BasePath dir
        self.base_path: Path = self.ctx.base_path
        self.simdir_ctrl: Path = self.base_path / self.ctx.t0_spinup.strftime("%Y/%m/%d/%H")



    def insert_analysis_section(
            self,
            ds_restart: xr.Dataset,
            ds_analysis: xr.Dataset,
            ztop_idx:int,
            ):
        """ 
        This function inserts the assimialted sub-domain back into the full domain 
        """
        # Work on a copy to avoid mutating the original by surprise
        out = ds_restart.copy(deep=True)
        for var in self.ctx.state_vars:
            # analysis array for this variable
            arr = ds_analysis[var]
            arr=arr.squeeze()
            
            # Reorder analysis dims to match the target variable’s dims
            target_dims = out[var].dims
            keep_dims_in_order = [dim for dim in target_dims if dim in arr.dims]
            arr = arr.transpose(*keep_dims_in_order)

            # Make sure shapes match the indexer region (otherwise make it fail if sth is off)
            tgt_shape = out[var].isel(zf=slice(None,ztop_idx)).shape
            if arr.shape != tgt_shape:
                raise ValueError(
                    f"Shape mismatch for {var}: analysis {arr.shape} vs target slice {tgt_shape} "
                    f"(dims target={target_dims}, analysis={arr.dims})")
            
            # Assign back into the region (isel for positional assignment)
            out[var].isel(zf=slice(None,ztop_idx))[:] = arr

        return out




    def rtpp(self, ds_prior, ds_post, alpha:float, ens_dim="ensemble"):
        """ Relaxation to Prior Perturbation implementation """
        # 1) Means
        xb_mean = ds_prior.mean(dim=ens_dim)
        xa_mean = ds_post.mean(dim=ens_dim)

        # 2) Perturbations
        xb_prime = ds_prior - xb_mean # background perts
        xa_prime = ds_post  - xa_mean # analysis perts

        # 3) RTPP perturbations
        xa_prime_rtpp = alpha * xa_prime + (1.0-alpha) * xb_prime

        # 4) Rebuild ensemble with analysis mean
        ds_post_rtpp = xa_mean + xa_prime_rtpp
        return ds_post_rtpp


    def __call__(
            self, 
            ds_obs:xr.Dataset, 
            ztop_assim:int,
            do_assimilation:bool=True, 
            weight_save_path:str|None=None,
            return_posterior:bool=False,
            ):
        
        """
        Orchestrates the Assimilation step:
            1) Load Prior data
            2) DA update step
        """
        
        ## --- 1) LOAD GRASP data
        ens_loader = EnsembleLoader(
            data_folder=self.simdir_ctrl, 
            timestamp=self.tda, 
            nmembers=self.ctx.n_members)
        ds_prior = ens_loader(fname_base="graspOutRestart")
        ds_prior = ds_prior[self.ctx.state_vars]
        
        ds_prior_small = ds_prior.isel(zf=slice(None,ztop_assim))


        if do_assimilation:
            
            # --- 2) Data assimilation Update
            ds_post = run_letkf(
                ds_prior=ds_prior_small,
                ds_obs=ds_obs, 
                loc_radius_m=self.ctx.radius,
                vert_loc=self.ctx.vert_loc,
                inflation=self.ctx.inflation,
                R_std=self.ctx.r2,
                state_vars=self.ctx.state_vars,
                varnames=self.ctx.state_vars,
                gpu=self.gpu,
                weight_save_path=weight_save_path)


            # --- 3) Optional: Relaxation To Prior Perturbation using alpha=0.75
            if self.apply_rtpp:
                # if isinstance(self.rttp_factor, float):
                ds_post = self.rtpp(ds_prior_small, ds_post, alpha=self.rttp_factor)



            # --- 5) Project data back s.t. it fits with Restart-file structure 
            ds_post = self.insert_analysis_section(
                ds_restart=ds_prior.squeeze(),      # the big one you want to update
                ds_analysis=ds_post,           # analysis on the small window
                ztop_idx=ztop_assim
            )
            
            # --- 5) Project data back s.t. it fits with Restart-file structure 
            ds_post2 = ens_loader.restagger_u_v(ds=ds_post)
    
        else:
            ds_post2=ds_prior

        # --- 6) Save Posterior state as new input state
        ens_loader.save_posterior_ensemble(ds_post=ds_post2)

        if return_posterior:
            return ds_post2
            






