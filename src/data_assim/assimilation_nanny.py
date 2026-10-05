import xarray as xr
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import shutil
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
                 simdir_ctrl: Path,
                 tstart_window: datetime,
                 dt_window: int, 
                 ensemble_nr: int | None, 
                 verbose: int = 1,
                 ):
        
        super().__init__(simdir_ctrl, ensemble_nr, verbose)
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
            dtwrite_restart=f"{self.dt_window}s",
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




    def copy_restart_inout(
            self,
            fname_in: str = "graspOutRestart",
            fname_out: str = "graspInRestart",
            file_appendix: str = ".meso.nc"
            ):
        time_string = self.tda.strftime("%Y%m%d%H%M")
        # --- Loop through all ensembles
        nmembers = self.ctx.n_members
        for i in range(1, nmembers+1):
            fdir = self.ctx.simdir_ctrl / f"ensemble_{str(i).zfill(2)}"
            fname_old = f"{fname_in}_{time_string}{file_appendix}"
            fname_new = f"{fname_out}_{time_string}{file_appendix}"
            fin  = fdir / fname_old
            fout = fdir / fname_new
            
            if not fin.is_file():
                raise ValueError(f"Restart file: {fin} does not exist!")
        
            shutil.copy(fin, fout)




    def insert_analysis_section(
            self,
            ds_restart: xr.Dataset,
            ds_analysis: xr.Dataset,
            ztop_idx:int|None,
            vertical_coord: str = "zf",
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
            tgt_shape = out[var].isel({vertical_coord: slice(None,ztop_idx)}).shape
            if arr.shape != tgt_shape:
                raise ValueError(
                    f"Shape mismatch for {var}: analysis {arr.shape} vs target slice {tgt_shape} "
                    f"(dims target={target_dims}, analysis={arr.dims})")
            
            # Assign back into the region (isel for positional assignment)
            out[var].isel({vertical_coord:slice(None,ztop_idx)})[:] = arr

        return out


    def overwrite_nwp_with_analysis(
            self,
            ds_nwp_post: xr.Dataset,
            ):

        folder_struct = self.ctx.t0_spinup.strftime("%Y/%m/%d/%H")
        id_membs = ds_nwp_post.ensemble.values
        
        for id_memb in id_membs:
            
            # --- Load original dataset
            nwp_path = self.ctx.simdir_ctrl / f"ensemble_{str(id_memb).zfill(2)}"
            ds_og = xr.open_dataset(f"{nwp_path}/graspInNWP.meso.nc")
            ds_nwpin = ds_og.copy(deep=True)

            # -- Select correct member and slices of origi NWPIn and of posterior
            ds_A = (ds_nwp_post
                    .sel(ensemble=id_memb)
                    .rename({"xf":"x", "yf":"y", "zf":"z"})
                    .squeeze(dim=["z", "time"])
                    )

            # -- Modify pressure gradient at current time and second highest level
            z, t = ds_A.z.item(), ds_A.time.item()
            ds_nwpin["dpdx"].loc[{'z': z, 'time': t}] = ds_A["dpdx"]
            ds_nwpin["dpdy"].loc[{'z': z, 'time': t}] = ds_A["dpdy"]

            # -- Save back to file
            ds_nwpin.to_netcdf(f"{nwp_path}/nwp_tmp.meso.nc")
            ds_prior = ds_og.sel(time=t).expand_dims(time=[t])
            ds_prior.to_netcdf(f"{nwp_path}/graspInNWP.prior.{t.strftime('%Y%m%d%H%M')}.nc")

            # -- Save dataset back
            ds_nwpin_full = utils_aspire.transfer_vars_to_graspInNWP(
                base_file = Path(f"{nwp_path}/graspInNWP.meso.nc"),
                member_file = Path(f"{nwp_path}/nwp_tmp.meso.nc"),
                variables = ["dpdx", "dpdy"],
                output_file = Path(f"{nwp_path}/graspInNWP.meso.nc"),
            )



    def rtpp(self, ds_prior, ds_post, alpha:float, ens_dim="ensemble"):
        """ Relaxation to Prior Perturbation implementation """
        # 1) Means
        xb_mean = ds_prior.mean(dim=ens_dim)
        xa_mean = ds_post.mean(dim=ens_dim)

        # 2) Perturbations
        xb_prime = ds_prior - xb_mean # background perts
        xa_prime = ds_post  - xa_mean # analysis perts

        # 3) RTPP perturbations
        xa_prime_rtpp = alpha * xb_prime + (1.0-alpha) * xa_prime

        # 4) Rebuild ensemble with analysis mean
        ds_post_rtpp = xa_mean + xa_prime_rtpp
        return ds_post_rtpp


    def __call__(
            self, 
            ds_obs:xr.Dataset, 
            state_vars: list[str],
            ztop_assim:int|None,
            do_assimilation:bool, 
            weight_save_path:str|None=None,
            return_posterior:bool=False,
            ):
        
        """
        Orchestrates the Assimilation step:
            1) Load Prior data
            2) DA update step
        """
        

        nwp_assim = True if ("dpdx" in state_vars or "dpdy" in state_vars) else False
        fname = "graspOutRestart" if not nwp_assim else "graspInNWP"
        fname_psudo = "graspOutRestart"

        ## --- 1) LOAD GRASP data
        ens_loader = EnsembleLoader(
            data_folder=self.ctx.simdir_ctrl,
            timestamp=self.tda,
            nmembers=self.ctx.n_members)
        ds_prior = ens_loader(fname_base=fname)
        
        
        if nwp_assim:
            zm2 = ds_prior.isel(z=-1).z.item()
            ds_prior_small = (ds_prior
                  .rename({"x":"xf", "y":"yf", "z":"zf"})
                  .isel(zf=-2)
                  .expand_dims(zf=[zm2])
                  )
            vert_loc = None
        else:
            ds_prior_small = ds_prior.isel(zf=slice(None,ztop_assim))
            vert_loc = self.ctx.vert_loc
        
        # Load pseudo state for pred-obs
        ds_prior_pseudo = ens_loader(fname_base=fname_psudo)
        ds_prior_pseudo = ds_prior_pseudo.isel(zf=slice(None,ztop_assim))



        if do_assimilation:
            # --- 2) Data assimilation Update
            ds_post = run_letkf(
                ds_prior=ds_prior_small,
                ds_prior_pseudo=ds_prior_pseudo,
                ds_obs=ds_obs, 
                loc_radius_m=self.ctx.radius,
                vert_loc=vert_loc,
                inflation=self.ctx.inflation,
                R_std=self.ctx.obs_std,
                state_vars=state_vars,
                obs_vars=self.ctx.obs_vars,
                gpu=self.gpu,
                weight_save_path=weight_save_path
                )

            # --- 3) Optional: Relaxation To Prior Perturbation using alpha=0.75
            if self.apply_rtpp:
                # if isinstance(self.rttp_factor, float):
                ds_post = self.rtpp(ds_prior_small, ds_post, alpha=self.rttp_factor)
            
            # --- 5) Project data back s.t. it fits with Restart-file structure 
            if nwp_assim:
                # ==================
                # NWP assim
                # ==================
                self.overwrite_nwp_with_analysis(ds_nwp_post=ds_post)
                self.copy_restart_inout()
            
            else:
                # ==================
                # Meso assim
                # ==================
                ds_post = self.insert_analysis_section(
                    ds_restart=ds_prior.squeeze(),      # the big one you want to update
                    ds_analysis=ds_post,           # analysis on the small window
                    ztop_idx=ztop_assim
                )            
                # --- 5) Project data back s.t. it fits with Restart-file structure 
                ds_post = ens_loader.restagger_u_v(ds=ds_post)

                # --- 6) Save Posterior state as new input state
                ens_loader.save_posterior_ensemble(ds_post=ds_post)

        else:
            ds_post=ds_prior

        if return_posterior:
            return ds_post
            




class AssimilationConductor:
    def __init__(
            self, 
            config_path: Path, 
            base_dir: Path,
            obs_file: Path,
            ztop_assim_idx: int = 30,
            save_weights: bool = True,
            gpu: bool = True,
            ):
        
        # --- Context 
        config = self._load_config(config_path / "config.yml")
        config["PATHS"]["experiment_dir"] = base_dir
        self.ctx = ExperimentContext(config)
        self.gpu = gpu
        self.save_weights = save_weights

        # --- Timing 
        self.t_now: datetime = self.ctx.t0_spinup

        # --- Paths
        self.path_assim_obs: Path = base_dir / "obs_data" / "assimilated"
        self.checkpoint_file: Path = base_dir / "checkpoint.yml"
        self.obs_file: Path = obs_file
        
        # -- Assim specifics
        self.assim_schedule: list = self._create_assim_schedule()
        self.current_schedule_idx = 0
        self.ztop_assim_idx = ztop_assim_idx


        if self.save_weights:
            weight_save_path = base_dir / "output" / "letkf_weights"
            weight_save_path.mkdir(parents=True, exist_ok=True)


    def _load_config(self, ymlpath:Path):
        """ Load config.yml file """
        with ymlpath.open("r") as file:
            return yaml.safe_load(file)


    def load_obs_4_da(
            self, 
            obs_file:Path, 
            tda_window:datetime
            ) -> xr.Dataset:
        """ Function selects correct times from dictionary of LiDARs/Tower that will be assimilated """
        ds_obs = xr.open_dataset(obs_file)
        # -- 1) Select time 
        ds_obs = ds_obs.sel(time=tda_window, method="nearest", tolerance="5min")
        ds_obs = ds_obs.expand_dims(time=[tda_window])

        return ds_obs
    
    def _create_assim_schedule(self):
        """ Create a list that stores the times for each simulation (0: before the first run, to the time when the last run is started))"""
        seconds_dawindow = self.ctx.dt_da
        n_windows = self.ctx.n_da

        init_times = []
        for i in range(n_windows):
            t_cyc = self.ctx.t0_da + i*timedelta(seconds=seconds_dawindow)
            init_times.append(t_cyc)
        return init_times


    def spinup(
            self, 
            ensemble_run:bool=True
            ):
        """ Run Spin-Up as simple forward step for ensemble and control simulation"""
        # --- FORWARD Run Members
        iterator = range(1, self.ctx.n_members + 1)if ensemble_run else range(0,1)
        for i in iterator:
            print(f"SpinUp for member {i:2d}")
            forward = ForwardOperator(
                simdir_ctrl=self.ctx.simdir_ctrl,
                ensemble_nr=i, 
                tstart_window=self.ctx.t0_spinup,
                dt_window=self.ctx.dt_spinup 
                )        

            if ensemble_run:
                fnml = self.ctx.simdir_ctrl / f"ensemble_{str(i).zfill(2)}" / "graspIn.meso.nml"
            else:
                fnml = self.ctx.simdir_ctrl / "graspIn.meso.nml"
            forward(f_namelist=fnml, read_restart=False, write_restart=True)


    def forward_step(
            self, 
            t_init: datetime,
            ensemble_run:bool=True
            ):
        """ If ensemble_run is False, the control simulation is run """
    
        iterator = range(1, self.ctx.n_members + 1)if ensemble_run else range(0,1)
        for i in iterator:
            forward = ForwardOperator(
                simdir_ctrl=self.ctx.simdir_ctrl,
                ensemble_nr=i, 
                tstart_window=t_init, 
                dt_window=self.ctx.dt_da 
                )        

            if ensemble_run:
                fnml = self.ctx.simdir_ctrl / f"ensemble_{str(i).zfill(2)}" / "graspIn.meso.nml"
            else:
                fnml = self.ctx.simdir_ctrl / "graspIn.meso.nml"
            forward(f_namelist=fnml, read_restart=True, write_restart=True)

        # --- Update current time-step
        if ensemble_run:
            self.current_schedule_idx += 1



    def forecast_step(
            self, 
            tinit_forecast: datetime,
            ensemble_run:bool=True,
            ):
        """ Run pseudo forecast after data-assimilation is done until the end of the available time """
        # --- FORWARD Run Members
        
        simtype = "Ensemble" if ensemble_run else "Control"
        print(f"{simtype} Run Predictions at: {self.ctx.t0_pred} for dtpred: {self.ctx.dt_pred} secs")

        iterator = range(1, self.ctx.n_members + 1)if ensemble_run else range(0,1)
        for i in iterator:
            forward = ForwardOperator(
                context=self.ctx, 
                simdir_ctrl=self.ctx.simdir_ctrl,
                ensemble_nr=i, 
                tstart_window=tinit_forecast, 
                dt_window=self.ctx.dt_pred 
                )        

            if ensemble_run:
                fnml = self.ctx.simdir_ctrl / f"ensemble_{str(i).zfill(2)}" / "graspIn.meso.nml"
            else:
                fnml = self.ctx.simdir_ctrl / "graspIn.meso.nml"
            forward(f_namelist=fnml, read_restart=True, write_restart=True)



    def assim_step(
            self, 
            ztop_assim:int, 
            obs_file:Path,
            do_assimilation:bool,
            return_posterior:bool=False, 
            weight_save_path=None,
            ):
        """ Compute data assimilation update at given time-step"""

        # TimeStep of DA
        if self.current_schedule_idx <= len(self.assim_schedule)-1:
            t_cyc = self.assim_schedule[self.current_schedule_idx]
        else:
            print(f"Last Assimilation step reached at {self.assim_schedule[-1]}")
            return self.assim_schedule[-1]

        # --- 3.3 Prepare Observations at current timestep t or during Assim_window
        print(f"Prepare Observations")
        ds_obs_da = self.load_obs_4_da(obs_file=obs_file, tda_window=t_cyc)

        # --- 3.4 Perform ASSIMILATION STEP
        print(f"\nAssimilation step")
        assimilation = AssimilationOperator(
            context=self.ctx,
            t_assim=t_cyc,
            gpu=self.gpu,
            )
        
        assimilation(
            ds_obs=ds_obs_da, 
            state_vars=self.ctx.state_vars,
            ztop_assim=ztop_assim, 
            do_assimilation=do_assimilation, 
            return_posterior=return_posterior, 
            weight_save_path=weight_save_path)


    def __call__(self, do_assimilation:bool, run_control:bool):

        # --- 1) Run Spin-Up (if not done already)
        print("\nRun SpinUp\n----------")
        if run_control:
            self.spinup(ensemble_run=False) # for control
        self.spinup() # for ensembles

        # --- 2) Loop over acssimilation schedule and perform data-assimilation tasks
        for idx, t_assim in enumerate(self.assim_schedule):
            
            print(f"\nAssimilation step {idx+1:2d}")
            self.assim_step(
                ztop_assim=self.ztop_assim_idx, 
                do_assimilation=do_assimilation,
                obs_file=self.obs_file, 
                return_posterior=False,
                )
            
            if idx == len(self.assim_schedule)-1:
                print("DONE with Assimilation Part")   
            else:
                t0 = (t_assim).strftime("%Y-%m-%d %H:%M")
                t1 = (t_assim + timedelta(seconds=self.ctx.dt_da)).strftime("%Y-%m-%d %H:%M\n")
                print(f"\nForward Step of ensemble from {t0} to {t1}\n")
                self.forward_step(t_init=t_assim)
                if run_control:
                    self.forward_step(t_init=t_assim, ensemble_run=False)

        # --- 3) Forecast step
        print(f"\nForecasting with lead time of: {self.ctx.dt_pred/3600:.1f} hours")
        t0_forecast = self.assim_schedule[-1]
        if run_control:
            self.forecast_step(tinit_forecast=t0_forecast, ensemble_run=False)
        self.forecast_step(tinit_forecast=t0_forecast, ensemble_run=True)






