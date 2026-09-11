import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta






# ---- DATA LOADING FUNCTIONS
# **********************************************************************

def load_exp_data(
        expname:str, 
        base_dir:Path, 
        xslice:slice,
        yslice:slice, 
        zslice:slice,
        adjust_coords_for_rs:bool, 
        round_predobs_to:None|str,
        load_simdata_field:bool,
        load_predobs:bool,
        load_restart_fields:bool,
        load_ens_perts:bool, 
        cutoff_time: datetime | timedelta | None,
        ):
    

    def load_ds(path, xslice=xslice, yslice=yslice, zslice=zslice):
        ds = xr.open_dataset(path)
        ds = ds.isel(xf=xslice, yf=yslice, zf=zslice)
        ds = crop_time(ds, cutoff_time)
        return ds


    def split_control(ds):
        ds_ctrl = ds.sel(member=0)
        ds_ens  = ds.drop_sel(member=0)
        return ds_ens, ds_ctrl

    
    # ======================= DATA LOADING =========================================
    datadict = {}

    # --- LOAD SimData (EnsembleMean and Control)
    if load_simdata_field:
        datadict["ds_da"]   = load_ds(f"{base_dir}/{expname}/data_assim/run/output/graspOutSimdata.ensemble.nc")
        datadict["ds_ctrl"] = load_ds(f"{base_dir}/{expname}/data_assim/run/output/graspOutSimdata.control.nc")

    # --- Load Restart files
    if load_restart_fields:
        datadict["ds_rsi"]      = load_ds(f"{base_dir}/{expname}/data_assim/run/output/graspRestartIn.ensemble.nc")
        datadict["ds_rso"]      = load_ds(f"{base_dir}/{expname}/data_assim/run/output/graspRestartOut.ensemble.nc")
        datadict["ds_rsi_ctrl"] = load_ds(f"{base_dir}/{expname}/data_assim/run/output/graspRestartIn.control.nc")
        datadict["ds_rso_ctrl"] = load_ds(f"{base_dir}/{expname}/data_assim/run/output/graspRestartOut.control.nc")
        

    # --- Load Predicted-Observations (Restart-Out and Simdata)
    if load_predobs:
        ds_po_sd  = xr.open_dataset(f"{base_dir}/{expname}/data_assim/run/output/graspOutSimdata.PredObs.nc")
        ds_po_rso = xr.open_dataset(f"{base_dir}/{expname}/data_assim/run/output/graspRestartOut.PredObs.nc")
        
        rsi_predobs = Path(f"{base_dir}/{expname}/data_assim/run/output/graspRestartIn.PredObs.nc")
        if rsi_predobs.is_file():
            ds_po_rsi = xr.open_dataset(rsi_predobs)
        else:
            ds_po_rsi = None
        
        # fast crop BEFORE resample
        ds_po_sd  = crop_time(ds_po_sd, cutoff_time)
        ds_po_rso = crop_time(ds_po_rso, cutoff_time)
        if ds_po_rsi is not None:
            ds_po_rsi = crop_time(ds_po_rsi, cutoff_time)

                
        # Round time if desired
        if round_predobs_to is not None:
            if not isinstance(round_predobs_to, str): raise ValueError(f"round_predobs_to must be string indicating resample interval (e.g. '10min')")
            ds_po_sd  = ds_po_sd.resample(time=round_predobs_to, label="right", closed="right").mean() 
        ds_po_sd_ens, ds_po_sd_ctrl  = split_control(ds_po_sd)
        ds_po_rso_ens,ds_po_rso_ctrl = split_control(ds_po_rso)
        if ds_po_rsi is not None:
            ds_po_rsi_ens,ds_po_rsi_ctrl = split_control(ds_po_rsi)
        else:
            ds_po_rsi_ens,ds_po_rsi_ctrl = None, None

        datadict["ds_po_sd_ens"]  = ds_po_sd_ens
        datadict["ds_po_sd_ctrl"] = ds_po_sd_ctrl
        datadict["ds_po_rso_ens"] = ds_po_rso_ens
        datadict["ds_po_rso_ctrl"]= ds_po_rso_ctrl
        datadict["ds_po_rsi_ens"] = ds_po_rsi_ens
        datadict["ds_po_rsi_ctrl"]= ds_po_rsi_ctrl


    # --- Load EnsPerturbations
    if load_ens_perts:
        rsi_pert = Path(f"{base_dir}/{expname}/data_assim/run/output/graspRestartIn.EnsPerts.nc")
        rso_pert = Path(f"{base_dir}/{expname}/data_assim/run/output/graspRestartOut.EnsPerts.nc")
        
        if rso_pert.is_file():
            ds_rso_pert = load_ds(rso_pert)
        else:
            ds_rso_pert = None
            print(f"WARNING: graspRestartOut.EnsPerts.nc does not exist for: {expname}")
            print(f"{base_dir}/{expname}/data_assim/run/output/graspRestartOut.EnsPerts.nc\n")
        if rsi_pert.is_file():
            ds_rsi_pert = load_ds(rsi_pert)
        else:
            ds_rsi_pert = None
            print(f"WARNING: graspRestartIn.EnsPerts.nc does not exist for: {expname}")
            
        datadict["ds_rso_pert"] = ds_rso_pert
        datadict["ds_rsi_pert"] = ds_rsi_pert

    if adjust_coords_for_rs:
        xvals, yvals = datadict["ds_da"]["xf"].values, datadict["ds_da"]["yf"].values
        datadict["ds_rsi"]["xf"], datadict["ds_rsi"]["yf"] = xvals, yvals
        datadict["ds_rso"]["xf"], datadict["ds_rso"]["yf"] = xvals, yvals
        datadict["ds_rsi_ctrl"]["xf"], datadict["ds_rsi_ctrl"]["yf"] = xvals, yvals
        datadict["ds_rso_ctrl"]["xf"], datadict["ds_rso_ctrl"]["yf"] = xvals, yvals
        
        if load_ens_perts and ds_rso_pert is not None:
            datadict["ds_rso_pert"]["xf"], datadict["ds_rso_pert"]["yf"] = xvals, yvals
        if load_ens_perts and ds_rsi_pert is not None:
            datadict["ds_rsi_pert"]["xf"], datadict["ds_rsi_pert"]["yf"] = xvals, yvals


    return datadict



def crop_time(ds: xr.Dataset, cutoff: datetime | timedelta | None) -> xr.Dataset:
    if cutoff is None:
        return ds
    if ("time" not in ds.dims) and ("time" not in ds.coords):
        return ds

    if isinstance(cutoff, timedelta):
        t0 = ds.indexes["time"][0] if "time" in ds.indexes else ds["time"].to_index()[0]
        t0 = pd.Timestamp(t0)
        cutoff_ts = t0 + cutoff - pd.Timedelta(minutes=5)
    else:
        cutoff_ts = pd.Timestamp(cutoff)
    return ds.sel(time=slice(None, cutoff_ts))




def load_spectra_dataset(expnames: dict, base_dir: Path, varname:str="M"):
    ds_enspert   = xr.Dataset()
    ds_increment = xr.Dataset()

    for key, expname in expnames.items():
        datadir = base_dir / expname / "data_assim" / "run" / "output"

        # --- Load ensemble-perturbation spectra
        file_enspert = datadir / "spectra_ensemble_perturbation.nc"
        if file_enspert.is_file():
            spectra_enspert = xr.open_dataset(file_enspert)
            ds_enspert[key] = spectra_enspert[varname]
        else:
            print(f"Warning: ensemble-perturbation spectra do not exist for file:\n{file_enspert}")


        # --- Load analysis increment spectra
        file_incr = datadir / "spectra_analysis_increment.nc"
        if file_incr.is_file():
            spectra_incr = xr.open_dataset(file_incr)
            ds_increment[key] = spectra_incr[varname]
        else:
            print(f"Warning: analysis-increment spectra do not exist for file:\n{file_incr}")

    return ds_enspert, ds_increment


def load_experiments(
        base_dir:Path,
        expnames:dict, 
        x_slice:slice,
        y_slice:slice,
        z_slice:slice,
        adjust_coords_for_rs:bool=False, 
        round_predobs_to=None,
        load_simdata_field=True,
        load_predobs=True,
        load_restart_fields=True,
        load_ens_perts=True,
        load_spectra=True,
        cutoff_time:datetime|None=None
        ):
    

    datadict = dict()
    

    # --- Load Simulation Output
    for key, name in expnames.items():
        exp_data = load_exp_data(
            name, base_dir, x_slice, y_slice, z_slice,
            adjust_coords_for_rs, round_predobs_to,
            load_simdata_field, load_predobs,
            load_restart_fields, load_ens_perts,
            cutoff_time, 
        )
        datadict[key] = exp_data
    
    # --- Load Observation Output
    random_expname = next(iter(expnames.values())) # NOTE: Observation are the same for all ens-innit methods!!
    obs_path = f"{base_dir}/{random_expname}/obs_data"
    ds_assim = xr.open_dataset(f"{obs_path}/assimilated/ds_obs_assim.nc")
    ds_valid = xr.open_dataset(f"{obs_path}/validation/ds_os_valid.nc")
    datadict["obs_assim"] = ds_assim
    datadict["obs_valid"] = ds_valid
    

    if load_spectra:
        ds_spectra_enspert, ds_spectra_increment = load_spectra_dataset(expnames=expnames, base_dir=base_dir, varname="M")
        return datadict, ds_spectra_enspert, ds_spectra_increment
    
    else:
        return datadict


def set_zlevels_predobs(
        ds_predobs,
        lidar_dict
        ):
    # Select a reference z_obs-list
    zmin = np.inf
    zmax = 0
    for key, ds_lidar in lidar_dict.items():
        z = ds_lidar["zf"].dropna(dim="zf", how="all").values
        # Adjust ma/min height
        zmax = np.max(z) if np.max(z) > zmax else zmax
        zmin = np.min(z) if np.min(z) < zmin else zmin
    
    return ds_predobs.sel(zf=slice(zmin,zmax))








# -------------------------------------------------------------------------------------------------------------------
# ============================================
# --- Innovation DF -- MULTI-CASE HANDLING ---
# ============================================

def flatten_obs_pred_pair_stationwise(
    y_obs: xr.DataArray,
    Hxf: xr.DataArray,
    min_finite_members: int = 20,
    control:bool = False
):
    """
    Robust flattening when y_obs and Hxf may have different zf grids/lengths.

    For each station:
      1) Keep only zf levels where obs is finite
      2) Restrict those zf levels to ones available in Hxf
      3) Extract y and Hx and concatenate across stations

    Inputs at time t:
      y_obs dims: (station, zf)
      Hxf   dims: (station, zf, member)

    Returns:
      y : (N,)
      x : (N,K)
      nr_NaN : int
    """
    # Ensure consistent dim order
    y_obs = y_obs.transpose("station", "zf")
    if control:
        Hxf = Hxf.transpose("station", "zf")
    else:
        Hxf = Hxf.transpose("station", "zf", "member")

    lidar_name = []
    yobs_list = []
    Hxf_list = []
    zf_list = []
    nr_NaN = 0

    for st in y_obs["station"].values:
        y_st = y_obs.sel(station=st)     # (zf,)
        x_st = Hxf.sel(station=st)       # (zf, member)

        # 1) keep only finite obs zf levels
        ok = np.isfinite(y_st.values)
        nr_NaN += int((~ok).sum())

        if not ok.any():
            print(f"\nWARNING: All obs. NaN in station: {st}\n")
            continue

        # zf values where obs is finite
        z_keep = y_st["zf"].values[ok]

        # 2) Ensure that zf-vals are the same in both arrays
        z_keep = np.intersect1d(z_keep, x_st["zf"].values)

        # 3) select and extract values
        y_keep = y_st.sel(zf=z_keep).values              # (n_i,)
        x_keep = x_st.sel(zf=z_keep).values              # (n_i, K)
        

        if not control:
            finite_members = np.isfinite(x_keep).sum(axis=1)
            keep2 = finite_members >= min_finite_members
            y_keep = y_keep[keep2]
            x_keep = x_keep[keep2, :]

        if y_keep.size == 0:
            print(f"\nWARNING: No more Obs. left for station: {st}\n")
            continue

        # Hard fail if NaNs remain in predicted obs
        if not np.isfinite(x_keep).all():
            print()
            print(np.isfinite(x_keep))
            print(x_keep)
            print()
            raise ValueError(f"NaNs in predicted observations after filtering at station={st}")

        # Convert lidat string to a list of length n_i with, where all elements are same obs-name
        lidars = [st] * y_keep.shape[0]

        lidar_name.append(lidars)
        yobs_list.append(y_keep)
        Hxf_list.append(x_keep)
        zf_list.append(z_keep)



    obs_stn = np.concatenate(lidar_name, axis=0)
    y_out = np.concatenate(yobs_list, axis=0)
    x_out = np.concatenate(Hxf_list, axis=0)
    z_out = np.concatenate(zf_list, axis=0)

    return y_out, x_out, z_out, obs_stn, nr_NaN




def get_innovation_df(da_hxf:xr.DataArray, da_obs:xr.DataArray, control:bool=False):
    """
    This function collects innovations over all timesteps, levels and obs-stations 
    in a pd.DataFrame.
    It does so for a specific case and ensemble-innit method.

    Inputs at time t:
      da_hxf xr.DataArray:   dims: (station, zf, member, time)
      Hda_obs xr.DataArray:  dims: (station, zf, time)

    Returns:
      pd.DataFrame with columns: 
        [Hxf_m<ID> (all K-ens members) | y (observation) | zf (int) | time (datetime) | obs_station (name of station) | time_index10 (how many 10min steps after t0)]
    """

    times = da_hxf.time.values
    lidars = da_obs.station.values

    nan_dict = {}

    ASSIM_STEPS = 6
    ASSIM_WINDOW = 1800 #seconds
    DT_OBS = 600 # second

    OBS_PER_WINDOW = int(ASSIM_WINDOW/DT_OBS)

    rows = []
    for idx, t in enumerate(times):

        # --- Select predicted-obs (SimData) for slected-stations and at given time t
        Hxf_t = da_hxf.sel(time=t, station=lidars)
        obs_t = da_obs.sel(time=t)

        # --- Flatten Obs and PredObs at given time t across all stations and heights zf
        output = flatten_obs_pred_pair_stationwise(obs_t, Hxf_t, control=control)
        y_out, Hxf_out, zf_out, lidar_name, nr_NaN = output


        # --- Create pd.DataFrame with flattened yo and H(x)
        if not control:
            ens_size = Hxf_out.shape[1]
            Hxf_cols = [f"Hx_m{k:02d}" for k in range(1,ens_size+1)]
        else:
            Hxf_cols = ["Hxf_bar"]

        df_t = pd.DataFrame(Hxf_out, columns=Hxf_cols)
        df_t["y"]            = y_out
        df_t["zf"]           = zf_out
        df_t["time"]         = pd.Timestamp(t.values) if hasattr(t, "values") else t
        df_t["obs_station"]  = lidar_name
        df_t["time_index10"] = idx+1
        df_t["assim_step"]   = np.where(df_t["time_index10"].between(1, OBS_PER_WINDOW*ASSIM_STEPS), 
                                        ((df_t["time_index10"] - 1) // OBS_PER_WINDOW + 1),
                                        np.nan, 
                                        )

        # -- Make columne name to none
        df_t["assim_step"] = df_t["assim_step"].astype("Int64")
        df_t.loc[df_t["assim_step"].isna(), "assim_step"] = None

        rows.append(df_t)
        
        # Keep track of NaN values
        nan_dict[t] = nr_NaN
    return pd.concat(rows, ignore_index=True)




# -------------------------------------------------------------------------------------------------------------------
# =====================================
# --- LOAD-DATA Summary
# =====================================


def load_case_data_dfinnov(
        date_list, 
        expnames:dict,
        variable:str,
        obs_type:str|None,
        x_slice=slice(64, 192),
        y_slice=slice(76, 204),
        z_slice=slice(0,23),
        round_predobs_to:str|None=None, #"10min",
        adjust_coords_for_rs:bool=False,
        load_control=True, 
        control_method="rn",
        cutoff_time: datetime | timedelta | None = None,
        load_simdata_field:bool=True,
        load_predobs:bool=True,
        load_restart_fields:bool=True,
        load_ens_perts:bool=True,
        load_spectra:bool = True,
        calc_innov:bool=True
        ):


    # ------------------
    # Load Ouput Fields
    # ------------------
    datadict = {}

    spectra_dict = {}

    for dtime in date_list:
        # --- TIMESTAMP
        tstamp = datetime.strptime(dtime, "%Y%m%d_%H")

        time_folder = tstamp.strftime('%Y%m%d_%H')
        base_dir = Path(f"/home/maxf/projects/REFORM/grasp/data_assimilation/ensemble4da/experiments/{time_folder}")


        output = load_experiments(
            base_dir=base_dir, 
            expnames=expnames,
            x_slice=x_slice,
            y_slice=y_slice,
            z_slice=z_slice,
            adjust_coords_for_rs = adjust_coords_for_rs, 
            round_predobs_to     = round_predobs_to,
            load_simdata_field   = load_simdata_field,
            load_predobs         = load_predobs,
            load_restart_fields  = load_restart_fields,
            load_ens_perts       = load_ens_perts,
            load_spectra         = load_spectra,
            cutoff_time          = cutoff_time
            )

        if load_spectra:
            exp_dict, ds_spectra_enspert, ds_spectra_increment = output
            spectra_dict[dtime] = dict(enspert=ds_spectra_enspert,
                                       increment=ds_spectra_increment)
        else:
            exp_dict = output

        datadict[dtime] = exp_dict


    # ----------------------------
    # Compute Innovations Stats
    # ----------------------------
    innov_dict = {}
    if load_predobs and calc_innov:
        if obs_type is None:
            raise ValueError("To calculate Innovations stats, specify if validation of assimilation obs. should be used with kword: 'obs_type'!!!")
        # --- Create dict with all innovations
        # --- Loop over all methods
        for ensgen in list(expnames):
            df_list = []
            # --- Loop over all cases
            for date, casedict in datadict.items():
                # Set Data
                da_hxf = casedict[ensgen]["ds_po_sd_ens"][variable]
                da_obs = casedict[obs_type][variable]
                
                # Compute innov dataframe
                innov_df = get_innovation_df(da_hxf=da_hxf, da_obs=da_obs)

                # Compute Ensemble Mean
                ens_mask = innov_df.columns.str.contains('Hx*')
                ens_avg = innov_df.loc[:,ens_mask].mean(axis=1)
                ens_spd = innov_df.loc[:,ens_mask].std(axis=1)
                innov_df["Hxf_bar"] = ens_avg
                innov_df["Hxf_std"] = ens_spd
                df_list.append(innov_df)
            # Concat to one large dataset
            df_ensgen = pd.concat(df_list, ignore_index=True)
            innov_dict[ensgen] = df_ensgen

        # --- Load Control case
        if load_control:
            ensgen=control_method
            df_list = []
            for date, casedict in datadict.items():
                # Set Data
                da_hxf = casedict[ensgen]["ds_po_sd_ctrl"][variable]
                da_obs = casedict[obs_type][variable]
                # Compute innov dataframe
                innov_df = get_innovation_df(da_hxf=da_hxf, da_obs=da_obs, control=True)

                df_list.append(innov_df)
            df_ensgen = pd.concat(df_list, ignore_index=True)
            innov_dict["ctrl"] = df_ensgen



    if len(datadict) == 1:
        datadict = datadict[list(datadict)[0]]
    if len(innov_dict) == 1:
        innov_dict = innov_dict[list(innov_dict)[0]]


    if load_spectra:
        return datadict, innov_dict, spectra_dict
    else:
        return datadict, innov_dict

