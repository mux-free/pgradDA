import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta




# **********************************************************************
# ---- Load summaraized Data
def load_exp_data(
        base_dir:Path, 
        expname:str, 
        adjust_coords_for_rs: bool, 
        round_predobs_to:None|str,
        load_simdata_field:bool,
        load_predobs:bool,
        load_restart_fields:bool,
        load_ens_perts:bool, 
        yslice:slice=slice(None,None), 
        xslice:slice=slice(None,None),
        zslice:slice=slice(None,None),
        cutoff_time: datetime | timedelta | None = None,
        ):
    """ Function that loads summarized data """

    def load_ds(path, xslice=xslice, yslice=yslice, zslice=zslice):
        ds = xr.open_dataset(path)
        ds = ds.isel(xf=xslice, yf=yslice, zf=zslice)
        ds = crop_time(ds, cutoff_time)
        return ds


    def split_control(ds):
        ds_ctrl = ds.sel(ensemble=0)
        ds_ens  = ds.drop_sel(ensemble=0)
        return ds_ens, ds_ctrl

    
    # ======================= DATA LOADING =========================================
    datadict = {}

    # --- LOAD SimData (EnsembleMean and Control)
    if load_simdata_field:
        datadict["ds_da"]   = load_ds(f"{base_dir}/{expname}/sim_output/graspOutSimdata.ensemble.nc")
        datadict["ds_ctrl"] = load_ds(f"{base_dir}/{expname}/sim_output/graspOutSimdata.control.nc")

    # --- Load Restart files
    if load_restart_fields:
        datadict["ds_rsi"]      = load_ds(f"{base_dir}/{expname}/sim_output/graspRestartIn.ensemble.nc")
        datadict["ds_rso"]      = load_ds(f"{base_dir}/{expname}/sim_output/graspRestartOut.ensemble.nc")
        datadict["ds_rsi_ctrl"] = load_ds(f"{base_dir}/{expname}/sim_output/graspRestartIn.control.nc")
        datadict["ds_rso_ctrl"] = load_ds(f"{base_dir}/{expname}/sim_output/graspRestartOut.control.nc")
        

    # --- Load Predicted-Observations (Restart-Out and Simdata)
    if load_predobs:
        ds_po_sd  = xr.open_dataset(f"{base_dir}/{expname}/sim_output/graspOutSimdata.PredObs.nc")
        ds_po_rso = xr.open_dataset(f"{base_dir}/{expname}/sim_output/graspRestartOut.PredObs.nc")
        
        rsi_predobs = Path(f"{base_dir}/{expname}/sim_output/graspRestartIn.PredObs.nc")
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
        rsi_pert = Path(f"{base_dir}/{expname}/sim_output/graspRestartIn.EnsPerts.nc")
        rso_pert = Path(f"{base_dir}/{expname}/sim_output/graspRestartOut.EnsPerts.nc")
        
        if rso_pert.is_file():
            ds_rso_pert = load_ds(rso_pert)
        else:
            ds_rso_pert = None
            print(f"WARNING: graspRestartOut.EnsPerts.nc does not exist for: {expname}")
            print(f"{base_dir}/{expname}/sim_output/graspRestartOut.EnsPerts.nc\n")
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
        datadir = base_dir / expname / "sim_output"

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
        obs_file:Path,
        adjust_coords_for_rs: bool = False, 
        round_predobs_to: str|None = None,
        load_simdata_field: bool = True,
        load_predobs: bool = True,
        load_restart_fields: bool = True,
        load_ens_perts: bool = True,
        load_spectra: bool = True,
        y_slice: slice = slice(None,None),
        x_slice: slice = slice(None,None),
        z_slice: slice = slice(None,None),
        cutoff_time: datetime | None = None,
        ):
    

    datadict = dict()
    

    # --- Load Simulation Output
    for name in expnames:
        exp_data = load_exp_data(
            expname=name, 
            base_dir=base_dir, 
            xslice=x_slice, 
            yslice=y_slice, 
            zslice=z_slice,
            adjust_coords_for_rs=adjust_coords_for_rs, 
            round_predobs_to=round_predobs_to,
            load_simdata_field=load_simdata_field, 
            load_predobs=load_predobs,
            load_restart_fields=load_restart_fields, 
            load_ens_perts=load_ens_perts,
            cutoff_time=cutoff_time, 
        )
        datadict[name] = exp_data
    
    # --- Load Observation Output
    ds_obs = xr.open_dataset(obs_file)
    datadict["observations"] = ds_obs
    

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
    da_obs: xr.DataArray,
    Hxf: xr.DataArray,
    min_finite_members: int = 15,
    control:bool = False
):
    """
    Robust flattening when da_obs and Hxf may have different zf grids/lengths.

    For each station:
      1) Keep only zf levels where obs is finite
      2) Restrict those zf levels to ones available in Hxf
      3) Extract y and Hx and concatenate across stations

    Inputs at time t:
      da_obs dims: (station, zf)
      Hxf   dims: (station, zf, ensemble)

    Returns:
      y : (N,)
      x : (N,K)
      nr_NaN : int
    """
    # Ensure consistent dim order


    da_obs = da_obs.transpose("station", "zf")
    if control:
        Hxf = Hxf.transpose("station", "zf")
    else:
        Hxf = Hxf.transpose("station", "zf", "ensemble", ...)

    obs_name_list = []
    yobs_list = []
    Hxf_list = []
    zf_list = []
    nr_NaN = 0

    print("TEST1")

    for st in da_obs["station"].values:
        y_st = da_obs.sel(station=st)     # (zf,)
        x_st = Hxf.sel(station=st)       # (zf, ensemble)

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
        obs_names = [st] * y_keep.shape[0]

        obs_name_list.append(obs_names)
        yobs_list.append(y_keep)
        Hxf_list.append(x_keep)
        zf_list.append(z_keep)



    obs_stn = np.concatenate(obs_name_list, axis=0)
    y_out = np.concatenate(yobs_list, axis=0)
    x_out = np.concatenate(Hxf_list, axis=0)
    z_out = np.concatenate(zf_list, axis=0)

    return y_out, x_out, z_out, obs_stn, nr_NaN




def get_innovation_df(
        da_hxf:xr.DataArray, 
        da_obs:xr.DataArray, 
        control:bool=False,
        ASSIM_STEPS = 6,
        ASSIM_WINDOW = 1800, #seconds
        DT_OBS = 1800, # second
        ):
    """
    This function collects innovations over all timesteps, levels and obs-stations 
    in a pd.DataFrame.
    It does so for a specific case and ensemble-innit method.

    Inputs at time t:
      da_hxf xr.DataArray:   dims: (station, zf, ensemble, time)
      Hda_obs xr.DataArray:  dims: (station, zf, time)

    Returns:
      pd.DataFrame with columns: 
        [Hxf_m<ID> (all K-ens members) | y (observation) | zf (int) | time (datetime) | obs_station (name of station) | time_index10 (how many 10min steps after t0)]
    """

    obs_times = da_obs.time.values
    obs_names = da_obs.station.values

    nan_dict = {}

    OBS_PER_WINDOW = int(ASSIM_WINDOW/DT_OBS)

    rows = []
    for idx, t in enumerate(obs_times):
        print("here ", t)

        # --- Select predicted-obs (SimData) for slected-stations and at given time t
        Hxf_t = da_hxf.sel(station=obs_names).sel(method="nearest", tolerance=pd.Timedelta("5min"))
        da_obs_t = da_obs.sel(time=t, method="nearest", tolerance=pd.Timedelta("5min"))

        # --- Flatten Obs and PredObs at given time t across all stations and heights zf
        output = flatten_obs_pred_pair_stationwise(da_obs_t, Hxf_t, control=control)
        y_out, Hxf_out, zf_out, obs_names, nr_NaN = output


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
        df_t["obs_station"]  = obs_names
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

    print("\n\nIs this error here\n\n")
    df = pd.concat(rows, ignore_index=True)
    print("NO!")
    return df




# -------------------------------------------------------------------------------------------------------------------
# =====================================
# --- LOAD-DATA Summary
# =====================================


def load_case_data_dfinnov(
        base_dir:Path,
        expnames:dict,
        variable:str,
        obs_file: Path,
        x_slice=slice(None, None),
        y_slice=slice(None, None),
        z_slice=slice(None, None),
        round_predobs_to: str|None=None, #"10min",
        adjust_coords_for_rs:bool=False,
        load_control=True, 
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

    output = load_experiments(
        base_dir=base_dir, 
        expnames=expnames,
        obs_file=obs_file,
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
        datadict, ds_spectra_enspert, ds_spectra_increment = output
        spectra_dict = dict(enspert=ds_spectra_enspert,
                                    increment=ds_spectra_increment)
    else:
        datadict = output

   


    # ----------------------------
    # Compute Innovations Stats
    # ----------------------------
    innov_dict = {}
    if load_predobs and calc_innov:
        # --- Create dict with all innovations
        # --- Loop over all methods
        for ensgen in list(expnames):
           
            da_hxf = datadict[ensgen]["ds_po_sd_ens"][variable]
            da_obs = datadict["observations"][variable]
            
            # Compute innov dataframe
            df_ensgen = get_innovation_df(da_hxf=da_hxf, da_obs=da_obs)

            # Compute Ensemble Mean
            ens_mask = df_ensgen.columns.str.contains('Hx*')
            ens_avg = df_ensgen.loc[:,ens_mask].mean(axis=1)
            ens_spd = df_ensgen.loc[:,ens_mask].std(axis=1)
            df_ensgen["Hxf_bar"] = ens_avg
            df_ensgen["Hxf_std"] = ens_spd
            
            innov_dict[ensgen] = df_ensgen


        # --- Load Control case
        if load_control:
            ensgen=expnames[0]
            da_hxf = datadict[ensgen]["ds_po_sd_ctrl"][variable]
            da_obs = datadict["observations"][variable]
            df_ensgen = get_innovation_df(da_hxf=da_hxf, da_obs=da_obs, control=True)
            innov_dict["ctrl"] = df_ensgen

    if len(datadict) == 1:
        datadict = datadict[list(datadict)[0]]
    if len(innov_dict) == 1:
        innov_dict = innov_dict[list(innov_dict)[0]]


    if load_spectra:
        return datadict, innov_dict, spectra_dict
    else:
        return datadict, innov_dict

