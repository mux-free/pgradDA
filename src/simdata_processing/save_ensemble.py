import xarray as xr
import numpy as np
from pathlib import Path


# Add .../REFORM/src to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simdata_processing.simloader import load_data
import analysis_tools.spectra.spectral_plots as specplot

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "utils"))
from utils import save_ds # type: ignore


def predict_obs(
    ds_model: xr.Dataset,
    ds_obs_synth: xr.Dataset,
    variables=["u", "v"],
) -> xr.Dataset:
    """ Get ensemble states at synthetic metmast locations and heights --> predicted observations """
    ds_predobs = ds_model[list(variables)].interp(
        xf=ds_obs_synth["xf"],
        yf=ds_obs_synth["yf"],
        zf=ds_obs_synth["zf"],
        method="linear",
    )
    return ds_predobs
    



def save_summary_ds(
        exp_dir: Path,
        obs_file: Path,
        xy_range: tuple[int|None , int|None] = (None, None),
        z_range:  tuple[int|None , int|None] = (None, None),
        simdata: bool = True,
        save_mean_std_only: bool = False,
        restartin: bool = True,
        restartout: bool = True,
        predicted_obs: bool = True,
        ensemble_pert: bool = True,
        spectra_analysis: bool = True,
        spectra_enspert: bool = True,
        save_vars: list[str] = ["u","v","M"],
        ) -> None:
    """ 
    Given a directory with ensemble folders (e.g. /<HH>_m<ID>/), it load all members and calculates mean/std over them 
    for simdata (returns dims x,y,z,time) or restart files.

    """

    def split_ensemble_control(ds, ctrl_idx=0):
        ds_ctrl = ds.sel(ensemble=ctrl_idx)
        ds_ens  = ds.drop_sel(ensemble=ctrl_idx)
        return ds_ctrl, ds_ens


    def calc_mean_std(ds_ens):
        """ Helper function to calculate mean and standard deviation over ensemble members. Also returns cotnrol """
        # --- Define ensemble members and control
        ds_summary = xr.Dataset()
        # --- Loop over all variables of interest and create mean + avg
        for var in save_vars:
            da_mean = ds_ens[var].mean(dim="ensemble")
            da_std  = ds_ens[var].std(dim="ensemble")
            ds_summary[var+"_avg"] = da_mean
            ds_summary[var+"_std"] = da_std
        return ds_summary


    def calc_ens_perts(ds):
        ds_ens = ds.drop_sel(ensemble=0)
        ds_perts = xr.Dataset()
        for var in save_vars:
            da_mean = ds_ens[var].mean(dim="ensemble")
            ds_perts[var] = ds_ens[var] - da_mean
        return ds_perts


    def set_xf_yf(ds, xvlas, yvals):
        ds["xf"] = xvlas
        ds["yf"] = yvals
        return ds


    # ==================================================================================== 
    # LAOD DATA
    # ================
    # CREATE SAVE_PATH
    SAVE_DIR = Path(f"{exp_dir}/sim_output")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    ## --- LOAD DATA
    data_dict = load_data(
        exp_dir=exp_dir,
        load_rsi=restartin,
        load_rso=restartout,
        load_simdata=simdata,
        load_control=True,
        domain_subset = dict(xy_iboundary=xy_range, z_range=z_range),
    )
    

    # -- Calc offsets, s.t. they can be added to restart files
    xvals = data_dict["simdata"]["xf"].values
    yvals = data_dict["simdata"]["yf"].values

    ## --- SimData summary
    if simdata:
        ds_sd = data_dict["simdata"]
        ds_sd_ctrl, ds_sd_ens = split_ensemble_control(ds_sd, ctrl_idx=0)
        if save_mean_std_only:
            ds_sd_ens = calc_mean_std(ds_sd_ens)
        
        save_ds(ds_sd_ctrl, f"{SAVE_DIR}/graspOutSimdata.control.nc")
        save_ds(ds_sd_ens,  f"{SAVE_DIR}/graspOutSimdata.ensemble.nc")


    ## --- Calculate statistics
    if predicted_obs:
        ds_simdata = data_dict["simdata"]
        ds_rso = data_dict["restart_out"]
        ds_rsi = data_dict["restart_in"]
        
        # Adjust coordinates
        ds_rso = set_xf_yf(ds_rso, xvlas=xvals, yvals=yvals)
        ds_rsi = set_xf_yf(ds_rsi, xvlas=xvals, yvals=yvals)

        # Create PredObs
        ds_obs = xr.open_dataset(obs_file)
        ds_predobs_sd  = predict_obs(ds_simdata, ds_obs, save_vars)
        ds_predobs_rso = predict_obs(ds_rso, ds_obs, save_vars)
        ds_predobs_rsi = predict_obs(ds_rsi, ds_obs, save_vars)
        # Save Datasets
        save_ds(ds_predobs_sd,  f"{SAVE_DIR}/graspOutSimdata.PredObs.nc")
        save_ds(ds_predobs_rso, f"{SAVE_DIR}/graspRestartOut.PredObs.nc")
        save_ds(ds_predobs_rsi, f"{SAVE_DIR}/graspRestartIn.PredObs.nc")
    

    ## --- Save Ensemble Perturbations (ensemble - EnsMean)
    if ensemble_pert:
        ds_rso = data_dict["restart_out"]
        ds_rsi = data_dict["restart_in"]
        # Adjust coordinates
        ds_rso = set_xf_yf(ds_rso, xvlas=xvals, yvals=yvals)
        ds_rsi = set_xf_yf(ds_rsi, xvlas=xvals, yvals=yvals)

        # Get ensemble perturbation      (time, xf, yf, zf, ensemble)
        ds_perts_rso = calc_ens_perts(ds_rso)
        ds_perts_rsi = calc_ens_perts(ds_rsi)
        # Save
        save_ds(ds_perts_rso, f"{SAVE_DIR}/graspRestartOut.EnsPerts.nc")
        save_ds(ds_perts_rsi,  f"{SAVE_DIR}/graspRestartIn.EnsPerts.nc")



    if restartin:
        ds_rsi = data_dict["restart_in"]
        # --- Adjust coordinates
        ds_rsi = set_xf_yf(ds_rsi, xvlas=xvals, yvals=yvals)
        # -- Split control and ensemble members
        ds_rsi_ctrl, ds_rsi_ens = split_ensemble_control(ds_rsi, ctrl_idx=0)
        # --- Compute mean and standard dev from ensemble
        if save_mean_std_only:
            ds_rsi_ens = calc_mean_std(ds_rsi_ens)
        
        save_ds(ds_rsi_ctrl, f"{SAVE_DIR}/graspRestartIn.control.nc")
        save_ds(ds_rsi_ens,  f"{SAVE_DIR}/graspRestartIn.ensemble.nc")

    if restartout:
        ds_rso = data_dict["restart_out"]
        # --- Adjust coords
        ds_rso = set_xf_yf(ds_rso, xvlas=xvals, yvals=yvals)
        # -- Split control and ensemble members
        ds_rso_ctrl, ds_rso_ens = split_ensemble_control(ds_rso, ctrl_idx=0)
        # --- Compute mean and standard dev from ensemble
        if save_mean_std_only:
            ds_rso_ens = calc_mean_std(ds_rso_ens)
        save_ds(ds_rso_ctrl, f"{SAVE_DIR}/graspRestartOut.control.nc")
        save_ds(ds_rso_ens,  f"{SAVE_DIR}/graspRestartOut.ensemble.nc")



    # ====================================
    # ===  COMPUTE SPECTRA
    # ====================================
    def get_analysis_increment(ds_rsi, ds_rso, varname:str):
        da = (ds_rsi[varname+"_avg"] - ds_rso[varname+"_avg"])
        return da

    if ensemble_pert:
        if spectra_enspert:
            varname="M"
            zlev=200
            # Compute Spectra
            ds_perts_rso = xr.open_dataset(f"{SAVE_DIR}/graspRestartOut.EnsPerts.nc")
            da_spectra_enspert = specplot.compute_spectra(
                ds_perts_rso[varname], 
                varname=varname, 
                zlev=zlev, 
                skip_last_timestep=True)
            # Save spectra
            save_ds(da_spectra_enspert, f"{SAVE_DIR}/spectra_ensemble_perturbation.nc")

    if spectra_analysis:
        varname="M"
        zlev=200

        ds_rsi = xr.open_dataset(f"{SAVE_DIR}/graspRestartIn.ensemble.nc")
        ds_rso = xr.open_dataset(f"{SAVE_DIR}/graspRestartOut.ensemble.nc")

        da_increment = get_analysis_increment(ds_rsi, ds_rso, varname)
        da_spectra_analysis = specplot.compute_spectra(
            da_increment, 
            varname=varname, 
            zlev=zlev, 
            skip_last_timestep=True)

        # Save spectra
        save_ds(da_spectra_analysis, f"{SAVE_DIR}/spectra_analysis_increment.nc")




# def cleanup_simdata(basedir, exp_name, sim_folder, fname_pattern:list|tuple):
#     """ Remove Simdata files to save space """
    
#     if not isinstance(fname_pattern, (list,tuple)):
#         fname_pattern = [fname_pattern]

#     for pattern in fname_pattern:
#         exp_path = f"{basedir}/{exp_name}/{sim_folder}"
#         file_list = sorted(glob.glob(f"{exp_path}/20*/**/{pattern}*.nc", recursive=True))
#         for f in file_list:
#             os.remove(f)



