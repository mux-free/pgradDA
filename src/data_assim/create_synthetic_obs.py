from pathlib import Path
import xarray as xr
import numpy as np
import sys
import yaml
sys.path.insert(1, '../../utils')
import utils_aspire
#######################################
## --- Create synthetic observations
#######################################


def synthobs_from_truth(
    truth_simdata_path: Path,
    file_synth_obs: Path,
    sigma_obs: float,
    obs_ix: list[int] = [64], #[32, 96, 64, 32, 96]
    obs_iy: list[int] = [64], #[32, 32, 64, 96, 96]
    ztop_obs: int = 2000,
    z_step: int = 2,
    ti_start: int = 2,
    t_step: int = 3,
    random_seed: int = 14,
    save_ds: bool = True
    ) -> None:
    
    # --- Control randomness
    rng = np.random.default_rng(seed=random_seed)

    # --- Load Data and unstagger
    ds = xr.open_dataset(f"{truth_simdata_path}/graspOutSimdata.meso.nc")
    ds = utils_aspire.unstagger_u_v(ds, xfoffset=0, yfoffset=0) # type: ignore
    
    # --- Select timing --> every 3rd step (30min) starting at half hours & Heights From index 0 to 10 (~500 meters)
    ds_obs = ds.isel(time=slice(ti_start, None, t_step)).sel(zf=slice(0,ztop_obs,z_step))
    ds_obs["time"] = ds_obs.time.dt.round("5min")
    ds_obs = ds_obs[["u","v"]]
    ds_obs["M"] = np.hypot(ds_obs["u"], ds_obs["v"])

    ## --- SetUp station dimension and select only these positions
    n_stations = len(obs_ix)
    station = np.arange(n_stations)
    ix = xr.DataArray(obs_ix,dims="station",coords={"station": station})
    iy = xr.DataArray(obs_iy,dims="station",coords={"station": station})
    ds_obs = ds_obs.isel(xf=ix, yf=iy)
    ds_obs = ds_obs.transpose("time","station","zf")

    ## --- Add observation errors
    ds_obs["u"] = ds_obs["u"] + rng.normal(loc=0.0, scale=sigma_obs, size=ds_obs["u"].shape)
    ds_obs["v"] = ds_obs["v"] + rng.normal(loc=0.0, scale=sigma_obs, size=ds_obs["v"].shape)

    ## --- Safe Dataset
    if save_ds:
        ds_obs.to_netcdf(file_synth_obs)

