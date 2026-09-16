from pathlib import Path
import xarray as xr
import numpy as np
import sys
sys.path.insert(1, '../../utils')
import utils_aspire
#######################################
## --- Create synthetic observations
#######################################

dir_truth = Path("/home/maxf/projects/REFORM/pgradDA/grasp/ecmwf_ensemble")
OBS_FILE = Path(f"{dir_truth}/synthetic_observation.nc")

SIGMA_U = 0.5
ZTOP_OBS = 25
OBS_DENSITY = 2
rng = np.random.default_rng(seed=14)


# --- Load Data and unstagger
ds = xr.open_dataset(f"{dir_truth}/run/2026/02/10/00/graspOutSimdata.meso.nc")
ds = utils_aspire.unstagger_u_v(ds, xfoffset=0, yfoffset=0) # type: ignore
# --- Select timing --> every 3rd step (30min) starting at half hours & Heights From index 0 to 10 (~500 meters)
ds_obs = ds.isel(time=slice(2, None, 3), zf=slice(0,ZTOP_OBS,OBS_DENSITY))
ds_obs["time"] = ds_obs.time.dt.round("5min")
ds_obs = ds_obs[["u","v"]]

## --- Observation in indices
obs_ix = [64] #[32, 96, 64, 32, 96]
obs_iy = [64] #[32, 32, 64, 96, 96]

## --- SetUp station dimension and select only these positions
n_stations = len(obs_ix)
station = np.arange(n_stations)
ix = xr.DataArray(obs_ix,dims="station",coords={"station": station})
iy = xr.DataArray(obs_iy,dims="station",coords={"station": station})
ds_obs = ds_obs.isel(xf=ix, yf=iy)

ds_obs = ds_obs.transpose("time","station","zf")

## --- Add observation errors
ds_obs["u"] = ds_obs["u"] + rng.normal(loc=0.0,scale=SIGMA_U,size=ds_obs["u"].shape)
ds_obs["v"] = ds_obs["v"] + rng.normal(loc=0.0,scale=SIGMA_U,size=ds_obs["v"].shape)

## --- Safe Dataset
ds_obs.to_netcdf(OBS_FILE)