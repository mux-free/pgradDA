# letkf_interface.py
"""
letkf_interface.py
====================
This script is an interface between GRASP-ensemble and Tobias Finn's **pytassim** 
implementation of the Local Ensemble Transform Kalman Filter (LETKF).

Expected Inputs:
    - The prior ensemble [xr.Dataset] in the native grid (with dimensions [zf, yf, xf, ensemble (, time)])
    - Dictionary with Wind-LiDAR measurements (lidar_dict) where key is the station name and value is xr.Dataset
        NOTE: The xf/yf-coords of each LiDAR location must be contained in the xr.Dataset!

It returns:
    - the **analysis** ensemble as an xr.DataArray
    - (optionally) the model-equivalent observations (y_b), in order to inspect innovations.
"""

# Python Modules
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# Pytassim Modules
import sys
sys.path.append("/home/maxf/projects/REFORM/src/letkf_pyt/torch-assimilate")
from pytassim.localization import GaspariCohn 
from pytassim.interface.letkf import LETKF 




def stack_state_for_pytassim(
    ds: xr.Dataset,
    state_vars: Sequence[str],
) -> xr.DataArray:
    """
    Stacks prior-state (which is a 3d field for each variable with dims z,y,x)

    Parameters:
    -----------
    ds: xr.Dataset      ---> Prior background state (all state vars) with dims (x,y,z)
    state_vars: List    ---> List of variable names that are used as state variables
    
    Returns DataArray with dims (var_name, time, ensemble, grid),
    where grid is a stacked in the order: (zf, yf, xf).
    
    NOTE: Assumes ds already has a 'time' dimension (which is 1d; i.e. only one value)
    """
    missing = [v for v in state_vars if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables missing from state: {missing}")

    da = ds[state_vars].to_array(dim="var_name")         # (var_name, time?, ensemble, zf, yf, xf)
    da = da.stack(grid=("zf", "yf", "xf"))
    # Ensure dimension order
    order = [d for d in ("var_name", "time", "ensemble", "grid") if d in da.dims]
    da = da.transpose(*order)
    return da



# ----------------------------------------------------------------------------- #
# Observation operator                                                          #
# ----------------------------------------------------------------------------- #
class MetMasObsOp:
    """Interpolate model state to MetMast observation points."""

    def __call__(self, ds_obs, state):

        # Model -> (var_name, time, ensemble, zf, yf, xf)
        if isinstance(state, xr.Dataset):
            state = state.to_array("var_name")
        elif "grid" in state.dims:
            state = state.unstack("grid")

        # Retireve MultiIndex
        mi = ds_obs.indexes["obs_grid_1"]

        # One vectorized indexer per observation
        def obs_coord(name):
            return xr.DataArray(
                mi.get_level_values(name).to_numpy(),
                dims="obs_grid_1",
            )

        # --- Create Observation operator        
        Hx = (
            state
            .sel(var_name=obs_coord("variable"))
            .interp(
                xf=obs_coord("xf"),
                yf=obs_coord("yf"),
                zf=obs_coord("zf"),
            )
            .drop_vars(["var_name", "xf", "yf", "zf"], errors="ignore")
            .assign_coords(xr.Coordinates.from_pandas_multiindex(mi, "obs_grid_1"))
            .transpose("time", "ensemble", "obs_grid_1")
        )

        return Hx






# ----------------------------------------------------------------------------- #
# Build observation dataset for pytassim                                        #
# ----------------------------------------------------------------------------- #
def build_obs_ds(
    ds_obs: xr.Dataset,
    analysis_time: datetime,
    varnames=("u", "v"),
    R_std: float = 0.5,
    time_tolerance: str = "2min",
) -> xr.Dataset:
    """
    Build pytassim observation dataset for synthetic MetMast (vertical profiles) Observations

    Expected structure of ds_obs:
        - u(time, station, zf)
        - v(time, station, zf)
        where 'time' may either be a scalar coordinate or a proper dimension.
    with coordinates: xf(station), yf(station), zf(zf)

    Output
    ------
    xr.Dataset that contains two xr.DataArray:
    observations : (time, obs_grid_1)
    covariance   : (obs_grid_1)

    obs_grid_1 is a MultiIndex:
        (variable, station, xf, yf, zf)
    """

    target = pd.Timestamp(analysis_time)
    tol = pd.Timedelta(time_tolerance)

    
    # ========================================================================================================
    # --- Validation of observation Dataset
    # =======================================
    if "time" not in ds_obs.coords:
        raise ValueError("ds_obs must contain a 'time' coordinate.")
    
    obs_time = ds_obs["time"]

    # --- Check correct time-step is selected
    if obs_time.ndim == 0:
        # Single synthetic observation time stored as scalar
        actual_time = pd.Timestamp(obs_time.values)
        dt = abs(actual_time - target)
        if dt > tol:
            raise ValueError(f"Observation time {actual_time} differs from analysis time {target} by {dt}, exceeding tolerance {tol}")

        ds_t = ds_obs

    elif obs_time.ndim == 1 and "time" in ds_obs.dims:
        # Time series of synthetic observations
        ds_t = ds_obs.sel(
            time=np.datetime64(target),
            method="nearest",
            tolerance=tol,
        )
    else:
        raise ValueError(f"Unsupported time coordinate: dims={obs_time.dims}, shape={obs_time.shape}")

    # --- Ensure station dimension is present
    if "station" not in ds_t.dims:
        raise ValueError("Expected a 'station' dimension.\n")
    # ========================================================================================================



    # --------------------------------------------    
    # --- 1) Build variable + station + zf array
    # --------------------------------------------    
    obs_blocks = []
    for var in varnames:
        da = ds_t[var]

        # Remove scalar time coordinate/dimension if present
        if "time" in da.dims:
            da = da.squeeze("time", drop=True)
        # Expected is one vertical profile per station
        if set(da.dims) != {"station", "zf"}:
            raise ValueError(f"Expected {var!r} to have dimensions ('station', 'zf'), got {da.dims}")

        # -- Add variable dimension
        da = da.transpose("station", "zf")
        da = da.expand_dims(variable=[var])
        obs_blocks.append(da)

    # This results in: variable × station × zf
    da_obs = xr.concat(
        obs_blocks,
        dim="variable",
    )

    # --------------------------------------------    
    # --- 2) Stack into obs_grid_1
    # --------------------------------------------    
    da_obs = da_obs.stack(
        obs_grid_1=(
            "variable",
            "station",
            "zf",
        )
    )

    # Unpack the multi-index
    da_obs = da_obs.reset_index("obs_grid_1")

    # -------------------------------------------------------
    # --- 3) Get xf/yf coords of stations and assign to obs
    # -------------------------------------------------------
    station_index = da_obs["station"].values
    xf_per_obs = ds_t["xf"].values[station_index]
    yf_per_obs = ds_t["yf"].values[station_index]

    da_obs = da_obs.assign_coords(
        xf=("obs_grid_1", xf_per_obs),
        yf=("obs_grid_1", yf_per_obs),
    )

    # --- 4) Drop unavailable observations
    da_obs = da_obs.dropna(dim="obs_grid_1")

    # -------------------------------------------------------
    # 7. Construct meaningful observation MultiIndex
    # -------------------------------------------------------
    da_obs = da_obs.set_index(
        obs_grid_1=(
            "variable",
            "station",
            "xf",
            "yf",
            "zf",
        )
    )

    # --- 5) Add time-dimesnion and 
    da_obs = (
        da_obs
        .expand_dims(time=[np.datetime64(target)])
        .transpose("time", "obs_grid_1")
        .rename("observations")
    )

    # ----------------------------------------
    # --- 6) Add Observation-error covariance
    # ----------------------------------------
    n_obs = da_obs.sizes["obs_grid_1"]
    covariance = xr.DataArray(
        np.full(n_obs, R_std**2),
        coords={"obs_grid_1": da_obs["obs_grid_1"]},
        dims=("obs_grid_1",),
        name="covariance",
    )

    # ------------------------------------------
    # --- 7) Final pytassim Observation Dataset
    # ------------------------------------------
    obs_ds = xr.Dataset(
        {
            "observations": da_obs,
            "covariance": covariance,
        }
    )

    return obs_ds



# ----------------------------------------------------------------------------- #
# Localization distance                                                         #
# ----------------------------------------------------------------------------- #
def make_gc_distance(xf_vals: np.ndarray, yf_vals: np.ndarray, zf_vals: np.ndarray|None = None):
    xf_vals = np.asarray(xf_vals, dtype=float)
    yf_vals = np.asarray(yf_vals, dtype=float)
    
    if zf_vals is not None:
        zf_vals = np.asarray(zf_vals, dtype=float)

    def _dist(state_id, _obs_grid_ignored):
        
        # state_id is stacked as [time, zf, yf, xf]
        yf0 = float(state_id[-2])
        xf0 = float(state_id[-1])
        dx = xf0 - xf_vals
        dy = yf0 - yf_vals
        dh = np.hypot(dx, dy)  # shape (n_obs,)
        if zf_vals is not None:
            zf0 = float(state_id[-3])
            dv = np.abs(zf0 - zf_vals)                     
            return dh, dv

        else:
            return dh
    
    return _dist



# ----------------------------------------------------------------------------- #
# Run LETKF distance                                                            #
# ----------------------------------------------------------------------------- #
def run_letkf(
    ds_prior: xr.Dataset,
    ds_obs: xr.Dataset,
    loc_radius_m: int,
    vert_loc: int|None,
    inflation: float,
    R_std: float = 0.5,
    state_vars: Sequence[str] = ["u", "v", "Thl", "qt"],
    varnames: Sequence[str] = ("u", "v"),
    gpu: bool = True,
    weight_save_path:str|None=None,
    weight_grid: None | xr.DataArray = None,

) -> xr.Dataset:

    # --- 1) Build observation dataset (3D or 4D-flat)
    analysis_time = np.asarray(ds_prior["time"].values)[0]


    ds_obs = build_obs_ds(
        ds_obs,
        varnames=varnames, 
        R_std=R_std,
        time_tolerance="2min", 
        analysis_time=analysis_time,
    )

    # --- 2) Observation operator
    op = MetMasObsOp()
    ds_obs.attrs["operator"] = op
    ds_obs.obs.operator = op


    # --- 3) Localization
    if vert_loc is not None:
        dist_fn = make_gc_distance(ds_obs["xf"].values, ds_obs["yf"].values, ds_obs["zf"].values)
        loc = GaspariCohn((loc_radius_m, vert_loc), dist_func=dist_fn)
    else:
        dist_fn = make_gc_distance(ds_obs["xf"].values, ds_obs["yf"].values)
        loc = GaspariCohn((loc_radius_m,), dist_func=dist_fn)


    # --- 4) Pack state for pytassim (Here ds_prior is instantaneous restart field, as this one will be assimilated)
    state_stacked = stack_state_for_pytassim(ds_prior, state_vars=state_vars)

    # --- 5) Assimilation update
    letkf = LETKF(
        localization=loc, 
        inf_factor=inflation, 
        gpu=gpu,
        weight_save_path=weight_save_path
        )
    
    analysis_stacked = letkf.assimilate(state_stacked, (ds_obs,))
    an = analysis_stacked.unstack("grid").to_dataset(dim="var_name")
    for var in varnames:
        if var in ds_prior:
            an[var].attrs.update(ds_prior[var].attrs)

    # WEIRD HACK TO AVOID: "RuntimeError: lazy wrapper should be called at most once"
    try:
        an = an.compute()
    except RuntimeError: 
        an = an.compute()

    return an

