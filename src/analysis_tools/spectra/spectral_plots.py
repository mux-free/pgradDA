import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import spectral





def _get_square_domain_length(da: xr.DataArray) -> float:
    dy = da["yf"][1] - da["yf"][0]
    dx = da["xf"][1] - da["xf"][0]
    Ly = da["yf"][-1] - da["yf"][0] + dy
    Lx = da["xf"][-1] - da["xf"][0] + dx

    if not np.isclose(Lx, Ly):
        raise ValueError(f"Spectra must be computed on a square domain. Got Lx={Lx}, Ly={Ly}")

    return float(Lx)


def _compute_mean_spectrum_at_time(da_t: xr.DataArray, return_each_member=False):
    """
    Compute spectrum for one time slice.

    If a 'member' dimension exists:
      - return_each_member=False: return member-mean spectrum, shape (k,)
      - return_each_member=True:  return all member spectra, shape (member, k)

    If no 'member' dimension is present:
      - always return shape (k,)
    """
    if "member" in da_t.dims:
        other_dims = set(da_t.dims) - {"member", "xf", "yf"}
        if other_dims:
            raise ValueError(
                f"Unexpected extra dimensions besides ('member', 'xf', 'yf'): {da_t.dims}"
            )

        bins_list = []
        kvals_ref = None

        for mem in da_t["member"].values:
            da_mem = da_t.sel(member=mem)
            bins_f, kvals_f = spectral.S_kh(da_mem)

            if kvals_ref is None:
                kvals_ref = kvals_f
            elif not np.allclose(kvals_f, kvals_ref):
                raise ValueError("k-values differ between ensemble members, cannot combine safely.")

            bins_list.append(bins_f)

        bins_arr = np.stack(bins_list, axis=0)   # Shape: (member, k)

        if return_each_member:
            return bins_arr, kvals_ref
        else:
            return bins_arr.mean(axis=0), kvals_ref # Shape: (k,)

    # --- Deterministic case
    other_dims = set(da_t.dims) - {"xf", "yf"}
    if other_dims:
        raise ValueError(f"Unexpected extra dimensions for 2D field: {da_t.dims}")

    bins_f, kvals_f = spectral.S_kh(da_t)  # Shape: (k,)
    return bins_f, kvals_f


def compute_spectra(
        da, 
        varname:str, 
        zlev:int|None=None, 
        skip_last_timestep:bool=False,
        return_each_member=False
        ):
    
    # Select vertical height
    if zlev is not None:
        da = da.sel(zf=zlev,method="nearest")

    available_times = da["time"].values
    if skip_last_timestep:
        requested_times = available_times[:-1]
    else:
        requested_times = available_times

    Lx = _get_square_domain_length(da)

    # --- Compute spectra per timestep
    spec_list = []
    actual_times = []
    kvals_global = None

    for t in requested_times:
        bins_f, kvals_f = _compute_mean_spectrum_at_time(
            da.sel(time=t),
            return_each_member=return_each_member
        )

        if kvals_global is None:
            kvals_global = kvals_f
        elif not np.allclose(kvals_f, kvals_global):
            raise ValueError(f"k-values differ across times; cannot stack safely at time {t}.")

        spec_list.append(bins_f)
        actual_times.append(np.datetime64(t))

    wavelength_global = (Lx / kvals_global) / 1000.0
    spectra_arr = np.stack(spec_list, axis=0)

    if return_each_member:
        dims = ("time", "member", "k")
        coords={"time": np.array(actual_times), "member":da["member"].values, "k": kvals_global}
        attrs={"long_name": "spectrum","zlev": zlev}
    else:
        dims = ("time", "k")
        coords={"time": np.array(actual_times), "k": kvals_global}
        attrs={"long_name": "spectrum","zlev": zlev, "ensemble_mean": str("member" in da.dims)}


    da_out = xr.DataArray(
                spectra_arr,
                dims=dims,
                coords=coords,
                attrs=attrs
                )
    
    ds_out = xr.Dataset(data_vars = {varname: da_out})

    ds_out = ds_out.assign_coords(
        wavelength_km=("k", wavelength_global)
    )

    ds_out.attrs.update(
        {
            "varname": varname,
            "zlev": zlev,
            "domain_length_m": Lx,
        }
    )

    return ds_out




def plot_spectra_dataset(
    ds_spectra: xr.Dataset,
    premultiplied:bool =True,
    fig_savedir: Path | None = None,
    filename: str | None = None,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.0, 3.0),
    return_fig_axes:bool=False,
    pert_type:str="",
    time_axis="absolute",
    ds_spectra_spread:xr.Dataset|None=None,
    color_exp_dict:dict={}
    ):
    
    """
    Plot spectra from a dataset created by compute_spectra_dataset().

    Parameters
    ----------
    ds_spectra : xr.Dataset
        Dataset returned by compute_spectra_dataset.
    fig_savedir : Path or None
        Directory where figure is saved.
    """

    # Get times when spectra are plotted
    # minutes_da = ds_spectra.time / np.timedelta64(1, "m")
    minutes_da = (
        ds_spectra.time - ds_spectra.time.isel(time=0)) / np.timedelta64(1, "m")
    ds_spectra["time"] = minutes_da


    if "date" in ds_spectra.coords:
        ds_spectra_avg = ds_spectra.mean(dim="date")
        ds_spectra_std = ds_spectra.std(dim="date")
    else:
        ds_spectra_avg = ds_spectra
        ds_spectra_std = None


    nplots = len(minutes_da)
    ncols = min(ncols, nplots)
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=False,
        sharey=False,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        dpi=300,
        squeeze=False,
    )
    axs = axes.flatten()

    wavelength_km = ds_spectra["wavelength_km"].values

    methods = list(ds_spectra.data_vars)

    for i, time in enumerate(minutes_da):
        ax = axs[i]

        for method in methods:
            da_avg = ds_spectra_avg[method].sel(time=time)
            if ds_spectra_std is not None:
                da_std = ds_spectra_std[method].sel(time=time)
            else:
                ds_spectra_spread= None


            color = color_exp_dict.get(method, "k")
            
            # --- Get Spectrum Values and premultiply (if desired)
            k = ds_spectra["k"].values if premultiplied else 1
            y = k * da_avg.values
            
            if ds_spectra_spread is not None:
                y_std = k * da_std.values
            else:
                y_std = 0

            # --- Plot
            if ds_spectra_spread is not None:
                ax.loglog(wavelength_km, y, label=method, color=color)                
                ax.loglog(wavelength_km, y - y_std, label=method, color=color, linestyle="--")
                ax.loglog(wavelength_km, y + y_std, label=method, color=color, linestyle="--")


            else:
                ax.loglog(wavelength_km, y, label=method, color=color)


        ax.set_xlim(np.ceil(wavelength_km.max() / 1000) * 1000, 1)
        ax.grid(alpha=0.25)

        if i == 0:
            ax.legend(loc="lower left")
        if i % ncols == 0:
            ax.set_ylabel("k * S(k)" if premultiplied else "S(k)")
        if i >= nplots - ncols:
            ax.set_xlabel("Wavelength [km]")

        if time_axis == "absolute":
            tstamp = pd.Timestamp(time).strftime("%Y-%m-%d %H:%M")
        else:
            tstamp = f"DA iter {(time):2}"
            
        ax.set_title(tstamp)

    for j in range(nplots, len(axs)):
        axs[j].axis("off")

    
    
    fig.suptitle(f"Spectra -- {pert_type.replace('_', ' ')}")

    fig.tight_layout()

    if fig_savedir is not None:
        fig_savedir = Path(fig_savedir)
        fig_savedir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"spectra_{pert_type}.png"

        fig.savefig(fig_savedir / filename, dpi=200, bbox_inches="tight")

    if return_fig_axes:
        return fig, axes









def plot_avg_spectra_ds(
    spectra_dict: xr.Dataset,
    premultiplied:bool =True,
    methods=["wn","rn","nmc","era","bv"],
    fig_savedir: Path | None = None,
    filename: str | None = None,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (4.0, 3.0),
    return_fig_axes:bool=False,
    plot_spread:bool=True,
    time_collapse:str|int="mean",
    color_exp_dict:dict={}, 
    max_wavenumber:int=64
    ):
    
    """
    Plot spectra from a dataset created by compute_spectra_dataset().

    Parameters
    ----------
    ds_spectra : xr.Dataset
        Dataset returned by compute_spectra_dataset.
    fig_savedir : Path or None
        Directory where figure is saved.
    """


    def pepare_spec_data(method, data_type, time_collapse):

        dates = list(spectra_dict)
        wavelength_km = None
        
        data_list = []
        wavelength_km = None

        for date in dates:
            
            if date == "20250724_02": #or date=="20250910_05" or date=="20251003_06" or date=="20251201_15":
                continue
            
            try: 
                data = spectra_dict[date][data_type][method]
            except KeyError as e:
                print(date)
                raise e

            # -- Select data
            if time_collapse=="mean":
                data = data.mean(dim="time")
            elif isinstance(time_collapse, int):
                data = data.isel(time=time_collapse)
            else:
                tlen = data["time"].values.shape[0]
                raise ValueError(f"Keyword 'time_collapse' must either be 'mean' or int from 0 to {tlen}")
            
            # -- Filter spectra to only represent 4 to 256km (cut-off 256 to 512 part)
            if data["k"].shape[0] > max_wavenumber:
                data = data.isel(k=slice(1, None, 2))
                # -- Relabel k from 2,4,6,...,128 to 1,2,3,...,64
                data = data.assign_coords(k=("k", (data["k"].values / 2)))

            k_km = data["wavelength_km"].values            
            if wavelength_km is None:
                wavelength_km = k_km
            elif wavelength_km.shape != k_km.shape:
                raise ValueError(
                    f"Shape Mismatch between weavelength-array for dates {date}\n"
                    f"New wave-length array: {k_km.shape}    Previous array: {wavelength_km.shape}\n"
                    f"New:\n{k_km}\n\nOld:\n{wavelength_km}")

            elif not np.allclose(wavelength_km, k_km):
                raise ValueError(f"wavelength_km differs for date {date}")


            data = data.expand_dims(experiment_date=[date])
            data_list.append(data)

        data_all = xr.concat(data_list, dim="experiment_date", join="exact")

        output = dict(
            mean = data_all.mean(dim="experiment_date"),
            median = data_all.median(dim="experiment_date"),
            q25 = data_all.quantile(0.25, dim="experiment_date"),
            q75 = data_all.quantile(0.75, dim="experiment_date"),
            waelenght = wavelength_km)
        
        return output



    nplots = 2#len(methods)
    ncols = 2#min(ncols, nplots)
    nrows = 1#int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=False,
        sharey=True,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        dpi=300,
        squeeze=False,
    )
    axs = axes.flatten()
    

    for idx, spec_type in enumerate(["increment", "enspert"]):
        for idx2, method in enumerate(methods):
            
            ax = axs[idx]

            c = color_exp_dict[method]
            output = pepare_spec_data(method, data_type=spec_type, time_collapse=time_collapse)
            
            if premultiplied:
                k = output["mean"]["k"].values
            else:
                k=1

            yavg = k*output["mean"].values
            # ymed  = k*output["median"].values
            y25 = k*output["q25"].values
            y75 = k*output["q75"].values

            ax.loglog(output["waelenght"], yavg, color=c, label=method, linewidth=2.5)
            if plot_spread:
                # ax.loglog(output["waelenght"], ymed, color=c, label=method, linestyle="--")
                ax.fill_between(output["waelenght"], y25, y75, color=c, alpha=0.2)


            ax.set_xlim(np.ceil(output["waelenght"].max() / 1000) * 1000, 1)
            ax.grid(alpha=0.25)

            ax.set_ylim(1e-6,1e-1)

            if isinstance(time_collapse, str):
                t_add = " (TimeAvg)"
            else:
                t_add = f" (tidx: {time_collapse})"

            title = "Ensemble Perutbations" if spec_type=="enspert" else "Analysis Increment"
            ax.set_title(title+t_add)


            if idx == 0:
                axs[idx].legend(loc="lower left")
            if idx % ncols == 0:
                axs[idx].set_ylabel("k * S(k)" if premultiplied else "S(k)")
            if idx >= nplots - ncols:
                axs[idx].set_xlabel("Wavelength [km]")


    
    fig.suptitle(f"Averaged Spectra")

    fig.tight_layout()

    if fig_savedir is not None:
        fig_savedir = Path(fig_savedir)
        fig_savedir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"Spectra_DAcyclesDates.png"

        fig.savefig(fig_savedir / filename, dpi=200, bbox_inches="tight")

    if return_fig_axes:
        return fig, axes
    



def prepare_avg_timeseries(spectra_dict):
    
    ds_list = []
    for key, value in spectra_dict.items():
        ds_sub = value["increment"].copy(deep=True)        
        # Get a string datetime representation
        t0_date = ds_sub.time[0].dt.strftime("%Y%m%d_%H").item()
        # Change timing to relative time after t0
        t0 = ds_sub.time[0] - pd.Timedelta(minutes=30)
        ds_sub["time"] = ds_sub["time"] - t0
        ds_sub = ds_sub.expand_dims({"date": [str(t0_date)]})
        # Check that we only look to inner-domain --> k in 1,64
        ds_list.append(ds_sub.sel(k=slice(0,64)))

    ds_all = xr.concat(ds_list, dim="date", join="exact")

    minutes_da = ds_all.time / np.timedelta64(1, "m")
    ds_all["time"] = minutes_da.astype("int")


    return ds_all






## ==========================================================================================================
## ==========================================================================================================
## ==========================================================================================================
## Spatial correlation Plot
## ------------------------


def compute_radial_correlation(
    da: xr.DataArray,
    zlev: float,
    dx_km: float = 2.0,
    buffer_g: int = 12,
    bin_km: float = 4.0,
    max_d_km: float = 100.0,
    enforce_zero_mean: bool = True,
) -> xr.DataArray:
    """
    Compute isotropic (radially averaged) spatial correlation function on one z level
    from ensemble perturbations.

    Parameters
    ----------
    da : xr.DataArray
        Dimensions must include: ('time', 'xf', 'yf', 'zf', 'member')
        Values are ensemble perturbations (or anomalies).
    zlev : float
        Target level in same units as zf coordinate (e.g. 200).
    dx_km : float
        Grid spacing in km (2 km for you).
    buffer_g : int
        Buffer in grid points cropped on each side before computing correlations.
    bin_km : float
        Radial bin size in km.
    max_d_km : float
        Maximum radius in km for the output curve.
    enforce_zero_mean : bool
        If True, subtract member-mean at each grid point (and time) to enforce anomalies.

    Returns
    -------
    xr.DataArray
        1D correlation vs radius. Dimension: 'r_km'
    """

    # --- 1) select level
    da2 = da.sel(zf=zlev, method="nearest")

    # --- 2) crop buffer
    nx = da2.sizes["xf"]
    ny = da2.sizes["yf"]
    if 2 * buffer_g >= min(nx, ny):
        raise ValueError("buffer_g too large for domain size.")

    da2 = da2.isel(
        xf=slice(buffer_g, nx - buffer_g),
        yf=slice(buffer_g, ny - buffer_g),
    )

    # Ensure order (time, member, yf, xf) for contiguous arrays
    da2 = da2.transpose("time", "member", "yf", "xf")

    # Optionally enforce anomalies
    if enforce_zero_mean:
        da2 = da2 - da2.mean("member")


    arr = da2.values
    nt, nm, nyi, nxi = arr.shape

    # --- 3) accumulate autocorrelation (via FFT) over time+member
    # Autocorr in periodic sense; cropping buffer reduces edge effects substantially.
    ac_sum = np.zeros((nyi, nxi), dtype=np.float64)
    n_used = 0

    for t in range(nt):
        for m in range(nm):
            x = arr[t, m, :, :]

            # Check if NaN's are present!
            if not np.isfinite(x).all():
                raise ValueError("Found NaNs. Need a masked correlation variant.")

            # FFT-based autocorrelation: ifft2(|F|^2)
            F = np.fft.fft2(x)
            ac = np.fft.ifft2(F * np.conj(F)).real

            ac_sum += ac
            n_used += 1

    ac_mean = ac_sum / max(n_used, 1)

    # --- 4) shift so that zero-lag is at the center (more intuitive for distances)
    ac_centered = np.fft.fftshift(ac_mean)

    # --- 5) normalize to correlation
    # zero-lag is at center after fftshift
    c0 = ac_centered[nyi // 2, nxi // 2]
    if c0 <= 0:
        raise ValueError(f"Non-positive variance at zero lag: {c0}")
    corr2d = ac_centered / c0

    # --- 6) radial binning
    # lag coordinates: [-N/2 .. N/2-1] * dx
    iy = np.arange(-nyi // 2, nxi // 2 + (nyi % 2))  # careful with even sizes
    ix = np.arange(-nxi // 2, nxi // 2 + (nxi % 2))
    # But corr2d is exactly (nyi, nxi), so simpler:
    iy = np.arange(-nyi//2, nyi//2)
    ix = np.arange(-nxi//2, nxi//2)

    YY, XX = np.meshgrid(iy * dx_km, ix * dx_km, indexing="ij")
    R = np.sqrt(XX**2 + YY**2)

    # bins
    bin_edges = np.arange(0.0, max_d_km + bin_km, bin_km)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    corr_1d = np.full(bin_centers.shape, np.nan, dtype=np.float64)

    for i in range(len(bin_centers)):
        r0, r1 = bin_edges[i], bin_edges[i + 1]
        mask = (R >= r0) & (R < r1)
        if np.any(mask):
            corr_1d[i] = corr2d[mask].mean()

    return xr.DataArray(
        corr_1d,
        coords={"r_km": bin_centers},
        dims=("r_km",),
        name="Correlation",
        attrs={
            "dx_km": dx_km,
            "buffer_g": buffer_g,
            "bin_km": bin_km,
            "max_d_km": max_d_km,
            "zf_selected": float(da2["zf"].values) if "zf" in da2.coords else zlev,
            "note": "Computed from ensemble perturbations using FFT autocorrelation, then radially averaged.",
        },
    )



def plot_horizontal_correlation(
        dict_of_methods, 
        ensemble_pert_key, 
        variable, 
        crop_time_by=1,
        zlev=200,
        dx_km=2,
        buffer_g=0,
        bin_km=2,
        max_d_km=120,
        enforce_zero_mean=False,
        color_exp_dict:dict={},
        fig_savedir:Path|None=None
        ):

    fig,ax=plt.subplots(figsize=(6,3))
    for key, value in dict_of_methods.items():
        da_wind = value[ensemble_pert_key][variable].isel(time=slice(None,-crop_time_by))
        corr = compute_radial_correlation(
            da=da_wind,          # (time, xf, yf, zf, member)
            zlev=zlev,
            dx_km=dx_km,
            buffer_g=buffer_g,
            bin_km=bin_km,
            max_d_km=max_d_km,
            enforce_zero_mean=enforce_zero_mean, 
        )

        color_exp_dict = {}
        foo = color_exp_dict.get(key,None)
        c = color_exp_dict.get(key, "k") 
        label = foo if foo is not None else key
        corr.plot(color=c, label=label, ax=ax)
    
    ax.grid(alpha=.25)
    ax.legend()
    ax.set_xlabel("Horizonatl distance [km]")
    ax.set_ylabel("Correlation")

    # ax.set_title("Spatial Correlation of Ens. Perts -- Avg. over all Analysises")

    fig.tight_layout()

    if fig_savedir is not None:
        fig.savefig(f"{fig_savedir}/horizontal_correlation_z{zlev}.png")



