from collections.abc import Sequence

import numpy as np
import xarray as xr


class ObsOp:

    def __init__(
        self,
        obs_type: str,
        pred_vars: Sequence[str] | None = None,
    ) -> None:

        self.obs_type = obs_type
        self.pred_vars = list(pred_vars) if pred_vars is not None else ["u", "v"]

    # -----------------------------------------------------------------

    def _interp_xy(
        self,
        ds: xr.Dataset,
        xcoord: float,
        ycoord: float,
    ) -> xr.Dataset:
        """Interpolate field to an observation location."""

        return ds.interp(
            xf=xcoord,
            yf=ycoord,
            method="linear",
        )

    # -----------------------------------------------------------------

    def _interp_z(
        self,
        ds: xr.Dataset,
        z_list: Sequence[float],
    ) -> xr.Dataset:
        """Interpolate field to prescribed vertical levels."""

        return ds.interp(
            zf=np.asarray(z_list),
            method="linear",
        )


class ObsOpSynthetic(ObsOp):
    """
    Observation operator for synthetic data.

    Parameters
    ----------
    pred_vars
        Variables to extract, e.g. ["u", "v"].

    coord_idx
        List of (ix, iy) index pairs specifying synthetic station locations.

    z_list
        Vertical levels to interpolate to.
    """

    def __init__(
        self,
        pred_vars: Sequence[str],
        coord_idx: Sequence[tuple[int, int]],
        z_list: Sequence[float],
    ) -> None:

        super().__init__("synthetic metmast", pred_vars)

        self.coord_idx = list(coord_idx)
        self.z_list = list(z_list)

    # -----------------------------------------------------------------

    def __call__(self, ds_Xf: xr.Dataset) -> xr.Dataset:

 
        for dim in ("xf", "yf", "zf"):
            if dim not in ds_Xf.dims:
                raise ValueError(
                    f"Input dataset must contain dimension {dim!r}. "
                    f"Found dimensions: {tuple(ds_Xf.dims)}"
                )

        # Keep only variables required by the observation operator
        ds_Xf = ds_Xf[self.pred_vars]

        # --- 2) Extract/interpolate each synthetic station
        Yf = []

        for icoord in self.coord_idx:

            if len(icoord) != 2:
                raise ValueError("Each element of coord_idx must contain exactly two indices: (ix, iy).")
            ix, iy = icoord

            # Select horizontal grid point.
            # xf/yf are retained as scalar coordinates at this stage.
            ds_station = ds_Xf.isel(xf=ix,yf=iy)
            # Interpolate vertically
            ds_station = self._interp_z(ds_station, self.z_list)
            Yf.append(ds_station)

        if not Yf:
            raise ValueError("coord_idx contains no station locations.")

        # --- 3) Concatenate stations
        ds_Yf = xr.concat(Yf, dim=xr.IndexVariable("station", np.arange(len(Yf))))

        return ds_Yf