# letkf_local.py
import numpy as np
import xarray as xr
from functools import partial
import numpy as np
from numpy.linalg import eig, inv



def proj_matrix(N: int) -> np.ndarray:
    """Return the projection matrix Π = (I - 1 1ᵀ / N) / √(N-1).

    Parameters
    ----------
    N : int
        Ensemble size.
    """
    one = np.ones((N, 1))
    return (np.eye(N) - (one @ one.T) / N) / np.sqrt(N - 1)


def anomalies(matrix: np.ndarray) -> np.ndarray:
    """Return anomaly matrix A = E Π where E has shape (state, N)."""
    return matrix @ proj_matrix(matrix.shape[1])


def symm_matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Return the symmetric square root of a symmetric p.d matrix.

    Parameters
    ----------
    M : np.ndarray
        Symmetric p.d matrix.

    Returns
    -------
    np.ndarray
        Symmetric square root satisfying S Sᵀ = M.
    """
    # Eigen‐decomposition – guaranteed real for symmetric M
    vals, vecs = eig(M)
    # Numerical guard: clip negative tiny eigenvalues to zero
    vals = np.where(vals < 0, 0.0, vals)
    S = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    return (S + S.T) * 0.5  # enforce exact symmetry


def letkf_update(
    Z: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
    Y=None,
    H=None,
    inflation: float = 1.0,
    return_weights: bool = False,
):
    """Local Ensemble Transform Kalman Filter (global variant).

    Implements the analysis step of LETKF as described by Hunt et al. (2007).

    Parameters
    ----------
    Z : np.ndarray (n, N)
        Forecast/"background" ensemble matrix, columns are ensemble members.
    y_obs : np.ndarray (m,) or (m, 1)
        Observation vector at the current analysis time.
    H : callable or np.ndarray
        Either (i) a function mapping ensemble states (n, N) -> (m, N)
        that applies the nonlinear/linear observation operator to every
        ensemble member, **or** (ii) a constant observation matrix H of
        shape (m, n) for a *linear* operator.  In the latter case a matrix
        multiplication is used.
    R : np.ndarray (m, m)
        Observation-error covariance matrix.
    inflation : float, optional (default=1.0)
        Multiplicative covariance inflation factor (rho in Hunt et al.).
        Values >1 inflate the background spread.
    return_weights : bool, optional
        If *True* the function returns (Za, w_bar, W), where ``Za`` is the
        analysis ensemble, ``w_bar`` the analysis weight mean, and ``W`` the
        weight perturbation matrix in ensemble space.

    Returns
    -------
    Za : np.ndarray (n, N)
        Analysis ensemble matrix.
    w_bar : np.ndarray (N, 1), optional
        Mean weights in ensemble space.
    W : np.ndarray (N, N), optional
        Weight perturbations (columns sum to zero).
    """
    n, N = Z.shape

    # 1. Background mean and perturbations
    z_bar = Z.mean(axis=1, keepdims=True)
    Xb = Z - z_bar  # shape: (n, N)

    # # 2. Apply observation operator to each ensemble member
    # if callable(H):
    #     Y = H(Z)  
    # else:  # assume matrix
    #     Y = H @ Z

    m = Y.shape[0] # expected shape (m, N)
    y_bar = Y.mean(axis=1, keepdims=True)
    Yb = Y - y_bar  # shape (m, N)

    # 3. Pre‐compute inverses
    Rinv = inv(R)

    # 4. Ensemble‐space background covariance and its inverse
    P_tilde_inv = (N - 1) / inflation * np.eye(N) + Yb.T @ Rinv @ Yb
    P_tilde = inv(P_tilde_inv)  # shape (N, N)

    # 5. Analysis weights
    innov = y_obs.reshape(m, 1) - y_bar  # (m,1)
    w_bar = P_tilde @ (Yb.T @ Rinv @ innov)  # (N,1)

    # 6. Weight perturbation transform (symmetric square root)
    W = symm_matrix_sqrt((N - 1) * P_tilde)  # (N,N), columns sum to 0

    # 7. Analysis ensemble in state space
    Za = z_bar + Xb @ (w_bar + W)

    if return_weights:
        return Za, w_bar, W
    return Za






def letkf_local(
        prior_ens: xr.Dataset,        # (member, zf, xf, yf)
        predobs_ens: xr.Dataset,      # (member, station, zf)
        y_obs: xr.DataArray,          # (station, zf)
        R_std: float = 0.5,           # lidar σ
        loc_rad: float = 20e3,        # Radius in meters
        ):
    
    """
    My implementation of the LETKF quite strictly following Hunt et al. (2007)
    Assumptions:
    ------------
        - Flat earth --> Euclidean distance in metres is fine.
        - Diagonal and constant in time R-matrix (Observation-Error covariance)
        - Additional localisation (Gaspari-Cohn) and inflation within the local radius
    """


    # --- 1) Select the

    # --- 1) Establish Background Observation Ensemble
    """ This is already given as input """
    # --- 2) Obtain Background Perturbation matrix X^b
    X_b   = ens_b_obs - ens_b_obs.mean(dim="member")
    # --- 3) Select necessary Grid-points for Observations PER GRID-POINT


    pass

def gaspari_cohn(r, cut):
    q = np.abs(r) / cut
    w = np.where(q <= 1,
                 1 - 5*q**2 + 5*q**3 + 0.6666667*q**4,
                 np.where(q <= 2,
                          0.6666667*(2 - q)**4,
                          0.0))
    return w



def letkf_local_foofoo(prior: xr.Dataset,
                yob: xr.DataArray,            # (member, station, zf)
                y_obs: xr.DataArray,          # (station, zf)
                R_std: float = 0.5,           # lidar σ
                loc_rad: float = 20e3,        # 20 km
                inflation: float = 1.07):
    """
    Vectorised LETKF with horizontal Gaspari-Cohn localisation.
    Assumes flat earth --> Euclidean distance in metres is fine.
    """
    memb   = prior.dims['member']
    xf, yf = prior['xf'], prior['yf']
    stn_x, stn_y = yob['xf'], yob['yf']

    # --- reshape -> matrices ------------------------------------------------
    Xb = prior.to_array().stack(state=['variable', 'zf', 'yf', 'xf']).transpose('state', 'member').values            # (n, N)
    Yb = yob.stack(obs=['station', 'zf']).transpose('obs', 'member').values
    y  = y_obs.stack(obs=['station', 'zf']).values

    # Pre-pack static terms
    N  = memb
    P  = np.eye(N) - 1/N
    R  = np.eye(y.size) * R_std**2

    Xa = np.empty_like(Xb)                      # to be filled column-wise

    # --- column loop (can be joblib / dask) ---------------------------------
    for j, (xj, yj) in enumerate(np.broadcast(xf, yf)):
        # 1) find which obs lie inside localisation radius
        dist = np.hypot(stn_x - xj, stn_y - yj)
        w    = gaspari_cohn(dist, loc_rad)      # (station,)
        use  = w > 0
        if not np.any(use):
            Xa[j] = Xb[j]                      # no obs => copy background
            continue

        # 2) build local obs vectors/matrices
        idx = np.repeat(use, y_obs.shape[-1])   # each zf level
        w_full = np.repeat(w[use], y_obs.shape[-1])
        R_loc  = R[np.ix_(idx, idx)] / (w_full[:, None]*w_full)

        Yb_loc = Yb[idx]
        y_loc  = y[idx]

        # 3) perform “standard” LETKF in ensemble space
        yb_bar = Yb_loc.mean(1)
        d      = y_loc - yb_bar
        Yb_p   = (Yb_loc - yb_bar[:, None]) / np.sqrt(N-1)

        Pa_inv = (N-1)*np.eye(N) + (Yb_p.T @ np.linalg.solve(R_loc, Yb_p))
        Wa     = np.linalg.solve(Pa_inv, (N-1)*P + (Yb_p.T @ np.linalg.solve(R_loc, d)))

        Xa[j]  = Xb[j] + Xb[:, j][:, None] @ (Wa/np.sqrt(N-1) - P).T

    # --- reshape back -------------------------------------------------------
    anal = (xr.DataArray(
        Xa, coords=dict(state=prior.to_array().stack(state=['variable','zf','yf','xf']).state,
                        member=prior.member))
           .unstack('state')
           .to_dataset('variable'))
    anal['member'] = prior.member
    return anal * inflation                   # simple multiplicative inflation



