"""functions for plotting different types of spectra

1D to 1D
P_k: Power spectrum of a 1D (x or t) signal as a function of wavenumber or frequency k
S_k: Power spectral density of a 1D (x or t) signal as a function of wavenumber or frequency k

2D to 1D
P_kh: Power spectrum of a 2D (x,y) signal as a function of horizontal wavenumber |k|
S_kh: Power spectral density of a 2D (x,y) signal as a function of horizontal wavenumber |k|
note that the power of wavenumbers between 1 and sqrt(2) is not included, meaning that the sum of the power spectrum does not completely retrieve the variance


2D to 2D
P_kx_kt: Power spectral density of a 2D (t,x) signal as a function of wavenumber kx and frequency kt
S_kx_kt: Power spectral density of a 2D (t,x) signal as a function of wavenumber kx and frequency kt
note that the power of horizontal wavenumbers between 1 and sqrt(2) is not included, meaning that the sum of the power spectrum does not completely retrieve the variance

3D to 2D
P_kh_kt: Power spectrum of a 3D signal (t,x,y) as a function of horizontal wavenumber |k| and frequency kt
S_kh_kt: Power spectral density of a 3D signal (t,x,y) as a function of horizontal wavenumber |k| and frequency kt
note that the power of horizontal wavenumbers between 1 and sqrt(2) is not included, meaning that the sum of the power spectrum does not completely retrieve the variance
"""

import numpy as np
import scipy.stats as stats


def P_k(f):
    """Power spectrum of 1D data (in either space or time)

    Parameters
    ----------
    f : array_like
        the data to compute the spectrum on, either spatial or temporal 1D data.

    Returns
    -------
    Pbins: array
        The power spectrum
    kvals : array
        The dimensionless wavenumbers.  To convert to wavelength: lambda = L / kvals. To convert to frequency omega = k / T, where T is the time span of the the signal.
    """
    N = f.shape[0]
    A = np.fft.fftn(f) / f.size  # the Fourier coefficients
    P = np.abs(A) ** 2  # the Fourier amplitudes
    k = (
        np.fft.fftfreq(N) * N
    )  # the dimensionless wavenumbers (1 is one wavelength in the full domain, N/2 is the Nyquist wavelength)
    knrm = (
        k**2
    ) ** 0.5  # the norm of the wavevector (everything becomes positive, so that we only have to define positive bins)
    kbins = np.arange(0.5, N // 2 + 1, 1)  # the wavenumber bins (only positive)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])  # the midpoints of those bins
    Pbins, _, _ = stats.binned_statistic(
        knrm.flatten(), P.flatten(), statistic="sum", bins=kbins
    )  # Power spectrum
    return Pbins, kvals


def S_k(f):
    """Power spectral density of 1D data (in either space or time)

    Parameters
    ----------
    f : 1D array
        the data to compute the spectrum on, either spatial or temporal 1D data.

    Returns
    -------
    Pbins: array
        The power spectral density
    kvals : array
        The dimensionless wavenumbers.  To convert to wavelength: lambda = L / kvals. To convert to frequency omega = k / T, where T is the time span of the the signal.
    """
    N = f.shape[0]
    A = np.fft.fftn(f) / f.size  # the Fourier coefficients
    P = np.abs(A) ** 2  # the Fourier amplitudes
    k = (
        np.fft.fftfreq(N) * N
    )  # the dimensionless wavenumbers (1 is one wavelength in the full domain, N/2 is the Nyquist wavelength)
    knrm = (
        k**2
    ) ** 0.5  # the norm of the wavevector (everything becomes positive, so that we only have to define positive bins)
    kbins = np.arange(0.5, N // 2 + 1, 1)  # the wavenumber bins (only positive)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])  # the midpoints of those bins
    Sbins, _, _ = stats.binned_statistic(
        knrm.flatten(), P.flatten(), statistic="mean", bins=kbins
    )  # Power spectral density
    return Sbins, kvals


def P_kh(f):
    """
    Power spectrum of a 2D (x,y) signal as a function of horizontal wavenumber |k|.
    Note that the power of wavenumbers between 1 and sqrt(2) is not included, meaning that the sum of the power spectrum does not completely retrieve the variance

    Parameters
    ----------
    f : 2D array in which both dimensions are equivalen (e.g. x and y, but not x and t.)

    Returns
    -------
    Pbins: 1D array
        The power spectrum
    kvals : array
        The dimensionless wavenumbers.  To convert to wavelength: lambda = L / kvals. To convert to frequency omega = k / T, where T is the time span of the the signal.

    """
    N, N = f.shape
    A = np.fft.fftn(f) / f.size  # the Fourier coefficients
    P = np.abs(A) ** 2  # the Fourier amplitudes
    kx, ky = (
        np.fft.fftfreq(N) * N,
        np.fft.fftfreq(N) * N,
    )  # the dimensionless wavenumbers (1 is one wavelength in the full domain, N/2 is the Nyquist wavelength)
    Kx, Ky = np.meshgrid(kx, ky)
    knrm = (
        Kx**2 + Ky**2
    ) ** 0.5  # the norm of the wavevector (everything becomes positive, so that we only have to define positive bins)
    kbins = np.arange(0.5, N // 2 + 1, 1)  # the wavenumber bins (only positive)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])  # the midpoints of those bins
    Pbins, _, _ = stats.binned_statistic(
        knrm.flatten(), P.flatten(), statistic="sum", bins=kbins
    )  # Power spectrum
    return Pbins, kvals


def S_kh(f):
    """
    Power spectral density of a 2D (x,y) signal as a function of horizontal wavenumber |k|.
    Note that the power of wavenumbers between 1 and sqrt(2) is not included, meaning that the sum of the power spectrum does not completely retrieve the variance


    Parameters
    ----------
    f : 2D array in which both dimensions are equivalen (e.g. x and y, but not x and t.)

    Returns
    -------
    Pbins: 1D array
        The power spectral density
    kvals : array
        The dimensionless wavenumbers.  To convert to wavelength: lambda = L / kvals. To convert to frequency omega = k / T, where T is the time span of the the signal.

    """
    N, N = f.shape
    A = np.fft.fftn(f) / f.size  # the Fourier coefficients
    P = np.abs(A) ** 2  # the Fourier amplitudes
    kx, ky = (
        np.fft.fftfreq(N) * N,
        np.fft.fftfreq(N) * N,
    )  # the dimensionless wavenumbers (1 is one wavelength in the full domain, N/2 is the Nyquist wavelength)
    
    Kx, Ky = np.meshgrid(kx, ky)
    knrm = (Kx**2 + Ky**2) ** 0.5  # the norm of the wavevector (everything becomes positive, so that we only have to define positive bins)
    kbins = np.arange(0.5, N // 2 + 1, 1)  # the wavenumber bins (only positive)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])  # the midpoints of those bins
    Sbins, _, _ = stats.binned_statistic(
        knrm.flatten(), P.flatten(), statistic="mean", bins=kbins
    )  # Power spectrum
    return Sbins, kvals


def P_kx_kt(f):
    """
    Power spectrum of a 2D (t,x) signal as a function of wavenumber kx and frequency kt


    Parameters
    ----------
    f : 2D array in with non-equivalent dimensions t,x

    Returns
    -------
    Pbins: 2D array
        The power spectrum
    kxvals: 1D array
        The dimensionless wavenumbers in space.
    ktvals: 1D array
        The dimensionless wavenumbers in time.
    """

    M, N = f.shape
    A = np.fft.fftn(f) / f.size  # the Fourier coefficients
    P = np.abs(A) ** 2  # the Fourier amplitudes
    kx, kt = (
        np.fft.fftfreq(N) * N,
        np.fft.fftfreq(M) * M,
    )  # the dimensionless wavenumbers (1 is one wavelength in the full domain, N/2 is the Nyquist wavelength)
    kxnrm, ktnrm = (kx**2) ** 0.5, (
        kt**2
    ) ** 0.5  # the norm of the wavevector (everything becomes positive, so that we only have to define positive bins)
    kxbins, ktbins = np.arange(0.5, N // 2 + 1, 1), np.arange(
        0.5, M // 2 + 1, 1
    )  # the wavenumber bins
    kxvals, ktvals = 0.5 * (kxbins[1:] + kxbins[:-1]), 0.5 * (
        ktbins[1:] + ktbins[:-1]
    )  # the midpoints of those bins
    # input shape of the first binning is M x N, output is shape M x kxbins
    Pbins, _, _ = stats.binned_statistic(
        kxnrm, P, statistic="sum", bins=kxbins
    )  # Power spectrum
    # input shape of the second binning should be kxbins x M
    PPbins, _, _ = stats.binned_statistic(
        ktnrm, Pbins.transpose(), statistic="sum", bins=ktbins
    )  # Power spectrum
    # output shape is kxbins x ktbins
    return PPbins, kxvals, ktvals


def S_kx_kt(f):
    """
    Power spectral density of a 2D (t,x) signal as a function of wavenumber kx and frequency kt


    Parameters
    ----------
    f : 2D array in with non-equivalent dimensions t,x

    Returns
    -------
    Sbins: 2D array
        The power spectral density
    kxvals: 1D array
        The dimensionless wavenumbers in space.
    ktvals: 1D array
        The dimensionless wavenumbers in time.
    """

    M, N = f.shape
    A = np.fft.fftn(f) / f.size  # the Fourier coefficients
    P = np.abs(A) ** 2  # the Fourier amplitudes
    kx, kt = (
        np.fft.fftfreq(N) * N,
        np.fft.fftfreq(M) * M,
    )  # the dimensionless wavenumbers (1 is one wavelength in the full domain, N/2 is the Nyquist wavelength)
    
    kxnrm, ktnrm   = (kx**2) ** 0.5, (kt**2) ** 0.5  # the norm of the wavevector (everything becomes positive, so that we only have to define positive bins)
    kxbins, ktbins = np.arange(0.5, N // 2 + 1, 1), np.arange(0.5, M // 2 + 1, 1)  # the wavenumber bins
    kxvals, ktvals = 0.5 * (kxbins[1:] + kxbins[:-1]), 0.5 * (ktbins[1:] + ktbins[:-1])  # the midpoints of those bins
    
    # input shape of the first binning is M x N, output is shape M x kxbins
    Sbins, _, _ = stats.binned_statistic(kxnrm, P, statistic="mean", bins=kxbins)  # Power spectrum
    
    # input shape of the second binning should be kxbins x M
    SSbins, _, _ = stats.binned_statistic(
        ktnrm, Sbins.transpose(), statistic="sum", bins=ktbins
    )  # Power spectrum
    # output shape is kxbins x ktbins
    # to plot with increasing size and period, use ax.invert_xaxis() and ax.invert_yaxis()
    return SSbins, kxvals, ktvals


def P_kh_kt(f):
    """
    Power spectrum of a 3D signal (t,x,y) as a function of spatial wavenumber |k| and frequency kt
    note that the power of the spatial wavenumbers between 1 and sqrt(2) is not included, meaning that the sum of the power spectrum does not completely retrieve the variance

    Parameters
    ----------
    f : 3D array in with dimensions t,x,y. Size of dimensions x and y should be equal.

    Returns
    -------
    Pbins: 2D array
        The power spectrum
    khvals: 1D array
        The dimensionless wavenumbers in space.
    ktvals: 1D array
        The dimensionless wavenumbers in time.
    """
    M, N, N = f.shape
    # the DFT
    A = np.fft.fftn(f) / f.size
    # the power spectrum
    P = (np.abs(A)) ** 2
    kx, ky, kt = (
        np.fft.fftfreq(N) * N,
        np.fft.fftfreq(N) * N,
        np.fft.fftfreq(M) * M,
    )  # the dimensionless wavenumbers (1 is one wavelength in the full domain, N/2 is the Nyquist wavelength)
    Kx, Ky = np.meshgrid(kx, ky)
    khnrm, ktnrm = (Kx**2 + Ky**2) ** 0.5, (
        kt**2
    ) ** 0.5  # the norm of the wavevector (everything becomes positive, so that we only have to define positive bins)
    khbins = np.arange(0.5, N // 2 + 1, 1)
    ktbins = np.arange(0.5, M // 2 + 1, 1)  # the wavenumber bins (only positive)
    
    khvals = 0.5 * (khbins[1:] + khbins[:-1]) 
    ktvals = 0.5 * (ktbins[1:] + ktbins[:-1])  # the midpoints of those bins    f: input with shape Nt, Nx, Ny = Nx

    # shape to be put into the kh binning is M x N^2, returns shape M x kbins
    Pbins, _, _ = stats.binned_statistic(khnrm.flatten(), P.reshape(M, N**2), statistic="sum", bins=khbins)  # Power spectrum
    
    # shape to be be put into the kt binning is kbins x M (transpose of previous result), this returns shape khbins x ktbins:
    PPbins, _, _ = stats.binned_statistic(ktnrm, Pbins.transpose(), statistic="sum", bins=ktbins)  # Power spectrum
    # the result now has shape khbins x ktbins
    # to plot with increasing size and period, use ax.invert_xaxis() and ax.invert_yaxis() and np.swapaxes(PPbins,0,1)

    # power at wavenumbers higher than khbins (up to sqrt(2)*max(khbins))
    residual_power = P.sum() - PPbins.sum()

    return PPbins, khvals, ktvals


def S_kh_kt(f):
    """
    Power spectral density of a 3D signal (t,x,y) as a function of spatial wavenumber |k| and frequency kt
    note that the power of the spatial wavenumbers between 1 and sqrt(2) is not included, meaning that the sum of the power spectrum does not completely retrieve the variance

    Parameters
    ----------
    f : 3D array in with dimensions t,x,y. Size of dimensions x and y should be equal.

    Returns
    -------
    Pbins: 2D array
        The power spectral density
    khvals: 1D array
        The dimensionless wavenumbers in space.
    ktvals: 1D array
        The dimensionless wavenumbers in time.
    """
    M, N, N = f.shape
    
    # the DFT
    A = np.fft.fftn(f) / f.size
    
    # the power spectrum
    P = (np.abs(A)) ** 2
    kx, ky, kt = (
        np.fft.fftfreq(N) * N,
        np.fft.fftfreq(N) * N,
        np.fft.fftfreq(M) * M,
    )  # the dimensionless wavenumbers (1 is one wavelength in the full domain, N/2 is the Nyquist wavelength)
    
    Kx, Ky = np.meshgrid(kx, ky)
    khnrm, ktnrm = (Kx**2 + Ky**2) ** 0.5, (kt**2) ** 0.5  # the norm of the wavevector (everything becomes positive, so that we only have to define positive bins)
    khbins, ktbins = np.arange(0.5, N // 2 + 1, 1), np.arange(0.5, M // 2 + 1, 1)  # the wavenumber bins (only positive)
    khvals, ktvals = 0.5 * (khbins[1:] + khbins[:-1]), 0.5 * (ktbins[1:] + ktbins[:-1])  # the midpoints of those bins

    # shape to be put into the kh binning is M x N^2, returns shape M x kbins
    Sbins, _, _ = stats.binned_statistic(khnrm.flatten(), P.reshape(M, N**2), statistic="mean", bins=khbins)  # Power spectrum
    
    # shape to be be put into the kt binning is kbins x M (transpose of previous result), this returns shape khbins x ktbins:
    SSbins, _, _ = stats.binned_statistic(ktnrm, Sbins.transpose(), statistic="mean", bins=ktbins)  # Power spectrum
    
    # the result now has shape khbins x ktbins
    # to plot with increasing size and period, use ax.invert_xaxis() and ax.invert_yaxis()
    return SSbins, khvals, ktvals


def make_kmask(size, kmin, kmax):
    """ "
    makes a 'mask' of values for the wavenumber norm. Everything between kmin and kmax is assinged .5, everything else 1.
    size: size of the array to compute the spectrum on
    kmin, kmax: in between these values, the mask is .5, elsewhere 1. Note that these are the integer wavenumbers. So, the period corresponding to them is T/kmin and T/kmax.
    """
    k = np.fft.fftfreq(size) * size  # wavenumbers as integers
    knrm = (k**2) ** 0.5  # their norm
    kmask = np.ones_like(knrm)
    kmask[(knrm > kmin)] = 0.5
    kmask[(knrm > kmax)] = 1
    return kmask


def reconstruct(f, kmask):
    """ " given 1D input data (f) and a mask of the norm of k (kmask), reconstruct the input data. The Fourier coefficients are multiplied with the mask.
    Workflow:

    no_bins = 50
    kmask = make_kmask_other_signal(f_original, f_desired, no_bins=no_bins)
    f_reconstructed = reconstruct(f_original, kmask)
    """
    N = f.shape[0]
    A = np.fft.fftn(f) / f.size  # the Fourier coefficients
    P = np.abs(A) ** 2  # the Fourier amplitudes
    k = (
        np.fft.fftfreq(N) * N
    )  # the dimensionless wavenumbers (1 is one wavelength in the full domain, N/2 is the Nyquist wavelength)
    knrm = (
        k**2
    ) ** 0.5  # the norm of the wavevector (everything becomes positive, so that we only have to define positive bins)
    A_modified = A * kmask
    return np.real(np.fft.ifftn(A_modified * f.size))


def smooth_spectrum(P, k, no_bins):
    """
    averages a spectrum over logarithmically increasing bin widths
    """

    bins = (
        np.unique(np.ceil(np.logspace(0, np.log(k[-1]), no_bins, base=np.e))) - 0.5
    )  # to prevent empty bins
    
    bin_widths = np.ediff1d(bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    P, _, _ = stats.binned_statistic(k, P, statistic="mean", bins=bins)

    return P, bin_centers


def smooth_and_interpolate(x, data, no_bins):
    """
    smooths data by binning and computing the mean within each bin. Then resamples to the original points.

    for spectra: x should be np.arange(data.size), or np.arange(k.size). Cannot use the real k values.
    bins are logarithmically spaces between

    """
    bins = (
        np.unique(np.ceil(np.logspace(0, np.log(x[-1]), no_bins, base=np.e))) - 0.5
    )  # to prevent empty bins

    statistic, bins, _ = stats.binned_statistic(
        x,
        data,
        bins=bins,
    )
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # fig, ax = plt.subplots()
    # plt.loglog(x, data, alpha = .5)
    # plt.loglog(x, np.interp(x, bin_centers, statistic), alpha = .5)
    # plt.vlines(bins, ax.get_ylim()[0], ax.get_ylim()[1])
    # plt.show()

    return np.interp(x, bin_centers, statistic)


def make_kmask_other_signal(f_original, f_desired, no_bins):
    """ "
    makes a 'mask' of values for the wavenumber norm from another signal.
    f_original: the data (!) of which the spectrum needs to be modified
    f_desired: the data of which we want to impose the spectrum on data_original
    returns: a kmask to use for the function reconstruct
    """
    P_original = np.abs(np.fft.fftn(f_original)) ** 2
    P_desired = np.abs(np.fft.fftn(f_desired)) ** 2
    k = np.fft.fftfreq(f_original.size) * f_original.size
    mask = smooth_and_interpolate(
        np.arange(k.size), P_desired, no_bins=no_bins
    ) / smooth_and_interpolate(np.arange(k.size), P_original, no_bins=no_bins)
    return mask


def shift(
    arr, num, fill_value=np.nan
):  # from https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def increments(data, tau):
    return data - shift(data, tau)


def structure_function(data, shifts, moments=np.arange(1, 5), standardized=True):
    """
    data: np array with (wind) data
    shifts: the range of time shifts to compute the structure functions for

    returns: array size no. shifts x no. moments
    can be plotted easily with plt.loglog(shifts, structure_function(data, shifts))
    """
    if standardized:
        data = (data - data.mean()) / data.std()  # for the standardized moment
    data_shifted = np.array(
        [
            np.array([np.abs(increments(data, shift)) for shift in shifts]) ** moment
            for moment in moments
        ]
    )
    return np.nanmean(np.abs(data_shifted), axis=-1).T


def P_kh_hist(data_list, custom_bins=False, bins=np.arange(100), skip=1):
    """
    temporal histogram of the spatial power spectrum. Inspired by Selz, T., Bierdel, L., & Craig, G. C. (2019). 
    Estimation of the Variability of Mesoscale Energy Spectra with Three Years of COSMO-DE Analyses. Journal of the Atmospheric Sciences, 
    76(2), 627–637. https://doi.org/10.1175/jas-d-18-0155.1

    data_list: a list of xarray DataArray's with t, x and y dimension (x and y should be the same size and spacing). For the energy spectrum: [ds.u, ds.v]
    bins: to bin the power in, default is log-spaced between the lowest and highest power.
    skip: take every n'th value in time, because this calculation can get heavy.

    returns:
    P_hist: temporal histogram of the spatial power spectrum
    bin_centers: bins of the power
    k: dimensionless wavenumber

    """
    _, k = S_kh(data_list[0].isel(time=0))
    # _, k = wp.post.spectral.S_kh(data_list[0].isel(time=0))

    P = np.sum(
        np.array(
            [
                np.array(
                    [
                        # (wp.post.spectral.P_kh(data.sel(time=t)))[0]
                        (P_kh(data.sel(time=t)))[0]
                        for t in data.time[0::skip]
                    ]
                )
                for data in data_list
            ]
        ),
        0,
    )

    if not custom_bins:
        if np.nanmin(P) > 0:
            bins = np.logspace(np.log10(np.nanmin(P)), np.log10(np.nanmax(P)), 100)
        if np.nanmin(P) == 0:
            bins = np.logspace(-10, np.log10(np.nanmax(P)), 100)
    b = np.histogram(P, bins=bins)[1]
    bin_centers = (b[1:] + b[:-1]) / 2
    P_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 0, P)
    P_99 = np.apply_along_axis(lambda a: np.nanpercentile(a, 90), 0, P)
    P_1 = np.apply_along_axis(lambda a: np.nanpercentile(a, 10), 0, P)
    """ # plotting example
    P_hist, bin_centers, k = P_kh_hist([ds.u, ds.v])
    fig, ax = plt.subplots()
    plt.contourf(k, bin_centers,P_hist)
    plt.loglog(k, k**(-5/3))
    plt.loglog(k, k**(-3))
    ax.set_xscale('log')
    ax.set_yscale('log')
    """
    return P_hist, bin_centers, k






def compute_spectra_zfavg_new(data, spectral_density=True):
    """
    Computes the 1D spectra for each vertical level of a 2D field using the P_kh/S_kh 
    functions and then averages the spectra over all vertical levels.
    
    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        3D data field containing vertical levels (expcects dim-name "zf")

    use_spectral_density
        If True, use the S_kh function (spectral density using mean aggregation).
        Otherwise, use P_kh (power spectrum using summed energy).
    
    Returns
    -------
    result : dict
        A dictionary containing:
            'spectrum'    : The averaged 1D spectrum over all vertical levels.
            'wavenumbers' : The corresponding wavenumber array.
            'n_levels'    : Number of vertical levels averaged.
    """
    # Determine how many vertical levels and extract slices.
    spectra = []
    n_levels = 0
    common_kvals = None
    
    common_khvals = None
    common_ktvals = None

    # Support for xarray DataArray (with coordinate 'zf') or a plain numpy array
    if hasattr(data, 'dims') and 'zf' in data.dims and 'time' in data.dims:
        vertical_levels = data["zf"].values
        for level in vertical_levels:
            # Select the 2D field for the current vertical level
            field = data.sel(zf=level).values

            # Compute the spectrum on this 2D field using the chosen function
            if spectral_density:
                spec, khvals, ktvals = S_kh_kt(field)
            else:
                spec, khvals, ktvals = P_kh_kt(field)
            spectra.append(spec)
            n_levels += 1
            # Save common kvals from the first level (assumed identical for all levels)
            if common_khvals is None:
                common_khvals = khvals
            if common_ktvals is None:
                common_ktvals = ktvals

    elif hasattr(data, 'coords') and 'zf' in data.coords:
        vertical_levels = data["zf"].values
        for level in vertical_levels:
            # Select the 2D field for the current vertical level
            field = data.sel(zf=level).values

            # Compute the spectrum on this 2D field using the chosen function
            if spectral_density:
                spec, kvals = S_kh(field)
            else:
                spec, kvals = P_kh(field)
            spectra.append(spec)
            n_levels += 1
            # Save common kvals from the first level (assumed identical for all levels)
            if common_kvals is None:
                common_kvals = kvals
    else:
        raise KeyError(f"Data is expected to be xr.DataArray with vertical dim-name 'zf', but data is of type: {type(data)}")

    # Convert list of spectra to a NumPy array and average over vertical levels
    spectra      = np.array(spectra)
    avg_spectrum = np.mean(spectra, axis=0)

    # Save result
    result = dict()
    result["spectrum"] = avg_spectrum
    result["n_levels"] = n_levels
    if common_kvals is not None:
        result["wavenumbers"] = common_kvals        
    elif common_khvals is not None:
        result["khvals"] = common_khvals

        
    return result
