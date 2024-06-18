import warnings

import healpy as hp
import numpy as np
import rubin_sim.maf as maf
from scipy.optimize import minimize_scalar

from .constants import A_wfd
from .galaxy_distribution import number_density
from .utils import data_dir

# Load the CMB cross-correlation cache
_snr_cache_file = data_dir / "cmb_xcorr_snr_cache.npz"
if _snr_cache_file.exists():
    _snr_cache = np.load(_snr_cache_file, allow_pickle=True)
else:
    warnings.warn(
        "The CMB cross-correlation SNR cache does not exist. "
        "You should run `cmb_lensing_snr.ipynb` before computing SNR metrics."
    )
    _snr_cache = None


def density_for_quantile(
    q: float,
    m5: maf.MetricBundle,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate number density of LBGs for the given dropout-band m5 quantile.

    Parameters
    ----------
    q : float
        Quantile of 5-sigma limiting magnitudes. E.g. if q=0.25, then the
        cut in the dropout band is set to the 25th percentile of extinction-
        corrected depths.
    m5 : maf.MetricBundle
        MAF metric bundle containing 5-sigma limiting magnitudes in the
        specified band.
    band : str
        The band corresponding to the m5 metrics
    snr_floor: float, default=3
        The minimum SNR in the dropout band.
    dropout: float, default=1
        The change in color required for dropout selection.

    Returns
    -------
    float
        Number density of LBGs in deg^{-2}
    """
    # Get the limiting magnitudes from the metric bundle
    m5_array = m5.metric_values[m5.metric_values.mask == False].data

    # Get the limit in the dropout band
    drop_lim = np.quantile(m5_array, q)

    # Compute the number density
    return number_density(
        m5=drop_lim,
        band=band,
        snr_floor=snr_floor,
        dropout=dropout,
    )


def _fwfd_from_quantile(
    q: float,
    m5: maf.MetricBundle,
) -> float:
    """Calculate f_wfd corresponding to the quantile of m5.

    Parameters
    ----------
    q : float
        Quantile of 5-sigma limiting magnitudes. E.g. if q=0.25, then the
        cut in the dropout band is set to the 25th percentile of extinction-
        corrected depths.
    m5 : maf.MetricBundle
        MAF metric bundle containing 5-sigma limiting magnitudes.

    Returns
    -------
    float
        Fraction of fiducial WFD survey area that is deeper than the m5
        corresponding to the quantile
    """
    # Fraction deeper than m5 quantile
    f_deeper = 1 - q

    # Number of pixels in this bundle
    npix = (~m5.metric_values.mask).sum()

    # Area of each pixel
    pix_area = hp.nside2pixarea(m5.slicer.nside, degrees=True)

    # Total area deeper than m5 quantile
    area_deeper = f_deeper * npix * pix_area

    # Fraction of fiducial WFD deeper than cut
    f_wfd = area_deeper / A_wfd

    return f_wfd


def fwfd_for_density(
    n: float,
    m5: maf.MetricBundle,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate fraction of WFD deep enough to supply requested number density.

    Parameters
    ----------
    n : float
        The requested number density in units deg^{-2}
    m5 : maf.MetricBundle
        MAF metric bundle containing 5-sigma limiting magnitudes in the
        specified band.
    band : str
        The band corresponding to the m5 metrics
    snr_floor: float, default=3
        The minimum SNR in the dropout band.
    dropout: float, default=1
        The change in color required for dropout selection.

    Returns
    -------
    float
        Fraction of fiducial WFD that is deep enough to supply the requested
        number density.

    Raises
    ------
    RuntimeError
        If scipy optimizer fails
    """
    # Solve for the sky fraction
    res = minimize_scalar(
        lambda q: np.abs(
            n
            - density_for_quantile(
                q=q,
                m5=m5,
                band=band,
                snr_floor=snr_floor,
                dropout=dropout,
            )
        ),
        bounds=(0, 1),
    )

    # Make sure it worked!
    if not res.success:
        raise RuntimeError("Solving for fsky failed.")

    # Calculate f_wfd
    f_wfd = _fwfd_from_quantile(res.x, m5)

    return f_wfd


def _calc_snr(
    q: float,
    m5: maf.MetricBundle,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate the CMB xcorr SNR corresponding to the quantile.

    Uses a cache for speed.

    Parameters
    ----------
    q : float
        Quantile of 5-sigma limiting magnitudes. E.g. if q=0.25, then the
        cut in the dropout band is set to the 25th percentile of extinction-
        corrected depths.
    m5 : maf.MetricBundle
        MAF metric bundle containing 5-sigma limiting magnitudes in the
        specified band.
    band : str
        The band corresponding to the m5 metrics
    snr_floor: float, default=3
        The minimum SNR in the dropout band.
    dropout: float, default=1
        The change in color required for dropout selection.

    Returns
    -------
    float
        Signal-to-noise ratio of LBG x CMB Lensing correlation
    """
    # Get the number density
    n = density_for_quantile(
        q=q,
        m5=m5,
        band=band,
        snr_floor=snr_floor,
        dropout=dropout,
    )

    # Calculate f_wfd
    f_wfd = _fwfd_from_quantile(q, m5)

    # Calculate SNR
    n_grid = np.array(_snr_cache["n"].tolist()[band])
    snr_grid = np.array(_snr_cache["snr"].tolist()[band])
    snr = np.interp(n, n_grid, snr_grid * np.sqrt(f_wfd))

    return snr


def optimize_snr(
    m5: maf.MetricBundle,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> tuple[float, float, float]:
    """Optimize LBG x CMB Lensing correlation by varying number density and f_wfd.

    Parameters
    ----------
    m5 : maf.MetricBundle
        MAF metric bundle containing 5-sigma limiting magnitudes in the
        specified band.
    band : str
        The band corresponding to the m5 metrics
    snr_floor: float, default=3
        The minimum SNR in the dropout band.
    dropout: float, default=1
        The change in color required for dropout selection.

    Returns
    -------
    float
        Maximized SNR
    float
        Number density corresponding to maximum SNR
    float
        f_wfd corresponding to maximum SNR

    Raises
    ------
    RuntimeError
        If scipy optimizer fails
    """
    # Maximize SNR
    res = minimize_scalar(
        lambda q: -_calc_snr(
            q=q,
            m5=m5,
            band=band,
            snr_floor=snr_floor,
            dropout=dropout,
        ),
        bounds=(0, 1),
    )

    # Make sure it worked!
    if not res.success:
        raise RuntimeError("Maximizing SNR failed.")

    # Calculate corresponding number density
    snr = -res.fun
    n = density_for_quantile(
        q=res.x,
        m5=m5,
        band=band,
        snr_floor=snr_floor,
        dropout=dropout,
    )
    f_wfd = _fwfd_from_quantile(res.x, m5)

    return snr, n, f_wfd


def total_lbgs(
    m5: maf.MetricBundle,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> tuple[float, float, float]:
    """Calculate total number of LBGs, ignoring uniformity.

    Parameters
    ----------
    m5 : maf.MetricBundle
        MAF metric bundle containing 5-sigma limiting magnitudes in the
        specified band.
    band : str
        The band corresponding to the m5 metrics
    snr_floor: float, default=3
        The minimum SNR in the dropout band.
    dropout: float, default=1
        The change in color required for dropout selection.

    Returns
    -------
    float
        Total number of LBGs detected across the footprint, irrespective
        of uniformity.
    """
    # Get the limiting magnitudes from the metric bundle
    m5_array = m5.metric_values[m5.metric_values.mask == False].data

    # Create an interpolation grid for number density
    m_grid = np.linspace(m5_array.min(), m5_array.max(), 1_000)
    n_grid = number_density(
        m5=m_grid,
        band=band,
        snr_floor=snr_floor,
        dropout=dropout,
    )

    # Interpolate densities
    n = np.interp(m5_array, m_grid, n_grid)

    # Multiply by pixel area
    pix_area = hp.nside2pixarea(m5.slicer.nside, degrees=True)
    N = n * pix_area

    # Sum over all pixels
    return N.sum()
