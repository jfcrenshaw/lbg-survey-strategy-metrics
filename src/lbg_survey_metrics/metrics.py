import healpy as hp
import numpy as np
import rubin_sim.maf as maf
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar

from .constants import A_wfd
from .utils import cache_cmb_snr, cache_number_density, cache_pz_stat

# Create 2D interpolators
_number_density_interpolator = {
    band: RegularGridInterpolator(
        (cache_number_density["m5"], cache_number_density["m5"]),
        cache_number_density["n"][band],
        bounds_error=False,
        fill_value=0,
    )
    for band in cache_number_density["n"]
}
_pz_mean_interpolator = {
    band: RegularGridInterpolator(
        (cache_pz_stat["m5"], cache_pz_stat["m5"]),
        cache_pz_stat["pz_mean"][band],
        bounds_error=False,
        fill_value=0,
    )
    for band in cache_pz_stat["pz_mean"]
}
_pz_sig_interpolator = {
    band: RegularGridInterpolator(
        (cache_pz_stat["m5"], cache_pz_stat["m5"]),
        cache_pz_stat["pz_sig"][band],
        bounds_error=False,
        fill_value=0,
    )
    for band in cache_pz_stat["pz_sig"]
}


def get_m5_ranges(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    quantile_cut: float = 0.25,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> dict:
    """Get m5 ranges, assuming the specified cuts

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    quantile_cut: float, default=0.25
        Quantile of m5 magnitudes to make depth cut.
    cut_on_drop: bool, default=True
        Whether create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

    Returns
    -------
    float
        Minimum of detection band range, given the specified cuts
    float
        Maximum of detection band range, given the specified cuts
    """
    # Create cut
    if cut_on_drop:
        drop_cut = np.quantile(m5_drop[m5_drop.mask == False].data, quantile_cut)
        det_cut = drop_cut + 2.5 * np.log10(5 / snr_floor) - dropout
    else:
        drop_cut = -np.inf
        det_cut = np.quantile(m5_det[m5_det.mask == False].data, quantile_cut)

    # Create a mask for the maps
    mask = m5_drop.mask | m5_det.mask | (m5_drop < drop_cut) | (m5_det < det_cut)

    ranges = {
        "drop_cut": drop_cut,
        "drop_min": m5_drop[mask == False].min(),
        "drop_max": m5_drop[mask == False].max(),
        "det_cut": det_cut,
        "det_min": m5_det[mask == False].min(),
        "det_max": m5_det[mask == False].max(),
        "mask": mask,
    }

    return ranges


def _create_map(
    interpolator: RegularGridInterpolator,
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    quantile_cut: float,
    cut_on_drop: bool,
    snr_floor: float,
    dropout: float,
) -> np.ma.MaskedArray:
    """Create the map using cuts and interpolator.

    Parameters
    ----------
    interpolator: scipy.interpolate.RegularGridInterpolator
        Interpolator for the metric to evaluate across the map.
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    quantile_cut: float
        Quantile of m5 magnitudes to make depth cut.
    cut_on_drop: bool
        Whether create the cut from the dropout band.
    snr_floor: float
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

    Returns
    -------
    np.ma.MaskedArray
        Mask of relative density variations.
    """
    # Get band ranges
    ranges = get_m5_ranges(
        m5_drop=m5_drop,
        m5_det=m5_det,
        quantile_cut=quantile_cut,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )

    # Evaluate the metric
    metric = np.ma.array(interpolator((m5_det, ranges["det_cut"])), fill_value=np.nan)
    metric.mask = ranges["mask"]

    return metric


def map_number_density(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    band: str,
    quantile_cut: float = 0.25,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> np.ma.MaskedArray:
    """Return the LBG number density map.

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    band: str
        Name of the dropout band
    quantile_cut: float, default=0.25
        Quantile of m5 magnitudes to make depth cut.
    cut_on_drop: bool, default=True
        Whether to create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

    Returns
    -------
    np.ma.MaskedArray
        Mask of relative density variations.
    """
    # Evaluate number density across map
    return _create_map(
        interpolator=_number_density_interpolator[band],
        m5_drop=m5_drop,
        m5_det=m5_det,
        quantile_cut=quantile_cut,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )


def map_pz_mean(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    band: str,
    quantile_cut: float = 0.25,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> np.ma.MaskedArray:
    """Return the mean redshift map.

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    band: str
        Name of the dropout band
    quantile_cut: float, default=0.25
        Quantile of m5 magnitudes to make depth cut.
    cut_on_drop: bool, default=True
        Whether create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

    Returns
    -------
    np.ma.MaskedArray
        Mask of relative density variations.
    """
    # Evaluate mean redshift across map
    return _create_map(
        interpolator=_pz_mean_interpolator[band],
        m5_drop=m5_drop,
        m5_det=m5_det,
        quantile_cut=quantile_cut,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )


def map_pz_sig(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    band: str,
    quantile_cut: float = 0.25,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> np.ma.MaskedArray:
    """Return the redshift sigma map.

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    band: str
        Name of the dropout band
    quantile_cut: float, default=0.25
        Quantile of m5 magnitudes to make depth cut.
    cut_on_drop: bool, default=True
        Whether create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

    Returns
    -------
    np.ma.MaskedArray
        Mask of relative density variations.
    """
    # Evaluate redshift sigma across map
    return _create_map(
        interpolator=_pz_sig_interpolator[band],
        m5_drop=m5_drop,
        m5_det=m5_det,
        quantile_cut=quantile_cut,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )


def density_for_quantile(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    band: str,
    quantile_cut: float = 0.25,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Return the mean LBG number density at the quantile cut.

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    band: str
        Name of the dropout band
    quantile_cut: float, default=0.25
        Quantile of m5 magnitudes to make depth cut.
    cut_on_drop: bool, default=True
        Whether to create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

    Returns
    -------
    float
        Average LBG number density in deg^-2
    """
    n = map_number_density(
        m5_drop=m5_drop,
        m5_det=m5_det,
        band=band,
        quantile_cut=quantile_cut,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )
    return n.mean()


def _fwfd_from_quantile(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    quantile_cut: float = 0.25,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate f_wfd corresponding to the quantile of m5.

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    quantile_cut: float, default=0.25
        Quantile of m5 magnitudes to make depth cut.
    cut_on_drop: bool, default=True
        Whether create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

    Returns
    -------
    float
        Fraction of fiducial WFD survey area that is deeper than the m5
        corresponding to the quantile
    """
    # Fraction deeper than m5 quantile
    f_deeper = 1 - quantile_cut

    # Get limits
    ranges = get_m5_ranges(
        m5_drop=m5_drop,
        m5_det=m5_det,
        quantile_cut=quantile_cut,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )

    # Number of pixels passing cuts
    npix = (~ranges["mask"]).sum()

    # Area of each pixel
    nside = hp.npix2nside(m5_drop.size)
    pix_area = hp.nside2pixarea(nside, degrees=True)

    # Total area deeper than m5 quantile
    area_deeper = f_deeper * npix * pix_area

    # Fraction of fiducial WFD deeper than cut
    f_wfd = area_deeper / A_wfd

    return f_wfd


def fwfd_for_density(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    band: str,
    n: float,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate fraction of WFD deep enough for requested number density.

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    band: str
        Name of the dropout band
    n: float
        Requested mean number density of LBGs (deg^-2)
    cut_on_drop: bool, default=True
        Whether to create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.
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
        lambda quantile_cut: np.abs(
            n
            - density_for_quantile(
                m5_drop=m5_drop,
                m5_det=m5_det,
                band=band,
                quantile_cut=quantile_cut,
                cut_on_drop=cut_on_drop,
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
    f_wfd = _fwfd_from_quantile(
        m5_drop=m5_drop,
        m5_det=m5_det,
        quantile_cut=res.x,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )

    return f_wfd


def _calc_snr(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    band: str,
    quantile_cut: float = 0.25,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate CMB x-corr corresponding to quantile.

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    band: str
        Name of the dropout band
    quantile_cut: float, default=0.25
        Quantile of m5 magnitudes to make depth cut.
    cut_on_drop: bool, default=True
        Whether to create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

    Returns
    -------
    float
        Signal-to-noise ratio of LBG x CMB Lensing correlation
    """
    # Get number density
    n = density_for_quantile(
        m5_drop=m5_drop,
        m5_det=m5_det,
        band=band,
        quantile_cut=quantile_cut,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )

    # Calculate f_wfd
    f_wfd = _fwfd_from_quantile(
        m5_drop=m5_drop,
        m5_det=m5_det,
        quantile_cut=quantile_cut,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )

    # Calculate SNR
    n_grid = cache_cmb_snr["n"][band]
    snr_grid = cache_cmb_snr["snr"][band]
    snr = np.interp(n, n_grid, snr_grid * np.sqrt(f_wfd))

    return snr


def optimize_snr(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    band: str,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> tuple[float, float, float]:
    """Optimize LBG x CMB Lensing correlation by varying number density and f_wfd.

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    band: str
        Name of the dropout band
    cut_on_drop: bool, default=True
        Whether to create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

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
        lambda quantile_cut: -_calc_snr(
            m5_drop=m5_drop,
            m5_det=m5_det,
            band=band,
            quantile_cut=quantile_cut,
            cut_on_drop=cut_on_drop,
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
        m5_drop=m5_drop,
        m5_det=m5_det,
        band=band,
        quantile_cut=res.x,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )
    f_wfd = _fwfd_from_quantile(
        m5_drop=m5_drop,
        m5_det=m5_det,
        quantile_cut=res.x,
        cut_on_drop=cut_on_drop,
        snr_floor=snr_floor,
        dropout=dropout,
    )

    return snr, n, f_wfd


def total_lbgs(
    m5_drop: np.ma.MaskedArray,
    m5_det: np.ma.MaskedArray,
    band: str,
    cut_on_drop: bool = True,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate total number of LBGs, ignoring uniformity.

    Parameters
    ----------
    m5_drop: np.ma.MaskedArray
        Map of 5-sigma depths in the dropout band.
    m5_det: np.ma.MaskedArray
        Map of 5-sigma depths in the detection band.
    band: str
        Name of the dropout band
    cut_on_drop: bool, default=True
        Whether to create the cut from the dropout band.
    snr_floor: float, default=3
        Minimum SNR in the dropout band.
        Only relevant if cut_on_drop is True.
    dropout: float, default=1
        Minimum change in color for a dropout.
        Only relevant if cut_on_drop is True.

    Returns
    -------
    float
        Total number of LBGs detected across the footprint, irrespective
        of uniformity.
    """
    if cut_on_drop:
        drop_cut = m5_drop
        det_cut = drop_cut + 2.5 * np.log10(5 / snr_floor) - dropout
    else:
        drop_cut = -np.inf * m5_drop
        det_cut = m5_det

    # Interpolate densities
    n = _number_density_interpolator[band]((det_cut, det_cut))

    # Multiply by pixel area
    nside = hp.npix2nside(m5_drop.size)
    pix_area = hp.nside2pixarea(nside, degrees=True)
    N = n * pix_area

    # Sum over all pixels
    return np.nansum(N)
