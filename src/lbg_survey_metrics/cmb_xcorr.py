import numpy as np
import pyccl as ccl

from .constants import *
from .galaxy_distribution import number_density, redshift_distribution
from .utils import get_lensing_noise

# Create LCDM Cosmology
cosmo = ccl.CosmologyVanillaLCDM()

# Load lensing noise
ell, Nkk = get_lensing_noise()


def create_lbg_tracer(m5: float, band: str) -> ccl.tracers.NzTracer:
    """Create number density tracer for LBG dropouts.

    Parameters
    ----------
    m5: float
        5-sigma limit in the detection band
    band: str
        Name of dropout band

    Returns
    -------
    ccl.tracers.NzTracer
        Number counts tracer for LBGs
    """
    # Get the redshift distribution
    _z, _pz = redshift_distribution(m5=m5, band=band)

    # Increase redshift sample density
    z = np.linspace(1, 8, 1000)
    pz = np.interp(z, _z, _pz)

    # Create the tracer
    tracer = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z, pz),
        bias=(z, 0.28 * (1 + z) ** 1.6),
    )

    return tracer


def calc_cross_spectra(
    m5: float, band: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate angular cross-spectra of LBGs and CMB Lensing.

    Parameters
    ----------
    m5: float
        5-sigma limit in the detection band
    band: str
        Name of dropout band

    Returns
    -------
    np.ndarray
        Set of multipoles, ell
    np.ndarray
        Cgg -- LBG autospectrum
    np.ndarray
        Ckg -- LBG x CMB Lensing spectrum
    np.ndarray
        Ckk -- CMB Lensing autospectrum

    """
    # Create tracers
    lbg_tracer = create_lbg_tracer(m5=m5, band=band)
    cmb_lensing = ccl.CMBLensingTracer(cosmo, z_source=1100)

    # Calculate cross-spectra
    Cgg = ccl.angular_cl(cosmo, lbg_tracer, lbg_tracer, ell)
    Ckg = ccl.angular_cl(cosmo, cmb_lensing, lbg_tracer, ell)
    Ckk = ccl.angular_cl(cosmo, cmb_lensing, cmb_lensing, ell)

    return ell, Cgg, Ckg, Ckk


def calc_LBGxCMB_snr(m5: float, band: str) -> float:
    """Calculate the SNR of LBG x CMB Lensing.

    Parameters
    ----------
    m5: float
        5-sigma limit in the detection band
    band: str
        Name of dropout band

    Returns
    -------
    float
        SNR of the LBG x CMB Lensing signal
    """
    # Calculate spectra
    ell, Cgg, Ckg, Ckk = calc_cross_spectra(m5=m5, band=band)

    # Get the number density
    n = number_density(m5=m5, band=band)

    # Calculate variance
    var = (
        ((Ckk + Nkk) * (Cgg + 1 / (n * deg2_per_ster)) + Ckg**2)
        / (2 * ell + 1)
        / (A_wfd / A_sky)
    )

    # Calculate weighted SNR
    snr = np.sqrt(np.sum(Ckg**2 / var))

    return snr
