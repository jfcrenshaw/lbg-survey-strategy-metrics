import numpy as np
import pyccl as ccl

from .constants import *
from .galaxy_distribution import number_density, redshift_distribution
from .utils import *


# Create LCDM Cosmology
cosmo = ccl.CosmologyVanillaLCDM()

# Minimum-variance baseline forecast for SO lensing noise
lensing_noise = np.genfromtxt(
    data_dir / "inputs/nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat"
)
ell = lensing_noise[:, 0]
Nkk = lensing_noise[:, 7]


def create_lbg_tracer(
    m5: float,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> ccl.tracers.NzTracer:
    """Create number density tracer for LBG dropouts.

    Parameters
    ----------
    m5: float
        The 5-sigma limit in the dropout band
    band: str
        Either "u", "g", or "r".
    snr_floor: float, default=3
        The minimum SNR in the dropout band.
    dropout: float, default=1
        The change in color required for dropout selection.

    Returns
    -------
    ccl.tracers.NzTracer
        Number counts tracer for LBGs
    """
    # Get the redshift distribution
    _z, _pz = redshift_distribution(
        m5=m5, band=band, snr_floor=snr_floor, dropout=dropout
    )

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
    m5: float,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate angular cross-spectra of LBGs and CMB Lensing.

    Parameters
    ----------
    m5: float
        The 5-sigma limit in the dropout band
    band: str
        Either "u", "g", or "r".
    snr_floor: float, default=3
        The minimum SNR in the dropout band.
    dropout: float, default=1
        The change in color required for dropout selection.

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
    lbg_tracer = create_lbg_tracer(m5, band, snr_floor, dropout)
    cmb_lensing = ccl.CMBLensingTracer(cosmo, z_source=1100)

    # Calculate cross-spectra
    Cgg = ccl.angular_cl(cosmo, lbg_tracer, lbg_tracer, ell)
    Ckg = ccl.angular_cl(cosmo, cmb_lensing, lbg_tracer, ell)
    Ckk = ccl.angular_cl(cosmo, cmb_lensing, cmb_lensing, ell)

    return ell, Cgg, Ckg, Ckk


def calc_LBGxCMB_snr(
    m5: float,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate the SNR of LBG x CMB Lensing.

    Parameters
    ----------
    m5: float
        The 5-sigma limit in the dropout band
    band: str
        Either "u", "g", or "r".
    snr_floor: float, default=3
        The minimum SNR in the dropout band.
    dropout: float, default=1
        The change in color required for dropout selection.

    Returns
    -------
    float
        SNR of the LBG x CMB Lensing signal
    """
    # Calculate spectra
    ell, Cgg, Ckg, Ckk = calc_cross_spectra(m5, band, snr_floor, dropout)

    # Get the number density
    n = number_density(m5, band, snr_floor, dropout)

    # Calculate variance
    var = (
        ((Ckk + Nkk) * (Cgg + 1 / (n * deg2_per_ster)) + Ckg**2)
        / (2 * ell + 1)
        / (A_wfd / A_sky)
    )

    # Calculate weighted SNR
    snr = np.sqrt(np.sum(Ckg**2 / var))

    return snr
