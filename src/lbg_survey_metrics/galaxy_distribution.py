import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.modeling.models import Schechter1D

from .constants import deg2_per_ster
from .utils import get_completeness

# Define Schechter LF params for each sample
# from Table 6 of arXiv:2108.01090
lf_params = {
    "u": {
        "log_phi_star": -2.84,
        "M_star": -20.91,
        "alpha": -1.68,
    },
    "g": {
        "log_phi_star": -2.52,
        "M_star": -20.49,
        "alpha": -1.59,
    },
    "r": {
        "log_phi_star": -3.16,
        "M_star": -21.09,
        "alpha": -1.79,
    },
}


def schecter_lf(
    m_grid: np.ndarray,
    log_phi_star: float = -2.84,
    M_star: float = -20.91,
    alpha: float = -1.68,
    redshift: float | np.ndarray = 3,
) -> np.ndarray:
    """Schecter Luminosity Function on grid of apparent magnitudes.

    Defaults are for z~3 u-dropout Luminosity Function from Table 6
    of Harikane et al. 2022.

    Parameters
    ----------
    m_grid: np.ndarray
        Array of apparent AB magnitudes on which to calculate the
        luminosity function.
    log_phi_star: float, default=-2.84
        Natural log of phi_star, the normalization of the luminosity
        function in units of mag^-1 Mpc^-3
    M_star: float, default=-20.91
        The characteristic absolute magnitude where the power-law form
        of the luminosity function cuts off.
    alpha: float, default=-1.68
        The power law index, also known as the faint-end slope.
    redshift: float or np.ndarray, default=3
        Redshift used for converting apparent magnitudes into absolute
        magnitudes.

    Returns
    -------
    np.ndarray
        Number density in units mag^-1 Mpc^-3
    """
    # Convert observed magnitudes to absolute
    DL = cosmo.luminosity_distance(redshift).to(u.pc).value  # Lum. Dist. in pc
    M_grid = m_grid[..., None] - 5 * np.log10(DL / 10) + 2.5 * np.log10(1 + redshift)

    # Calculate luminosity function in absolute magnitudes
    schechter = Schechter1D(10**log_phi_star, M_star, alpha)

    return schechter(M_grid)


def _num_den_per_z(
    m5: np.ndarray,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the un-normalized redshift distribution.

    Parameters
    ----------
    m5: np.ndarray
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
        Redshift grid
    np.ndarray
        Number density of galaxies in each redshift bin
    """
    # Create apparent magnitude grid
    drop_cut = m5 + 2.5 * np.log10(5 / snr_floor)
    det_cut = drop_cut - dropout
    det_grid = np.linspace(20, det_cut, 1_001)

    # Load completeness
    z, C = get_completeness(band)

    # Calculate luminosity function
    LF = schecter_lf(det_grid, redshift=z, **lf_params[band])

    # Calculate differential comoving volume (Mpc^-3 deg^-2)
    dV = cosmo.differential_comoving_volume(z).value / deg2_per_ster

    # Integrate luminosity function to get number density of galaxies
    # in each redshift bin
    nz = np.trapz(dV * LF * C, det_grid[..., None], axis=0)

    return z, nz


def number_density(
    m5: np.ndarray,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate number density per deg^2.

    Parameters
    ----------
    m5: np.ndarray
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
        The total number density of galaxies in units deg^-2.
    """
    # Get number of galaxies in each redshift bin
    z, nz = _num_den_per_z(m5=m5, band=band, snr_floor=snr_floor, dropout=dropout)

    # Integrate over redshift bins
    n = np.trapz(nz, z, axis=-1)

    return n


def redshift_distribution(
    m5: float,
    band: str,
    snr_floor: float = 3,
    dropout: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the redshift distribution of the dropout sample.

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
        Redshift array
    np.ndarray
        Normalized redshift distribution
    """
    # Make sure m5 is an array
    m5 = np.atleast_1d(m5)

    # Get number of galaxies in each redshift bin
    z, nz = _num_den_per_z(m5=m5, band=band, snr_floor=snr_floor, dropout=dropout)

    # Integrate over redshift bins
    n = np.trapz(nz, z, axis=-1)

    # Normalize redshift distribution
    pz = nz / n[:, None]

    return z, pz.squeeze()
