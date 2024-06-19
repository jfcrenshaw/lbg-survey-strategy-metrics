import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.modeling.models import Schechter1D

from .constants import deg2_per_ster
from .utils import get_completeness


# Double power-law params from Table 6 of https://arxiv.org/abs/2108.01090
lf_params = {
    "u": {
        "M_star": -21.30,
        "log_phi_star": -3.23,
        "alpha": -1.89,
        "beta": -4.78,
    },
    "g": {
        "M_star": -20.99,
        "log_phi_star": -3.00,
        "alpha": -1.86,
        "beta": -4.77,
    },
    "r": {
        "M_star": -21.54,
        "log_phi_star": -3.63,
        "alpha": -2.01,
        "beta": -4.91,
    },
    "i": {
        "M_star": -21.03,
        "log_phi_star": -3.52,
        "alpha": -2.08,
        "beta": -4.57,
    },
    "z": {
        "M_star": -20.12,
        "log_phi_star": -3.05,
        "alpha": -1.89,
        "beta": -3.81,
    },
}


def dpl_lf(
    M_grid: np.ndarray,
    band: str,
) -> np.ndarray:
    """Double power law luminosity function.

    Uses parameters from Table 6 of https://arxiv.org/abs/2108.01090

    Parameters
    ----------
    M_grid : np.ndarray
        Absolute magnitudes
    band : str
        Name of Rubin band.

    Returns
    -------
    np.ndarray
        Number density in units mag^-1 Mpc^-3
    """
    # Unpack parameters
    M_star = lf_params[band]["M_star"]
    phi_star = 10 ** lf_params[band]["log_phi_star"]
    alpha_p1 = lf_params[band]["alpha"] + 1
    beta_p1 = lf_params[band]["beta"] + 1

    # Calculate LF
    dM = M_grid - M_star
    LF = np.log(10)/2.5 * phi_star / (10 ** (0.4 * alpha_p1 * dM) + 10 ** (0.4 * beta_p1 * dM))

    return LF


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

    # Convert observed magnitudes to absolute
    DL = cosmo.luminosity_distance(z).to(u.pc).value  # Lum. Dist. in pc
    M_grid = det_grid[..., None] - 5 * np.log10(DL / 10) + 2.5 * np.log10(1 + z)

    # Calculate luminosity function
    LF = dpl_lf(M_grid, band)

    # Calculate dV/dz (Mpc^-3 deg^-2)
    dV = cosmo.differential_comoving_volume(z).value / deg2_per_ster

    # Integrate luminosity function to get number density of galaxies
    # in each redshift bin
    nz = np.trapz(dV * LF * C, M_grid, axis=0)

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
