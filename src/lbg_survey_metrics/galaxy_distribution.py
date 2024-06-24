import astropy.units as u
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo

from .completeness import completeness
from .constants import deg2_per_ster
from .utils import data_dir

# Load the luminosity function params
_lf_params = pd.read_csv(
    data_dir / "inputs" / "lf_params.dat",
    sep="\s+",
).to_dict()["Value"]


def dpl_lf(M: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Double power law luminosity function.

    Using the evolving LF model of https://arxiv.org/abs/2207.02233

    Parameters
    ----------
    M: np.ndarray
        Absolute magnitude in detection band
    z: np.ndarray
        Redshift

    Returns
    -------
    np.ndarray
        Number density in units mag^-1 Mpc^-3
    """
    # Calculate DPL params at this redshift
    log_phi_star = _lf_params["Pg0"] + _lf_params["Pg1"] * (1 + z)
    M_star = _lf_params["Mg0"] + _lf_params["Mg1"] * (1 + z)
    alpha_p1 = _lf_params["Ag0"] + _lf_params["Ag1"] * (1 + z) + 1
    beta_p1 = _lf_params["Bg0"] + _lf_params["Bg1"] * (1 + z) + 1

    # Calculate LF
    norm = np.log(10) / 2.5 * 10 ** (log_phi_star)
    dM = M - M_star
    LF = norm / (10 ** (0.4 * alpha_p1 * dM) + 10 ** (0.4 * beta_p1 * dM))

    return LF


def dndz(
    m5: np.ndarray,
    band: str,
    m5_cut: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate projected number density per redshift.

    Parameters
    ----------
    m5: np.ndarray
        5-sigma limit in the detection band
    band: str
        Name of dropout band
    m5_cut: float or None
        Limiting magnitude to set detection threshold. If None, m5 is used.
        Setting this separately allows you to investigate non-uniformity.
        I.e., you can cut on a single magnitude, but the regular m5 parameter
        still varies the completeness function.

    Returns
    -------
    np.ndarray
        Redshift grid
    np.ndarray
        Number density of galaxies in each redshift bin
    """
    # Make sure m5 is an array
    m5 = np.atleast_1d(m5)

    # If m5_cut is None, just use m5
    m5_cut = m5 if m5_cut is None else m5_cut

    # Create grid of apparent magnitudes and redshifts
    m = np.linspace(20, m5_cut, 1_001).T[..., None]
    z = np.arange(2, 8, 0.1)[None, None, :]

    # Convert apparent to absolute magnitude
    DL = cosmo.luminosity_distance(z).to(u.pc).value  # Lum. Dist. in pc
    M = m - 5 * np.log10(DL / 10) + 2.5 * np.log10(1 + z)

    # Calculate luminosity function
    LF = dpl_lf(M, z)

    # Calculate completeness
    C = completeness(m, z, band, m5[:, None, None])

    # Calculate dV/dz (Mpc^-3 deg^-2)
    dVdz = cosmo.differential_comoving_volume(z).value / deg2_per_ster

    # Integrate luminosity function to get number density of galaxies
    # in each redshift bin
    nz = np.trapz(LF * C * dVdz, M, axis=1)

    return z.squeeze(), nz


def number_density(
    m5: np.ndarray,
    band: str,
    m5_cut: np.ndarray | None = None,
) -> float:
    """Calculate number density per deg^2.

    Parameters
    ----------
    m5: np.ndarray
        5-sigma limit in the detection band
    band: str
        Name of dropout band
    m5_cut: float
        Limiting magnitude to set detection threshold. If None, then m5 is
        used. Setting this separately allows you to investigate non-uniformity.
        I.e., you can cut on a single magnitude, but the regular m5 parameter
        still varies the completeness function.

    Returns
    -------
    float
        The total number density of galaxies in units deg^-2.
    """
    # Get number of galaxies in each redshift bin
    z, nz = dndz(m5=m5, band=band, m5_cut=m5_cut)

    # Integrate over redshift bins
    n = np.trapz(nz, z, axis=-1)

    return n


def redshift_distribution(
    m5: float,
    band: str,
    m5_cut: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the redshift distribution of the dropout sample.

    Parameters
    ----------
    m5: float
        5-sigma limit in the detection band
    band: str
        Name of dropout band
    m5_cut: float
        Limiting magnitude to set detection threshold. If None, then m5 is
        used. Setting this separately allows you to investigate non-uniformity.
        I.e., you can cut on a single magnitude, but the regular m5 parameter
        still varies the completeness function.

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
    z, nz = dndz(m5=m5, band=band, m5_cut=m5_cut)

    # Integrate over redshift bins
    n = np.trapz(nz, z, axis=-1)

    # Normalize redshift distribution
    pz = nz / n[:, None]

    return z, pz.squeeze()
