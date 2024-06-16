import numpy as np
from astropy.modeling.models import Schechter1D
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


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
    redshift: float = 3,
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
    redshift: float, default=3
        Redshift used for converting apparent magnitudes into absolute
        magnitudes.

    Returns
    -------
    np.ndarray
        Number density in units mag^-1 Mpc^-3
    """
    # Convert observed magnitudes to absolute
    DL = cosmo.luminosity_distance(redshift).to(u.pc).value  # Lum. Dist. in pc
    M_grid = m_grid - 5 * np.log10(DL / 10) + 2.5 * np.log10(1 + redshift)

    # Calculate luminosity function in absolute magnitudes
    schechter = Schechter1D(10**log_phi_star, M_star, alpha)

    return schechter(M_grid)


def number_density(
    u5: float,
    band: str,
    redshift: float,
    dz: float = 1,
    snr_floor: float = 3,
    dropout: float = 1,
) -> float:
    """Calculate number density per deg^2.

    Parameters
    ----------
    u5: float
        The u band 5-sigma limit
    band: str
        Either "u", "g", or "r".
    redshift: float
        The central redshift used for evaluating comoving quantities.
    dz: float, default=1
        The full width of the redshift bin
    snr_floor: float, default=3
        The minimum SNR in the dropout band.
    dropout: float, default=1
        The change in color required for dropout selection.

    Returns
    -------
    float
        The total number density of galaxies in units deg^-2.
    """
    # Create apparent magnitude grid
    drop_cut = u5 + 2.5 * np.log10(5 / snr_floor)
    det_cut = drop_cut - dropout
    det_grid = np.linspace(20, det_cut)

    # Calculate the luminosity function
    LF = schecter_lf(det_grid, redshift=redshift, **lf_params[band])

    # Calculate comoving depth of redshift bin (Mpc)
    chi_far = cosmo.comoving_distance(redshift + dz / 2)
    chi_near = cosmo.comoving_distance(redshift - dz / 2)
    dchi = chi_far - chi_near

    # Calculate number density (mag^-1 Mpc^-2)
    n_dm = (LF / u.Mpc**3) * dchi

    # Convert to mag^-1 deg^-2
    deg_per_Mpc = cosmo.arcsec_per_kpc_comoving(redshift).to(u.deg / u.Mpc)
    n_dm /= deg_per_Mpc**2

    # Integrate the luminosity function
    n = np.trapz(n_dm, det_grid, axis=0)

    return n.value