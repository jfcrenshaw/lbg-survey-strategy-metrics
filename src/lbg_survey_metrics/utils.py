import os
from pathlib import Path

import numpy as np

# Define the data directory
data_dir = Path(__file__).parents[2] / "data"


# Set the bandpass directory
try:
    bandpass_dir = Path(os.environ["RUBIN_SIM_DATA_DIR"]) / "throughputs" / "baseline"
except KeyError:
    bandpass_dir = None


def get_bandpass(band: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the bandpass for the band.

    Parameters
    ----------
    band: str
        The name of the band

    Returns
    -------
    np.ndarray
        Wavelength grid in Angstroms
    np.ndarray
        Bandpass transmission
    """
    # Check the bandpass directory exists
    if bandpass_dir is None or not bandpass_dir.exists():
        raise RuntimeError(
            "The bandpass directory does not seem to exist. "
            "Perhaps you need to bin bin/setup_rubin_sim.sh."
        )

    # Load from file
    w, R = np.genfromtxt(bandpass_dir / f"total_{band}.dat", unpack=True)

    # Convert nm -> Angstrom
    w *= 10

    return w, R


def get_lensing_noise() -> tuple[np.ndarray, np.ndarray]:
    """Load the lensing noise.

    This is the minimum-variance baseline forcast for the Simons Obs.

    Returns
    -------
    np.ndarray
        A grid of multipoles (ell)
    np.ndarray
        Lensing noise as a function of ell (Nkk)
    """
    lensing_noise = np.genfromtxt(
        data_dir / "inputs/nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat"
    )
    ell = lensing_noise[:, 0]
    Nkk = lensing_noise[:, 7]

    return ell, Nkk


# Set the cache directory
cache_dir = data_dir / "caches"

# Load the caches
cache_file_pz_stat = cache_dir / "cache_pz_stat.npz"
if cache_file_pz_stat.exists():
    _cache_pz_stat = np.load(cache_file_pz_stat, allow_pickle=True)
    cache_pz_stat = {
        "m5": _cache_pz_stat["m5"],
        "pz_mean": _cache_pz_stat["pz_mean"].tolist(),
        "pz_sig": _cache_pz_stat["pz_sig"].tolist(),
    }
else:
    cache_pz_stat = None


cache_file_cmb_snr = cache_dir / "cache_cmb_xcorr_snr.npz"
if cache_file_cmb_snr.exists():
    _cache_cmb_snr = np.load(cache_file_cmb_snr, allow_pickle=True)
    cache_cmb_snr = {
        "m5": _cache_cmb_snr["m5"],
        "n": _cache_cmb_snr["n"].tolist(),
        "snr": _cache_cmb_snr["snr"].tolist(),
    }
else:
    cache_cmb_snr = None
