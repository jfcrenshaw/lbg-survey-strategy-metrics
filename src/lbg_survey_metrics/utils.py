from pathlib import Path

import numpy as np

# Define the data directory
data_dir = Path(__file__).parents[2] / "data"


def get_completeness(band: str) -> tuple[np.ndarray, np.ndarray]:
    """Get the completeness curve for the band.

    The completeness is defined as the fraction of galaxies
    that pass the LBG cut.

    Parameters
    ----------
    band: str
        The name of the band

    Returns
    -------
    np.ndarray
        Redshift grid for completeness curve
    np.ndarray
        Completeness
    """
    return np.genfromtxt(data_dir / "inputs" / f"completeness_{band}.dat", unpack=True)
