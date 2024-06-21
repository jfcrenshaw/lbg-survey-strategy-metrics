import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .utils import data_dir

# Create dictionary for completeness metadata
_completeness_meta = {
    band: dict(m5=float(m5))
    for band, m5 in np.genfromtxt(
        data_dir / "inputs" / "completeness_m5.dat", dtype=str
    )
}

# Load completeness tables
_completeness_data = {}
for band in "ugriz":
    # Load data
    df = pd.read_csv(
        data_dir / "inputs" / f"completeness_{band}.dat",
        sep="  ",
        header=5,
        engine="python",
    )

    # Calculate differential mags wrt the limiting mag
    dm = df.columns.to_numpy(dtype=float) - _completeness_meta[band]["m5"]
    df.columns = dm

    _completeness_data[band] = df


# Create completeness interpolators
_completeness_interpolators = {}
for band in "ugriz":
    # Get the completeness table
    df = _completeness_data[band]

    # Get redshift and magnitude grids
    z = df.index.to_numpy()
    dm = df.columns.to_numpy(dtype=float)

    # Create the interpolator
    _completeness_interpolators[band] = RegularGridInterpolator(
        (z, dm),
        df.values,
        bounds_error=False,
        fill_value=None,  # linear extrapolation
    )

    # Save zmin, zmax in the metadata
    _completeness_meta[band]["zmin"] = z.min()
    _completeness_meta[band]["zmax"] = z.max()


def completeness(m, z, band, m5):
    """Calculate completeness.

    Parameters
    ----------
    m: np.ndarray
        Apparent magnitudes
    z: np.ndarray
        Redshifts
    band: str
        Name of detection band
    m5: float
        Limiting magnitude in detection band

    Returns
    -------
    np.ndarray
        Completenesses evaluated on the m, z grid
    """
    # Calculate completeness
    C = np.clip(_completeness_interpolators[band]((z, m - m5)), 0, 1)

    # Set vals outside redshift range to zero
    zmin = _completeness_meta[band]["zmin"]
    zmax = _completeness_meta[band]["zmax"]
    C *= (z >= zmin) & (z <= zmax)

    return C
