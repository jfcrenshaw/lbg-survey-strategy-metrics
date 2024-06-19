import numpy as np

from lbg_survey_metrics.cmb_xcorr import calc_LBGxCMB_snr
from lbg_survey_metrics.galaxy_distribution import number_density, redshift_distribution
from lbg_survey_metrics.utils import *

# Setup the cache directory
if not cache_dir.exists():
    cache_dir.mkdir(parents=True)

# Limiting magnitude grid for caches below
# Limits are extreme so we can see CMB SNR plateau on both ends
m5 = np.linspace(22, 31, 100)

# Calculate redshift distributions
mean = {}
sig = {}
for i, band in enumerate("ugriz"):
    # Get redshift distributions for range of limiting magnitudes
    z, pz = redshift_distribution(m5, band)

    mean[band] = np.trapz(z * pz, z, axis=-1)

    std = np.sqrt(np.trapz((z[None, :] - mean[band][:, None]) ** 2 * pz, z, axis=-1))
    sig[band] = std / (1 + mean[band])

# Save cache
np.savez(cache_file_pz_stat, m5=m5, pz_mean=mean, pz_sig=sig)

# Calculate the CMB SNR across a wide range of number densities
n = {}
snr = {}
for band in "ugriz":
    n[band] = number_density(m5, band)
    snr[band] = np.array([calc_LBGxCMB_snr(m, band) for m in m5])

# Save cache
np.savez(cache_file_cmb_snr, m5=m5, n=n, snr=snr)

