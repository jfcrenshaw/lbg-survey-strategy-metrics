import numpy as np

from lbg_survey_metrics.cmb_xcorr import calc_LBGxCMB_snr
from lbg_survey_metrics.galaxy_distribution import number_density, redshift_distribution
from lbg_survey_metrics.utils import (
    cache_dir,
    cache_file_cmb_snr,
    cache_file_number_density,
    cache_file_pz_stat,
)

# Setup the cache directory
if not cache_dir.exists():
    cache_dir.mkdir(parents=True)

# Cache number density
m5 = np.arange(22, 29, 0.1)
cache_number_density = {}
for band in "ugriz":
    cache_number_density[band] = np.array(
        [number_density(m5, band, m5_cut) for m5_cut in m5]
    ).T
np.savez(cache_file_number_density, m5=m5, n=cache_number_density)

# Cache photo-z distribution statistics
m5 = np.arange(22, 29, 0.1)
cache_pz_mean = {}
cache_pz_sig = {}
for band in "ugriz":
    mean = []
    sig = []
    for m5_cut in m5:
        z, pz = redshift_distribution(m5, band, m5_cut)
        mean_i = np.trapz(z * pz, z, axis=-1)
        sig_i = np.sqrt(np.trapz((z[None, :] - mean_i[:, None]) ** 2 * pz, z, axis=-1))
        sig_i /= 1 + mean_i
        mean.append(mean_i)
        sig.append(sig_i)
    cache_pz_mean[band] = np.array(mean).T
    cache_pz_sig[band] = np.array(sig).T
np.savez(cache_file_pz_stat, m5=m5, pz_mean=cache_pz_mean, pz_sig=cache_pz_sig)

# Cache CMB SNR
m5 = np.linspace(22, 31, 100)
n = {}
snr = {}
for band in "ugriz":
    n[band] = number_density(m5, band)
    snr[band] = np.array([calc_LBGxCMB_snr(m, band) for m in m5])
np.savez(cache_file_cmb_snr, m5=m5, n=n, snr=snr)
