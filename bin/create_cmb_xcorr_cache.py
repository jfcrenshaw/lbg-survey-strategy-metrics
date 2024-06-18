import numpy as np

from lbg_survey_metrics.cmb_xcorr import calc_LBGxCMB_snr
from lbg_survey_metrics.galaxy_distribution import number_density
from lbg_survey_metrics.utils import data_dir

# Calculate the SNR across a wide range of number densities
# WARNING: this cell takes a few minutes to run
m5 = np.linspace(22, 31, 100)
n = {}
snr = {}
for band in "ugr":
    n[band] = number_density(m5, band)
    snr[band] = [calc_LBGxCMB_snr(m, band) for m in m5]

# Save cache
np.savez(data_dir / "cmb_xcorr_snr_cache.npz", m5=m5, n=n, snr=snr)
