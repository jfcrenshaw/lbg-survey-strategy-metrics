import itertools

import numpy as np
import pandas as pd
import rubin_sim.maf as maf

from lbg_survey_metrics.metrics import (
    density_for_quantile,
    fwfd_for_density,
    optimize_snr,
    total_lbgs,
)
from lbg_survey_metrics.utils import data_dir

# Directory where m5 files are stored
m5_dir = data_dir / "rubin_sim_m5"

# All params associated with different u band strategy m5 files
bands = list("ugriz")
years = [1, 4, 7, 10]
scales = [1.0, 1.1, 1.2, 1.5]
expts = [30, 38, 45, 60]

# Metadata on selection for bands
det_band = dict(u="r", g="i", r="z", i="y", z="y")
cut_on_drop = dict(u=True, g=True, r=False, i=False, z=False)

# Loop over all strategies and calculate metrics
u_strat = []
for band, year, scale, expt in itertools.product(bands, years, scales, expts):
    try:
        # Load depth in dropout band
        drop_file = f"internal_u_expt{expt}_nscale{str(scale).replace('.', '_')}v3_4_{year}yrs_ExgalM5_{band}.npz"
        m5_drop = maf.MetricBundle.load(m5_dir / drop_file).metric_values

        # Load depth in detection band
        det_file = f"internal_u_expt{expt}_nscale{str(scale).replace('.', '_')}v3_4_{year}yrs_ExgalM5_{det_band[band]}.npz"
        m5_det = maf.MetricBundle.load(m5_dir / det_file).metric_values
    except:
        u_strat.append([band, year, scale, expt] + 9 * [np.nan])
        continue

    # Calculate densities for different depth quantiles
    n_5 = density_for_quantile(
        m5_drop, m5_det, band, quantile_cut=0.05, cut_on_drop=cut_on_drop[band]
    )
    n_25 = density_for_quantile(
        m5_drop, m5_det, band, quantile_cut=0.25, cut_on_drop=cut_on_drop[band]
    )

    # Calculate f_wfd for different mean number densities
    fwfd_100 = fwfd_for_density(
        m5_drop, m5_det, band, n=100, cut_on_drop=cut_on_drop[band]
    )
    fwfd_400 = fwfd_for_density(
        m5_drop, m5_det, band, n=400, cut_on_drop=cut_on_drop[band]
    )
    fwfd_1000 = fwfd_for_density(
        m5_drop, m5_det, band, n=1000, cut_on_drop=cut_on_drop[band]
    )

    # Optimize the SNR
    snr_optim, n_optim, fwfd_optim = optimize_snr(
        m5_drop, m5_det, band, cut_on_drop=cut_on_drop[band]
    )

    # Calculate the total number of LBGs, ignoring uniformity
    Ntot = total_lbgs(m5_drop, m5_det, band, cut_on_drop=cut_on_drop[band])

    u_strat.append(
        [
            band,
            year,
            scale,
            expt,
            n_5,
            n_25,
            fwfd_100,
            fwfd_400,
            fwfd_1000,
            snr_optim,
            n_optim,
            fwfd_optim,
            Ntot,
        ]
    )

# Convert metrics to pandas dataframe
metrics = pd.DataFrame(
    u_strat,
    columns=[
        "band",
        "year",
        "scale",
        "expt",
        "n_5",
        "n_25",
        "fwfd_100",
        "fwfd_400",
        "fwfd_1000",
        "snr_optim",
        "n_optim",
        "fwfd_optim",
        "Ntot",
    ],
)

# Save metrics
metrics.to_pickle(data_dir / "metrics_uband_strategy.pkl")
