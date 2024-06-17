# lbg-survey-strategy-metrics
Survey Strategy Metrics for Detecting LBGs with the Rubin Observatory

Installation:
```bash
mamba env create -f environment.yaml
mamba activate lbg-survey-strat
python -m ipykernel install --user --name lbg-survey-strat --display-name "LBG Survey Strat"
```
or equivalent.

The notebooks in the top-level directory create all the plots in the paper.
They can be run in any order, but note that `cmb-lensing-snr.ipynb` creates `data/cmb_xcorr_snr_vals.npz` which is used in `metrics.ipynb`.
This data file is already packaged with the repo, so you don't actually have to run the former notebook before the latter, but if you change something in the former it might affect the latter.

To run `metrics.ipynb` there are several scripts you must run first:

1. Source `bin/setup_rubin_sim.sh`. This will set `RUBIN_SIM_DATA_DIR` and download the required data if not already present. If you already have the rubin sim data downloaded somewhere other than the default (`data/rubin_sim_data`), you can set this location by calling the setup script with the path as an argument.

2. Source `bin/setup_runs.sh`. This will set `RUBIN_SIM_RUN_DIR` and, if not already present, will download the survey simulation runs we are analyzing. If you already have the rubin sim runs downloaded somewhere other than the default (`data/rubin_sim_data`), you can set this location by calling the setup script with the path as an argument.

3. Calculate the 5-sigma depths for all of the runs/years by running `bin/calc_m5.py`.

Note you also have to run step 1 in order to run `mean_igm_extinction.ipynb`.
Downloading all the data takes about 80 minutes and consumes NN GB.