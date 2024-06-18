# lbg-survey-metrics
Survey Strategy Metrics for Detecting LBGs with the Rubin Observatory

Installation (from the root directory):
```bash
mamba env create -f environment.yaml
mamba activate lbg-survey-strat
python -m ipykernel install --user --name lbg-survey-strat --display-name "LBG Survey Strat"
pip install -e .
```
or equivalent.

The scripts in `bin/` are meant to be run in the following order:
1. Source `bin/setup_rubin_sim.sh`. This will set `RUBIN_SIM_DATA_DIR` and download the required data if not already present. If you already have the rubin sim data downloaded somewhere other than the default (`data/rubin_sim_data`), you can set this location by calling the setup script with the path as an argument.

2. Source `bin/setup_runs.sh`. This will set `RUBIN_SIM_RUN_DIR` and, if not already present, will download the survey simulation runs we are analyzing. If you already have the rubin sim runs downloaded somewhere other than the default (`data/rubin_sim_data`), you can set this location by calling the setup script with the path as an argument.

3. Calculate the 5-sigma depths for all of the runs/years by running `bin/calc_m5.py`.

4. Create the CMB lensing cache by running `bin/create_cmb_xcorr_cache.py`

Note you will have to source the first 2 bash scripts every time you start a new session.
The first time you run the first two commands, it typically takes a few hours and consumes 23 GB to download all the data.

The notebooks in the top-level directory create all the plots in the paper and can be run in any order.