#!/bin/bash

# This script downloads the runs we are analyzing. If the data already 
# exists, it won't be downloaded again. By default, 
# RUBIN_SIM_RUNS_DIR=data/rubin_sim_runs
# If you want to set a different directory (e.g. because you have already 
# downloaded the data somewhere) you can call this script with an argument.
# E.g., `source download_runs_3p4.sh /path/to/rubin/sim/runs`

# Export the rubin sims run 3.4 directory
if [ -z "$1" ]; then
    DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
    export RUBIN_SIM_RUNS_DIR="${DIR}/data/rubin_sim_runs"
else
    export RUBIN_SIM_RUNS_DIR=$1
fi

# Create list of runs to download
BASE_URL="https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs3.4"
declare -a runs=(
    "baseline/baseline_v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt30_nscale0.9v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt30_nscale1.0v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt30_nscale1.1v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt30_nscale1.2v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt30_nscale1.5v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt38_nscale0.9v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt38_nscale1.0v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt38_nscale1.1v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt38_nscale1.2v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt38_nscale1.5v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt45_nscale0.9v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt45_nscale1.0v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt45_nscale1.1v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt45_nscale1.2v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt45_nscale1.5v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt60_nscale0.9v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt60_nscale1.0v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt60_nscale1.1v3.4_10yrs.db"
    "vary_u_exp_n/internal_u_expt60_nscale1.2v3.4_10yrs.db"
    "noroll/noroll_v3.4_10yrs.db",
    "roll_uniform_early_half/roll_uniform_early_half_mjdp0_v3.4_10yrs.db"
    "roll_3/roll_3_v3.4_10yrs.db"
)

# Loop through runs and download what does not exist
for run in "${runs[@]}"
do
    if ! [ -f "${RUBIN_SIM_RUNS_DIR}/${run}" ]; then
        dir=$(dirname "${run}")
        wget -P "${RUBIN_SIM_RUNS_DIR}/${dir}" "${BASE_URL}/${run}"
    fi
done