import argparse
import os
from pathlib import Path

import numpy as np
import rubin_sim.maf as maf
from rubin_scheduler.scheduler.utils import SkyAreaGenerator
from rubin_sim.maf.metrics.exgal_m5 import ExgalM5

# Define the directory where the m5 maps are saved
out_dir = Path(__file__).parents[1] / "data" / "rubin_sim_m5"


# Function to compute m5 maps
def run_metric_bundles():
    # List of sims to run
    try:
        run_dir = Path(os.environ["RUBIN_SIM_RUNS_DIR"])
    except KeyError:
        raise RuntimeError(
            "Environment variable RUBIN_SIM_RUNS_DIR is not set. "
            "You need to run bin/setup_runs.sh."
        )
    run_list = list(run_dir.rglob("*.db"))

    # Create sky map
    nside = 128
    surveyAreas = SkyAreaGenerator(nside=nside)
    map_footprints, map_labels = surveyAreas.return_maps()
    slicer = maf.HealpixSubsetSlicer(
        nside=nside,
        hpid=np.where(map_labels == "lowdust")[0],
        use_cache=False,
    )

    # Instantiate extragalactic depth metric
    metric = ExgalM5()

    # Loop over all the runs
    for file in run_list:
        m5_bundles = []
        # Loop over survey years
        for year in [1, 4, 7, 10]:
            # Calculate days until this year
            days = year * 365.25

            # Loop over bands
            for band in "ugrizy":
                # Constraint on exposures
                constraint = f"filter='{band}' and note not like 'DD%' and night <= {days} and note not like 'twilight_near_sun' "

                # Add the metric bundle
                m5_bundles.append(
                    maf.MetricBundle(
                        metric=metric,
                        slicer=slicer,
                        constraint=constraint,
                        run_name=file.stem,
                    )
                )

        # Calculate all metrics from group
        m5_group = maf.MetricBundleGroup(m5_bundles, str(file), out_dir=str(out_dir))
        m5_group.run_all()


# Function to rename files of m5 maps
def rename_files():
    # Get all the metrics
    files = list(out_dir.glob("*"))
    for file in files:
        try:
            # Get the number of years for this file
            base, days, _ = file.stem.split("_and_note_not_like")
            days = float(".".join(days.split("_")[-2:]))
            years = int(days / 365.25)

            # Split the base again
            base, band = base.split("10yrs")

            # Create the new name
            new_name = base + f"{years}yrs" + band

            # Rename the file
            file.rename(file.with_stem(new_name))
        except:
            pass


if __name__ == "__main__":
    # Argument for forcing a re-run
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Whether to force a new run, even if outputs already exist.",
    )
    args = parser.parse_args()

    # Do outputs exist?
    if out_dir.exists() and len(list(out_dir.glob("*"))) > 0 and not args.force:
        print(
            "Results of calc_m5.py already exist. "
            "If you want to force a re-run, run with flag '-f'."
        )
    else:
        run_metric_bundles()
        rename_files()
