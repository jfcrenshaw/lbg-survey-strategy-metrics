import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_u_strategy_matrix(
    data: pd.DataFrame,
    band: str,
    metric: str,
    title: str | None = None,
    normalize: bool = True,
    two_dec: bool = True,
    box_fid: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot matrix for u band strategy metrics.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing u band strategy metrics.
        Created by `bin/calc_ustrat_metrics.py`
    band: str
        Name of the dropout band
    metric: str
        Name of the metric to plot
    title: str, default=None
        Title of the plot
    normalize: bool, default=True
        Whether to normalize all values to the baseline strategy
    two_dec: bool, default=True
        Whether to display two decimal points for printed metrics.
        If False, only one decimal place is shown
    box_fid: bool, default=True
        Whether to draw dashed red box around the fiducial strategy
        (i.e. 1.1x visits, 38s exposures)

    Returns
    -------
    plt.Figure
        The matplotlib figure
    plt.Axes
        The plot axes
    """
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), constrained_layout=True, dpi=150)

    # Query metrics for the band and create a table with the relevant
    # metric for different u-band strategies
    table = pd.pivot_table(
        data.query(f"band == '{band}'"),
        metric,
        ["year", "scale", "expt"],
        dropna=False,
    )

    for i, year in enumerate([1, 10]):
        # Extract metric values
        vals = table.loc[year].to_numpy().reshape(4, 4).T

        if normalize:
            # Normalize metric to baseline strategy
            vals /= vals[0, 0]

        # Plot metric colors
        axes[i].imshow(
            vals,
            cmap="coolwarm",
            vmin=np.nanmin(vals),
            vmax=np.nanmax(vals),
            origin="lower",
        )

        # Plot metric labels
        for (j, k), z in np.ndenumerate(vals):
            if np.isfinite(z):
                if two_dec:
                    axes[i].text(
                        k, j, f"{z:.2f}", ha="center", va="center", fontsize=12
                    )
                else:
                    axes[i].text(
                        k, j, f"{z:.1f}", ha="center", va="center", fontsize=12
                    )

        # Set axes properties
        axes[i].set(
            xticks=np.arange(4),
            xticklabels=table.index.get_level_values(1).unique(),
            xlabel="Fraction of $u$ band visits",
            yticks=np.arange(4),
            yticklabels=table.index.get_level_values(2).unique(),
            ylabel="$u$ band exposure time",
            title=f"Year {year}",
        )

        if box_fid:
            # Box the fiducial strategy
            axes[i].plot(
                [0.5, 1.5, 1.5, 0.5, 0.5],
                [0.5, 0.5, 1.5, 1.5, 0.5],
                c="C3",
                lw=1,
                ls="--",
            )

    # Set the title
    fig.suptitle(title)

    return fig, axes


def plot_map(
    values: np.ma.MaskedArray,
    title: str | None = None,
    n_dec: int = 2,
    sub: int | None = None,
) -> None:
    """Plot metric on a Mollweide map.

    Parameters
    ----------
    values: np.ma.MaskedArray
        Metric values
    title: str or None, default=None
        Title for map
    n_dec: int, default=2
        Number of decimals to display in colorbar ticks
    sub: int or None, default=None
        Integer indicating with subplot to plot the map on. For example, if you
        do `fig, axes = plt.subplots(2, 2)`, and want to put this map on the
        lower right subplot, you set `sub=224`.
    """
    # Don't allow healpy to override font sizes
    fontsize = {
        "xlabel": None,
        "ylabel": None,
        "title": None,
        "xtick_label": None,
        "ytick_label": None,
        "cbar_label": None,
        "cbar_tick_label": None,
    }

    # Get value limits
    limit = np.abs(values).max()

    # Plot map
    hp.projview(
        values,
        title=title,
        sub=sub,
        cmap="coolwarm",
        min=-limit,
        max=+limit,
        cbar_ticks=[-limit, 0, +limit],
        format=f"%.{n_dec}f",
        fontsize=fontsize,
    )
