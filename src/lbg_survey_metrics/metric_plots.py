import matplotlib.pyplot as plt
import numpy as np
import healpy as hp


def plot_u_strategy_matrix(
    data, band, metric, title=None, vmin=None, vmax=None, two_dec=False, norm=1
):
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), constrained_layout=True, dpi=150)

    for i, year in enumerate([1, 10]):
        # Accumulate metric values
        vals = []
        for scale in data[band][year]:
            for expt in data[band][year][scale]:
                vals.append(data[band][year][scale][expt][metric])
        vals = np.array(vals).reshape(4, 4) / norm

        # Plot metric colors
        axes[i].imshow(vals, cmap="coolwarm", vmin=vmin, vmax=vmax, origin="lower")

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
            xticklabels=data[band][year].keys(),
            xlabel="Fraction of $u$ band visits",
            yticks=np.arange(4),
            yticklabels=data[band][year][1.0].keys(),
            ylabel="$u$ band exposure time",
            title=f"Year {year}",
        )

        axes[i].plot(
            [-0.48, 0.5, 0.5, -0.48, -0.48],
            [-0.48, -0.48, 0.5, 0.5, -0.48],
            c="C3",
            lw=1,
            zorder=10,
        )
        axes[i].plot(
            [0.5, 1.5, 1.5, 0.5, 0.5], [0.5, 0.5, 1.5, 1.5, 0.5], c="C3", lw=1, ls="--"
        )

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
