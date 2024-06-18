import matplotlib.pyplot as plt
import numpy as np


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

    fig.suptitle(title)
