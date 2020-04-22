import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
from matplotlib.backends.backend_pdf import PdfPages

TO_WATCH = [245, 184, 249, 173, 103, 65, 113, 19]


def generate_gif(
    scorer,
    data_abs_xy,
    activations,
    directory,
    filename,
    epoch,
    plot_graphs=True,  # pylint: disable=unused-argument
    nbins=20,  # pylint: disable=unused-argument
    cm="jet",
    sort_by_score_60=True,
):
    """Plotting function."""

    # Concatenate all trajectories
    xy = data_abs_xy.reshape(-1, data_abs_xy.shape[-1])
    act = activations.reshape(-1, activations.shape[-1])
    n_units = act.shape[1]
    # Get the rate-map for each unit
    s = [scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i]) for i in range(n_units)]
    # Get the scores
    score_60, score_90, max_60_mask, max_90_mask, sac = zip(*[scorer.get_scores(rate_map) for rate_map in s])
    # Plot
    cols, rows = 8, 1
    fig = plt.figure(figsize=(10, 2))
    plt.suptitle(f"Epoch{epoch}", fontsize=14, horizontalalignment="left")
    for i in range(len(TO_WATCH)):
        rf = plt.subplot(rows, cols, i+1)
        if i < n_units:
            index = TO_WATCH[i]
            title = "%d (%.2f)" % (index, score_60[index])
            # Plot the autocorrelation of the activation maps
            im = scorer.plot_sac(sac[index], mask_params=max_60_mask[index], ax=rf, title=title, cmap=cm, animated=True)
    plt.savefig(f"images/test_2/test_{epoch}.png")
    # plt.show()
    return
