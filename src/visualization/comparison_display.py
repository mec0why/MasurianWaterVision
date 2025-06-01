import matplotlib.pyplot as plt
import numpy as np
from .constants import WATER_MASK_DIFF_CMAP


def plot_water_mask_comparison(eopatch, idx1, idx2, titles=None):
    mask1 = eopatch.mask["WATER_MASK"][idx1, :, :, 0]
    mask2 = eopatch.mask["WATER_MASK"][idx2, :, :, 0]

    diff_mask = np.zeros_like(mask1, dtype=np.uint8)
    diff_mask[np.logical_and(~mask1, mask2)] = 1
    diff_mask[np.logical_and(mask1, ~mask2)] = 2
    diff_mask[np.logical_and(mask1, mask2)] = 3

    if titles:
        title_mask1, title_mask2, fig_super_title = titles
        subplot_diff_title = "Analiza porównawcza"
    else:
        timestamp1_str = eopatch.timestamps[idx1].strftime('%Y-%m-%d')
        timestamp2_str = eopatch.timestamps[idx2].strftime('%Y-%m-%d')
        title_mask1 = f"Maska: {timestamp1_str}"
        title_mask2 = f"Maska: {timestamp2_str}"
        fig_super_title = f"Porównanie masek wodnych: {timestamp1_str} vs {timestamp2_str}"
        subplot_diff_title = "Różnice"

    data_aspect_ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(
        eopatch.bbox.max_y - eopatch.bbox.min_y)

    plot_height = 5
    single_plot_width = plot_height * data_aspect_ratio if data_aspect_ratio > 0 else plot_height

    total_fig_width = single_plot_width * 3 + single_plot_width * 0.2 + 1.5
    total_fig_width = max(12, total_fig_width)
    fig_height = plot_height + 1.5

    fig = plt.figure(figsize=(total_fig_width, fig_height))

    gs_width_ratios = [single_plot_width] * 3 + [single_plot_width * 0.2]
    gs = fig.add_gridspec(1, 4, width_ratios=gs_width_ratios, wspace=0.4 if data_aspect_ratio > 0.5 else 0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cbar_ax = fig.add_subplot(gs[0, 3])

    fig.suptitle(fig_super_title, fontsize=16)

    ax1.imshow(mask1, cmap='Blues')
    ax1.set_title(title_mask1, fontsize=12)
    ax1.axis('off')

    ax2.imshow(mask2, cmap='Blues')
    ax2.set_title(title_mask2, fontsize=12)
    ax2.axis('off')

    im = ax3.imshow(diff_mask, cmap=WATER_MASK_DIFF_CMAP, vmin=0, vmax=3)
    ax3.set_title(subplot_diff_title, fontsize=12)
    ax3.axis('off')

    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
    cbar.set_ticklabels(['Brak wody', 'Nowa woda', 'Zanikła woda', 'Stała woda'])
    cbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)

    return fig, [ax1, ax2, ax3]