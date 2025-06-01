import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import sobel
from skimage.morphology import closing, disk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .constants import NDVI_CMAP


def _calculate_ndvi_values(eopatch, idx):
    red = eopatch.data["BANDS"][idx, :, :, 2]
    nir = eopatch.data["BANDS"][idx, :, :, 3]
    ndvi = np.zeros_like(red, dtype=float)
    valid_mask = (nir + red) > 0
    ndvi[valid_mask] = (nir[valid_mask].astype(float) - red[valid_mask].astype(float)) / (
                nir[valid_mask].astype(float) + red[valid_mask].astype(float))
    return ndvi


def plot_rgb_w_water(eopatch, idx, title=None):
    ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(eopatch.bbox.max_y - eopatch.bbox.min_y)
    fig, ax = plt.subplots(figsize=(ratio * 10, 10))

    ax.imshow(np.clip(2.5 * eopatch.data["BANDS"][idx][:, :, [2, 1, 0]], 0, 1))

    observed = closing(eopatch.mask["WATER_MASK"][idx, :, :, 0], disk(1))
    nominal = sobel(eopatch.mask_timeless["NOMINAL_WATER"][:, :, 0])
    observed = sobel(observed)

    nominal = np.ma.masked_where(~nominal.astype(bool), nominal)
    observed = np.ma.masked_where(~observed.astype(bool), observed)

    ax.imshow(nominal, cmap=plt.cm.Reds)
    ax.imshow(observed, cmap=plt.cm.Blues)

    if title:
        ax.set_title(title)

    ax.axis("off")
    plt.tight_layout()

    return fig, ax


def plot_ndvi(eopatch, idx, title=None, compare_idx=None):
    ndvi = _calculate_ndvi_values(eopatch, idx)
    ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(eopatch.bbox.max_y - eopatch.bbox.min_y)

    if compare_idx is None:
        fig, ax = plt.subplots(figsize=(ratio * 10, 10))
        im = ax.imshow(ndvi, cmap=NDVI_CMAP, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax).set_label('NDVI (indeks -1 do 1)')
        ax.set_title(title if title else f"NDVI - {eopatch.timestamps[idx]}")
        ax.axis("off")
        plt.tight_layout()
        return fig, ax
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ratio * 20, 10))
        ndvi_compare = _calculate_ndvi_values(eopatch, compare_idx)

        im1 = ax1.imshow(ndvi, cmap=NDVI_CMAP, vmin=-1, vmax=1)
        ax1.set_title(f"NDVI - {eopatch.timestamps[idx]}")
        ax1.axis("off")

        im2 = ax2.imshow(ndvi_compare, cmap=NDVI_CMAP, vmin=-1, vmax=1)
        ax2.set_title(f"NDVI - {eopatch.timestamps[compare_idx]}")
        ax2.axis("off")

        cax = make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.2)
        fig.colorbar(im1, cax=cax).set_label('NDVI (indeks -1 do 1)', fontsize=12)

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

        plt.subplots_adjust(wspace=0.05, top=0.9)
        return fig, (ax1, ax2)