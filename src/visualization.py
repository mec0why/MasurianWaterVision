import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.morphology import closing, disk


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


def get_water_level_data(eopatch, max_coverage=1.0):
    dates = np.asarray(eopatch.timestamps)

    cloud_filter = eopatch.scalar["COVERAGE"][:, 0] < max_coverage

    median_level = np.median(eopatch.scalar["WATER_LEVEL"])
    outlier_filter = eopatch.scalar["WATER_LEVEL"][:, 0] >= (median_level * 0.96)

    valid_data = np.logical_and(cloud_filter, outlier_filter)
    valid_indices = np.where(valid_data)[0]

    return dates, valid_data, valid_indices


def plot_water_levels(eopatch, max_coverage=1.0):
    dates, valid_data, valid_indices = get_water_level_data(eopatch, max_coverage)

    fig, ax = plt.subplots(figsize=(20, 7))

    ax.plot(
        dates[valid_data],
        eopatch.scalar["WATER_LEVEL"][valid_data],
        "bo-",
        alpha=0.7,
        label="Poziom wody"
    )

    ax.plot(
        dates[valid_data],
        eopatch.scalar["COVERAGE"][valid_data],
        "--",
        color="gray",
        alpha=0.7,
        label="Zachmurzenie"
    )

    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("Data")
    ax.set_ylabel("Poziom wody / Zachmurzenie")
    ax.set_title("Poziom wody w jeziorze Świętajno")
    ax.grid(axis="y")
    ax.legend()

    plt.tight_layout()
    return ax, valid_data, valid_indices