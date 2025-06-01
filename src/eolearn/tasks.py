import numpy as np
from skimage.filters import threshold_otsu
from eolearn.core import EOTask, FeatureType


class AddValidDataMaskTask(EOTask):


    def execute(self, eopatch):
        is_data_mask = eopatch[FeatureType.MASK, "IS_DATA"].astype(bool)
        cloud_mask = ~eopatch[FeatureType.MASK, "CLM"].astype(bool)
        eopatch[FeatureType.MASK, "VALID_DATA"] = np.logical_and(is_data_mask, cloud_mask)
        return eopatch


def calculate_coverage(array):
    return 1.0 - np.count_nonzero(array) / np.size(array)


class AddValidDataCoverageTask(EOTask):


    def execute(self, eopatch):
        valid_data = eopatch[FeatureType.MASK, "VALID_DATA"]
        time, height, width, channels = valid_data.shape

        coverage = []
        for t in range(time):
            data = valid_data[t].reshape(height * width * channels)
            coverage.append(calculate_coverage(data))

        eopatch[FeatureType.SCALAR, "COVERAGE"] = np.array(coverage)[:, np.newaxis]
        return eopatch


class ValidDataCoveragePredicate:


    def __init__(self, threshold):
        self.threshold = threshold


    def __call__(self, array):
        return calculate_coverage(array) < self.threshold


class IceFilterTask(EOTask):


    def __init__(self, threshold=0.6):
        self.threshold = threshold


    def execute(self, eopatch):
        ndsi = eopatch.data["NDSI"][..., 0]
        nominal_water = eopatch.mask_timeless["NOMINAL_WATER"][..., 0].astype(bool)

        valid_indices = []
        for i in range(len(eopatch.timestamps)):
            ndsi_mean = np.nanmean(ndsi[i][nominal_water])
            if ndsi_mean < self.threshold:
                valid_indices.append(i)

        if not valid_indices and len(eopatch.timestamps) > 0:
            return eopatch.temporal_subset([])

        return eopatch.temporal_subset(valid_indices)


def detect_water(ndwi):
    otsu_thr = 1.0

    if len(np.unique(ndwi)) > 1:
        ndwi[np.isnan(ndwi)] = -1
        otsu_thr = threshold_otsu(ndwi)

    return ndwi > otsu_thr


class WaterDetectionTask(EOTask):


    def execute(self, eopatch):
        water_masks = []
        water_levels = []

        if eopatch.data["NDWI"] is None or len(eopatch.data["NDWI"]) == 0:
            eopatch[FeatureType.MASK, "WATER_MASK"] = np.array([])
            eopatch[FeatureType.SCALAR, "WATER_LEVEL"] = np.array([])
            return eopatch

        for ndwi_idx, ndwi in enumerate(eopatch.data["NDWI"]):
            mask = detect_water(ndwi[..., 0])

            current_nominal_water = eopatch.mask_timeless["NOMINAL_WATER"]
            if current_nominal_water.ndim == 2:
                current_nominal_water = current_nominal_water[..., np.newaxis]

            final_mask = mask[..., np.newaxis] * current_nominal_water

            water_masks.append(final_mask)

            total_water_pixels = np.count_nonzero(final_mask)
            total_nominal_pixels = np.count_nonzero(current_nominal_water)
            water_level = total_water_pixels / total_nominal_pixels if total_nominal_pixels > 0 else 0
            water_levels.append(water_level)

        eopatch[FeatureType.MASK, "WATER_MASK"] = np.array(water_masks) if water_masks else np.empty(
            (0, *eopatch.data["NDWI"].shape[1:-1], 1), dtype=bool)
        eopatch[FeatureType.SCALAR, "WATER_LEVEL"] = np.array(water_levels)[
            ..., np.newaxis] if water_levels else np.empty((0, 1), dtype=float)

        return eopatch