import numpy as np
from skimage.filters import threshold_otsu
from eolearn.core import EOTask, EOWorkflow, FeatureType, OutputTask, linearly_connect_tasks
from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.geometry import VectorToRasterTask
from eolearn.io import SentinelHubInputTask
from sentinelhub import DataCollection


def create_download_task(cache_folder="cached_data"):
    return SentinelHubInputTask(
        data_collection=DataCollection.SENTINEL2_L1C,
        bands_feature=(FeatureType.DATA, "BANDS"),
        resolution=10,
        maxcc=0.1,
        bands=["B02", "B03", "B04", "B08"],
        additional_data=[
            (FeatureType.MASK, "dataMask", "IS_DATA"),
            (FeatureType.MASK, "CLM")
        ],
        cache_folder=cache_folder,
    )


def create_ndwi_task():
    return NormalizedDifferenceIndexTask(
        (FeatureType.DATA, "BANDS"),
        (FeatureType.DATA, "NDWI"),
        (1, 3)
    )


def create_nominal_water_task(gdf):
    return VectorToRasterTask(
        gdf,
        (FeatureType.MASK_TIMELESS, "NOMINAL_WATER"),
        values=1,
        raster_shape=(FeatureType.MASK, "IS_DATA"),
        raster_dtype=np.uint8,
    )


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

        coverage = np.apply_along_axis(
            calculate_coverage,
            1,
            valid_data.reshape((time, height * width * channels))
        )

        eopatch[FeatureType.SCALAR, "COVERAGE"] = coverage[:, np.newaxis]
        return eopatch


class ValidDataCoveragePredicate:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        return calculate_coverage(array) < self.threshold


def create_cloud_filter_task(threshold=0.05):
    return SimpleFilterTask(
        (FeatureType.MASK, "VALID_DATA"),
        ValidDataCoveragePredicate(threshold)
    )


class WaterDetectionTask(EOTask):
    @staticmethod
    def detect_water(ndwi):
        otsu_thr = 1.0

        if len(np.unique(ndwi)) > 1:
            ndwi[np.isnan(ndwi)] = -1
            otsu_thr = threshold_otsu(ndwi)

        return ndwi > otsu_thr

    def execute(self, eopatch):
        water_masks = np.asarray([
            self.detect_water(ndwi[..., 0]) for ndwi in eopatch.data["NDWI"]
        ])

        water_masks = water_masks[..., np.newaxis] * eopatch.mask_timeless["NOMINAL_WATER"]

        water_levels = np.asarray([
            np.count_nonzero(mask) / np.count_nonzero(eopatch.mask_timeless["NOMINAL_WATER"])
            for mask in water_masks
        ])

        eopatch[FeatureType.MASK, "WATER_MASK"] = water_masks
        eopatch[FeatureType.SCALAR, "WATER_LEVEL"] = water_levels[..., np.newaxis]

        return eopatch


def create_workflow(gdf, cloud_threshold=0.05):
    download_task = create_download_task()
    calculate_ndwi = create_ndwi_task()
    add_nominal_water = create_nominal_water_task(gdf)
    add_valid_mask = AddValidDataMaskTask()
    add_coverage = AddValidDataCoverageTask()
    remove_cloudy_scenes = create_cloud_filter_task(cloud_threshold)
    water_detection = WaterDetectionTask()
    output_task = OutputTask("final_eopatch")

    workflow_nodes = linearly_connect_tasks(
        download_task,
        calculate_ndwi,
        add_nominal_water,
        add_valid_mask,
        add_coverage,
        remove_cloudy_scenes,
        water_detection,
        output_task,
    )

    workflow = EOWorkflow(workflow_nodes)

    return workflow, workflow_nodes[0]