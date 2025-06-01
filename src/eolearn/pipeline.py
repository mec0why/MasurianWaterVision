import numpy as np
from eolearn.core import EOWorkflow, FeatureType, OutputTask, linearly_connect_tasks
from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.geometry import VectorToRasterTask
from eolearn.io import SentinelHubInputTask
from sentinelhub import DataCollection
from config import settings
from .tasks import AddValidDataMaskTask, AddValidDataCoverageTask, ValidDataCoveragePredicate, IceFilterTask, WaterDetectionTask


def create_download_task(cache_folder=None, maxcc=None, resolution=None):
    if cache_folder is None:
        cache_folder = settings.CACHE_FOLDER
    if resolution is None:
        resolution = settings.SATELLITE_IMAGE_RESOLUTION
    if maxcc is None:
        maxcc = settings.MAX_CLOUD_COVERAGE_DOWNLOAD
    return SentinelHubInputTask(
        data_collection=DataCollection.SENTINEL2_L1C,
        bands_feature=(FeatureType.DATA, "BANDS"),
        resolution=resolution,
        maxcc=maxcc,
        bands=["B02", "B03", "B04", "B08", "B11"],
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


def create_ndsi_task():
    return NormalizedDifferenceIndexTask(
        (FeatureType.DATA, "BANDS"),
        (FeatureType.DATA, "NDSI"),
        (1, 4)
    )


def create_nominal_water_task(gdf):
    return VectorToRasterTask(
        gdf,
        (FeatureType.MASK_TIMELESS, "NOMINAL_WATER"),
        values=1,
        raster_shape=(FeatureType.MASK, "IS_DATA"),
        raster_dtype=np.uint8,
    )


def create_cloud_filter_task(threshold=0.05):
    return SimpleFilterTask(
        (FeatureType.MASK, "VALID_DATA"),
        ValidDataCoveragePredicate(threshold)
    )


def create_workflow(gdf, cloud_threshold=0.05, ice_threshold=0.6,
                    max_cloud_coverage_download=None, resolution=None):
    if max_cloud_coverage_download is None:
        max_cloud_coverage_download = settings.MAX_CLOUD_COVERAGE_DOWNLOAD
    download_task = create_download_task(maxcc=max_cloud_coverage_download, resolution=resolution)
    calculate_ndwi = create_ndwi_task()
    calculate_ndsi = create_ndsi_task()
    add_nominal_water = create_nominal_water_task(gdf)
    add_valid_mask = AddValidDataMaskTask()
    add_coverage = AddValidDataCoverageTask()
    remove_cloudy_scenes = create_cloud_filter_task(cloud_threshold)
    remove_ice_scenes = IceFilterTask(ice_threshold)
    water_detection = WaterDetectionTask()
    output_task = OutputTask("final_eopatch")

    workflow_nodes = linearly_connect_tasks(
        download_task,
        calculate_ndwi,
        calculate_ndsi,
        add_nominal_water,
        add_valid_mask,
        add_coverage,
        remove_cloudy_scenes,
        remove_ice_scenes,
        water_detection,
        output_task,
    )

    workflow = EOWorkflow(workflow_nodes)

    return workflow, workflow_nodes[0]