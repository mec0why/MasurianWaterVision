import geopandas as gpd
import shapely.wkt
from sentinelhub import CRS, BBox


def load_water_boundary(wkt_file_path):
    with open(wkt_file_path, "r") as f:
        dam_wkt = f.read()

    return shapely.wkt.loads(dam_wkt)


def create_bounding_box(geometry, buffer_percentage=0.1):
    minx, miny, maxx, maxy = geometry.bounds

    delx = maxx - minx
    dely = maxy - miny
    minx = minx - delx * buffer_percentage
    maxx = maxx + delx * buffer_percentage
    miny = miny - dely * buffer_percentage
    maxy = maxy + dely * buffer_percentage

    return BBox((minx, miny, maxx, maxy), crs=CRS.WGS84)


def create_geodataframe(geometry):
    return gpd.GeoDataFrame(crs=CRS.WGS84.pyproj_crs(), geometry=[geometry])