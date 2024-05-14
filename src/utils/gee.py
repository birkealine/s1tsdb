import ee
import geetools.tools._deprecated_image as image_module # birke: adpated because otherwise doesnt work with new geetools version (not sure which one Olivier used)
from geetools.tools._deprecated_imagecollection import mergeGeometries # birke: adpated because otherwise doesnt work with new geetools version (not sure which one Olivier used)
from shapely.geometry import MultiPolygon, Polygon
from typing import List, Union


def init_gee(project="rmac-ethz"):
    """Initialize GEE. Works also when working through ssh"""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=project)


def shapely_to_gee(geo: Union[Polygon, MultiPolygon]) -> ee.Geometry:
    """Transforms shapely geometry into GEE geometry"""
    if geo.__geo_interface__["type"] == "Polygon":
        return ee.Geometry.Polygon(geo.__geo_interface__["coordinates"])
    elif geo.__geo_interface__["type"] == "MultiPolygon":
        return ee.Geometry.MultiPolygon(geo.__geo_interface__["coordinates"])
    else:
        raise TypeError


def gee_download_still_going() -> bool:
    """Whether some tasks are still running in GEE."""

    # Flags for tasks that are not finished yet
    not_done = ["UNSUBMITTED", "READY", "RUNNING"]
    return any([any([t["state"].__eq__(s) for s in not_done]) for t in ee.data.getTaskList()])


def get_tasks_already_launched() -> List[str]:
    """Get name of tasks that have been launched in GEE."""
    not_done = ["UNSUBMITTED", "READY", "RUNNING"]
    return [t["description"] for t in ee.data.getTaskList() if any([t["state"].__eq__(s) for s in not_done])]


def custom_mosaic_same_day(collection: ee.ImageCollection, date_field: str = "date") -> ee.ImageCollection:
    """
    Mosaic together tiles from the same date.

    Custom function that apply mosaic for tiles with the same dates and simply copy the properties
    from one of the tiles. Except the ID, the others properties are the same for all tiles mosaic
    (eg orbit_number or orbit_direction)

    Args:
        collection (ee.ImageCollection): The collection with maybe multiple tiles with same date
        date_field (str): The name of the date field in the collection

    Returns:
        ee.ImageCollection: New collection with tiles combined together.
    """

    # Identify unique dates
    def _list_unique_dates(d_, l_):
        l_ = ee.List(l_)
        return ee.Algorithms.If(l_.contains(d_), l_, l_.add(d_))

    all_dates = collection.aggregate_array(date_field)
    unique_dates = ee.List(all_dates.iterate(_list_unique_dates, ee.List([])))

    def get_unique_img(date) -> ee.Image:
        """Either keep the image if alone, or combine with mosaic"""
        date = ee.Date(date)
        filtered = collection.filterDate(date, date.advance(1, "day"))  # single date

        def custom_mosaic(col):
            # adapted from geetools.tools -> mosaicSameDay
            first_img = col.first()
            bands = first_img.bandNames()

            mos = col.mosaic()
            mos = ee.Image(mos.copyProperties(first_img))
            # System properties are not copied by default.
            mos = mos.set("system:index", first_img.get("system:index"))
            mos = mos.set("system:time_start", first_img.get("system:time_start"))
            mos = mos.set("system:footprint", mergeGeometries(collection), "mosaic", True)
            mos = mos.select(bands)

            def reproject(bname, mos):
                mos = ee.Image(mos)
                mos_bnames = mos.bandNames()
                bname = ee.String(bname)
                proj = first_img.select(bname).projection()

                newmos = ee.Image(
                    ee.Algorithms.If(
                        mos_bnames.contains(bname),
                        image_module.replace(mos, bname, mos.select(bname).setDefaultProjection(proj)),
                        mos,
                    )
                )
                return newmos

            return ee.Image(bands.iterate(reproject, mos))

        # Whether the list has one tile or more
        condition = ee.Number(filtered.size()).eq(ee.Number(1))
        return ee.Algorithms.If(condition, filtered.first(), custom_mosaic(filtered))

    new_collection = ee.ImageCollection.fromImages(unique_dates.map(get_unique_img))
    return new_collection
