import rasterio
import geopandas
from pyproj import CRS


__all__ = [
            mosaic_tiles, 
            mask_raster
        ]

def mosaic_tiles(dirpath, search_criteria, out_fp, epsg=4326):
    
    """ Mosaic raster tiles
    
    Args:
        dirpath (str): 
        search_criteria (str): 
        out_fp (str): 
        epsg (int): EPSG coordinate refernce number
        
    Returns:
        None
    
    """
    # locate raster tiles
    q = os.path.join(dirpath, search_criteria)
    dem_fps = glob.glob(q)

    # create list with mosaic tiles
    src_files_to_mosaic = []
    for fp in dem_fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    # create mosaic 
    mosaic, out_trans = rasterio.merge.merge(src_files_to_mosaic)

    # Update metadata
    out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans, 
                    "crs": CRS.from_epsg(epsg)
                    })
    
    # Save to file
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)

def mask_raster(in_raster_path, out_raster_path, mask):
    
    """ Clip extent of a raster using mask shape.
    
    Args:
        in_raster_path (str): 
        out_raster_path (str):
        mask (str or geodaframe):
        
    Return:
        None
    
    """

    # Import mask
    if type(mask)==str:
        gdf = geopandas.read_file(mask)
    else:
        gdf = mask

    # Define mask shapes
    shapes = list(gdf['geometry'].values)
    
    # Mask raster 
    with rasterio.open(in_raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    
    # Update metadata
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
    # Save to file
    with rasterio.open(out_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)