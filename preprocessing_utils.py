import numpy as np 
import pandas as pd
import os, glob
from pyproj import CRS
import geopandas
import rasterio

__all__ = ['import_dataframe',
           'drop_df_NaNs',
           'clip_time_period',
           'list_bc_stations',
           'log_transform',
           'dry_days_binomial',
           'disjunctive_union_lists',
           'mosaic_tiles',
           'mask_raster',
           'FilterCompleteStationYears',
           'FilterByList'
           ]

def import_dataframe(path, verbose=False):
    """ Reads DataFrame from path
    
    Args:
        path (string) : source path 

    Returns:
        Dataframe : pandas Dataframes
    """
    
    df = pd.read_pickle(path)
    
    if verbose: 
        print(f"Length of imported dataframe: {len(df)}")
    
    return df

def drop_df_NaNs(df, series='Prec', verbose=False):
    """ Replaces non-numeric values from Dataframe series 
    to NaN values, so that it can be read as float.

    Args:	
        df (DataFrame) : pandas Dataframe containing series of interest
        series (string) : name of pandas DataFrame series 

    Returns:
        df (Dataframe) : 
    """

    df[series].replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df[series].replace(r'.', np.nan, regex=True, inplace=True)
    df.dropna(inplace=True)
    
    if verbose: 
        print(f"Length of dataframe after dropping NaNs: {len(df)}")
    
    return df

def clip_time_period(df, start, end, verbose=False):
    """Returns a Dataframe clipped to the time period specified by the start and end dates.

    Args:
        df : 
        start : 
        end : 

    Returns:
        df : 
    """

    df_clip = df[df['Date'].between(start,end)]

    if verbose:
        print(f"Length of clipped dataframe: {len(df_clip)}")
    return df_clip

def FilterCompleteStationYears(df):
    grouped_df = df.groupby(['Station','Year']).count().reset_index()
    pairs = list(grouped_df[grouped_df['Prec']>=365][['Station','Year']].apply(tuple,1))
    df = df[df[['Station','Year']].apply(tuple, 1).isin(pairs)]
    return df

def FilterByList(df,series,value_list):
    return df[df[series].isin(value_list)]

def filter_by_series(df, series, value):
    return df[df[series]==value]

def list_bc_stations(df):
    df['BC_diff'] = df['wrf_prcp'] - df['wrf_bc_prcp']
    aux = df.groupby('Station').sum()
    aux.reset_index(inplace=True)
    return aux['Station'][aux['BC_diff']!=0].unique()

def log_transform(df, cols, epsilon=1):
    """Log-transformation Pandas dataframe columns.

    Args:
        df : dataframe
            Input DataFrame
        cols : list
            List of variables/series to log-transform
        epsilon : float
            Shift applied to variables before log-transformation (to avoid log(zero))
    
    Returns:
        df : updated DataFrame with additonal columns for log-transformed variables. 
    """
    
    for col in cols:
        new_col_name = col + "_log"
        df[new_col_name] = df[col].transform(lambda x: np.log(x + epsilon))
    return df

def dry_days_binomial(df, probability=None, verbose=False): 
    """ Draw values from random variable with Binomial probability and append 
    it as an additional series to an existing Dataframe. 

    Args:
        df (DataFrame): Input DataFrame
        probability (float, optional): Binomial distribution probability. Must be a value in interval [0,1]
        verbose (Boolean, optional): Defaults to False.
    
    Returns:
        series
    """

    if probability == None:
        probability = sum(df['Prec']>0)/sum(df['Prec']==0)
    else:
        pass    
    
    if verbose:
        print(f"Probability is {probability}")
    
    return np.random.binomial(1,probability,len(df))


def disjunctive_union_lists(li1, li2):
    """ Returns the disjunctive union or symmetric difference between to sets, expressed as lists.
    
    Args:
        li1 (list): first set.
        li2 (list): second set.
    
    Returns:
        list
    """
    
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

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
        