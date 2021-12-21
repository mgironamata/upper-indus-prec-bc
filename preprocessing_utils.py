import numpy as np 
import pandas as pd
import os, glob
from pyproj import CRS
import geopandas
import rasterio
from sklearn.model_selection import KFold

import torch
from torch.utils.data import TensorDataset, DataLoader 

__all__ = ['import_dataframe',
           'drop_df_NaNs',
           'clip_time_period',
           'list_bc_stations',
           'log_transform',
           'dry_days_binomial',
           'disjunctive_union_lists',
           'mosaic_tiles',
           'mask_raster',
           'filter_complete_station_years',
           'FilterByList',
           'sort_by_quantile',
           'add_year_month_season',
           'season_apply',
           'add_yesterday_observation',
           'create_cv_held_out_sets',
           'create_station_dataframe',
           'create_input_data',
           'create_dataset'   
           ]

def sort_by_quantile(st):
    """Re-arrange dataframe of station data so that model simulations and observations match 
    based on quantiles, for each station separately
    
    Inputs:
    
    Outputs:
    
    """
    
    QM_data = {}
    list_stations = st['Station'].unique()
    
    for i, s in enumerate(list_stations):
        QM_data[s] = st[st['Station']==s].sort_values(by='wrf_prcp')
        QM_data[s]['Prec'] = QM_data[s]['Prec'].sort_values().values

        if i == 0:
            QM_df = QM_data[s]
        else:
            QM_df = QM_df.append(QM_data[s])

    QM_df.reset_index(drop=True)
    
    return QM_df

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

def add_year_month_season(st):
    st['year'] = pd.DatetimeIndex(pd.to_datetime(st['Date'], format='%Y-%m-%d')).year
    st['month'] = pd.DatetimeIndex(pd.to_datetime(st['Date'], format='%Y-%m-%d')).month
    st['season'] = st.apply(season_apply, axis=1) 
    return st

def filter_complete_station_years(df):
    grouped_df = df.groupby(['Station','year']).count().reset_index()
    pairs = list(grouped_df[grouped_df['Prec']>=365][['Station','year']].apply(tuple,1))
    df = df[df[['Station','year']].apply(tuple, 1).isin(pairs)]
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

def season_apply(df):
    # Premonsoon: Apr-May
    if (df['month'] >= 4) and (df['month'] <= 5):
        return 'Premonsoon (AM)'
    # Monsoon: June-Jul-Aug-Sept
    elif (df['month'] >= 6) and (df['month'] <= 9):
        return 'Monsoon (JJAS)'
    # Postmonsoon: Oct-Nov-Dec
    elif (df['month'] >= 10) and (df['month'] <= 12):
        return 'Postmonsoon (OND)'
    # Winter: Jan-Feb-Mar
    elif (df['month'] >= 1) or (df['month'] <= 3):
        return 'Winter (JFM)'

def add_yesterday_observation(st):
    """Add observed precipitation value from previous day"""
    st['obs_yesterday'] = st.groupby('Station')['Prec'].shift(1)
    st.dropna(inplace=True)
    return st

def create_cv_held_out_sets(st_names, non_bc_st_names, split_by = 'station', include_non_bc_stations = True):

    if split_by == 'station':
        
        split_dict = {}
        kf = KFold(n_splits=10,shuffle=True)
        #kf.get_n_splits(st_names)

        for i, (train_index, test_index) in enumerate(kf.split(st_names)):
            k = round(len(train_index)*0.8)
            split_dict[f'k{i}'] = {f'train': list(st_names[train_index[:k]]) + non_bc_st_names if include_non_bc_stations else list(st_names[train_index[:k]]),
                                   f'val' : st_names[train_index[k:]],
                                   f'test' : st_names[test_index[:]]
                                }
            
    elif split_by == 'year':
        
        split_dict = {}
        kf = KFold(n_splits=5, shuffle=True)
        years=np.array(range(1998,2008))

        for i, (train_index, test_index) in enumerate(kf.split(years)):
            k = round(len(train_index)*0.75)
            split_dict[f'k{i}'] = {f'train': list(years[train_index[:k]]),
                                   f'val' : years[train_index[k:]],
                                   f'test' : years[test_index[:]]
                                }
    
    return split_dict


def create_station_dataframe(TRAIN_PATH, start, end, add_yesterday=True, basin_filter = None, filter_incomplete_years = True):

    st = (import_dataframe(TRAIN_PATH)
        .pipe(drop_df_NaNs, series='Prec')
        .pipe(clip_time_period, start, end)
        .pipe(add_year_month_season)
        )  

    # Add yesterday's observation
    if add_yesterday: st = add_yesterday_observation(st)

    # Filter incomplete years
    if filter_incomplete_years: st = filter_complete_station_years(st)

    # Filter by basin
    if basin_filter is not None: st = st[st['Basin']==basin_filter]

    # st['set'] = "train" 

    # st_val = (import_dataframe(TEST_PATH) # Import dataframe
    #     .pipe(drop_df_NaNs, series='Prec') # Drop NaNs
    #     .pipe(clip_time_period, start, end) # Clip data temporally 
    # )

    # st_val['set'] = "test"

    # # Append validation stations to training 
    # st = st.append(st_val)

    return st

def create_input_data(st, predictors, predictand, split_dict, split_by='station', sort_by_quantile_flag=False):

    data = {}

    X = st[predictors].to_numpy()

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)

    for i in range(len(split_dict)):

        if split_by=='station':
            
            data[f'set_train_{i}'] = st[st['Station'].isin(split_dict[f'k{i}']['train'])]
            data[f'set_val_{i}'] = st[st['Station'].isin(split_dict[f'k{i}']['val'])]
            data[f'set_test_{i}'] = st[st['Station'].isin(split_dict[f'k{i}']['test'])]
                                    
            if sort_by_quantile_flag:
                data[f'set_train_{i}'] = sort_by_quantile(data[f'set_train_{i}'])
                data[f'set_val_{i}'] = sort_by_quantile(data[f'set_val_{i}'])
                                    
            data[f'X_train_{i}'] = (data[f'set_train_{i}'][predictors].to_numpy() - x_mean) / x_std
            data[f'X_val_{i}'] = (data[f'set_val_{i}'][predictors].to_numpy() - x_mean) / x_std
            data[f'X_test_{i}'] = (data[f'set_test_{i}'][predictors].to_numpy() - x_mean) / x_std

            data[f'Y_train_{i}'] = data[f'set_train_{i}'][predictand].to_numpy()
            data[f'Y_val_{i}'] = data[f'set_val_{i}'][predictand].to_numpy()
            data[f'Y_test_{i}'] = data[f'set_test_{i}'][predictand].to_numpy()

        
        elif split_by=='year':
        
            data[f'X_train_{i}'] = (st[st['year'].isin(split_dict[f'k{i}']['train'])][predictors].to_numpy() - x_mean) / x_std
            data[f'X_val_{i}'] = (st[st['year'].isin(split_dict[f'k{i}']['val'])][predictors].to_numpy() - x_mean) / x_std
            data[f'X_test_{i}'] = (st[st['year'].isin(split_dict[f'k{i}']['test'])][predictors].to_numpy() - x_mean) / x_std

            data[f'Y_train_{i}'] = st[st['year'].isin(split_dict[f'k{i}']['train'])][predictand].to_numpy()
            data[f'Y_val_{i}'] = st[st['year'].isin(split_dict[f'k{i}']['val'])][predictand].to_numpy()
            data[f'Y_test_{i}'] = st[st['year'].isin(split_dict[f'k{i}']['test'])][predictand].to_numpy()

    # data['X_test'] = (st[st['Station'].isin(st_names_dict['test'])][predictors].to_numpy() - x_mean) / x_std
    # data['Y_test'] = st[st['Station'].isin(st_names_dict['test'])][predictand].to_numpy()

    # data['X_test2'] = (st_val[predictors].to_numpy() - x_mean) / x_std
    # data['Y_test2'] = st_val[predictand].to_numpy()

    return data, x_mean, x_std

def create_dataset(data, split, d):
    tensor_x = torch.Tensor(data[f'X_{split}'][:,:d])
    tensor_y = torch.Tensor(data[f'Y_{split}'][:,:d])
    return TensorDataset(tensor_x, tensor_y)