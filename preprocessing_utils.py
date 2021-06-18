import numpy as np 
import pandas as pd

__all__ = ['import_dataframe',
           'drop_df_NaNs',
           'clip_time_period',
           'list_bc_stations',
           'log_transform',
           'dry_days_binomial',
           'disjunctive_union_lists']

def import_dataframe(path, verbose=True):
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

def drop_df_NaNs(df, series='Prec', verbose=True):
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

def clip_time_period(df, start, end, verbose=True):
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
    """Returns the disjunctive union or symmetric difference between to sets, expressed as lists.
    
    Args:
        li1 (list): first set.
        li2 (list): second set.
    
    Returns:
        list
    """
    
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))


