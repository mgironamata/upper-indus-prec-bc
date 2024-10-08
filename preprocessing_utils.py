import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
import random, pickle

import torch
from torch.utils.data import TensorDataset, DataLoader 

random.seed(42)

__all__ = [
            # 'import_dataframe',
            # '_drop_dataframe_nan_values',
            '_filter_complete_station_years',
            # 'clip_time_period',
           'list_bc_stations',
            # 'log_transform',
           'dry_days_binomial',
           'disjunctive_union_lists',
           'FilterByList',
           'sort_by_quantile',
            # 'add_year_month_season',
            # 'season_apply',
            'add_yesterday_observation',
            # 'create_cv_held_out_sets',
            'create_station_dataframe',
           'create_input_data',
           'create_dataset',
           'DataPreprocessing'  
           ]

class DataPreprocessing():

    def __init__(self, 
                train_path : str, 
                start : str, 
                end : str, 
                add_yesterday : bool = True, 
                basin_filter : str = None, 
                split_bias_corrected_only = True, # if True, validation and test sets will only include bias corrected stations.
                filter_incomplete_years = True,
                include_non_bc_stations = True,
                split_by = 'station'
                ) -> None:
        
        self.train_path = train_path
        self.start = start
        self.end = end
        self.add_yesterday = add_yesterday
        self.basin_filter = basin_filter
        self.filter_incomplete_years = filter_incomplete_years
        self.split_bias_corrected_only = split_bias_corrected_only
        self.include_non_bc_stations = include_non_bc_stations
        self.split_by = split_by

        
        # Create station dataframe
        self.st = create_station_dataframe(self.train_path, self.start, self.end, add_yesterday=True, basin_filter=None, filter_incomplete_years = filter_incomplete_years)
        self.station_dict = self.create_station_dict()
        self.st['StationNum'] = pd.factorize(self.st.Station)[0]

    def split_stations(self, sort_by_elev = True):

        self.sort_by_elev = sort_by_elev

        if self.sort_by_elev:
            self.st = self.st.sort_values(by='Z', ascending=True)

        if self.split_bias_corrected_only:
            if self.sort_by_elev:
                raise ValueError("sort_by_elev must be False if split_bias_corrected_only is True")

            #  List bias-corrected and non-bias-corrected stations
            self.bc_stations = list_bc_stations(self.st)
            self.non_bc_stations = disjunctive_union_lists(self.st['Station'].unique(), self.bc_stations)

            # Set of stations to be split into training, validation and test held out sets.
            self.st_names = list(set(self.bc_stations) & set(self.st['Station'].unique()))

        else:

            # self.st_names = self.st['Station'][self.st['set']=='train'].unique()
            self.bc_stations = self.st['Station'].unique()
            self.non_bc_stations = []
            self.st_names = self.st['Station'].unique()
            self.st_elevs = self.st['Z'].unique()

        self.st_names = np.array(self.st_names)

        # np.random.shuffle(self.st_names) # shuffle stations 

        split = round(len(self.st_names) * 0.2)

        self.st_names_dict = {}

        # st_names_dict['train'] = list(st_names)#[:split*4])
        # st_names_dict['val'] = list(st_names_test)#[split*4:])
        # st_names_dict['test'] = list(st_names_test)

        self.st_names_dict['train'] = list(self.st_names[:split*3])    
        self.st_names_dict['val'] = list(self.st_names[split*3:split*4])
        self.st_names_dict['test'] = list(self.st_names[split*4:split*5]) 

        if (self.split_bias_corrected_only) & (self.include_non_bc_stations):
            self.st_names_dict['train'] += self.non_bc_stations

        # print("%s stations used for training, %s used for validation, and %s used testing" % (len(self.st_names_dict['train']), len(self.st_names_dict['val']), len(self.st_names_dict['test'])))
        
        if self.split_by == 'region':
            if self.sort_by_elev:
                raise ValueError("sort_by_elev must be False if split_by is region")
            self.split_dict = self.split_dict_by_region()
            
        else:
            self.split_dict = create_cv_held_out_sets(st_names = self.st_names, 
                                                  non_bc_st_names = self.non_bc_stations,
                                                  split_by = self.split_by, 
                                                  include_non_bc_stations = self.include_non_bc_stations,
                                                  sort_by_elevation = self.sort_by_elev)

    def split_dict_by_region(self):
        self.regions = self.st.Region.unique()

        split_dict = {}

        for i,r in enumerate(self.regions):
            train_val_split = self.st[self.st['Region']!=r].Station.unique()
            random.shuffle(train_val_split, )
            test_split = self.st[self.st['Region']==r].Station.unique()
            train_split = train_val_split[:round(len(train_val_split)*0.8)]
            val_split = train_val_split[round(len(train_val_split)*0.8):]
            split_dict[f'k{i}'] = {"train":train_split, "val":val_split, "test":test_split}
        
        return split_dict

    def create_station_dict(self):
        self.station_dict = {}
        for s in self.st['Station'].unique():
            self.station_dict[s] = self.st[self.st['Station']==s]

    def load_split_dict(self, pickle_path = "split_dict.pickle"):
        self.loaded_dictionary = pickle.load(open(pickle_path, "rb"))
        self.split_dict = self.loaded_dictionary.copy()
    
    def dump_split_dict(self, dump_path = "split_dict.pickle"):
        with open(dump_path, 'wb') as handle:
            pickle.dump(self.split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def print_split_dict(self):
        
        print("# Split dict:")
        for k,v in self.split_dict.items():
            print(" " + k)
            for kk,vv in v.items():
                print(f"  {kk} = {len(vv)}")
                print(vv)

    def input_data(self, predictors, predictand, sort_by_quantile=False):
        
        self.sort_by_quantile = sort_by_quantile
        self.data, self.x_mean, self.x_std = create_input_data(self.st, predictors, predictand, self.split_dict, split_by=self.split_by, sort_by_quantile_flag=self.sort_by_quantile)

        self.years_dict = {}
        self.years_dict['train'] = list(range(1998,2008))

        self.years_list = disjunctive_union_lists(list(self.st['year'].unique()),list(range(1998,2008)))
        random.Random(5).shuffle(self.years_list)

        self.years_dict['val'] = self.years_list[:10] #list(range(2004,2006))
        self.years_dict['test'] = self.years_list[10:] #list(range(2006,2008))
        
        self.d = len(predictors) # define number of input dimensions

        self.splits = ['train', 'val', 'test']

        # years = range(1998,2005)

        for i in self.splits:
            
            if (self.split_by=='station') or (self.split_by=='region'):
                self.data[f'X_{i}'] = (self.st[self.st['Station'].isin(self.st_names_dict[f'{i}'])][predictors].to_numpy() - self.x_mean) / self.x_std
                self.data[f'Y_{i}'] = self.st[self.st['Station'].isin(self.st_names_dict[f'{i}'])][predictand].to_numpy()
                self.data[f'S_{i}'] = self.st[self.st['Station'].isin(self.st_names_dict[f'{i}'])][['StationNum']].to_numpy()

            elif self.split_by=='year':
                self.data[f'X_{i}'] = (self.st[self.st['year'].isin(self.years_dict[f'{i}'])][predictors].to_numpy() - self.x_mean) / self.x_std
                self.data[f'Y_{i}'] = self.st[self.st['year'].isin(self.years_dict[f'{i}'])][predictand].to_numpy()
                self.data[f'S_{i}'] = self.st[self.st['year'].isin(self.years_dict[f'{i}'])]['StationNum'].to_numpy()
                
        self.train_dataset = create_dataset(data=self.data,split='train',d=self.d)
        self.val_dataset = create_dataset(data=self.data,split='val',d=self.d)
        self.test_dataset = create_dataset(data=self.data,split='test',d=self.d)
            
        # train_tensor_x = torch.Tensor(data['X_train'][:,:d]) # transform to torch tensor
        # train_tensor_y = torch.Tensor(data['Y_train'][:,:d])
        # train_dataset = TensorDataset(train_tensor_x,train_tensor_y) # create your dataset

        # val_tensor_x = torch.Tensor(data['X_val'][:,:d]) # transform to torch tensor
        # val_tensor_y = torch.Tensor(data['Y_val'][:,:d])
        # val_dataset = TensorDataset(val_tensor_x,val_tensor_y) # create your dataset

        # test_tensor_x = torch.Tensor(data['X_test'][:,:d]) # transform to torch tensor
        # test_tensor_y = torch.Tensor(data['Y_test'][:,:d])
        # test_dataset = TensorDataset(test_tensor_x,test_tensor_y) # create your dataset

        # test2_tensor_x = torch.Tensor(data['X_test2'][:,:d]) # transform to torch tensor
        # test2_tensor_y = torch.Tensor(data['Y_test2'][:,:d])
        # test2_dataset = TensorDataset(test2_tensor_x,test2_tensor_y) # create your dataset

def sort_by_quantile(st, sort_by = 'precip_norris'):
    """Re-arrange dataframe of station data so that model simulations and observations match 
    based on quantiles, for each station separately
    
    Inputs:
    
    Outputs:
    
    """
    
    QM_data = {}
    list_stations = st['Station'].unique()
    
    for i, s in enumerate(list_stations):
        QM_data[s] = st[st['Station']==s].sort_values(by=sort_by)
        QM_data[s]['Prec'] = QM_data[s]['Prec'].sort_values().values

        if i == 0:
            QM_df = QM_data[s]
        else:
            QM_df = pd.concat([QM_df,QM_data[s]])

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

def _drop_dataframe_nan_values(df, series='Prec', verbose=False):
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

def replace_negative_values(df, series='Prec', verbose=False):

    # df[series][df[series] < 0] = 0
    df.loc[df[series] < 0, series] = 0
    
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
    # st['season'] = st.apply(season_apply, axis=1) 
    return st

def _filter_complete_station_years(df):
    grouped_df = df.groupby(['Station','year']).count().reset_index()
    pairs = list(grouped_df[grouped_df['Prec']>=365][['Station','year']].apply(tuple,1))
    df = df[df[['Station','year']].apply(tuple, 1).isin(pairs)]
    return df

def FilterByList(df,series,value_list):
    return df[df[series].isin(value_list)]

def filter_by_series(df, series, value):
    return df[df[series]==value]

def list_bc_stations(df, raw_field = 'wrf_prcp', bc_field = 'wrf_bc_prcp'):
    df['BC_diff'] = df[raw_field] - df[bc_field]
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

def season_apply(df):
    # Premonsoon: Apr-May
    if (df['month'] >= 4) and (df['month'] <= 5):
        return 'AM'
    # Monsoon: June-Jul-Aug-Sept
    elif (df['month'] >= 6) and (df['month'] <= 9):
        return 'JJAS'
    # Postmonsoon: Oct-Nov-Dec
    elif (df['month'] >= 10) and (df['month'] <= 12):
        return 'OND'
    # Winter: Jan-Feb-Mar
    elif (df['month'] >= 1) or (df['month'] <= 3):
        return 'JFM'

def add_yesterday_observation(st):
    """Add observed precipitation value from previous day"""
    st['obs_yesterday'] = st.groupby('Station')['Prec'].shift(1)
    st.dropna(inplace=True)
    return st

def create_cv_held_out_sets(st_names, non_bc_st_names = [], split_by = 'station', include_non_bc_stations = True, sort_by_elevation = False):

    if split_by == 'station':
        
        split_dict = {}
        if sort_by_elevation:
            kf = KFold(n_splits=10,shuffle=False)
        else:
            kf = KFold(n_splits=10,shuffle=True, random_state=42)
        #kf.get_n_splits(st_names)

        for i, (train_index, test_index) in enumerate(kf.split(st_names)):
            k = round(len(train_index)*0.8)
            split_dict[f'k{i}'] = {f'train': list(st_names[train_index[:k]]) + non_bc_st_names if include_non_bc_stations else list(st_names[train_index[:k]]),
                                   f'val' : st_names[train_index[k:]],
                                   f'test' : st_names[test_index[:]]
                                }
            
    elif split_by == 'year':
        
        split_dict = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        years=np.array(range(1998,2008))

        for i, (train_index, test_index) in enumerate(kf.split(years)):
            k = round(len(train_index)*0.75)
            split_dict[f'k{i}'] = {f'train': list(years[train_index[:k]]),
                                   f'val' : years[train_index[k:]],
                                   f'test' : years[test_index[:]]
                                }
    
    return split_dict

def calculate_doy_columns(st):
    st['doy'] = st['Date'].dt.dayofyear
    st['doy_sin'] = st['doy'].transform(lambda x: np.sin(x))
    st['doy_cos'] = st['doy'].transform(lambda x: np.cos(x))

    return st

def create_station_dataframe(TRAIN_PATH: str, start: str, end: str, add_yesterday: str = True, 
                             basin_filter = None, filter_incomplete_years = True):

    st = (import_dataframe(TRAIN_PATH)
        .pipe(_drop_dataframe_nan_values, series='Prec')
        .pipe(replace_negative_values, series='Prec')
        .pipe(clip_time_period, start, end)
        .pipe(add_year_month_season)
        .pipe(calculate_doy_columns)
        )  

    # Add yesterday's observation
    if add_yesterday: st = add_yesterday_observation(st)

    # Filter incomplete years
    if filter_incomplete_years: st = _filter_complete_station_years(st)

    # Filter by basin
    if basin_filter is not None: st = st[st['Basin']==basin_filter]

    # st['set'] = "train" 

    # st_val = (import_dataframe(TEST_PATH) # Import dataframe
    #     .pipe(_drop_dataframe_nan_values, series='Prec') # Drop NaNs
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

        if (split_by=='station') or (split_by=='region'):
            
            data[f'set_train_{i}'] = st[st['Station'].isin(split_dict[f'k{i}']['train'])]
            data[f'set_val_{i}'] = st[st['Station'].isin(split_dict[f'k{i}']['val'])]
            data[f'set_test_{i}'] = st[st['Station'].isin(split_dict[f'k{i}']['test'])]

        elif split_by=='year':

            data[f'set_train_{i}'] = st[st['year'].isin(split_dict[f'k{i}']['train'])]
            data[f'set_val_{i}'] = st[st['year'].isin(split_dict[f'k{i}']['val'])]
            data[f'set_test_{i}'] = st[st['year'].isin(split_dict[f'k{i}']['test'])]

        if sort_by_quantile_flag:
            data[f'set_train_{i}'] = sort_by_quantile(data[f'set_train_{i}'])
            data[f'set_val_{i}'] = sort_by_quantile(data[f'set_val_{i}'])
            data[f'set_test_{i}'] = sort_by_quantile(data[f'set_test_{i}'])
                       
        data[f'X_train_{i}'] = (data[f'set_train_{i}'][predictors].to_numpy() - x_mean) / x_std
        data[f'X_val_{i}'] = (data[f'set_val_{i}'][predictors].to_numpy() - x_mean) / x_std
        data[f'X_test_{i}'] = (data[f'set_test_{i}'][predictors].to_numpy() - x_mean) / x_std

        data[f'Y_train_{i}'] = data[f'set_train_{i}'][predictand].to_numpy()
        data[f'Y_val_{i}'] = data[f'set_val_{i}'][predictand].to_numpy()
        data[f'Y_test_{i}'] = data[f'set_test_{i}'][predictand].to_numpy()

        data[f'S_train_{i}'] = data[f'set_train_{i}'][['StationNum']].to_numpy()
        data[f'S_val_{i}'] = data[f'set_val_{i}'][['StationNum']].to_numpy()
        data[f'S_test_{i}'] = data[f'set_test_{i}'][['StationNum']].to_numpy()        
        
            # data[f'X_train_{i}'] = (st[st['year'].isin(split_dict[f'k{i}']['train'])][predictors].to_numpy() - x_mean) / x_std
            # data[f'X_val_{i}'] = (st[st['year'].isin(split_dict[f'k{i}']['val'])][predictors].to_numpy() - x_mean) / x_std
            # data[f'X_test_{i}'] = (st[st['year'].isin(split_dict[f'k{i}']['test'])][predictors].to_numpy() - x_mean) / x_std

            # data[f'Y_train_{i}'] = st[st['year'].isin(split_dict[f'k{i}']['train'])][predictand].to_numpy()
            # data[f'Y_val_{i}'] = st[st['year'].isin(split_dict[f'k{i}']['val'])][predictand].to_numpy()
            # data[f'Y_test_{i}'] = st[st['year'].isin(split_dict[f'k{i}']['test'])][predictand].to_numpy()

            # data[f'S_train_{i}'] = data[f'set_train_{i}'][['StationNum']].to_numpy()
            # data[f'S_val_{i}'] = data[f'set_val_{i}'][['StationNum']].to_numpy()
            # data[f'S_test_{i}'] = data[f'set_test_{i}'][['StationNum']].to_numpy()

    # data['X_test'] = (st[st['Station'].isin(st_names_dict['test'])][predictors].to_numpy() - x_mean) / x_std
    # data['Y_test'] = st[st['Station'].isin(st_names_dict['test'])][predictand].to_numpy()

    # data['X_test2'] = (st_val[predictors].to_numpy() - x_mean) / x_std
    # data['Y_test2'] = st_val[predictand].to_numpy()

    return data, x_mean, x_std

def create_dataset(data, split, d):
    tensor_x = torch.Tensor(data[f'X_{split}'][:,:d])
    tensor_y = torch.Tensor(data[f'Y_{split}'][:,:d])
    return TensorDataset(tensor_x, tensor_y)