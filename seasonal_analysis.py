from audioop import add
from selectors import SelectSelector
from typing import Dict, List
import pandas as pd
from utils import count_zeros, SMAPE, sample

__all__ = ['SeasonalAnalysis',
           ]

def squared_error(x : pd.Series, y : pd.Series) -> pd.Series:
    return (x - y)**2

def absolute_error(x : pd.Series, y : pd.Series) -> pd.Series:
    return abs(x - y)

def error(x : pd.Series, y : pd.Series) -> pd.Series:
    return x - y

def mean_of_samples(df : pd.DataFrame, field_prefix : str, n_samples : int) -> pd.Series:
    return df[[f'{field_prefix}_{i}' for i in range(n_samples)]].mean(axis=1)

def rearrange_dataframe(df : pd.DataFrame, melt_id_vars : List[str], melt_value_vars : List[str]) -> pd.DataFrame:
    return df.reset_index().melt(id_vars=melt_id_vars, value_vars=melt_value_vars)

class SeasonalAnalysis:

    def __init__(self, 
                df : pd.DataFrame, 
                columns : List(str), 
                sample_cols : List[str], 
                add_cols : List[str],
                n_samples : int,
                group_by_fields : List[str] = ['Station','season','year']):
        
        self.df  = df
        self.columns = columns
        self.sample_cols = sample_cols
        self.add_cols = add_cols
        self.n_samples = n_samples
        self.group_by_fields = group_by_fields
        
    def aggregate_precipitation_intensity_predictions(self) -> pd.DataFrame:
        return self.df.groupby(self.group_by_fields).sum()[(self.columns + self.sample_cols + self.add_cols)].copy()

    def aggregate_precipitation_occurrence_predicitons(self) -> pd.DataFrame:
        return self.df.groupby(self.group_by_fields, as_index=False)[(self.columns + self.sample_cols + self.add_cols)].agg([count_zeros]).droplevel(level=1, axis=1) # Crate dataframe for precipitation occurrence
        

    def absolute_error_dry_days(self) -> pd.DataFrame:
        
        keep_columns = []

        # DRY DAYS - ABSOLUTE ERROR 
        for i in range(self.n_samples):
            self.df[f'edd_mlp_{i}'] = absolute_error(self.df[f'sample_{i}'], self.df['Prec'])
            keep_columns.append(f'edd_mlp_{i}')

        self.df['edd_mlp'] = self.df[[f'edd_mlp_{i}' for i in range(self.n_samples)]].mean(axis=1)
        keep_columns.append('edd_mlp')
        
        for c in self.columns:
            self.df[f'edd_{c}'] = abs(self.df[c] - self.df['Prec'])
            keep_columns.append(f'edd_{c}')

        return self.df[keep_columns]

    def seasonal_analysis(self) -> pd.DataFrame: 
        """Groups results by station, season and year and computes various assessment metrics."""

        # THIS FUNCTION IS NOT PART OF A CLASS HOWEVER MOST OBJECTS ARE CREATED NOT AS PROPERTIES BUT OUTSIDE OF CLASS. REFACTOR. 

        df = self.aggregate_precipitation_intensity_predictions() # group by station, season and year (default)
        
        df_dry_days = self.aggregate_precipitation_occurrence_predicitons()

        # df = st_test.groupby(['Station','season','year']).sum()[(columns + sample_cols + add_cols)].copy()

        df_dry_days = self.absolute_error_dry_days()
        
        # PRECIPITATION - SQUARED ERROR 
        for c in self.columns:
            df[f'se_{c}'] = squared_error(df[c], df['Prec'])
            
    #     df['se_wrf_prcp'] = (df['wrf_prcp'] - df['Prec'])**2
    #     df['se_wrf_bc_prcp'] = (df['wrf_bc_prcp'] - df['Prec'])**2

        df['sample'] = df[self.sample_cols].mean(axis=1)

        for i in range(self.n_samples):
            df[f'se_mlp_{i}'] = squared_error(df[f'sample_{i}'], df['Prec'])**2

        df['se_mlp'] = mean_of_samples(df, 'se_mlp', self.n_samples)

        # PRECIPITATION - ERROR & ABSOLUTE ERROR
        for c in self.columns:
            df[f'e_{c}'] = error(df[c], df['Prec'])
            df[f'ae_{c}'] = absolute_error(df[c], df['Prec'])
        
    #     df['e_wrf_prcp'] = (df['wrf_prcp'] - df['Prec'])
    #     df['e_wrf_bc_prcp'] = (df['wrf_bc_prcp'] - df['Prec'])  

    #     df['ae_wrf_prcp'] = abs(df['wrf_prcp'] - df['Prec'])
    #     df['ae_wrf_bc_prcp'] = abs(df['wrf_bc_prcp'] - df['Prec'])

        for i in range(self.n_samples):
            df[f'e_mlp_{i}'] = error(df[f'sample_{i}'], df['Prec'])
            df[f'ae_mlp_{i}'] = absolute_error(df[f'sample_{i}'], df['Prec'])


        df['e_mlp'] = mean_of_samples(df, 'e_mlp', self.n_samples)
        df['ae_mlp'] = mean_of_samples(df, 'ae_mlp', self.n_samples)

        # PRECIPITATION - ABSOLUTE ERROR REDUCTION
        for c in self.columns:
            df[f'aer_{c}'] = df['ae_wrf_prcp'] - df[f'ae_{c}'] 
        
        df['aer_mlp'] = df['ae_wrf_prcp'] - df['ae_mlp']
        #df['aer_wrf_bc_prcp'] = df['ae_wrf_prcp'] - df['ae_wrf_bc_prcp']

        # PRECIPITATION - MSE IMPROVEMENT RATIO
        for c in self.columns:
            df[f'imp_{c}'] = 1 - df[f'se_{c}']/(df['se_wrf_prcp'])
        
        #df['imp_wrf_bc_prcp'] = 1 - df['se_wrf_bc_prcp']/(df['se_wrf_prcp'])
        
        df['imp_mlp'] = 1 - df['se_mlp']/(df['se_wrf_prcp'])

        # PRECIPITATION - SMAPE
        for c in self.columns:
            df[f'smape_{c}'] = df.apply(SMAPE, axis=1, args=(c, 'Prec'))
        
    #     df['smape_wrf_prcp'] = df.apply(SMAPE, axis=1, args=('wrf_prcp','Prec')) 
    #     df['smape_wrf_bc_prcp'] = df.apply(SMAPE, axis=1, args=('wrf_bc_prcp','Prec')) 
        
        df['smape_mlp'] = df.apply(SMAPE, axis=1, args=('sample','Prec')) 

        if 'mean' in self.add_cols:
            df['smape_mlp_mean'] = df.apply(SMAPE, axis=1, args=('mean','Prec'))  
        if 'median' in self.add_cols:
            df['smape_mlp_median'] = df.apply(SMAPE, axis=1, args=('median','Prec')) 

        df = pd.merge(df,df_dry_days,on=['Station','season','year'])

        return df

    def seasonal_summaries(self) -> Dict[str, pd.DataFrame]:
        
        # Totals
        value_vars = ['Prec','wrf_prcp','wrf_bc_prcp','sample'] + self.add_cols
        self.totals = rearrange_dataframe(self.df, self.group_by_fields, value_vars)
        
        # Error
        value_vars = [f'e_{c}' for c in self.columns] + ['e_mlp']
        self.e = rearrange_dataframe(self.df, self.group_by_fields, value_vars)

        # MAE
        value_vars = [f'ae_{c}' for c in self.columns] + ['ae_mlp']
        self.ae = rearrange_dataframe(self.df, self.group_by_fields, value_vars)

        # SE
        value_vars = [f'se_{c}' for c in self.columns] + ['se_mlp']
        self.se = rearrange_dataframe(self.df, self.group_by_fields, value_vars)

        # MAE REDUCTION
        value_vars = [f'aer_{c}' for c in self.columns] + ['aer_mlp']
        self.aer = rearrange_dataframe(self.df, self.group_by_fields, value_vars)

        # ERROR IN DRY DAYS 
        value_vars = [f'edd_{c}' for c in self.columns] + ['edd_mlp']
        self.edd = rearrange_dataframe(self.df, self.group_by_fields, value_vars)

        # MSE IMPROVEMENT RATIO   
        value_vars = [f'imp_{c}' for c in self.columns] + ['imp_mlp']
        self.improvement = rearrange_dataframe(self.df, self.group_by_fields, value_vars)

        # SMAPE
        value_vars = [f'smape_{c}' for c in self.columns] + ['smape_mlp'] + [f'smape_mlp_{i}' for i in self.add_cols]
        self.smape = rearrange_dataframe(self.df, self.group_by_fields, value_vars)

        d = {}
        
        d['totals'] = self.totals
        d['e'] = self.e
        d['ae'] = self.ae
        d['se'] = self.se
        d['aer'] = self.aer
        d['edd'] = self.edd
        d['improvement'] = self.improvement
        d['smape'] = self.smape
        
        return d