from typing import Dict, List
import pandas as pd
from utils import count_zeros, SMAPE, sample

__all__ = ['SeasonalAnalysis',
            ]

def squared_error(x : pd.Series, y : pd.Series) -> pd.Series:
    "Returns squared error between 2 pd.Series"
    return (x - y)**2

def absolute_error(x : pd.Series, y : pd.Series) -> pd.Series:
    "Returns absolute error between 2 pd.Series"
    return abs(x - y)

def error(x : pd.Series, y : pd.Series) -> pd.Series:
    "Returns error between 2 pd.Series"
    return x - y

def mean_of_samples(df : pd.DataFrame, field_prefix : str, n_samples : int) -> pd.Series:
    "Returns average of all samples as pd.Series"
    return df[[f'{field_prefix}_{i}' for i in range(n_samples)]].mean(axis=1)

def rearrange_dataframe(df : pd.DataFrame, melt_id_vars : List[str], melt_value_vars : List[str]) -> pd.DataFrame:
    "Returns rearranged pd.Dataframe"
    return df.reset_index().melt(id_vars=melt_id_vars, value_vars=melt_value_vars)

class SeasonalAnalysis:
    "Constructs an object to perform seasonal analyses of daily data"

    def __init__(self, 
                df : pd.DataFrame, 
                columns : List[str], 
                sample_columns : List[str], 
                additional_columns : List[str],
                n_samples : int,
                group_by_fields : List[str] = ['Station','season','year']):
        
        self.df  = df
        self.columns = columns
        self.sample_columns = sample_columns
        self.additional_columns = additional_columns
        self.n_samples = n_samples
        self.group_by_fields = group_by_fields
        
    def _aggregate_precipitation_intensity_predictions(self) -> pd.DataFrame:
        return self.df.groupby(self.group_by_fields).sum()[(self.columns + self.sample_columns + self.additional_columns)].copy()

    def _aggregate_precipitation_occurrence_predicitons(self) -> pd.DataFrame:
        return self.df.groupby(self.group_by_fields)[(self.columns + self.sample_columns + self.additional_columns)].agg([count_zeros]).droplevel(level=1, axis=1).copy() # Crate dataframe for precipitation occurrence
        
    def _absolute_error_dry_days(self) -> pd.DataFrame:
        
        keep_columns = []

        for idx in self.df_agg_dry_days.index.names: keep_columns.append(idx)

        # DRY DAYS - ABSOLUTE ERROR 
        for i in range(self.n_samples):
            self.df_agg_dry_days[f'edd_mlp_{i}'] = absolute_error(self.df_agg_dry_days[f'sample_{i}'], self.df_agg_dry_days['Prec'])
            keep_columns.append(f'edd_mlp_{i}')

        self.df_agg_dry_days['edd_mlp'] = self.df_agg_dry_days[[f'edd_mlp_{i}' for i in range(self.n_samples)]].mean(axis=1)
        keep_columns.append('edd_mlp')
        
        for c in self.columns:
            self.df_agg_dry_days[f'edd_{c}'] = abs(self.df_agg_dry_days[c] - self.df_agg_dry_days['Prec'])
            keep_columns.append(f'edd_{c}')

        self.df_agg_dry_days = self.df_agg_dry_days.reset_index()

        return self.df_agg_dry_days[keep_columns]

    def _seasonal_analysis(self) -> pd.DataFrame: 
        """Groups results by station, season and year and computes various assessment metrics."""

        # THIS FUNCTION IS NOT PART OF A CLASS HOWEVER MOST OBJECTS ARE CREATED NOT AS OUTSIDE PROPERTIES BUT OF CLASS. REFACTOR. 

        self.df_agg = self._aggregate_precipitation_intensity_predictions() # group by station, season and year (default)
        
        self.df_agg_dry_days = self._aggregate_precipitation_occurrence_predicitons()

        # df = st_test.groupby(['Station','season','year']).sum()[(columns + sample_columns + additional_columns)].copy()

        self.df_agg_dry_days = self._absolute_error_dry_days()
        
        # PRECIPITATION - SQUARED ERROR 
        for c in self.columns:
            self.df_agg[f'se_{c}'] = squared_error(self.df_agg[c], self.df_agg['Prec'])
            
    #     df['se_wrf_prcp'] = (df['wrf_prcp'] - df['Prec'])**2
    #     df['se_wrf_bc_prcp'] = (df['wrf_bc_prcp'] - df['Prec'])**2

        self.df_agg['sample'] = self.df_agg[self.sample_columns].mean(axis=1)

        for i in range(self.n_samples):
            self.df_agg[f'se_mlp_{i}'] = squared_error(self.df_agg[f'sample_{i}'], self.df_agg['Prec'])**2

        self.df_agg['se_mlp'] = mean_of_samples(self.df_agg, 'se_mlp', self.n_samples)

        # PRECIPITATION - ERROR & ABSOLUTE ERROR
        for c in self.columns:
            self.df_agg[f'e_{c}'] = error(self.df_agg[c], self.df_agg['Prec'])
            self.df_agg[f'ae_{c}'] = absolute_error(self.df_agg[c], self.df_agg['Prec'])
        
    #     self.df_agg['e_wrf_prcp'] = (self.df_agg['wrf_prcp'] - self.df_agg['Prec'])
    #     self.df_agg['e_wrf_bc_prcp'] = (self.df_agg['wrf_bc_prcp'] - self.df_agg['Prec'])  

    #     self.df_agg['ae_wrf_prcp'] = abs(self.df_agg['wrf_prcp'] - self.df_agg['Prec'])
    #     self.df_agg['ae_wrf_bc_prcp'] = abs(self.df_agg['wrf_bc_prcp'] - self.df_agg['Prec'])

        for i in range(self.n_samples):
            self.df_agg[f'e_mlp_{i}'] = error(self.df_agg[f'sample_{i}'], self.df_agg['Prec'])
            self.df_agg[f'ae_mlp_{i}'] = absolute_error(self.df_agg[f'sample_{i}'], self.df_agg['Prec'])


        self.df_agg['e_mlp'] = mean_of_samples(self.df_agg, 'e_mlp', self.n_samples)
        self.df_agg['ae_mlp'] = mean_of_samples(self.df_agg, 'ae_mlp', self.n_samples)

        # PRECIPITATION - ABSOLUTE ERROR REDUCTION
        for c in self.columns:
            self.df_agg[f'aer_{c}'] = self.df_agg['ae_wrf_prcp'] - self.df_agg[f'ae_{c}'] 
        
        self.df_agg['aer_mlp'] = self.df_agg['ae_wrf_prcp'] - self.df_agg['ae_mlp']
        #self.df_agg['aer_wrf_bc_prcp'] = self.df_agg['ae_wrf_prcp'] - self.df_agg['ae_wrf_bc_prcp']

        # PRECIPITATION - MSE IMPROVEMENT RATIO
        for c in self.columns:
            self.df_agg[f'imp_{c}'] = 1 - self.df_agg[f'se_{c}']/(self.df_agg['se_wrf_prcp'])
        
        #self.df_agg['imp_wrf_bc_prcp'] = 1 - self.df_agg['se_wrf_bc_prcp']/(self.df_agg['se_wrf_prcp'])
        
        self.df_agg['imp_mlp'] = 1 - self.df_agg['se_mlp']/(self.df_agg['se_wrf_prcp'])

        # PRECIPITATION - SMAPE
        for c in self.columns:
            self.df_agg[f'smape_{c}'] = self.df_agg.apply(SMAPE, axis=1, args=(c, 'Prec'))
        
    #     self.df_agg['smape_wrf_prcp'] = self.df_agg.apply(SMAPE, axis=1, args=('wrf_prcp','Prec')) 
    #     self.df_agg['smape_wrf_bc_prcp'] = self.df_agg.apply(SMAPE, axis=1, args=('wrf_bc_prcp','Prec')) 
        
        self.df_agg['smape_mlp'] = self.df_agg.apply(SMAPE, axis=1, args=('sample','Prec')) 

        if 'mean' in self.additional_columns:
            self.df_agg['smape_mlp_mean'] = self.df_agg.apply(SMAPE, axis=1, args=('mean','Prec'))  
        if 'median' in self.additional_columns:
            self.df_agg['smape_mlp_median'] = self.df_agg.apply(SMAPE, axis=1, args=('median','Prec')) 

        self.df_agg = pd.merge(self.df_agg,self.df_agg_dry_days,on=['Station','season','year'])

        return self.df_agg

    def seasonal_summaries(self) -> Dict[str, pd.DataFrame]:

        self.sa = self._seasonal_analysis()
        
        # Totals
        value_vars = ['Prec','wrf_prcp','wrf_bc_prcp','sample'] + self.additional_columns
        self.totals = rearrange_dataframe(self.sa, self.group_by_fields, value_vars)
        
        # Error
        value_vars = [f'e_{c}' for c in self.columns] + ['e_mlp']
        self.e = rearrange_dataframe(self.sa, self.group_by_fields, value_vars)

        # MAE
        value_vars = [f'ae_{c}' for c in self.columns] + ['ae_mlp']
        self.ae = rearrange_dataframe(self.sa, self.group_by_fields, value_vars)

        # SE
        value_vars = [f'se_{c}' for c in self.columns] + ['se_mlp']
        self.se = rearrange_dataframe(self.sa, self.group_by_fields, value_vars)

        # MAE REDUCTION
        value_vars = [f'aer_{c}' for c in self.columns] + ['aer_mlp']
        self.aer = rearrange_dataframe(self.sa, self.group_by_fields, value_vars)

        # ERROR IN DRY DAYS 
        value_vars = [f'edd_{c}' for c in self.columns] + ['edd_mlp']
        self.edd = rearrange_dataframe(self.sa, self.group_by_fields, value_vars)

        # MSE IMPROVEMENT RATIO   
        value_vars = [f'imp_{c}' for c in self.columns] + ['imp_mlp']
        self.improvement = rearrange_dataframe(self.sa, self.group_by_fields, value_vars)

        # SMAPE
        value_vars = [f'smape_{c}' for c in self.columns] + ['smape_mlp'] + [f'smape_mlp_{i}' for i in self.additional_columns]
        self.smape = rearrange_dataframe(self.sa, self.group_by_fields, value_vars)

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

if __name__ == "__main__":

    df = pd.read_csv("tempCSV4SA.csv")
    columns = ['Prec','wrf_prcp','wrf_bc_prcp','precip_norris']
    n_samples = 10
    sample_cols = [f'sample_{i}' for i in range(n_samples)]
    add_cols = []    

    SA = SeasonalAnalysis(df, columns, sample_columns=sample_cols, additional_columns=add_cols, n_samples=n_samples)

    d = SA.seasonal_summaries()

    print("Done")