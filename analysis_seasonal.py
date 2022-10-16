from typing import Dict, List
import pandas as pd
import numpy as np
from utils import count_zeros
from metrics import absolute_error, squared_error, error, SMAPE
from tabulate import tabulate
import scipy

"""
TO DO:
------
-> INTEGREATE TABLE PLOTTING FUNCTIONS INTO SEASONAL ANALYSIS CLASS

"""

__all__ = [ 
            'SeasonalAnalysis',
            'table_of_predictions_confidence_intervals',
            'table_of_predictions_for_metric',
            'table_of_predictions_ks_test'
          ]

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

        self.df_agg = self._aggregate_precipitation_intensity_predictions() # group by station, season and year (default)
        self.df_agg_dry_days = self._aggregate_precipitation_occurrence_predicitons()
        self.df_agg_dry_days = self._absolute_error_dry_days()
        
        for c in self.columns:
            self.df_agg[f'se_{c}'] = squared_error(self.df_agg[c], self.df_agg['Prec']) # squared error
            self.df_agg[f'e_{c}'] = error(self.df_agg[c], self.df_agg['Prec']) # simple error
            self.df_agg[f'ae_{c}'] = absolute_error(self.df_agg[c], self.df_agg['Prec']) # absolute error
            self.df_agg[f'smape_{c}'] = self.df_agg.apply(SMAPE, axis=1, args=(c, 'Prec'))
        
        for c in self.columns:
            self.df_agg[f'aer_{c}'] = self.df_agg['ae_wrf_prcp'] - self.df_agg[f'ae_{c}'] # absolute error reduction
            self.df_agg[f'imp_{c}'] = 1 - self.df_agg[f'se_{c}']/(self.df_agg['se_wrf_prcp']) # improvement ratio
            
        self.df_agg['sample'] = self.df_agg[self.sample_columns].mean(axis=1) # mean of samples

        # OPTION 1 - COMPUTE METRIC PER SAMPLE, THEN AVERAGE ACROSS SAMPLES
        for i in range(self.n_samples):
            self.df_agg[f'se_mlp_{i}'] = squared_error(self.df_agg[f'sample_{i}'], self.df_agg['Prec'])**2
            self.df_agg[f'e_mlp_{i}'] = error(self.df_agg[f'sample_{i}'], self.df_agg['Prec'])
            self.df_agg[f'ae_mlp_{i}'] = absolute_error(self.df_agg[f'sample_{i}'], self.df_agg['Prec'])
            self.df_agg[f'smape_mlp{i}'] = self.df_agg.apply(SMAPE, axis=1, args=(f'sample_{i}','Prec')) 
        
        self.df_agg['se_mlp'] = mean_of_samples(self.df_agg, 'se_mlp', self.n_samples)
        self.df_agg['e_mlp'] = mean_of_samples(self.df_agg, 'e_mlp', self.n_samples)
        self.df_agg['ae_mlp'] = mean_of_samples(self.df_agg, 'ae_mlp', self.n_samples)
        self.df_agg['smape_mlp'] = mean_of_samples(self.df_agg, 'smape_mlp',self.n_samples)
            
        self.df_agg['aer_mlp'] = self.df_agg['ae_wrf_prcp'] - self.df_agg['ae_mlp']   
        self.df_agg['imp_mlp'] = 1 - self.df_agg['se_mlp']/(self.df_agg['se_wrf_prcp'])
        
        # OPTION 2 - AVERAGE SAMPLES, THEN COMPUTE METRIC OF AVERAGED SAMPLE
        # self.df_agg['smape_mlp'] = self.df_agg.apply(SMAPE, axis=1, args=('sample','Prec')) 

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

def table_of_predictions_for_metric(predictions, seasons, columns, n_samples, sample_columns, additional_columns, metric = 'smape', prefix='smape'):

    table = []

    headers = ['Model']

    baseline_rows = {}
    for c in columns: baseline_rows[c] = [c]
    # row_a = ['Bann']
    # row_b = ['BannCorr']
    # row_c = ['Norr']

    for index, (k,v) in enumerate(predictions.items()):
        
        SA = SeasonalAnalysis(df = v['k_all'], 
                              columns=columns, 
                              sample_columns=sample_columns,
                              additional_columns=additional_columns,
                              n_samples=n_samples,
                              group_by_fields= ['Station', 'season', 'year'])

        d = SA.seasonal_summaries()

        r = d[metric].copy()
        
        row = [f'{k}']
        
        for season in seasons:
            
            if index == 0: 
                headers.append(f'{season} mean')
                headers.append(f'{season} median')
            
            rs = r[r['season']==season].copy()
            
            if index==0:
                for c in columns:
                    a = rs[rs['variable']==f'{prefix}_{c}']['value']
                    baseline_rows[c].append(f'{a.mean():.2f}')
                    baseline_rows[c].append(f'{a.median():.2f}')

                # a = rs[rs['variable']==f'{prefix}_wrf_prcp']['value']
                # b = rs[rs['variable']==f'{prefix}_wrf_bc_prcp']['value']
                # c = rs[rs['variable']==f'{prefix}_precip_norris']['value']
                # row_a.append(f'{a.mean():.2f}')
                # row_a.append(f'{a.median():.2f}')
                # row_b.append(f'{b.mean():.2f}')
                # row_b.append(f'{b.median():.2f}')
                # row_c.append(f'{c.mean():.2f}')
                # row_c.append(f'{c.median():.2f}')
                
            z = rs[rs['variable']==f'{prefix}_mlp']['value']
            row.append(f'{z.mean():.2f}')
            row.append(f'{z.median():.2f}')
                    
        if index==0:
            for c in columns: 
                table.append(baseline_rows[c])
            # table.append(row_a)
            # table.append(row_b)
            # table.append(row_c)
            
        table.append(row)
        
        # print(k)

    print(tabulate(table, headers, tablefmt='small', disable_numparse=True))

def table_of_predictions_confidence_intervals(predictions, seasons):
    """Prints table of results. 

    Previous name: table_of_results.
    
    Inputs:
        predictions : dict
    Returs:
        None
    
    """

    table = []
    headers = ['Model']

    for i, (k, v) in enumerate(predictions.items()):
        if i==0:
            row_a = ['WRF']
            row_b = ['BC WRF']
        row = [k]
        st_test = v['k_all']
        for season in seasons:
            if i==0:
                headers.append(f'{season} mean')
                headers.append(f'{season} median')
            
            ci_lista, ci_listb = [], []
            ci_list = []

            for s in st_test['Station'].unique():
                
                df = st_test.loc[(st_test['Station']==s) & (st_test['season']==season)].copy()
                
                if i == 0:
                    a_high = (df[f'wrf_prcp']>df['high_ci']).sum()
                    a_low = (df[f'wrf_prcp']<df['low_ci']).sum()
                    b_high = (df[f'wrf_prcp']>df['high_ci']).sum()
                    b_low = (df[f'wrf_prcp']<df['low_ci']).sum()
                    a_ci = (a_high + a_low) / len(df)
                    b_ci = (b_high + b_low) / len(df)

                    ci_lista.append(a_ci)
                    ci_listb.append(b_ci)
                
                c_high = (df[f'Prec']>df['high_ci']).sum()
                c_low = (df[f'Prec']<df['low_ci']).sum()
                
                c_ci = (c_high) / len(df)


                ci_list.append(c_ci)

            if i ==0:
                row_a.append(f'{np.mean(ci_lista):.4f}')
                row_a.append(f'{np.median(ci_lista):.4f}')
                row_b.append(f'{np.mean(ci_listb):.4f}')
                row_b.append(f'{np.median(ci_listb):.4f}')
            
            row.append(f'{np.mean(ci_list):.4f}')
            row.append(f'{np.median(ci_list):.4f}')
            
        if i== 0:
            table.append(row_a)
            table.append(row_b)
        
        print(k)
            
        table.append(row)     

    print(tabulate(table, headers, tablefmt='simple', disable_numparse=True))

def table_of_predictions_ks_test(predictions, seasons, columns, sample_columns, additional_columns):
    
    """[description].
    
    Previous name: table_of_predictions
    
    Inputs:
    
    Returns:
        None
        
    """

    table = []
    headers = ['Model']

    for i, (k, v) in enumerate(predictions.items()):
        if i==0:
            row_a = ['Bann']
            row_b = ['BannCorr']
            row_c = ['Norr']
            
        row = [k]
        st_test = v['k_all']
        for season in seasons:
            if i==0:
                headers.append(f'{season} mean')
                headers.append(f'{season} median')
            
            ks_lista, ks_listb, ks_listc = [], [], []
            ks_list = []

            for s in st_test['Station'].unique():
                df = st_test[st_test['Station']==s].copy()
                rvs = df[df['season']==season]['Prec']
                
                cdfa = df[df['season']==season]['wrf_prcp']
                cdfb = df[df['season']==season]['wrf_bc_prcp']
                cdfc = df[df['season']==season]['precip_norris']
                
                kstesta = scipy.stats.ks_2samp(rvs, cdfa, alternative='two-sided', mode='auto')
                kstestb = scipy.stats.ks_2samp(rvs, cdfb, alternative='two-sided', mode='auto')
                kstestc = scipy.stats.ks_2samp(rvs, cdfc, alternative='two-sided', mode='auto')
                
                ks_lista.append(kstesta)
                ks_listb.append(kstestb)
                ks_listc.append(kstestc)

                for sample in sample_columns:
                    cdf = np.array([])
                    cdf_s = np.array(df[df['season']==season][sample])
                    cdf = np.concatenate((cdf,cdf_s), axis=None)
                    
                kstest = scipy.stats.ks_2samp(rvs, cdf, alternative='two-sided', mode='auto')
                ks_list.append(kstest)
            
            if i ==0:
                row_a.append(f'{np.mean(ks_lista):.4f}')
                row_a.append(f'{np.median(ks_lista):.4f}')
                row_b.append(f'{np.mean(ks_listb):.4f}')
                row_b.append(f'{np.median(ks_listb):.4f}')
                row_c.append(f'{np.mean(ks_listc):.4f}')
                row_c.append(f'{np.median(ks_listc):.4f}')
            
            row.append(f'{np.mean(ks_list):.4f}')
            row.append(f'{np.median(ks_list):.4f}')
            
        if i== 0:
            table.append(row_a)
            table.append(row_b)
            table.append(row_c)
            
        print(k)
            
        table.append(row)
        
    print(tabulate(table, headers, tablefmt='simple', disable_numparse=True))
    