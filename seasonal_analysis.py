import pandas as pd
from utils import count_zeros, SMAPE

__all__ = ['seasonal_analysis',
           'seasonal_summaries',
           ]

def seasonal_analysis(st_test, columns, n_samples, sample_cols, add_cols):

    df = st_test.groupby(['Station','season','year']).sum()[columns].copy()

    df_dry_days = st_test.groupby(['Station','season','year'], as_index=False)[columns]
    df_dry_days = df_dry_days.agg([count_zeros]).droplevel(level=1, axis=1)

    # NUMBER OF DRY DAYS - ABSOLUTE ERROR 
    for i in range(n_samples):
        df_dry_days[f'edd_mlp_{i}'] = (df_dry_days[f'sample_{i}'] - df_dry_days['Prec'])

    df_dry_days['edd_mlp'] = abs(df_dry_days[[f'edd_mlp_{i}' for i in range(n_samples)]].mean(axis=1))
    df_dry_days['edd_wrf_prcp'] = abs(df_dry_days['wrf_prcp'] - df_dry_days['Prec'])
    df_dry_days['edd_wrf_bc_prcp'] = abs(df_dry_days['wrf_bc_prcp'] - df_dry_days['Prec'])

    df_dry_days = df_dry_days[['edd_wrf_prcp','edd_wrf_bc_prcp','edd_mlp']]

    # PRECIPITATION - SQUARED ERROR 
    df['se_wrf_prcp'] = (df['wrf_prcp'] - df['Prec'])**2
    df['se_wrf_bc_prcp'] = (df['wrf_bc_prcp'] - df['Prec'])**2

    df['sample'] = df[sample_cols].mean(axis=1)

    for i in range(n_samples):
        df[f'se_mlp_{i}'] = (df[f'sample_{i}'] - df['Prec'])**2


    df['se_mlp'] = df[[f'se_mlp_{i}' for i in range(n_samples)]].mean(axis=1)

    # PRECIPITATION - ERROR & ABSOLUTE ERROR
    df['e_wrf_prcp'] = (df['wrf_prcp'] - df['Prec'])
    df['e_wrf_bc_prcp'] = (df['wrf_bc_prcp'] - df['Prec'])  

    df['ae_wrf_prcp'] = abs(df['wrf_prcp'] - df['Prec'])
    df['ae_wrf_bc_prcp'] = abs(df['wrf_bc_prcp'] - df['Prec'])

    for i in range(n_samples):
        df[f'e_mlp_{i}'] = (df[f'sample_{i}'] - df['Prec'])
        df[f'ae_mlp_{i}'] = abs(df[f'sample_{i}'] - df['Prec'])

    df['e_mlp'] = df[[f'e_mlp_{i}' for i in range(n_samples)]].mean(axis=1)    
    df['ae_mlp'] = df[[f'ae_mlp_{i}' for i in range(n_samples)]].mean(axis=1)

    # PRECIPITATION - ABSOLUTE ERROR REDUCTION
    df['aer_mlp'] = df['ae_wrf_prcp'] - df['ae_mlp']
    df['aer_wrf_bc_prcp'] = df['ae_wrf_prcp'] - df['ae_wrf_bc_prcp']

    # PRECIPITATION - MSE IMPROVEMENT RATIO
    df['imp_wrf_bc_prcp'] = 1 - df['se_wrf_bc_prcp']/(df['se_wrf_prcp'])
    df['imp_mlp'] = 1 - df['se_mlp']/(df['se_wrf_prcp'])

    # PRECIPITATION - SMAPE
    df['smape_wrf_prcp'] = df.apply(SMAPE, axis=1, args=('wrf_prcp','Prec')) 
    df['smape_wrf_bc_prcp'] = df.apply(SMAPE, axis=1, args=('wrf_bc_prcp','Prec')) 
    df['smape_mlp'] = df.apply(SMAPE, axis=1, args=('sample','Prec')) 

    if 'mean' in add_cols:
        df['smape_mlp_mean'] = df.apply(SMAPE, axis=1, args=('mean','Prec'))  
    if 'median' in add_cols:
        df['smape_mlp_median'] = df.apply(SMAPE, axis=1, args=('median','Prec')) 

    df = pd.merge(df,df_dry_days,on=['Station','season','year'])

    return df

def seasonal_summaries(df, add_cols):

    # Totals
    totals = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=['Prec','wrf_prcp','wrf_bc_prcp','sample'] + add_cols
                         )
    
    # Error
    e = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=['e_wrf_prcp','e_wrf_bc_prcp','e_mlp']#,'imp_bc_wrf','imp_mlp','se_wrf','se_bc_wrf','se_mlp','edd_mlp','edd_bc'],
                         )

    # MAE
    ae = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=['ae_wrf_prcp','ae_wrf_bc_prcp','ae_mlp']#,'imp_bc_wrf','imp_mlp','se_wrf','se_bc_wrf','se_mlp','edd_mlp','edd_bc'],
                         )
    
    # SE
    se = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=['se_wrf_prcp','se_wrf_bc_prcp','se_mlp']#,'imp_bc_wrf','imp_mlp','se_wrf','se_bc_wrf','se_mlp','edd_mlp','edd_bc'],
                         )

    # MAE REDUCTION
    aer = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=['aer_wrf_bc_prcp','aer_mlp'],
                         )
    # ERROR IN DRY DAYS 
    edd = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=['edd_wrf_prcp','edd_wrf_bc_prcp','edd_mlp'],
                         )

    # MSE IMPROVEMENT RATIO                    
    improvement = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=['imp_wrf_bc_prcp','imp_mlp'],
                         )
    # SMAPE
    smape = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=['smape_wrf_prcp','smape_wrf_bc_prcp','smape_mlp'] + [f'smape_mlp_{i}' for i in add_cols]
                     )
        
    d = {}
    
    d['totals'] = totals
    d['e'] = e
    d['ae'] = ae
    d['se'] = se
    d['aer'] = aer
    d['edd'] = edd
    d['improvement'] = improvement
    d['smape'] = smape
    
    return d