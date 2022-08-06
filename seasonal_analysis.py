import pandas as pd
from utils import count_zeros, SMAPE

__all__ = ['seasonal_analysis',
           'seasonal_summaries',
           ]

def seasonal_analysis(st_test, columns, n_samples, sample_cols, add_cols):

    # Group by station, season and year
    df = st_test.groupby(['Station','season','year']).sum()[(columns + sample_cols + add_cols)].copy()

    # Crate dataframe for precipitation occurrence
    df_dry_days = st_test.groupby(['Station','season','year'], as_index=False)[(columns + sample_cols + add_cols)]
    df_dry_days = df_dry_days.agg([count_zeros]).droplevel(level=1, axis=1)

    list_of_edd_cols = []

    # DRY DAYS - ABSOLUTE ERROR 
    for i in range(n_samples):
        df_dry_days[f'edd_mlp_{i}'] = (df_dry_days[f'sample_{i}'] - df_dry_days['Prec'])
        list_of_edd_cols.append(f'edd_mlp_{i}')

    df_dry_days['edd_mlp'] = abs(df_dry_days[[f'edd_mlp_{i}' for i in range(n_samples)]].mean(axis=1))
    list_of_edd_cols.append('edd_mlp')
    
    for c in columns:
        df_dry_days[f'edd_{c}'] = abs(df_dry_days[c] - df_dry_days['Prec'])
        list_of_edd_cols.append(f'edd_{c}')
        
#     df_dry_days['edd_wrf_prcp'] = abs(df_dry_days['wrf_prcp'] - df_dry_days['Prec'])
#     df_dry_days['edd_wrf_bc_prcp'] = abs(df_dry_days['wrf_bc_prcp'] - df_dry_days['Prec'])

    df_dry_days = df_dry_days[list_of_edd_cols]

    # PRECIPITATION - SQUARED ERROR 
    for c in columns:
        df[f'se_{c}'] = (df[c] - df['Prec'])**2
        
#     df['se_wrf_prcp'] = (df['wrf_prcp'] - df['Prec'])**2
#     df['se_wrf_bc_prcp'] = (df['wrf_bc_prcp'] - df['Prec'])**2

    df['sample'] = df[sample_cols].mean(axis=1)

    for i in range(n_samples):
        df[f'se_mlp_{i}'] = (df[f'sample_{i}'] - df['Prec'])**2


    df['se_mlp'] = df[[f'se_mlp_{i}' for i in range(n_samples)]].mean(axis=1)

    # PRECIPITATION - ERROR & ABSOLUTE ERROR
    for c in columns:
        df[f'e_{c}'] = (df[c] - df['Prec'])
        df[f'ae_{c}'] = abs(df[c] - df['Prec'])
    
#     df['e_wrf_prcp'] = (df['wrf_prcp'] - df['Prec'])
#     df['e_wrf_bc_prcp'] = (df['wrf_bc_prcp'] - df['Prec'])  

#     df['ae_wrf_prcp'] = abs(df['wrf_prcp'] - df['Prec'])
#     df['ae_wrf_bc_prcp'] = abs(df['wrf_bc_prcp'] - df['Prec'])

    for i in range(n_samples):
        df[f'e_mlp_{i}'] = (df[f'sample_{i}'] - df['Prec'])
        df[f'ae_mlp_{i}'] = abs(df[f'sample_{i}'] - df['Prec'])

    df['e_mlp'] = df[[f'e_mlp_{i}' for i in range(n_samples)]].mean(axis=1)    
    df['ae_mlp'] = df[[f'ae_mlp_{i}' for i in range(n_samples)]].mean(axis=1)

    # PRECIPITATION - ABSOLUTE ERROR REDUCTION
    for c in columns:
        df[f'aer_{c}'] = df['ae_wrf_prcp'] - df[f'ae_{c}'] 
    
    df['aer_mlp'] = df['ae_wrf_prcp'] - df['ae_mlp']
    #df['aer_wrf_bc_prcp'] = df['ae_wrf_prcp'] - df['ae_wrf_bc_prcp']

    # PRECIPITATION - MSE IMPROVEMENT RATIO
    for c in columns:
        df[f'imp_{c}'] = 1 - df[f'se_{c}']/(df['se_wrf_prcp'])
    
    #df['imp_wrf_bc_prcp'] = 1 - df['se_wrf_bc_prcp']/(df['se_wrf_prcp'])
    
    df['imp_mlp'] = 1 - df['se_mlp']/(df['se_wrf_prcp'])

    # PRECIPITATION - SMAPE
    for c in columns:
        df[f'smape_{c}'] = df.apply(SMAPE, axis=1, args=(c, 'Prec'))
    
#     df['smape_wrf_prcp'] = df.apply(SMAPE, axis=1, args=('wrf_prcp','Prec')) 
#     df['smape_wrf_bc_prcp'] = df.apply(SMAPE, axis=1, args=('wrf_bc_prcp','Prec')) 
    
    df['smape_mlp'] = df.apply(SMAPE, axis=1, args=('sample','Prec')) 

    if 'mean' in add_cols:
        df['smape_mlp_mean'] = df.apply(SMAPE, axis=1, args=('mean','Prec'))  
    if 'median' in add_cols:
        df['smape_mlp_median'] = df.apply(SMAPE, axis=1, args=('median','Prec')) 

    df = pd.merge(df,df_dry_days,on=['Station','season','year'])

    return df

def seasonal_summaries(df, add_cols, cols):
    
    # Totals
    
    value_vars = cols + ['sample']
    totals = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=['Prec','wrf_prcp','wrf_bc_prcp','sample'] + add_cols
                         )
    
    # Error
    value_vars = [f'e_{c}' for c in cols] + ['e_mlp']
    e = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=value_vars #,'imp_bc_wrf','imp_mlp','se_wrf','se_bc_wrf','se_mlp','edd_mlp','edd_bc'],
                         )

    # MAE
    value_vars = [f'ae_{c}' for c in cols] + ['ae_mlp']
    ae = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=value_vars #,'imp_bc_wrf','imp_mlp','se_wrf','se_bc_wrf','se_mlp','edd_mlp','edd_bc'],
                         )
    
    # SE
    value_vars = [f'se_{c}' for c in cols] + ['se_mlp']
    se = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=value_vars #,'imp_bc_wrf','imp_mlp','se_wrf','se_bc_wrf','se_mlp','edd_mlp','edd_bc'],
                         )

    # MAE REDUCTION
    value_vars = [f'aer_{c}' for c in cols] + ['aer_mlp']
    aer = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=value_vars,
                         )
    # ERROR IN DRY DAYS 
    value_vars = [f'edd_{c}' for c in cols] + ['edd_mlp']
    edd = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=value_vars,
                         )

    # MSE IMPROVEMENT RATIO   
    value_vars = [f'imp_{c}' for c in cols] + ['imp_mlp']
    improvement = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=value_vars,
                         )
    # SMAPE
    value_vars = [f'smape_{c}' for c in cols] + ['smape_mlp']
    smape = df.reset_index().melt(id_vars=['Station','season','year'],
                          value_vars=value_vars + [f'smape_mlp_{i}' for i in add_cols]
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