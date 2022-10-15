import pandas as pd

__all__ = ['pairwise_errors']

def pairwise_errors(df : pd.DataFrame):
    "Computres pairwise metrics"

    df['se_wrf'] = (df['wrf_prcp'] - df['Prec'])**2 if ('wrf_prcp' in df.columns) else None
    df['se_bcp'] = (df['wrf_bc_prcp'] - df['Prec'])**2 if ('wrf_bc_prcp' in df.columns) else None
    df['se_mlp_mean'] = (df['mean'] - df['Prec'])**2 if ('mean' in df.columns) else None
    df['se_mlp_median'] = (df['median'] - df['Prec'])**2 if ('median' in df.columns) else None
    # df['se_mlp_median_gamma'] = (df['median_gamma'] - df['Prec'])**2 

    df['e_wrf'] = (df['wrf_prcp'] - df['Prec']) if ('wrf_prcp' in df.columns) else None
    df['e_bcp'] = (df['wrf_bc_prcp'] - df['Prec']) if ('wrf_bc_prcp' in df.columns) else None
    # df['e_mlp'] = (df['mean'] - df['Prec'])

    i=0

    while(f'sample_{i}' in df.columns):
        df[f'se_mlp_sample_{i}'] = (df[f'sample_{i}'] - df['Prec'])**2
        df[f'e_mlp_sample_{i}'] = (df[f'sample_{i}'] - df['Prec'])
        i += 1
    
    return df 