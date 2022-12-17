import pandas as pd
import numpy as np
import scipy 
import CRPS.CRPS as pscore
import properscoring as ps


__all__ = [ 
            'squared_error',
            'absolute_error',
            'error',
            'SMAPE',
            'BS',
            'QS',
            'CRPS_apply'
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

def SMAPE(df : pd.DataFrame, sim : str, obs : str):
    "Symmetrical Mean Absolute Percentage Error"
    if abs(df[sim] - df[obs]) == 0:
        return 0
    else:
        return abs(df[sim] - df[obs]) / (df[sim] + df[obs])

def BS(df : pd.DataFrame, sim : str, obs : str, wet_threshold : float = 0):
    "Brier score"
    if df[obs] > wet_threshold:
        return (df[sim] - 1)**2
    else:
        return df[sim]**2

def QS(df : pd.DataFrame, sim : str, obs : str, quantile : float):
    "Quantile Score"
    d = df[obs] - df[sim]
    if d < 0:
        return (quantile-1) * d
    else:
        return quantile * d

def CRPS_apply(df : pd.DataFrame, x : np.array = None, observation_series: str = 'Prec'):
    if x is None:
        # TO DO: INCLUDE OTHER PROBABILITY DISTRIBUTIONS OTHER THAT THE BERNOULLI GAMMA MIXTURE MODEL
        x = scipy.stats.gamma.ppf(q=np.linspace(0,1,100)[1:-1], a=df['alpha'], loc=0, scale=1/df['beta'])
    crps,fcrps,acrps = pscore(x, df[observation_series]).compute()
    return crps

def CRPS(df : pd.DataFrame, ensemble = None, num_quantiles : int = 100, limit = None):
    
    for idx, q in enumerate(np.linspace(0,1,num_quantiles+2)[1:-1]):
        if ensemble is None: 
            a = scipy.stats.gamma.ppf(q=q, a=df['alpha'], loc=0, scale=1/df['beta'])
            if limit is not None:
                a = np.where(a>limit,limit,a)
        else:  
            c = np.quantile(ensemble, q)
            if limit is not None:
                c = c.where(c>limit, limit, c)
            a = np.tile(c, len(ensemble))
        
        if idx == 0: 
            b = a.reshape([-1,1])
        else: 
            b = np.concatenate([b,a.reshape([-1,1])], axis=1)
    
    obs = df['Prec'].to_numpy()

    return ps.crps_ensemble(observations=obs, forecasts=b)
