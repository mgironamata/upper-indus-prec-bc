import pandas as pd

__all__ = [ 
            'squared_error',
            'absolute_error',
            'error',
            'SMAPE',
            'BS',
            'QS' 
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
