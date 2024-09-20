import pandas as pd
import numpy as np
import scipy 
import CRPS.CRPS as pscore
import properscoring as ps

from likelihoods import * 
import torch
import pdb

__all__ = [ 
            'squared_error',
            'absolute_error',
            'error',
            'SMAPE',
            'BS',
            'QS',
            'CRPS_apply',
            'ROC',
            'logprob'
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

# def CRPS_apply(df : pd.DataFrame, x : np.array = None, observation_series: str = 'Prec'):
#     if x is None:
#         # TO DO: INCLUDE OTHER PROBABILITY DISTRIBUTIONS OTHER THAT THE BERNOULLI GAMMA MIXTURE MODEL
#         x = scipy.stats.gamma.ppf(q=np.linspace(0,1,100)[1:-1], a=df['alpha'], loc=0, scale=1/df['beta'])
#     crps,fcrps,acrps = pscore(x, df[observation_series]).compute()
#     return crps

def CRPS_apply(df, dist='bgmm',mean='naive_bc',stdev=None):
    
    if dist=='bgmm':
        x0 = np.zeros(round(df['pi'] * 100))
        x1 = scipy.stats.gamma.ppf(q=np.linspace(0,1,100-len(x0))[1:-1], a=df['alpha'], loc=0, scale=1/df['beta'])
        x = np.concatenate([x0,x1])

    elif dist=='bernoulli':
        x0 = np.zeros(round(df['pi'] * 100))
        x1 = np.ones(100-len(x0))
        x = np.concatenate([x0,x1])

    elif dist=='gaussian':
        x = scipy.stats.norm.ppf(q=np.linspace(0,1,100)[1:-1], loc=df['mu'], scale=df['sigma'])
    
    elif dist=='gamma':
        x = scipy.stats.gamma.ppf(q=np.linspace(0,1,100)[1:-1], a=df['alpha'], loc=0, scale=1/df['beta'])
    
    elif dist=='lognormal':
        x = scipy.stats.lognorm.ppf(q=np.linspace(0,1,100)[1:-1], s=df['sigma'], loc=0, scale=np.exp(df['mu']))

    elif dist=='gumbel':
        x = scipy.stats.gumbel_r.ppf(q=np.linspace(0,1,100)[1:-1], loc=df['mu'], scale=df['beta'])
    
    elif dist=='halfnormal':
        x = scipy.stats.halfnorm.ppf(q=np.linspace(0,1,100)[1:-1], loc=0, scale=df['sigma'])

    elif dist=='bernoulli_gaussian':
        x0 = np.zeros(round(df['pi'] * 100))
        x1 = scipy.stats.norm.ppf(q=np.linspace(0,1,100-len(x0))[1:-1], loc=df['mu'], scale=df['sigma'])
        x = np.concatenate([x0,x1])
        
    elif dist=='bernoulli_gumbel':
        x0 = np.zeros(round(df['pi'] * 100))
        x1 = scipy.stats.gumbel_r.ppf(q=np.linspace(0,1,100-len(x0))[1:-1], loc=df['mu'], scale=df['beta'])
        x = np.concatenate([x0,x1])

    elif dist=='bernoulli_halfnormal':
        x0 = np.zeros(round(df['pi'] * 100))
        x1 = scipy.stats.halfnorm.ppf(q=np.linspace(0,1,100-len(x0))[1:-1], loc=0, scale=df['sigma'])
        x = np.concatenate([x0,x1])
    
    elif dist=='bernoulli_lognormal':
        # NEED TO CHECK THIS
        x0 = np.zeros(round(df['pi'] * 100))
        x1 = scipy.stats.lognorm.ppf(q=np.linspace(0,1,100-len(x0))[1:-1], s=df['sigma'], loc=0, scale=np.exp(df['mu']))
        x = np.concatenate([x0,x1])
        
    elif dist=='gaussian_from_deterministic':
        if stdev is None:
            raise ValueError('stdev must be specified for Gaussian distribution')
        if mean is None:
            raise ValueError('mean must be specified for Gaussian distribution')
        
        if isinstance(stdev, (int, float)): # if stdev is numeric, then it is a constant
            df['stdev'] = stdev
        elif isinstance(stdev, str): # if stdev is a string, then it is a column in the dataframe
            df['stdev'] = df[stdev]
        else:
            raise ValueError('stdev must be numeric or a string')
        
        x = scipy.stats.norm.ppf(q=np.linspace(0,1,100)[1:-1], loc=df[mean], scale=df['stdev'])
       
    crps,fcrps,acrps = pscore(ensemble_members = x, observation = df['Prec']).compute()
    
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

def ROC(df : pd.DataFrame, 
        obs : str, 
        sim : str = None, 
        wet_threshold : float = 0, 
        quantile : float = 0.5, 
        only_wet_days : bool = False,
        likelihood : str = None,
        ):
    "Returns hit rate and false alarm rate for a given quantile"

    if only_wet_days:
        df = df[df[obs]>0]

    if sim is None:
        q = (quantile - df['pi'])/(1-df['pi']) # probability of being in the wet class
        q = q.where(q>0, 0) # make sure q is positive
        if likelihood == 'bgmm':
            p = scipy.stats.gamma.ppf(q=q, a=df['alpha'], loc=0, scale=1/df['beta']) # quantile of the gamma distribution
        elif likelihood == 'bernoulli_lognormal':
            p = scipy.stats.lognorm.ppf(q=q, s=df['sigma'], loc=0, scale=np.exp(df['mu']))
        v = np.where(p>wet_threshold, 1, 0) # wet or dry
    else:
        v = np.array(df[sim]>wet_threshold)*1 # wet or dry for simulations

    o = np.array(df[obs]>wet_threshold)*1 # wet or dry for observations
    
    TP = np.sum(np.logical_and(v==1, o==1)) # true positive
    FP = np.sum(np.logical_and(v==1, o==0)) # false positive
    TN = np.sum(np.logical_and(v==0, o==0)) # true negative
    FN = np.sum(np.logical_and(v==0, o==1)) # false negative

    hit_rate = TP / (TP + FN)
    false_alarm_rate = FP / (FP + TN)

    return hit_rate, false_alarm_rate

def AUC(x, y):
    """
    Calculate the area under the ROC curve using trapezoidal integration.
    
    Parameters:
    - x: Array of False Alarm Rates
    - y: Array of Hit Rates
    
    Returns:
    - Area under the ROC curve
    """
    # Sort the arrays by x values (just in case they aren't sorted)
    sorted_indices = sorted(range(len(x)), key=lambda k: x[k])
    x = [x[i] for i in sorted_indices]
    y = [y[i] for i in sorted_indices]
    
    # Compute the AUC using trapezoidal integration
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
    
    return area

def logprob(df : pd.DataFrame, dist : str =None, det_mu_series : str = None, det_sigma_series : str = None, device = 'cpu'):
    if dist == 'bgmm':
        prec = torch.tensor(df['Prec'].values)
        alpha = torch.tensor(df['alpha'].values)
        beta = torch.tensor(df['beta'].values)
        pi = torch.tensor(df['pi'].values)
        return bernoulli_gamma_logpdf(obs=prec.float(), alpha=alpha, beta=beta, pi=pi, device=device)
    elif dist == 'bernoulli':
        prec = torch.tensor(df['Prec'].values)
        pi = torch.tensor(df['pi'].values)
        return bernoulli_logpdf(obs=prec.float(), pi=pi, device=device)
    elif dist == 'gaussian':
        prec = torch.tensor(df['Prec'].values)
        mu = torch.tensor(df['mu'].values)
        sigma = torch.tensor(df['sigma'].values)
        return gaussian_logpdf(obs=prec.float(), mu=mu, sigma=sigma, device=device)
    elif dist == 'gamma':
        prec = torch.tensor(df['Prec'].values)
        alpha = torch.tensor(df['alpha'].values)
        beta = torch.tensor(df['beta'].values)
        return gamma_logpdf(obs=prec.float(), alpha=alpha, beta=beta, device=device)   
    elif dist == 'lognormal':
        prec = torch.tensor(df['Prec'].values)
        mu = torch.tensor(df['mu'].values)
        sigma = torch.tensor(df['sigma'].values)
        return lognormal_logpdf(obs=prec.float(), mu=mu, sigma=sigma, device=device)
    elif dist == 'gumbel':
        prec = torch.tensor(df['Prec'].values)
        mu = torch.tensor(df['mu'].values)
        beta = torch.tensor(df['beta'].values)
        return gumbel_logpdf(obs=prec.float(), mu=mu, beta=beta, device=device)
    elif dist == 'halfnormal':
        prec = torch.tensor(df['Prec'].values)
        sigma = torch.tensor(df['sigma'].values)
        return halfnormal_logpdf(obs=prec.float(), sigma=sigma, device=device)
    elif dist == 'bernoulli_gaussian':
        prec = torch.tensor(df['Prec'].values)
        mu = torch.tensor(df['mu'].values)
        sigma = torch.tensor(df['sigma'].values)
        pi = torch.tensor(df['pi'].values)
        return bernoulli_gaussian_logpdf(obs=prec.float(), mu=mu, sigma=sigma, pi=pi, device=device)
    elif dist == 'bernoulli_gumbel':
        prec = torch.tensor(df['Prec'].values)
        mu = torch.tensor(df['mu'].values)
        beta = torch.tensor(df['beta'].values)
        pi = torch.tensor(df['pi'].values)
        return bernoulli_gumbel_logpdf(obs=prec.float(), mu=mu, beta=beta, pi=pi, device=device)
    elif dist == 'bernoulli_halfnormal':
        prec = torch.tensor(df['Prec'].values)
        sigma = torch.tensor(df['sigma'].values)
        pi = torch.tensor(df['pi'].values)
        return bernoulli_halfnormal_logpdf(obs=prec.float(), sigma=sigma, pi=pi, device=device)
    elif dist == 'bernoulli_lognormal':
        prec = torch.tensor(df['Prec'].values)
        mu = torch.tensor(df['mu'].values)
        sigma = torch.tensor(df['sigma'].values)
        pi = torch.tensor(df['pi'].values)
        return bernoulli_lognormal_logpdf(obs=prec.float(), mu=mu, sigma=sigma, pi=pi, device=device)
    elif dist == 'gaussian_from_deterministic':
        prec = torch.tensor(df['Prec'].values)
        mu = torch.tensor(df[det_mu_series].values)
        sigma = torch.tensor(df[det_sigma_series].values)
        return gaussian_logpdf(obs=prec.float(), mu=mu, sigma=sigma, device=device)
    
    else:
        raise ValueError('Distribution not yet implemented')
    
