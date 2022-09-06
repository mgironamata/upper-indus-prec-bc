from multiprocessing.sharedctypes import Value
import numpy as np
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as Fv

from torch.distributions.gamma import Gamma
from torch.distributions.gumbel import Gumbel
from torch.distributions.normal import Normal
from torch.distributions.half_normal import HalfNormal

import scipy.stats as stats 

import os

from models import MLP, SimpleRNN
from experiment import *
from runmanager import *

# from plot_utils import *
# from preprocessing_utils import *

from torch.utils.data import TensorDataset, DataLoader 

import pdb

__all__ =  [
            'gaussian_logpdf',
            'gamma_logpdf',
            'ggmm_logpdf',
            'bgmm_logpdf',
            'b2gmm_logpdf',
            'b2sgmm_logpdf',
            'train_epoch',
            'loss_fn',
            'gmm_fn',
            'sample',
            'mixture_percentile',
            'build_results_df',
            'RunningAverage',
            'pairwise_errors',
            'sample_mc',
            'truncate_sample',
            'count_zeros',
            'SMAPE',
            'add_to_dict',
            'make_predictions',
            'make_sequential_predictions',
            'multirun',
            'truncate_sample'
            ]
            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""Device to perform computations on."""

def _reduce(logp, reduction):
    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    elif reduction == 'batched_mean':
        return torch.mean(torch.sum(logp, 1))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')

def gaussian_logpdf(obs, mu, sigma, reduction='mean'):
    """Gamma mixture model log-density.

    Args:
        obs (torch.Tensor): Observed values.
        alpha (torch.Tensor)): Paramaters 'alpha' from Gamma distribution.
        beta (torch.Tensor): Pamaterers 'beta' from Gamma distribution. 
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """

    obs = obs.flatten()

    logp = Normal(loc=mu, scale=sigma).log_prob(obs)

    return _reduce(logp, reduction)

def gamma_logpdf(obs, alpha, beta, reduction='mean'):
    """Gamma mixture model log-density.

    Args:
        obs (torch.Tensor): Observed values.
        alpha (torch.Tensor)): Paramaters 'alpha' from Gamma distribution.
        beta (torch.Tensor): Pamaterers 'beta' from Gamma distribution. 
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """

    obs = obs.flatten()
    b_mask = obs == 0

    epsilon = 0.000001
    obs[b_mask] = obs[b_mask] + epsilon

    logp = Gamma(concentration=alpha, rate=beta).log_prob(obs)

    return _reduce(logp, reduction)

def ggmm_logpdf(obs, alpha1, alpha2, beta1, beta2, q, reduction='mean'):
    """Benroulli-Gamma-Gamma mexture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        alpha1 (torch.Tensor): 
        beta1 (torch.Tensor):
        alpha1 (torch.Tensor): 
        beta1 (torch.Tensor):
        q (torch.Tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """

    obs = obs.flatten()
    b_mask = obs == 0

    epsilon = 0.000001
    obs[b_mask] = obs[b_mask] + epsilon

    mixture_weights = torch.stack([q,1-q]).permute([1,0]) # REVIEW
    mixture_alphas = torch.stack([alpha1,alpha2]).permute([1,0])
    mixture_betas = torch.stack([beta1,beta2]).permute([1,0])

    mix = torch.distributions.Categorical(mixture_weights)
    comp = torch.distributions.Gamma(mixture_alphas, mixture_betas)
    gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)   

    logp = gmm.log_prob(obs)

    return _reduce(logp, reduction)

def bgmm_logpdf(obs, pi, alpha, beta, reduction='mean'):
    """Benroulli-Gamma mixture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        pi (torch.Tensor): 
        alpha (torch.Tensor): 
        beta (torch.Tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """
    
    #pdb.set_trace()

    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g_mask = obs > 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + Gamma(concentration=alpha[g_mask], rate=beta[g_mask]).log_prob(obs[g_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    return _reduce(logp, reduction)
     
def b2gmm_logpdf(obs, pi, alpha1, alpha2, beta1, beta2, q, reduction='mean'):
    """Benroulli-Gamma-Gamma mexture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        pi (torch.Tensor): 
        alpha1 (torch.Tensor): 
        beta1 (torch.Tensor):
        alpha1 (torch.Tensor): 
        beta1 (torch.Tensor):
        q (torch.Tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """

    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g_mask = obs > 0
    
    k = g_mask.shape[0]

    mixture_weights = torch.stack([q[g_mask],1-q[g_mask]]).permute([1,0]) # REVIEW
    mixture_alphas = torch.stack([alpha1[g_mask],alpha2[g_mask]]).permute([1,0])
    mixture_betas = torch.stack([beta1[g_mask],beta2[g_mask]]).permute([1,0])

    mix = torch.distributions.Categorical(mixture_weights)
    comp = torch.distributions.Gamma(mixture_alphas, mixture_betas)
    gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)   

    logp[g_mask] = torch.log(1-pi[g_mask]) + gmm.log_prob(obs[g_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    return _reduce(logp, reduction)

def b2sgmm_logpdf(obs, pi, alpha1, alpha2, beta1, beta2, q, t, reduction='mean'):
    """Benroulli-Gamma-Gamma mixture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        pi (torch.Tensor): 
        alpha (torch.Tensor): 
        beta (torch.Tensor):
        q (torch.Tensor):
        t (torch.Tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """
    
    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g1_mask = (obs > 0) * (obs < t)
    g2_mask = (obs > 0) * (obs >= t)

    logp[g1_mask] = torch.log((1-pi[g1_mask])) + torch.log((q[g1_mask])) + Gamma(concentration=alpha1[g1_mask], rate=beta1[g1_mask]).log_prob(obs[g1_mask])
    logp[g2_mask] = torch.log((1-pi[g2_mask])) + torch.log((1-q[g2_mask])) + Gamma(concentration=alpha2[g2_mask], rate=beta2[g2_mask]).log_prob(obs[g2_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    return _reduce(logp, reduction)

def bernoulli_gaussian_logpdf(obs, pi, mu, sigma, reduction='mean'):
    """Benroulli-Gaussian mixture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        pi (torch.Tensor): 
        mu (torch.Tensor): 
        sigma (torch.Tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """

    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g_mask = obs != 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + Normal(loc=mu[g_mask], scale=sigma[g_mask]).log_prob(obs[g_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    return _reduce(logp, reduction)

def bernoulli_loggaussian_logpdf(obs, pi, mu, sigma, reduction='mean'):
    """Benroulli-Gaussian mixture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        pi (torch.Tensor): 
        mu (torch.Tensor): 
        sigma (torch.Tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """

    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g_mask = obs != 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + Normal(loc=mu[g_mask], scale=sigma[g_mask]).log_prob(torch.log(obs[g_mask]))
    logp[b_mask] = torch.log(pi[b_mask])

    return _reduce(logp, reduction)

def bernoulli_gumbel_logpdf(obs, pi, mu, beta, reduction='mean'):
    """Benroulli-Gumbel mixture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        pi (torch.Tensor): 
        mu (torch.Tensor): 
        beta (torch.Tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """

    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g_mask = obs != 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + Gumbel(loc=mu[g_mask], scale=beta[g_mask]).log_prob(obs[g_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    return _reduce(logp, reduction)

def bernoulli_halfnormal_logpdf(obs, pi, sigma, reduction='mean'):
    """Benroulli-HalfNormal mixture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        pi (torch.Tensor): 
        sigma (torch.Tensor): 
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """

    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g_mask = obs != 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + HalfNormal(scale=sigma[g_mask]).log_prob(obs[g_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    return _reduce(logp, reduction)

def train_epoch(model, optimizer, train_loader, valid_loader, epoch, test_loader=None, print_progress=False):
    """Runs training for one epoch.

    Args:
        model
        optimizer
        train_loader
        valid_loader
        epoch
        test_loader
        print_prograss
    
    Returns:
        
    """

    train_losses = []
    valid_losses = []
    test_losses = []

    model.train()

    for i, (inputs, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels, inputs, model)                

        if loss.item() != 0: 
            
            loss.backward()
            
            for param in optimizer.param_groups[0]['params']:
                # Bit of regularisation
                nn.utils.clip_grad_value_(param, 1)

            optimizer.step()
        
        train_losses.append(loss.item())
        
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):

            outputs = model(inputs)
            loss = loss_fn(outputs, labels, inputs, model)
            valid_losses.append(loss.item())

        if test_loader != None:

            for i, (inputs, labels) in enumerate(test_loader):

                outputs = model(inputs)
                loss = loss_fn(outputs, labels, inputs, model)
                test_losses.append(loss.item())
             
            
    #mean_train_losses.append(np.mean(train_losses))
    #mean_valid_losses.append(np.mean(valid_losses))
    
    if print_progress:
        print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'\
         .format(epoch+1, np.mean(train_losses), np.mean(valid_losses)))
    
    return np.mean(train_losses), np.mean(valid_losses), np.mean(test_losses)

def loss_fn(outputs, labels, inputs, model, reduction='mean'):
    """Computes loss function (log-probability of labels).

    Args:
        outputs:
        labels
        model
        reduction

    Returns:
        loss
    """
        
    if model.likelihood == None:
        mask = inputs[:,0] > inputs[:,0].mode().values.item() 
        indices = mask.nonzero()
        ratio = len(indices)/len(outputs)
        loss = Fv.mse_loss(outputs[indices], labels[indices]) * ratio if len(indices)>0 else torch.tensor(0)

    elif model.likelihood == 'gaussian':
        mask = inputs[:,0] > inputs[:,0].mode().values.item() 
        indices = mask.nonzero()
        ratio = len(indices)/len(outputs)
        loss = -gaussian_logpdf(labels, mu=outputs[:,0][indices], sigma=outputs[:,1][indices], reduction=reduction) * ratio if len(indices)>0 else torch.tensor(0)
    
    elif model.likelihood == 'gamma':
        loss = -gamma_logpdf(labels, alpha=outputs[:,0], beta=outputs[:,1], reduction=reduction)

    elif model.likelihood == 'gamma_nonzero':
        mask = inputs[:,0] > inputs[:,0].mode().values.item() 
        indices = mask.nonzero()
        ratio = len(indices)/len(outputs)
        loss = -gamma_logpdf(labels, alpha=outputs[:,0][indices], beta=outputs[:,1][indices], reduction=reduction) * ratio if len(indices)>0 else torch.tensor(0)
    
    elif model.likelihood == 'ggmm':
        loss = -ggmm_logpdf(labels, alpha1=outputs[:,0], alpha2=outputs[:,1], 
                             beta1=outputs[:,2], beta2=outputs[:,3], q=outputs[:,4], reduction=reduction)

    elif model.likelihood == 'bgmm':
        loss = -bgmm_logpdf(labels, pi=outputs[:,0], alpha=outputs[:,1], beta=outputs[:,2], reduction=reduction)
    
    elif model.likelihood == 'b2gmm':
        loss = -b2gmm_logpdf(labels, pi=outputs[:,0], alpha1=outputs[:,1], alpha2=outputs[:,2], 
                             beta1=outputs[:,3], beta2=outputs[:,4], q=outputs[:,5], reduction=reduction)

    elif model.likelihood == 'b2sgmm':
        loss = -b2sgmm_logpdf(labels, pi=outputs[:,0], alpha1=outputs[:,1], alpha2=outputs[:,2], 
                             beta1=outputs[:,3], beta2=outputs[:,4], q=outputs[:,5], t=outputs[:,6], reduction=reduction)
    
    elif model.likelihood == 'bernoulli_gaussian':
        loss = -bernoulli_gaussian_logpdf(labels, pi=outputs[:,0], mu=outputs[:,1], sigma=outputs[:,2], reduction='mean')
    
    elif model.likelihood == 'bernoulli_loggaussian':
        loss = -bernoulli_loggaussian_logpdf(labels, pi=outputs[:,0], mu=outputs[:,1], sigma=outputs[:,2], reduction='mean')

    elif model.likelihood == 'bernoulli_gumbel':
        loss = -bernoulli_gumbel_logpdf(labels, pi=outputs[:,0], mu=outputs[:,1], beta=outputs[:,2], reduction='mean')

    elif model.likelihood == 'bernoulli_halfnormal':
        loss = -bernoulli_halfnormal_logpdf(labels, pi=outputs[:,0], sigma=outputs[:,1], reduction='mean')

    return loss

def gmm_fn(alpha1, alpha2, beta1, beta2, q):
    
    if type(q) == torch.Tensor:
        if len(q.shape) == 0:
            mixture_weights = torch.stack([q,1-q])
            mixture_alphas = torch.stack([alpha1,alpha2])
            mixture_betas = torch.stack([beta1,beta2])
        elif len(q.shape) == 1:
            mixture_weights = torch.stack([q,1-q]).permute([1,0])
            mixture_alphas = torch.stack([alpha1,alpha2]).permute([1,0])
            mixture_betas = torch.stack([beta1,beta2]).permute([1,0])
            
    else:
        mixture_weights = torch.stack([torch.tensor(q),torch.tensor(1-q)])
        mixture_alphas = torch.stack([torch.tensor(alpha1),torch.tensor(alpha2)])
        mixture_betas = torch.stack([torch.tensor(beta1),torch.tensor(beta2)])

    mix = torch.distributions.Categorical(mixture_weights)
    comp = torch.distributions.Gamma(mixture_alphas, mixture_betas)
    gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp) 
    
    return gmm

def sample(df, likelihood_fn='bgmm', sample_size=10000, series='uniform'):
    
    if likelihood_fn == 'gaussian':

        mu = df['mu']
        sigma = df['sigma']
        perc = df[series]
        occurrence = df['occurrence'] 

        quantile = perc
        ppf = 0
        while ppf <= 0:
            quantile = np.random.uniform(0,1)
            ppf = stats.norm.ppf(quantile,loc=mu, scale=sigma)
        return ppf * occurrence

    elif likelihood_fn == 'gamma':

        alpha = df['alpha']
        beta = df['beta']
        perc = df[series]

        quantile = perc
        return stats.gamma.ppf(quantile, a=alpha, loc=0, scale=1/beta)

    elif likelihood_fn == 'gamma_nonzero':

        alpha = df['alpha']
        beta = df['beta']
        perc = df[series]
        occurrence = df['occurrence'] 

        quantile = perc
        return stats.gamma.ppf(quantile, a=alpha, loc=0, scale=1/beta) * occurrence
    
    elif likelihood_fn == 'bgmm':

        pi = df['pi']
        alpha = df['alpha']
        beta = df['beta']
        perc = df[series] 

        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.gamma.ppf(quantile, a=alpha, loc=0, scale=1/beta)
        else:
            return 0
    
    elif likelihood_fn == 'b2gmm':
        pi = df['pi']
        alpha1 = df['alpha1']
        beta1 = df['beta1']
        alpha2 = df['alpha2']
        beta2 = df['beta2']
        q = df['q']
        perc = df[series]
        
        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            dist = gmm_fn(alpha1, alpha2, beta1, beta2 , q)
            return torch.quantile(dist.sample([sample_size]), quantile).numpy()
        else:
            return 0
    
    elif likelihood_fn == 'bernoulli_gaussian':
        pi = df['pi']
        mu = df['mu']
        sigma = df['sigma']
        perc = df[series] 

        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.norm.ppf(quantile, loc=mu, scale=sigma)
        else:
            return 0
    
    elif likelihood_fn == 'bernoulli_loggaussian':
        pi = df['pi']
        mu = df['mu']
        sigma = df['sigma']
        perc = df[series] 

        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.lognorm.ppf(quantile, s=sigma, scale=np.exp(mu))
        else:
            return 0

    elif likelihood_fn == 'bernoulli_gumbel':
        pi = df['pi']
        mu = df['mu']
        beta = df['beta']
        perc = df[series] 

        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.gumbel_r.ppf(quantile, loc=mu, scale=beta)
        else:
            return 0

    elif likelihood_fn == 'bernoulli_halfnormal':
        pi = df['pi']
        sigma = df['sigma']
        perc = df[series] 

        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.halfnorm.ppf(quantile, scale=sigma)
        else:
            return 0

def mixture_percentile(df, perc, likelihood_fn, sample_size=1000):

    if likelihood_fn == 'gamma':
        alpha = df['alpha']
        beta = df['beta']

        quantile = perc
        return stats.gamma.ppf(quantile, a=alpha, loc=0, scale=1/beta)
    
    elif likelihood_fn == 'ggmm':
        alpha1 = df['alpha1']
        beta1 = df['beta1']
        alpha2 = df['alpha2']
        beta2 = df['beta2']
        q = df['q']
        
        quantile = perc
        dist = gmm_fn(alpha1, alpha2, beta1, beta2 , q)
        return torch.quantile(dist.sample([sample_size]), quantile).numpy()

    elif likelihood_fn == 'bgmm':
        pi = df['pi']
        alpha = df['alpha']
        beta = df['beta']

        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.gamma.ppf(quantile, a=alpha, loc=0, scale=1/beta)
        else:
            return 0

    elif likelihood_fn == 'b2gmm':
        pi = df['pi']
        alpha1 = df['alpha1']
        beta1 = df['beta1']
        alpha2 = df['alpha2']
        beta2 = df['beta2']
        q = df['q']
        
        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            dist = gmm_fn(alpha1, alpha2, beta1, beta2 , q)
            return torch.quantile(dist.sample([sample_size]), quantile).numpy()
        else:
            return 0

    elif likelihood_fn == 'bernoulli_gaussian':
        pi = df['pi']
        mu = df['mu']
        sigma = df['sigma']
        
        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.norm.ppf(quantile, loc=mu, scale=sigma)
        else:
            return 0
        
    elif likelihood_fn == 'bernoulli_loggaussian':
        pi = df['pi']
        mu = df['mu']
        sigma = df['sigma']
        
        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.lognorm.ppf(quantile, s=sigma, scale=np.exp(mu)) #review
        else:
            return 0

    elif likelihood_fn == 'bernoulli_gumbel':
        pi = df['pi']
        mu = df['mu']
        beta = df['beta']
        
        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.gumbel_r.ppf(quantile, loc=mu, scale=beta)
        else:
            return 0

    elif likelihood_fn == 'bernoulli_halfnormal':
        pi = df['pi']
        sigma = df['sigma']
        
        if perc > pi:
            quantile = (perc - pi)/(1 - pi)
            return stats.halfnorm.ppf(quantile, scale=sigma)
        else:
            return 0
        

def build_results_df(df, test_dataset, st_names_test, model, p=0.05, x_mean=None, x_std=None,
                     confidence_intervals=False, draw_samples=True, n_samples=1, sequential_samples=False, threshold=None):
    
    if sequential_samples:
        seq_outputs_dict = {}
        for i in range(n_samples):
            outputs = make_sequential_predictions(model, test_dataset, x_mean, x_std, threshold=threshold)
            seq_outputs_dict[i] = outputs
        
    else:
        outputs = make_predictions(model, test_dataset)
    
    if type(st_names_test)==type(None):
        new_df = df.copy()
    else:  
        new_df = df[df['Station'].isin(st_names_test)].copy()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    if model.likelihood == 'gamma':
        new_df['alpha'] = outputs[:,0]
        new_df['beta'] = outputs[:,1]
        new_df['mean'] = new_df['alpha']/new_df['beta']

        new_df[f'perc_median'] = 0.5
        new_df[f'median'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_median')
    
    elif model.likelihood == 'gamma_nonzero':
        new_df['alpha'] = outputs[:,0]
        new_df['beta'] = outputs[:,1]
        new_df['mean'] = new_df['alpha']/new_df['beta']

        new_df['occurrence'] = new_df['wrf_prcp'].apply(lambda x: 1 if x>0 else 0)

        new_df[f'perc_median'] = 0.5
        new_df[f'median'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_median')
    
    elif model.likelihood == 'gaussian':
        new_df['mu'] = outputs[:,0]
        new_df['sigma'] = outputs[:,1]

        new_df['mean'] = new_df['mu']

        new_df['occurrence'] = new_df['wrf_prcp'].apply(lambda x: 1 if x>0 else 0)

    elif model.likelihood == 'ggmm':
        new_df['alpha1'] = outputs[:,0]
        new_df['alpha2'] = outputs[:,1]
        new_df['beta1'] = outputs[:,2]
        new_df['beta2'] = outputs[:,3]
        new_df['q'] = outputs[:,4]
        
        new_df['mean'] =  gmm_fn(alpha1 = outputs[:,0],
                                alpha2 =  outputs[:,1],
                                beta1 = outputs[:,2],
                                beta2 = outputs[:,3],
                                q = outputs[:,4]
                                ).mean
    
    elif model.likelihood == 'bgmm':
        new_df['pi'] = outputs[:,0]
        new_df['alpha'] = outputs[:,1]
        new_df['beta'] = outputs[:,2] 

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)
        new_df['mean'] = new_df['occurrence']*new_df['alpha']/new_df['beta']

        # new_df[f'perc_median'] = 0.5
        # new_df[f'median'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_median')

    elif model.likelihood == 'b2gmm':
        new_df['pi'] = outputs[:,0]
        new_df['alpha1'] = outputs[:,1]
        new_df['alpha2'] = outputs[:,2] 
        new_df['beta1'] = outputs[:,3]
        new_df['beta2'] = outputs[:,4]
        new_df['q'] = outputs[:,5]
        
        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)
        new_df['mean'] =  gmm_fn(alpha1 = outputs[:,1],
                                alpha2 =  outputs[:,2],
                                beta1 = outputs[:,3],
                                beta2 = outputs[:,4],
                                q = outputs[:,5]
                                ).mean * new_df['occurrence']
        
        new_df[f'perc_median'] = 0.5
        new_df[f'median'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_median')

    elif model.likelihood == 'bernoulli_gaussian':
        new_df['pi'] = outputs[:,0]
        new_df['mu'] = outputs[:,1]
        new_df['sigma'] = outputs[:,2] 

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)
        new_df['mean'] = new_df['occurrence']*new_df['mu']
    
    elif model.likelihood == 'bernoulli_loggaussian':
        new_df['pi'] = outputs[:,0]
        new_df['mu'] = outputs[:,1]
        new_df['sigma'] = outputs[:,2] 

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)
        # new_df['mean'] = new_df['occurrence']*new_df['mu']

        # new_df[f'perc_median'] = 0.5
        # new_df[f'median'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_median')

    elif model.likelihood == 'bernoulli_gumbel':
        new_df['pi'] = outputs[:,0]
        new_df['mu'] = outputs[:,1]
        new_df['beta'] = outputs[:,2] 

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)

    elif model.likelihood == 'bernoulli_halfnormal':
        new_df['pi'] = outputs[:,0]
        new_df['sigma'] = outputs[:,1]

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)

    elif model.likelihood == 'b2sgmm':

        new_df['pi'] = outputs[:,0]
        new_df['alpha1'] = outputs[:,1]
        new_df['alpha2'] = outputs[:,2] 
        new_df['beta1'] = outputs[:,3]
        new_df['beta2'] = outputs[:,4]
        new_df['q'] = outputs[:,5]
        new_df['t'] = outputs[:,6]

        ### work in progress ###

        #new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)

        #new_df['pi_q'] = new_df['pi'] + new_df['q']
        #new_df['high_gamma_occurrence'] = new_df['pi_q'].apply(lambda x: 1 if x < 0.5 else 0) # NOT CORRECT
        # new_df['high_gamma'] = 1 - new_df['']

        new_df['mean'] = new_df['occurrence']*(new_df['low_gamma_occurrence']*new_df['alpha1']/new_df['beta1'] + (1-new_df['low_gamma_occurrence'])*new_df['alpha2']/new_df['beta2'])
    
    elif type(model.likelihood) ==type(None):
        new_df['occurrence'] = new_df['wrf_prcp'].apply(lambda x: 1 if x>0 else 0)
        new_df['sample_0'] = outputs.squeeze() * new_df['occurrence']
        draw_samples = False
        confidence_intervals = False
    
    if draw_samples:
        if sequential_samples:
            for i in range(n_samples):
                outputs = seq_outputs_dict[i]
                new_df[f'sample_{i}'] = outputs[:,-1]
        else:
            for i in range(n_samples):
                new_df[f'uniform_{i}'] = new_df.apply(lambda x: np.random.uniform(0,1),axis=1)
                new_df[f'sample_{i}'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series=f'uniform_{i}').astype('float64')

    # new_df['median'] = new_df.apply(mixture_percentile, axis=1, args=(0.5, model.likelihood))
    # new_df['median'] = new_df['median'] * new_df['occurrence']

    # new_df['median_gamma'] = new_df.apply(mixture_percentile_gamma_only, axis=1, args=(0.5, model.likelihood))
    # new_df['median_gamma'] = new_df['median_gamma'] * new_df['occurrence']

    if confidence_intervals:

        new_df[f'perc_low_ci'] = p
        new_df[f'low_ci'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_low_ci')

        new_df[f'perc_high_ci'] = 1-p
        new_df[f'high_ci'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_high_ci')

        #new_df['low_ci'] = new_df.apply(mixture_percentile, axis=1, args=(p, model.likelihood))
        #new_df['high_ci'] = new_df.apply(mixture_percentile, axis=1, args=(1-p, model.likelihood))

    quantile = 0.9
    new_df[f'QS_quantile'] = quantile
    new_df['QS_sample'] = new_df.apply(sample, axis=1, args=(model.likelihood, 10000, 'QS_quantile'))
    new_df[f'BS'] = new_df.apply(BS, axis=1, args=('pi','Prec',0))
    new_df[f'QS'] = new_df.apply(QS, axis=1, args=('QS_sample', 'Prec', quantile))
                                              
    return new_df

def pairwise_errors(new_df):

    new_df['se_wrf'] = (new_df['wrf_prcp'] - new_df['Prec'])**2 if ('wrf_prcp' in new_df.columns) else None
    new_df['se_bcp'] = (new_df['wrf_bc_prcp'] - new_df['Prec'])**2 if ('wrf_bc_prcp' in new_df.columns) else None
    new_df['se_mlp_mean'] = (new_df['mean'] - new_df['Prec'])**2 if ('mean' in new_df.columns) else None
    new_df['se_mlp_median'] = (new_df['median'] - new_df['Prec'])**2 if ('median' in new_df.columns) else None
    # new_df['se_mlp_median_gamma'] = (new_df['median_gamma'] - new_df['Prec'])**2 

    new_df['e_wrf'] = (new_df['wrf_prcp'] - new_df['Prec']) if ('wrf_prcp' in new_df.columns) else None
    new_df['e_bcp'] = (new_df['wrf_bc_prcp'] - new_df['Prec']) if ('wrf_bc_prcp' in new_df.columns) else None
    # new_df['e_mlp'] = (new_df['mean'] - new_df['Prec'])

    i=0
    while(f'sample_{i}' in new_df.columns):
        new_df[f'se_mlp_sample_{i}'] = (new_df[f'sample_{i}'] - new_df['Prec'])**2
        new_df[f'e_mlp_sample_{i}'] = (new_df[f'sample_{i}'] - new_df['Prec'])
        i += 1
    
    return new_df 

class RunningAverage:
    """Maintain a running average."""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        """Reset the running average."""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """Update the running average.
        
        Args:
            val (float): Value to update with.
            n (int): Number elements used to compute `val`.
        """
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def mixture_percentile_gamma_only(df, perc, likelihood_fn, sample_size=1000):
    if likelihood_fn == 'bgmm':
        pi = df['pi']
        alpha = df['alpha']
        beta = df['beta']

        if perc > pi:
            return stats.gamma.ppf(perc, a=alpha, loc=0, scale=1/beta)
        else:
            return 0

    elif likelihood_fn == 'b2gmm':
        pi = df['pi']
        alpha1 = df['alpha1']
        beta1 = df['beta1']
        alpha2 = df['alpha2']
        beta2 = df['beta2']
        q = df['q']
        
        if perc > pi:
            dist = gmm_fn(alpha1, alpha2, beta1, beta2 , q)
            return torch.quantile(dist.sample([sample_size]), perc).numpy()
        else:
            return 0

def sample_mc(model, theta_dict):
    if model.likelihood == 'bgmm':
        pi = theta_dict['pi']
        alpha = theta_dict['alpha']
        beta = theta_dict['beta']
        
        perc = np.random.uniform(0,1)
        
        if perc > pi:
            quantile = (perc-pi)/(1-pi)
            return stats.gamma.ppf(quantile, a=alpha, loc=0, scale=1/beta)
        else:
            return 0
    
    elif model.likelihood == 'bernoulli_gaussian':
        pi = theta_dict['pi']
        mu = theta_dict['mu']
        sigma = theta_dict['sigma']
        
        perc = np.random.uniform(0,1)
        
        if perc > pi:
            quantile = (perc-pi)/(1-pi)
            return stats.norm.ppf(quantile, loc=mu, scale=sigma)
        else:
            return 0
    
    elif model.likelihood == 'bernoulli_loggaussian':
        pi = theta_dict['pi']
        mu = theta_dict['mu']
        sigma = theta_dict['sigma']
        
        perc = np.random.uniform(0,1)
        
        if perc > pi:
            quantile = (perc-pi)/(1-pi)
            return stats.lognorm.ppf(quantile, s=sigma, scale=np.exp(mu))
        else:
            return 0

def truncate_sample(x, threshold):
    if x<threshold:
        return x
    else:
        return threshold

def count_zeros(x,threshold=0):
    return np.sum(x<=threshold)

def SMAPE(df, sim, obs):
    if abs(df[sim] - df[obs]) == 0:
        return 0
    else:
        return abs(df[sim] - df[obs]) / (df[sim] + df[obs])

def BS(df, sim, obs, wet_threshold=0):
    if df[obs] > wet_threshold:
        return (df[sim] - 1)**2
    else:
        return df[sim]**2

def QS(df, sim, obs, quantile):
    d = df[obs] - df[sim]
    if d < 0:
        return (quantile-1) * d
    else:
        return quantile * d

def add_to_dict(xs,d):
    for x in xs:
        print(x)
        key = f'{x}'.split('=')[0]
        d[key] = x
    return d

def make_predictions(model, test_dataset):

    model.eval()

    with torch.no_grad():
        
        test_inputs = test_dataset.tensors[0]
        test_outputs = model(test_inputs)

        #   test2_inputs = test2_dataset.tensors[0]
        #   test2_outputs = model(test2_inputs)
    
    return test_outputs

def _get_theta_params(likelihood):
    if likelihood == 'bgmm':
        return ['pi','alpha','beta']
    elif likelihood == 'bernoulli_gaussian':
        return ['pi','mu','sigma']
    elif likelihood == 'bernoulli_loggaussian':
        return ['pi','mu','sigma']
    elif likelihood == 'gaussian':
        return ['mu','sigma']

def make_sequential_predictions(model, test_dataset, x_mean, x_std, threshold=None):

    model.eval()

    with torch.no_grad():
        
        for index in range(len(test_dataset.tensors[0])):

            if index==0:
                prev_prediction = torch.tensor(0)
            else:
                prev_prediction = torch.tensor(norm_sample)
                
            test_input = torch.cat([test_dataset.tensors[0][index,:-1], 
                                    prev_prediction.unsqueeze(0)])
            
            test_output = model(test_input.unsqueeze(0).float())

            # if test_output[0][0] == 0:
            #     print(test_input)
            
            theta_params = _get_theta_params(likelihood=model.likelihood)
            
            theta_dict = {key:test_output[0,i] for i, key in enumerate(theta_params)}
            
            sample = sample_mc(model, theta_dict)
            
            if threshold is not None:
                sample = truncate_sample(sample, threshold=threshold)
            
            norm_sample = (sample - x_mean[-1])/x_std[-1]
            
            if index==0:
                concat_test_outputs = torch.cat([test_output.squeeze(),
                                torch.tensor(sample).unsqueeze(0)]).unsqueeze(0)
            else:
                concat_test_output = torch.cat([test_output.squeeze(),
                                torch.tensor(sample).unsqueeze(0)]).unsqueeze(0)
                            
                concat_test_outputs = torch.cat([concat_test_outputs,
                                                 concat_test_output],dim=0)
            
            print(index) if index%10000==0 else None

    # test_outputs = concat_test_outputs[:,:-1]

    return concat_test_outputs

def multirun(data, predictors, params, epochs, split_by='station', sequential_samples=False, sample_threshold=None, n_samples=10, best_by='val', model_type='MLP'):

    m = RunManager()
    predictions={}

    for run in RunBuilder.get_runs(params): 
        
        d = len(predictors)
        
        train_tensor_x = torch.Tensor(data.data[f'X_train_{run.k}'][:,:d]) # transform to torch tensor
        train_tensor_y = torch.Tensor(data.data[f'Y_train_{run.k}'][:,:d]) # transform to torch tensor
        train_dataset = TensorDataset(train_tensor_x,train_tensor_y) # create training dataset

        val_tensor_x = torch.Tensor(data.data[f'X_val_{run.k}'][:,:d]) # transform to torch tensor
        val_tensor_y = torch.Tensor(data.data[f'Y_val_{run.k}'][:,:d]) # transform to torch tensor
        val_dataset = TensorDataset(val_tensor_x,val_tensor_y) # create test dataset
        
        test_tensor_x = torch.Tensor(data.data[f'X_test_{run.k}'][:,:d]) # transform to torch tensor
        test_tensor_y = torch.Tensor(data.data[f'Y_test_{run.k}'][:,:d]) # transform to torch tensor
        test_dataset = TensorDataset(test_tensor_x,test_tensor_y) # create test dataset

        if model_type == "MLP":
            network = MLP(in_channels=d, 
                hidden_channels=run.hidden_channels, 
                likelihood_fn=run.likelihood_fn,
                dropout_rate=run.dropout_rate,
                linear_model=run.linear_model,
                )
                
        elif model_type == "SimpleRNN":
            network = SimpleRNN(in_channels=d,
                                likelihood_fn=run.likelihood_fn)
        
        else:
            raise ValueError('No valid model specified')
        
        if model_type == "MLP":
            train_loader = DataLoader(dataset=train_dataset, batch_size=run.batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=run.batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=run.batch_size, shuffle=False)
        elif model_type == "SimpleRNN":
            train_loader = DataLoader(dataset=train_dataset, batch_size=run.batch_size, shuffle=False)
            val_loader = DataLoader(dataset=val_dataset, batch_size=run.batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=run.batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(network.parameters(), lr=run.lr)
        
        change_folder = True
        if change_folder:
            experiment_name = f'{run}'
            wd = WorkingDirectory(generate_root(experiment_name))
        
        m.begin_run(run, network, train_loader)
        
        train_losses = []
        val_losses = []
        test_losses = []
        decision_losses = []
        
        for epoch in range (epochs):
            
            m.begin_epoch()
            
            train_loss, val_loss, test_loss = train_epoch(network, 
                                                optimizer, 
                                                train_loader, 
                                                val_loader, 
                                                epoch=epoch,
                                                test_loader=test_loader, 
                                                print_progress=True)

            if best_by == 'val':
                decision_loss = val_loss
            elif best_by == 'train':
                decision_loss = train_loss
            elif best_by == 'test':
                decision_loss = test_loss

            m.epoch_loss = train_loss
            m.epoch_val_loss = val_loss
            m.epoch_test_loss = test_loss
            m.epoch_decision_loss = decision_loss
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            decision_losses.append(decision_loss)
                
            m.end_epoch()

            save_as_best = True if decision_loss == min(decision_losses) else False
            save_checkpoint(wd, network.state_dict(), is_best=save_as_best)
        
            PATH = os.path.join(wd.root, 'e_%s_loss_%.3f.pth.tar' % (epoch, val_loss))
            torch.save(network.state_dict(), PATH)
            
        load_best = True
        if load_best:
            network.load_state_dict(torch.load(os.path.join(wd.root,'model_best.pth.tar')))
        
        # if sequential_predictions:
        #     outputs = make_sequential_predictions(model=network, test_dataset=test_dataset, x_mean=data.x_mean, x_std=data.x_std)
        # else:
        #     outputs = make_predictions(model=network, test_dataset=test_dataset)

        # with torch.no_grad():
        #     outputs = network(test_tensor_x)
        #     # outputs = network(val_tensor_x)
        
        if split_by == 'year':
            input_df = data.st[(data.st['year'].isin(data.split_dict[f'k{run.k}']['test']))] 
            input_st_names = None
        elif split_by == 'station':
            input_df = data.st
            input_st_names = data.split_dict[f'k{run.k}']['test']

        st_test = build_results_df(df=input_df,
                                test_dataset=test_dataset,
                                st_names_test=input_st_names,                 
                                model=network,
                                x_mean=data.x_mean,
                                x_std=data.x_std,
                                confidence_intervals=True,
                                draw_samples=True,
                                n_samples=n_samples,
                                sequential_samples = sequential_samples, 
                                threshold=sample_threshold
                                )
        
        linear_flag = 'L' if run.linear_model else 'NL'
        
        key = f'{run.likelihood_fn}_{run.hidden_channels}_{linear_flag}_B={run.batch_size}_D={run.dropout_rate}'
        
        if not(key in predictions.keys()):
            predictions[key] = {} 
            
        predictions[key][f'k{run.k}'] = st_test
        
        SAVEPATH = os.path.join(wd.root, "st_test.pkl")
        st_test.to_pickle(SAVEPATH)
        
        m.end_run()    
        
    m.save('results')

    return st_test, predictions

def truncate_sample(x, threshold=300):
    if x>threshold:
        return threshold
    else:
        return x

# def brier_scores(df, columns, obs, wet_threshold):
#     for c in columns:
#         df[f'BS_{c}'] = df.apply(BS, axis=1, args=(c, obs, wet_threshold))

# def quantile_scores(df, columns, obs, quantile):    
#     for c in columns:
#         df[f'QS_{c}'] = df.apply(QS, axis=1, args=(c, obs, quantile))