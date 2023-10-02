import os
import numpy as np
import pandas as pd
from pandas._config import config
from math import prod
import pickle
import random, string

import torch
import torch.nn as nn
import torch.nn.functional as Fv
from torch.utils.data import TensorDataset, DataLoader, Dataset

from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift, GradientShap, FeatureAblation

import scipy.stats as stats 

from likelihoods import *
from metrics import *
from models import MLP, SimpleRNN, VGLM
from experiment import *
from runmanager import *
# from plot_utils import plot_losses
# from preprocessing_utils import *

import CONFIG

import pdb

__all__ =  [
            'RunningAverage',
            'train_epoch',
            'loss_fn',
            'gmm_fn',
            'sample_apply',
            'mixture_percentile',
            'build_results_df',
            'sample_mc',
            'truncate_sample',
            'count_zeros',
            'add_to_dict',
            'make_predictions',
            'make_sequential_predictions',
            'multirun',
            'truncate_sample'
            ]
            
device = CONFIG.device

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""Device to perform computations on."""

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

class CustomSimpleRNNDataset(Dataset):
    def __init__(self, predictors, predictands, stations, seq_length = 30):
        self.predictors = predictors
        self.predictands = predictands
        self.stations = stations
        self.seq_length = seq_length
        self.tensors = (self.predictors , self.predictands)
        
    def __len__(self):
        # prevents from sampling sequence indexes that are outside of data range
        return len(self.predictands) - self.seq_length 
    
    def __getitem__(self, idx):
        
        # checks that entire sequence belongs to same station
        while len(set(self.stations.squeeze().numpy()[idx : idx + self.seq_length]))>1:
            # otherwise, pick another index
            idx = np.random.randint(self.__len__())

        predictor = self.predictors[idx : idx + self.seq_length]
        predictand = self.predictands[idx : idx + self.seq_length]
        
        # station = self.stations[idx : idx + self.seq_length]
        # assert np.all(station.numpy() == station.numpy()[0])
        
        return predictor, predictand, 
        

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

    for idx, (predictors, labels) in enumerate(train_loader):
        
        predictors = predictors.to(device)
        labels = labels.to(device)

        # max_param = max(optimizer.param_groups[0]['params'])
        # min_param = min(optimizer.param_groups[0]['params'])

        optimizer.zero_grad()

        predictands = model(predictors)

        assert predictands.isnan().sum() == 0
        assert predictors.isnan().sum() == 0
        assert labels.isnan().sum() == 0

        loss = loss_fn(predictands, labels, predictors, model)  

        if loss.isnan():      

            for k, v in model.state_dict().items():
                if "weight" in k:
                    print(f"Max of {k}: {v.abs().max()}")

            print(f"Loss at index {idx}: {loss}")            
            print(f"Predictors at index {idx}: {predictors.abs().max()}")
            print(f"Labels at index {idx}: {labels.abs().max()}")

        if loss.item() != 0: 

            # loss.register_hook(lambda grad: print(grad)) prints dLoss/dLoss = 1   
            
            loss.backward()
            
            for param in optimizer.param_groups[0]['params']:
                nn.utils.clip_grad_value_(param, 1) # Bit of regularisation

            optimizer.step()
        
        train_losses.append(loss.item())
        
    model.eval()
    with torch.no_grad():
        for _, (predictors, labels) in enumerate(valid_loader):

            predictors = predictors.to(device)
            labels = labels.to(device)
            
            predictands = model(predictors)
            loss = loss_fn(predictands, labels, predictors, model)
            valid_losses.append(loss.item())

        if test_loader != None:

            for _, (predictors, labels) in enumerate(test_loader):

                predictors = predictors.to(device)
                labels = labels.to(device)

                predictands = model(predictors)
                loss = loss_fn(predictands, labels, predictors, model)
                test_losses.append(loss.item())
             
    #mean_train_losses.append(np.mean(train_losses))
    #mean_valid_losses.append(np.mean(valid_losses))
    
    if print_progress:
        print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'\
         .format(epoch+1, np.mean(train_losses), np.mean(valid_losses)))
    
    return np.mean(train_losses), np.mean(valid_losses), np.mean(test_losses)

def loss_fn(predictands : torch.tensor, labels : torch.tensor, predictors : torch.tensor, model : MLP, reduction : str = 'mean'):
    """Computes loss function (log-probability of labels).

    Args:
        predictands:
        labels
        model
        reduction

    Returns:
        loss
    """

    if predictands.dim() > 2: predictands = predictands.reshape(prod(predictands.shape[:-1]),-1)
    if predictors.dim() > 2: predictors = predictors.reshape(prod(predictors.shape[:-1]),-1)
    if labels.dim() > 2: labels = labels.reshape(prod(labels.shape[:-1]),-1)

    assert predictors.isnan().sum() == 0
    assert predictands.isnan().sum() == 0
    assert labels.isnan().sum() == 0

    # squeeze dimensions 0 and 1 for predictands, predictors...
        
    if model.likelihood == None:
        mask = predictors[:,0] > predictors[:,0].mode().values.item() 
        indices = mask.nonzero()
        ratio = len(indices)/len(predictands)
        loss = Fv.mse_loss(predictands[indices], labels[indices]) * ratio if len(indices)>0 else torch.tensor(0)

    elif model.likelihood == 'gaussian':
        mask = predictors[:,0] > predictors[:,0].mode().values.item() 
        indices = mask.nonzero()
        ratio = len(indices)/len(predictands)
        loss = -gaussian_logpdf(labels, mu=predictands[:,0][indices], sigma=predictands[:,1][indices], reduction=reduction) * ratio if len(indices)>0 else torch.tensor(0)
    
    elif model.likelihood == 'gamma':
        loss = -gamma_logpdf(labels, alpha=predictands[:,0], beta=predictands[:,1], reduction=reduction)

    elif model.likelihood == 'gamma_nonzero':
        mask = predictors[:,0] > predictors[:,0].mode().values.item() 
        indices = mask.nonzero()
        ratio = len(indices)/len(predictands)
        loss = -gamma_logpdf(labels, alpha=predictands[:,0][indices], beta=predictands[:,1][indices], reduction=reduction) * ratio if len(indices)>0 else torch.tensor(0)
    
    elif model.likelihood == 'ggmm':
        loss = -ggmm_logpdf(labels, alpha1=predictands[:,0], alpha2=predictands[:,1], 
                             beta1=predictands[:,2], beta2=predictands[:,3], q=predictands[:,4], reduction=reduction)

    elif model.likelihood == 'bgmm':
        loss = -bgmm_logpdf(labels, pi=predictands[:,0], alpha=predictands[:,1], beta=predictands[:,2], reduction=reduction)
    
    elif model.likelihood == 'b2gmm':
        loss = -b2gmm_logpdf(labels, pi=predictands[:,0], alpha1=predictands[:,1], alpha2=predictands[:,2], 
                             beta1=predictands[:,3], beta2=predictands[:,4], q=predictands[:,5], reduction=reduction)

    elif model.likelihood == 'b2sgmm':
        loss = -b2sgmm_logpdf(labels, pi=predictands[:,0], alpha1=predictands[:,1], alpha2=predictands[:,2], 
                             beta1=predictands[:,3], beta2=predictands[:,4], q=predictands[:,5], t=predictands[:,6], reduction=reduction)
    
    elif model.likelihood == 'bernoulli_gaussian':
        loss = -bernoulli_gaussian_logpdf(labels, pi=predictands[:,0], mu=predictands[:,1], sigma=predictands[:,2], reduction='mean')
    
    elif model.likelihood == 'bernoulli_loggaussian':
        loss = -bernoulli_loggaussian_logpdf(labels, pi=predictands[:,0], mu=predictands[:,1], sigma=predictands[:,2], reduction='mean')

    elif model.likelihood == 'bernoulli_gumbel':
        loss = -bernoulli_gumbel_logpdf(labels, pi=predictands[:,0], mu=predictands[:,1], beta=predictands[:,2], reduction='mean')

    elif model.likelihood == 'bernoulli_halfnormal':
        loss = -bernoulli_halfnormal_logpdf(labels, pi=predictands[:,0], sigma=predictands[:,1], reduction='mean')

    return loss

def gmm_fn(alpha1, alpha2, beta1, beta2, q):
    "Mixture model of two gamma distributions"
    
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

def cdf_apply(df: pd.DataFrame, likelihood_fn : str = 'bgmm', sample_size : int = 10000, obs : float = 10):
    
    if likelihood_fn == 'bgmm':
        
        pi = df['pi']
        alpha = df['alpha']
        beta = df['beta']
        # perc = df[series] 

        return pi + stats.gamma.cdf(obs, a=alpha, loc=0, scale=1/beta)*(1-pi)

    else:
        raise ValueError('Probability distribution not yet implemented')

def sample_apply(df : pd.DataFrame, likelihood_fn : str = 'bgmm', sample_size : int = 10000, series : str = 'uniform'):
    "Sample modelled distributions from a pd.Dataframe"
    
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

def mixture_percentile(df : pd.DataFrame, perc : float, likelihood_fn : str, sample_size : int = 1000):
    """Evalutates mixture model distribution of choice based on percentile value."""

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
        
def build_results_df(df, test_dataset, st_names_test, model, idx=0.05, x_mean=None, x_std=None,
                     confidence_intervals=False, draw_samples=True, n_samples=1, sequential_samples=False, threshold=None, model_type='MLP'):

    if sequential_samples:
        seq_predictors_dict = {}
        for i in range(n_samples):
            predictands = make_sequential_predictions(model, test_dataset, x_mean, x_std, threshold=threshold)
            seq_predictors_dict[i] = predictands
        
    else:
        predictands = make_predictions(model, test_dataset, model_type)

    predictands =predictands.cpu()
    
    if type(st_names_test)==type(None):
        new_df = df.copy()
    else:  
        new_df = df[df['Station'].isin(st_names_test)].copy()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    if model.likelihood == 'gamma':
        new_df['alpha'] = predictands[:,0]
        new_df['beta'] = predictands[:,1]
        new_df['mean'] = new_df['alpha']/new_df['beta']

        new_df[f'perc_median'] = 0.5
        new_df[f'median'] = new_df.apply(sample_apply, axis=1, likelihood_fn=model.likelihood, series='perc_median')
    
    elif model.likelihood == 'gamma_nonzero':
        new_df['alpha'] = predictands[:,0]
        new_df['beta'] = predictands[:,1]
        new_df['mean'] = new_df['alpha']/new_df['beta']

        new_df['occurrence'] = new_df['wrf_prcp'].apply(lambda x: 1 if x>0 else 0)

        new_df[f'perc_median'] = 0.5
        new_df[f'median'] = new_df.apply(sample_apply, axis=1, likelihood_fn=model.likelihood, series='perc_median')
    
    elif model.likelihood == 'gaussian':
        new_df['mu'] = predictands[:,0]
        new_df['sigma'] = predictands[:,1]

        new_df['mean'] = new_df['mu']

        new_df['occurrence'] = new_df['wrf_prcp'].apply(lambda x: 1 if x>0 else 0)

    elif model.likelihood == 'ggmm':
        new_df['alpha1'] = predictands[:,0]
        new_df['alpha2'] = predictands[:,1]
        new_df['beta1'] = predictands[:,2]
        new_df['beta2'] = predictands[:,3]
        new_df['q'] = predictands[:,4]
        
        new_df['mean'] =  gmm_fn(alpha1 = predictands[:,0],
                                alpha2 =  predictands[:,1],
                                beta1 = predictands[:,2],
                                beta2 = predictands[:,3],
                                q = predictands[:,4]
                                ).mean
    
    elif model.likelihood == 'bgmm':
        new_df['pi'] = predictands[:,0]
        new_df['alpha'] = predictands[:,1]
        new_df['beta'] = predictands[:,2] 

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)
        new_df['mean'] = new_df['occurrence']*new_df['alpha']/new_df['beta']

        # new_df[f'perc_median'] = 0.5
        # new_df[f'median'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_median')

    elif model.likelihood == 'b2gmm':
        new_df['pi'] = predictands[:,0]
        new_df['alpha1'] = predictands[:,1]
        new_df['alpha2'] = predictands[:,2] 
        new_df['beta1'] = predictands[:,3]
        new_df['beta2'] = predictands[:,4]
        new_df['q'] = predictands[:,5]
        
        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)

        new_df['mean'] =  gmm_fn(alpha1 = predictands[:,1],
                                alpha2 =  predictands[:,2],
                                beta1 = predictands[:,3],
                                beta2 = predictands[:,4],
                                q = predictands[:,5]
                                ).mean * new_df['occurrence']
        
        new_df[f'perc_median'] = 0.5
        new_df[f'median'] = new_df.apply(sample_apply, axis=1, likelihood_fn=model.likelihood, series='perc_median')

    elif model.likelihood == 'bernoulli_gaussian':
        new_df['pi'] = predictands[:,0]
        new_df['mu'] = predictands[:,1]
        new_df['sigma'] = predictands[:,2] 

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)
        new_df['mean'] = new_df['occurrence']*new_df['mu']
    
    elif model.likelihood == 'bernoulli_loggaussian':
        new_df['pi'] = predictands[:,0]
        new_df['mu'] = predictands[:,1]
        new_df['sigma'] = predictands[:,2] 

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)
        # new_df['mean'] = new_df['occurrence']*new_df['mu']

        # new_df[f'perc_median'] = 0.5
        # new_df[f'median'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_median')

    elif model.likelihood == 'bernoulli_gumbel':
        new_df['pi'] = predictands[:,0]
        new_df['mu'] = predictands[:,1]
        new_df['beta'] = predictands[:,2] 

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)

    elif model.likelihood == 'bernoulli_halfnormal':
        new_df['pi'] = predictands[:,0]
        new_df['sigma'] = predictands[:,1]

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)

    elif model.likelihood == 'b2sgmm':

        new_df['pi'] = predictands[:,0]
        new_df['alpha1'] = predictands[:,1]
        new_df['alpha2'] = predictands[:,2] 
        new_df['beta1'] = predictands[:,3]
        new_df['beta2'] = predictands[:,4]
        new_df['q'] = predictands[:,5]
        new_df['t'] = predictands[:,6]

        ### work in progress ###

        #new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)

        #new_df['pi_q'] = new_df['pi'] + new_df['q']
        #new_df['high_gamma_occurrence'] = new_df['pi_q'].apply(lambda x: 1 if x < 0.5 else 0) # NOT CORRECT
        # new_df['high_gamma'] = 1 - new_df['']

        new_df['mean'] = new_df['occurrence']*(new_df['low_gamma_occurrence']*new_df['alpha1']/new_df['beta1'] + (1-new_df['low_gamma_occurrence'])*new_df['alpha2']/new_df['beta2'])
    
    elif type(model.likelihood) ==type(None):
        new_df['occurrence'] = new_df['wrf_prcp'].apply(lambda x: 1 if x>0 else 0)
        new_df['sample_0'] = predictands.squeeze() * new_df['occurrence']
        draw_samples = False
        confidence_intervals = False
    
    if draw_samples:
        if sequential_samples:
            for i in range(n_samples):
                predictands = seq_predictors_dict[i]
                new_df[f'sample_{i}'] = predictands[:,-1]
        else:
            for i in range(n_samples):
                new_df[f'uniform_{i}'] = new_df.apply(lambda x: np.random.uniform(0,1),axis=1)
                new_df[f'sample_{i}'] = new_df.apply(sample_apply, axis=1, likelihood_fn=model.likelihood, series=f'uniform_{i}').astype('float64')

    # new_df['median'] = new_df.apply(mixture_percentile, axis=1, args=(0.5, model.likelihood))
    # new_df['median'] = new_df['median'] * new_df['occurrence']

    # new_df['median_gamma'] = new_df.apply(mixture_percentile_gamma_only, axis=1, args=(0.5, model.likelihood))
    # new_df['median_gamma'] = new_df['median_gamma'] * new_df['occurrence']

    if confidence_intervals:

        new_df[f'perc_low_ci'] = idx
        new_df[f'low_ci'] = new_df.apply(sample_apply, axis=1, likelihood_fn=model.likelihood, series='perc_low_ci')

        new_df[f'perc_high_ci'] = 1-idx
        new_df[f'high_ci'] = new_df.apply(sample_apply, axis=1, likelihood_fn=model.likelihood, series='perc_high_ci')

        #new_df['low_ci'] = new_df.apply(mixture_percentile, axis=1, args=(idx, model.likelihood))
        #new_df['high_ci'] = new_df.apply(mixture_percentile, axis=1, args=(1-idx, model.likelihood))

    quantile = 0.9
    new_df[f'QS_quantile'] = quantile
    new_df['QS_sample'] = new_df.apply(sample_apply, axis=1, args=(model.likelihood, 10000, 'QS_quantile'))
    new_df[f'BS'] = new_df.apply(BS, axis=1, args=('pi','Prec',0))
    new_df[f'QS'] = new_df.apply(QS, axis=1, args=('QS_sample', 'Prec', quantile))
                                              
    return new_df


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

def add_to_dict(xs,d):
    for x in xs:
        print(x)
        key = f'{x}'.split('=')[0]
        d[key] = x
    return d

def make_predictions(model, test_dataset, model_type = 'MLP'):

    model.eval()

    with torch.no_grad():

        test_predictors = test_dataset.tensors[0].to(device)    

        if model_type == 'SimpleRNN':
            test_predictors = torch.unsqueeze(test_predictors, 0)
        
        test_predictions = model(test_predictors)

        #   test2_predictors = test2_dataset.tensors[0]
        #   test2_predictors = model(test2_predictors)
    
    return test_predictions

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
            
            test_predictor = model(test_input.unsqueeze(0).float())

            # if test_predictor[0][0] == 0:
            #     print(test_input)
            
            theta_params = _get_theta_params(likelihood=model.likelihood)
            
            theta_dict = {key:test_predictor[0,i] for i, key in enumerate(theta_params)}
            
            sample = sample_mc(model, theta_dict)
            
            if threshold is not None:
                sample = truncate_sample(sample, threshold=threshold)
            
            norm_sample = (sample - x_mean[-1])/x_std[-1]
            
            if index==0:
                concat_test_predictors = torch.cat([test_predictor.squeeze(),
                                torch.tensor(sample).unsqueeze(0)]).unsqueeze(0)
            else:
                concat_test_predictor = torch.cat([test_predictor.squeeze(),
                                torch.tensor(sample).unsqueeze(0)]).unsqueeze(0)
                            
                concat_test_predictors = torch.cat([concat_test_predictors,
                                                 concat_test_predictor],dim=0)
            
            print(index) if index%10000==0 else None

    # test_predictors = concat_test_predictors[:,:-1]

    return concat_test_predictors

def multirun(data, predictors, params, epochs, split_by='station', 
             sequential_samples=False, sample_threshold=None, n_samples=10, draw_samples=True,
             best_by='val', use_device=device, load_run=None, feature_attribution=True, 
             save_to = '/data/hpcdata/users/marron31/', experiment_label = None, show_loss_plot = False):

    m = RunManager()
    predictions={}
    importance={}

    if experiment_label is None:
        random_label = ''.join([random.choice(string.ascii_letters) for i in range(10)])
    else:
        random_label = experiment_label

    for run in RunBuilder.get_runs(params):

        model_type = run.model_arch[0]
        hidden_channels = run.model_arch[1] 
        likelihood_fn = run.likelihood_fn
        dropout_rate = run.dropout_rate
        k = run.k
        batch_size = run.batch_size
        lr = run.lr

        if hasattr(run, 'random_noise'):
            random_noise = run.random_noise
        else:
            random_noise = 0

        d = len(predictors)
        
        train_tensor_x = torch.tensor(data.data[f'X_train_{k}'][:,:d],device='cpu', dtype=torch.float32) # transform to torch tensor
        train_tensor_y = torch.tensor(data.data[f'Y_train_{k}'][:,:d],device='cpu', dtype=torch.float32) # transform to torch tensor
        train_tensor_s = torch.tensor(data.data[f'S_train_{k}'][:,:d],device='cpu', dtype=torch.float32) # transform to torch tensor
        
        val_tensor_x = torch.tensor(data.data[f'X_val_{k}'][:,:d],device='cpu', dtype=torch.float32) # transform to torch tensor
        val_tensor_y = torch.tensor(data.data[f'Y_val_{k}'][:,:d],device='cpu', dtype=torch.float32) # transform to torch tensor        
        val_tensor_s = torch.tensor(data.data[f'S_val_{k}'][:,:d],device='cpu', dtype=torch.float32) # transform to torch tensor        
        
        test_tensor_x = torch.tensor(data.data[f'X_test_{k}'][:,:d],device='cpu', dtype=torch.float32) # transform to torch tensor
        test_tensor_y = torch.tensor(data.data[f'Y_test_{k}'][:,:d],device='cpu', dtype=torch.float32) # transform to torch tensor
        test_tensor_s = torch.tensor(data.data[f'S_test_{k}'][:,:d],device='cpu', dtype=torch.float32) # transform to torch tensor

        if model_type in ["VGLM","MLP"]:
            sequential_samples = False
            train_dataset = TensorDataset(train_tensor_x,train_tensor_y) # create training dataset
            val_dataset = TensorDataset(val_tensor_x,val_tensor_y) # create test dataset
            test_dataset = TensorDataset(test_tensor_x,test_tensor_y) # create test dataset
        elif model_type == "SimpleRNN":
            sequential_samples = True
            train_dataset = CustomSimpleRNNDataset(train_tensor_x,train_tensor_y,train_tensor_s) # create training dataset
            val_dataset = CustomSimpleRNNDataset(val_tensor_x,val_tensor_y,val_tensor_s) # create test dataset
            test_dataset = CustomSimpleRNNDataset(test_tensor_x,test_tensor_y,test_tensor_s) # create test dataset

        if model_type == "MLP":
            network = MLP(in_channels=d, 
                hidden_channels=hidden_channels, 
                likelihood_fn=likelihood_fn,
                dropout_rate=dropout_rate,
                random_noise=random_noise)
        elif model_type == "VGLM":
            network = VGLM(in_channels=d, 
                          likelihood_fn=likelihood_fn)  
        elif model_type == "SimpleRNN":
            network = SimpleRNN(in_channels=d,
                                likelihood_fn=likelihood_fn)
        else:
            raise ValueError('No valid model specified')

        network.to(use_device)

        num_workers = 0 if use_device == 'cpu' else 16

        if model_type in ["VGLM","MLP"]:
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        elif model_type == "SimpleRNN":
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        
        change_folder = True
        if change_folder:

            experiment_name = f'{run}'
            if load_run is None:
                wd = WorkingDirectory(generate_root(experiment_name, label_name=random_label))
            else:
                load_root = generate_root(experiment_name, 
                                            show_timestamp = False, 
                                            show_label = True, 
                                            label_name = random_label)
            
            MASTER_ROOT = f"{save_to}_experiments/{random_label}/"
            if not(os.path.isdir(MASTER_ROOT)):
                os.mkdir(MASTER_ROOT)
        
        m.begin_run(run, network, train_loader)
        
        train_losses = []
        val_losses = []
        test_losses = []
        decision_losses = []
        
        if load_run is None:
            
            for epoch in range (epochs):
            
                m.begin_epoch()
                
                train_loss, val_loss, test_loss = train_epoch(network,
                                                    optimizer,
                                                    train_loader,
                                                    val_loader,
                                                    epoch=epoch,
                                                    test_loader=test_loader,
                                                    print_progress=True)

                if best_by == 'val': decision_loss = val_loss
                elif best_by == 'train': decision_loss = train_loss
                elif best_by == 'test': decision_loss = test_loss

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
            
            # if show_loss_plot: plot_losses(train_losses, val_losses, test_losses)

        elif load_run is not None:
            network.load_state_dict(torch.load(os.path.join(load_root,'model_best.pth.tar')))
        
        # if sequential_predictions:
        #     predictands = make_sequential_predictions(model=network, test_dataset=test_dataset, x_mean=data.x_mean, x_std=data.x_std)
        # else:
        #     predictands = make_predictions(model=network, test_dataset=test_dataset)

        # with torch.no_grad():
        #     predictands = network(test_tensor_x)
        #     # predictands = network(val_tensor_x)        

        if split_by == 'year':
            input_df = data.st[(data.st['year'].isin(data.split_dict[f'k{k}']['test']))] 
            input_st_names = None
        elif split_by == 'station':
            input_df = data.st
            input_st_names = data.split_dict[f'k{k}']['test']

        st_test = build_results_df(df=input_df,
                                test_dataset=test_dataset,
                                st_names_test=input_st_names,                 
                                model=network,
                                x_mean=data.x_mean,
                                x_std=data.x_std,
                                confidence_intervals=False,
                                draw_samples=draw_samples,
                                n_samples=n_samples,
                                sequential_samples = sequential_samples, 
                                threshold=sample_threshold,
                                model_type=model_type
                                )
        
        key = f'{model_type}_{hidden_channels}_{likelihood_fn}_B={batch_size}_D={dropout_rate}_RN={random_noise}'
        print(key)
        
        if not(key in predictions.keys()):
            predictions[key] = {} 
            
        predictions[key][f'k{k}'] = st_test

        with open(os.path.join(MASTER_ROOT,'predictions.pkl'), 'wb') as handle:
            pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if feature_attribution:

            fa = FeatureAblation(network)

            for i in range(network.out_channels):
                fa_attr = fa.attribute(test_dataset.tensors[0], target=i)  

                if not(key in importance.keys()):
                    importance[key] = {}  
                
                if not(f'k{k}' in importance[key]):
                    importance[key][f'k{k}'] = {}

                importance[key][f'k{k}'][f'{i:.0f}'] = fa_attr
        
            with open(os.path.join(MASTER_ROOT,'importance.pkl'), 'wb') as handle:
                pickle.dump(importance, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if load_run is None:
            SAVEPATH = os.path.join(wd.root, "st_test.pkl")
            st_test.to_pickle(SAVEPATH)
        
        m.end_run()    
        
    m.save('results')

    # Create predictions for k_all
    for run in predictions.keys():
        for i in range(len(params['k'])):
            predictions[run][f'k{i}']['k_fold'] = i
            if i == 0:
                predictions[run]['k_all'] = predictions[run][f'k{i}']
            else:
                predictions[run]['k_all'] = predictions[run]['k_all'].append(predictions[run][f'k{i}'])
    
    with open(os.path.join(MASTER_ROOT,'predictions.pkl'), 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return st_test, predictions, importance

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