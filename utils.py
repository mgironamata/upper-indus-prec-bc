import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fv
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
import scipy.stats as stats 

import pdb

__all__ =  ['init_sequential_weights',
            'gaussian_logpdf',
            'gamma_logpdf',
            'ggmm_logpdf',
            'bgmm_logpdf',
            'b2gmm_logpdf',
            'b2sgmm_logpdf',
            'train_epoch',
            'loss_fn',
            'gmm_fn',
            'mixture_percentile',
            'build_results_df',
            'RunningAverage',
            'pairwise_errors'
            ]
            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""Device to perform computations on."""

def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.

    Args:
        model (:class:`nn.Sequential`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.

    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias)
    return model

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
                nn.utils.clip_grad_value_(param, 10)

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

def build_results_df(df, outputs, st_names_test, model, p=0.05, confidence_intervals=False, draw_samples=True, n_samples=1):
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

        new_df[f'perc_median'] = 0.5
        new_df[f'median'] = new_df.apply(sample, axis=1, likelihood_fn=model.likelihood, series='perc_median')

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
