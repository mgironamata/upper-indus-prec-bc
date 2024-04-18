import torch

from torch.distributions.gamma import Gamma
from torch.distributions.gumbel import Gumbel
from torch.distributions    .normal import Normal
from torch.distributions.half_normal import HalfNormal

import CONFIG

import pdb

__all__ = [ 'gaussian_logpdf',
            'gamma_logpdf',
            'gamma_gamma_logpdf',
            'gumbel_logpdf',
            'halfnormal_logpdf',
            'lognormal_logpdf',
            'bernoulli_gamma_logpdf',
            'b2gmm_logpdf',
            'b2sgmm_logpdf',
            'bernoulli_gaussian_logpdf',
            'bernoulli_lognormal_logpdf',
            'bernoulli_gumbel_logpdf',
            'bernoulli_halfnormal_logpdf'
          ]

device = CONFIG.device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Device to perform computations on."""

def _mask(logp, mask):
    if mask is not None:
        return logp[mask]
    return logp

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

def gaussian_logpdf(obs, mu, sigma, reduction='mean', mask=None, device=device):
    """Gaussian model log-density.

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

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def gamma_logpdf(obs, alpha, beta, reduction='mean', mask=None, device=device):
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

    if mask is not None:   
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def gamma_gamma_logpdf(obs, alpha1, alpha2, beta1, beta2, q, reduction='mean', mask=None, device=device):
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

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def bernoulli_gamma_logpdf(obs, pi, alpha, beta, reduction='mean', mask=None, device=device):
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
    logp = torch.zeros(obs.shape, device=device)
    
    b_mask = obs == 0
    g_mask = obs > 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + Gamma(concentration=alpha[g_mask], rate=beta[g_mask]).log_prob(obs[g_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)
     
def b2gmm_logpdf(obs, pi, alpha1, alpha2, beta1, beta2, q, reduction='mean', mask=None, device=device):
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
    logp = torch.zeros(obs.shape, device=device)
    
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

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def b2sgmm_logpdf(obs, pi, alpha1, alpha2, beta1, beta2, q, t, reduction='mean', mask=None, device=device):
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
    logp = torch.zeros(obs.shape, device=device)
    
    b_mask = obs == 0
    g1_mask = (obs > 0) * (obs < t)
    g2_mask = (obs > 0) * (obs >= t)

    logp[g1_mask] = torch.log((1-pi[g1_mask])) + torch.log((q[g1_mask])) + Gamma(concentration=alpha1[g1_mask], rate=beta1[g1_mask]).log_prob(obs[g1_mask])
    logp[g2_mask] = torch.log((1-pi[g2_mask])) + torch.log((1-q[g2_mask])) + Gamma(concentration=alpha2[g2_mask], rate=beta2[g2_mask]).log_prob(obs[g2_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def bernoulli_gaussian_logpdf(obs, pi, mu, sigma, reduction='mean', mask=None, device=device):
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
    logp = torch.zeros(obs.shape, device=device)
    
    b_mask = obs == 0
    g_mask = obs != 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + Normal(loc=mu[g_mask], scale=sigma[g_mask]).log_prob(obs[g_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def bernoulli_lognormal_logpdf(obs, pi, mu, sigma, reduction='mean', mask=None, device=device):
    """Benroulli-lognormal mixture model log-density.

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
    logp = torch.zeros(obs.shape, device=device)
    
    b_mask = obs == 0
    g_mask = obs != 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + Normal(loc=mu[g_mask], scale=sigma[g_mask]).log_prob(torch.log(obs[g_mask]))
    logp[b_mask] = torch.log(pi[b_mask])

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def bernoulli_gumbel_logpdf(obs, pi, mu, beta, reduction='mean', mask=None, device=device):
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
    logp = torch.zeros(obs.shape, device=device)
    
    b_mask = obs == 0
    g_mask = obs != 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + Gumbel(loc=mu[g_mask], scale=beta[g_mask]).log_prob(obs[g_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def bernoulli_halfnormal_logpdf(obs, pi, sigma, reduction='mean', mask=None, device=device):
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
    logp = torch.zeros(obs.shape, device=device)
    
    b_mask = obs == 0
    g_mask = obs != 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + HalfNormal(scale=sigma[g_mask]).log_prob(obs[g_mask])
    logp[b_mask] = torch.log(pi[b_mask])

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def halfnormal_logpdf(obs, sigma, reduction='mean', mask=None, device=device):
    """HalfNormal mixture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        sigma (torch.Tensor): 
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """
    
    obs = obs.flatten()

    logp = HalfNormal(scale=sigma).log_prob(obs)

    if mask is not None:   
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def lognormal_logpdf(obs, mu, sigma, reduction='mean', mask=None, device=device):
    """LogNormal mixture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        mu (torch.Tensor): 
        sigma (torch.Tensor): 
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        torch.Tensor: Log-density.
    """

    obs = obs.flatten()
    b_mask = obs == 0

    epsilon = 0.000001
    obs[b_mask] = obs[b_mask] + epsilon

    logp = Normal(loc=mu, scale=sigma).log_prob(torch.log(obs))

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

def gumbel_logpdf(obs, mu, beta, reduction='mean', mask=None, device=device):
    """Gumbel mixture model log-density.

    Args:
        obs (torch.Tensor): Inputs.
        mu (torch.Tensor): 
        beta (torch.Tensor): 
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns: 
        torch.Tensor: Log-density.
    """
    
    obs = obs.flatten()

    logp = Gumbel(loc=mu, scale=beta).log_prob(obs)

    if mask is not None:
        logp = _mask(logp, mask)

    return _reduce(logp, reduction)

