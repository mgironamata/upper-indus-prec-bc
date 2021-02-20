import numpy as np
import torch
import torch.nn as nn
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as Fv
from torch.distributions.gamma import Gamma
import scipy.stats as stats 


__all__ =  ['init_sequential_weights',
            'MLP',
            'bgmm_logpdf',
            'b2gmm_logpdf',
            'b2sgmm_logpdf',
            'train_epoch',
            'loss_fn',
            'gmm_fn',
            'mixture_percentile',
            'mixture_percentile_gamma_only',
            'build_results_df',
            ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""Device perform computations on."""

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

class MLP(nn.Module):
    """MLP with 1 hidden layer
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """

    def __init__(self,in_channels,hidden_channels=10,likelihood_fn='bgmm'):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.likelihood = likelihood_fn
        
        if self.likelihood == 'gamma':
            self.out_channels = 2
        if self.likelihood == 'ggmm':
            self.out_channels = 5   
        elif self.likelihood == 'bgmm':
            self.out_channels = 3
        elif self.likelihood == 'b2gmm':
            self.out_channels = 6
        elif self.likelihood == 'b2sgmm':
            self.out_channels = 7
 
        #self.f = self.build_weight_model()
        self.exp = torch.exp
        self.sigmoid = torch.sigmoid
        self.relu = nn.ReLU()

        # Hidden layers
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(self.in_channels, self.hidden_channels[0]))
        
        for k in range(len(self.hidden_channels)-1):
            self.hidden.append(nn.Linear(self.hidden_channels[k], self.hidden_channels[k+1]))

         # Output layer
        self.out = nn.Linear(self.hidden_channels[-1], self.out_channels)
        ####   

    def build_weight_model(self):
        """Returns a point-wise function that transforms the in_channels-dimensional
        input features to dimensionality out_channels.

        Returns:
          torch.nn.Module: Linear layer applied point-wise to channels.  
        """

        model = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels),
            #nn.ReLU(),
            #nn.Linear(self.hidden_channels,self.hidden_channels),
            #nn.ReLU(),
            #nn.Linear(self.hidden_channels,self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels,self.out_channels),
            )
        init_sequential_weights(model)
        return model

    def forward(self, x):
        
        #x = self.f(x)
        
        # Feedforward
        for layer in self.hidden[:]:
            x = self.relu(layer(x))
        x = self.out(x)
        ####
        if self.likelihood=='gamma':
            x[:,:] = self.exp(x[:,:]) # alpha, beta
            return x
        elif self.likelihood=='ggmm':
            x[:,:] = self.exp(x[:,:]) # alpha, beta
            return x
        elif self.likelihood=='bgmm':
            x[:,0] = self.sigmoid(x[:,0]) # pi
            x[:,1:] = self.exp(x[:,1:]) # alpha, beta
            return x
        elif self.likelihood=='b2gmm':
            x[:,0] = self.sigmoid(x[:,0]) # pi
            x[:,1:-1] = self.exp(x[:,1:-1]) #  alpha1, alpha2, beta1, beta2
            x[:,-1] = self.sigmoid(x[:,-1]) # q : weight parameter for gamma mixture model (#TO REVIEW)
            return x
        elif self.likelihood=='b2sgmm':
            x[:,0] = self.sigmoid(x[:,0]) # pi
            x[:,1:5] = self.exp(x[:,1:-2]) # alpha1, alpha2, beta1, beta2
            x[:,5] = self.sigmoid(x[:,-2]) # q : weight parameter for gamma mixture model (TO REVIEW)
            x[:,6] = self.exp(x[:,-1]) # t : threshold 
            return x

def gamma_logpdf(obs, alpha, beta, reduction='mean'):
    """Benroulli-Gamma mixture model log-density.

    Args:
        obs (tensor): Inputs.
        alpha (tensor): 
        beta (tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    
    #pdb.set_trace()

    obs = obs.flatten()
    b_mask = obs == 0

    epsilon = 1
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
        obs (tensor): Inputs.
        alpha1 (tensor): 
        beta1 (tensor):
        alpha1 (tensor): 
        beta1 (tensor):
        q (tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """

    obs = obs.flatten()
    b_mask = obs == 0

    epsilon = 1
    obs[b_mask] = obs[b_mask] + epsilon
    
    
    #pdb.set_trace()

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
        obs (tensor): Inputs.
        pi (tensor): 
        alpha (tensor): 
        beta (tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
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

def bgumm_logpdf(obs, pi, alpha, beta, reduction='mean'):
    """Benroulli-Gumbel mixture model log-density.

    Args:
        obs (tensor): Inputs.
        pi (tensor): 
        alpha (tensor): 
        beta (tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    
    #pdb.set_trace()

    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g_mask = obs > 0

    logp[g_mask] = torch.log((1-pi[g_mask])) + Gumbel(loc=alpha[g_mask], scale=beta[g_mask]).log_prob(obs[g_mask])
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
        obs (tensor): Inputs.
        pi (tensor): 
        alpha1 (tensor): 
        beta1 (tensor):
        alpha1 (tensor): 
        beta1 (tensor):
        q (tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """

    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g_mask = obs > 0
    
    k = g_mask.shape[0]
    
    #pdb.set_trace()

    mixture_weights = torch.stack([q[g_mask],1-q[g_mask]]).permute([1,0]) # REVIEW
    mixture_alphas = torch.stack([alpha1[g_mask],alpha2[g_mask]]).permute([1,0])
    mixture_betas = torch.stack([beta1[g_mask],beta2[g_mask]]).permute([1,0])

    mix = torch.distributions.Categorical(mixture_weights)
    comp = torch.distributions.Gamma(mixture_alphas, mixture_betas)
    gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)   

    logp[g_mask] = torch.log((1-pi[g_mask])) + gmm.log_prob(obs[g_mask])
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
        obs (tensor): Inputs.
        pi (tensor): 
        alpha (tensor): 
        beta (tensor):
        q (tensor):
        t (tensor):
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    
    obs = obs.flatten()
    logp = torch.zeros(obs.shape)
    
    b_mask = obs == 0
    g1_mask = (obs > 0) * (obs < t)
    g2_mask = (obs > 0) * (obs >= t)

    logp[g1_mask] = torch.log((1-pi[g1_mask])) + torch.log((q[g1_mask])) + Gamma(concentration=alpha1[g1_mask], rate=beta1[g1_mask]).log_prob(obs[g1_mask])
    logp[g2_mask] = torch.log((1-pi[g2_mask])) + torch.log((1- q[g2_mask])) + Gamma(concentration=alpha2[g2_mask], rate=beta2[g2_mask]).log_prob(obs[g2_mask])
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

def train_epoch(model, optimizer, train_loader, valid_loader, epoch, print_progress=False):
    
    model.train()
    train_losses = []
    valid_losses = []
    
    for i, (inputs, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_fn(outputs, labels, model)                 
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):

            outputs = model(inputs)
            
            loss = loss_fn(outputs, labels, model)
            
            valid_losses.append(loss.item())
            
    #mean_train_losses.append(np.mean(train_losses))
    #mean_valid_losses.append(np.mean(valid_losses))
    
    if print_progress:
        print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}'\
         .format(epoch+1, np.mean(train_losses), np.mean(valid_losses)))
    
    return np.mean(train_losses), np.mean(valid_losses)

def loss_fn(outputs, labels, model):
        
    if model.likelihood == 'gamma':
        loss = -gamma_logpdf(labels, alpha=outputs[:,0], beta=outputs[:,1], reduction='mean')
    
    elif model.likelihood == 'ggmm':
        loss = -ggmm_logpdf(labels, alpha1=outputs[:,0], alpha2=outputs[:,1], 
                             beta1=outputs[:,2], beta2=outputs[:,3], q=outputs[:,4], reduction='mean')

    elif model.likelihood == 'bgmm':
        loss = -bgmm_logpdf(labels, pi=outputs[:,0], alpha=outputs[:,1], beta=outputs[:,2], reduction='mean')
    
    elif model.likelihood == 'b2gmm':
        loss = -b2gmm_logpdf(labels, pi=outputs[:,0], alpha1=outputs[:,1], alpha2=outputs[:,2], 
                             beta1=outputs[:,3], beta2=outputs[:,4], q=outputs[:,5], reduction='mean')

    elif model.likelihood == 'b2sgmm':
        loss = -b2sgmm_logpdf(labels, pi=outputs[:,0], alpha1=outputs[:,1], alpha2=outputs[:,2], 
                             beta1=outputs[:,3], beta2=outputs[:,4], q=outputs[:,5], t=outputs[:,6], reduction='mean')
    
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

def build_results_df(df, outputs, st_names_test, model, p=0.05, confidence_intervals=False):
    new_df = df[df['Station'].isin(st_names_test)].copy()
    
    if model.likelihood == 'gamma':
        new_df['alpha'] = outputs[:,0]
        new_df['beta'] = outputs[:,1] 

        new_df['mean'] = new_df['alpha']/new_df['beta']

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
                                ).mean
        
        new_df['mean'] = new_df['occurrence'] * new_df['mean'] 

    elif model.likelihood == 'b2sgmm':

        new_df['pi'] = outputs[:,0]
        new_df['alpha1'] = outputs[:,1]
        new_df['alpha2'] = outputs[:,2] 
        new_df['beta1'] = outputs[:,3]
        new_df['beta2'] = outputs[:,4]
        new_df['q'] = outputs[:,5]
        new_df['t'] = outputs[:,6]

        ### work in progress ###

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)

        new_df['pi_q'] = new_df['pi'] + new_df['q']
        new_df['low_gamma'] = new_df['pi_q'].apply(lambda x: 1 if x < 0.5 else 0) # NOT CORRECT
        #new_df['high_gamma'] = 1 - new_df['']

        new_df['mean'] = new_df['occurrence']*new_df['alpha']/new_df['beta']
        
    new_df['median'] = new_df.apply(mixture_percentile, axis=1, args=(0.5, model.likelihood))
    #new_df['median'] = new_df['median'] * new_df['occurrence']

    new_df['median_gamma'] = new_df.apply(mixture_percentile_gamma_only, axis=1, args=(0.5, model.likelihood))
    #new_df['median_gamma'] = new_df['median_gamma'] * new_df['occurrence']
    
    new_df['se_wrf'] = (new_df['wrf_prcp'] - new_df['Prec'])**2
    new_df['se_bcp'] = (new_df['wrf_bc_prcp'] - new_df['Prec'])**2
    new_df['se_mlp'] = (new_df['mean'] - new_df['Prec'])**2
    
    new_df['e_wrf'] = (new_df['wrf_prcp'] - new_df['Prec'])
    new_df['e_bcp'] = (new_df['wrf_bc_prcp'] - new_df['Prec'])
    new_df['e_mlp'] = (new_df['mean'] - new_df['Prec'])

    new_df['se_mlp_median'] = (new_df['median'] - new_df['Prec'])**2 
    new_df['se_mlp_median_gamma'] = (new_df['median_gamma'] - new_df['Prec'])**2 

    if confidence_intervals:
        new_df['low_ci'] = new_df.apply(mixture_percentile, axis=1, args=(p, model.likelihood))
        new_df['high_ci'] = new_df.apply(mixture_percentile, axis=1, args=(1-p, model.likelihood))
                                              
    return new_df