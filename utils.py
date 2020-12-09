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
            'train_epoch',
            'loss_fn',
            'gmm_fn',
            'mixture_percentile',
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
        
        if self.likelihood == 'bgmm':
            self.out_channels = 3
        elif self.likelihood == 'b2gmm':
            self.out_channels = 6

            
        self.f = self.build_weight_model()
        self.exp = torch.exp
        self.sigmoid = torch.sigmoid
            

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
        
        x = self.f(x)
        
        if self.likelihood=='bgmm':
            x[:,0] = self.sigmoid(x[:,0])
            x[:,1:] = self.exp(x[:,1:])
            return x
        elif self.likelihood=='b2gmm':
            x[:,0] = self.sigmoid(x[:,0])
            x[:,1:-1] = self.exp(x[:,1:-1])
            x[:,-1] = self.sigmoid(x[:,-1]) #TO REVIEW
            return x

def bgmm_logpdf(obs, pi, alpha, beta, reduction='mean'):
    """Benroulli-Gamma mexture model log-density.

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
        
def b2gmm_logpdf(obs, pi, alpha1, beta1, alpha2, beta2, q, flag=1, reduction='mean'):
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
        
    if model.likelihood == 'bgmm':
        loss = -bgmm_logpdf(labels, pi=outputs[:,0], alpha=outputs[:,1], beta=outputs[:,2], reduction='mean')

    elif model.likelihood == 'b2gmm':
        loss = -b2gmm_logpdf(labels, pi=outputs[:,0], alpha1=outputs[:,1], alpha2=outputs[:,2], 
                             beta1=outputs[:,3], beta2=outputs[:,4], q=outputs[:,5], reduction='mean')
    return loss

def gmm_fn(pi, alpha1, alpha2, beta1, beta2, q):
    
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
    if likelihood_fn == 'bgmm':
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
            dist = gmm_fn(pi, alpha1, alpha2, beta1, beta2 , q)
            return torch.quantile(dist.sample([sample_size]), quantile).numpy()
        else:
            return 0

def build_results_df(df, outputs, st_names_test, model, p=0.05, confidence_intervals=False):
    new_df = df[df['Station'].isin(st_names_test)].copy()
    
    if model.likelihood == 'bgmm':
        new_df['pi'] = outputs[:,0]
        new_df['alpha'] = outputs[:,1]
        new_df['beta'] = outputs[:,2] 

        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)
        new_df['magnitude'] = new_df['occurrence']*new_df['alpha']/new_df['beta']

    elif model.likelihood == 'b2gmm':
        new_df['pi'] = outputs[:,0]
        new_df['alpha1'] = outputs[:,1]
        new_df['alpha2'] = outputs[:,2] 
        new_df['beta1'] = outputs[:,3]
        new_df['beta2'] = outputs[:,4]
        new_df['q'] = outputs[:,5]
        
        new_df['occurrence'] = new_df['pi'].apply(lambda x: 1 if x < 0.5 else 0)
        new_df['magnitude'] =  gmm_fn(pi = new_df,
                                         alpha1 = outputs[:,1],
                                         alpha2 =  outputs[:,2],
                                         beta1 = outputs[:,3],
                                         beta2 = outputs[:,4],
                                         q = outputs[:,5]
                                         ).mean
        
        new_df['magnitude'] = new_df['occurrence'] * new_df['magnitude'] 
        
    new_df['se_wrf'] = (new_df['model_precipitation'] - new_df['Prec'])**2
    new_df['se_bcp'] = (new_df['wrf_bcp'] - new_df['Prec'])**2
    new_df['se_mlp'] = (new_df['magnitude'] - new_df['Prec'])**2
    
    new_df['e_wrf'] = (new_df['model_precipitation'] - new_df['Prec'])
    new_df['e_bcp'] = (new_df['wrf_bcp'] - new_df['Prec'])
    new_df['e_mlp'] = (new_df['magnitude'] - new_df['Prec'])

    if confidence_intervals:
        new_df['low_ci'] = new_df.apply(mixture_percentile, axis=1, args=(p, model.likelihood))
        new_df['high_ci'] = new_df.apply(mixture_percentile, axis=1, args=(1-p, model.likelihood))
                                              
    return new_df