import numpy as np
import pandas as pd
import xarray as xr
import os

from math import pi

from preprocessing_utils import DataPreprocessing

import torch
from torch.utils.data import Dataset

# Import only required tools
from models import *
from utils import *
from runmanager import *
from experiment import *
from plot_utils import *
from preprocessing_utils import *
from elbo import *

# import Gamma distribution 
from torch.distributions import Gamma

pd.options.display.max_columns = None

np.random.seed(4)

import matplotlib
matplotlib.rc_file_defaults()

from plum import dispatch
from varz.torch import Vars
import torch.nn as nn
from stheno.torch import B, GP, EQ, Normal, Measure
from matrix import Diagonal

import pdb

# Detect device.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

__all__ = ['UpperIndusGridDataset', 'UpperIndusDataset', 'prepare_data', 'MultipleOptimizer', 'forward_backward_pass','MapDataset']

class UpperIndusGridDataset(Dataset):
    """Grid data for the Beas and Sutlej basins of the Upper Indus region"""
    
    def __init__(self, root_folder, train_mean, train_var):
        
        self.train_mean = np.expand_dims(train_mean, 1)
        self.train_var = np.expand_dims(train_var, 1)
        
        self.srtm = xr.open_dataset(os.path.join(root_folder,'srtm_reprojected_beas.nc'))
        self.era5_u = xr.open_dataset(os.path.join(root_folder,'era5_u_reprojected.nc'))
        self.era5_v = xr.open_dataset(os.path.join(root_folder,'era5_v_reprojected.nc'))
        self.wrf = xr.open_dataset(os.path.join(root_folder,'wrf.nc'))
        
        self.doy = pd.DatetimeIndex(self.wrf.time.values).dayofyear
        
        self.doysin = np.sin(self.doy*2*pi/365)
        self.doycos = np.cos(self.doy*2*pi/365)
        
        self.n = len(self.wrf.time)
        
    def __len__(self):
        return self.n
    
    def __getitem__(self,idx):
        
        srtm_arr = self.srtm.z.values.squeeze().flatten()
        mask = np.where(srtm_arr > 0)[0]
        
        wrf_arr = self.wrf.model_precipitation.isel(time=idx).values.flatten()
        
        era5_u_arr = self.era5_u.u.isel(time=idx).values.flatten()
        era5_v_arr = self.era5_v.v.isel(time=idx).values.flatten()
        
        x_arr, y_arr = np.meshgrid(self.srtm.x, self.srtm.y)
        x_arr, y_arr = x_arr.flatten(), y_arr.flatten()
        
        doysin_arr = np.full(x_arr.shape,self.doysin[idx]).flatten()
        doycos_arr = np.full(x_arr.shape,self.doycos[idx]).flatten()
        
        #print(srtm_arr.shape, wrf_arr.shape, x_arr.shape, y_arr.shape, doysin_arr.shape, doycos_arr.shape)
        
        array = np.stack([x_arr, y_arr, srtm_arr, wrf_arr, doysin_arr, doycos_arr,
                         era5_u_arr, era5_v_arr
                        ])[:,mask] 
        
        array_norm = (array - self.train_mean) / np.sqrt(self.train_var)
        
        return array_norm, array

class MapDataset(Dataset):

    def __init__(self, PATH, predictors, train_mean, train_var):

        self.predictors = predictors
        self.df = pd.read_csv(PATH)[predictors]

        # normalise df
        self.df_norm = (self.df - train_mean[:]) / np.sqrt(train_var[:])

        self.ds = self.df.to_xarray()
        self.ds_norm = self.df_norm.to_xarray()
        

    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return np.array(self.df_norm), np.array(self.df)
        #return self.ds_norm.to_array().values, self.ds.to_array().values

class UpperIndusDataset(Dataset):

    def __init__(self, TRAIN_PATH, start, end, predictant, predictors, stations=None):
        
        st, n, mean, var = prepare_data(TRAIN_PATH, start, end, predictant, predictors, stations=stations)
        
        self.stations = stations
        self.st = st
        self.ds = st.to_xarray()

        # create mask for missing data
        # self.mask = ~self.ds.isnull()

        # replace ds missing values by zeros
        self.ds = self.ds.fillna(0)

        self.n = n
        
        self.mean = mean
        self.var = var
        
    def __len__(self):
        return len(self.ds.Date)

    def __getitem__(self,idx):
        arr = self.ds.isel(Date=idx).to_array().values
        # mask = self.mask.isel(Date=idx).to_array().values
        return arr[1:,:], arr[0,:]

def prepare_data(TRAIN_PATH, start, end, predictant, predictors, stations=None):
    
    data = DataPreprocessing(TRAIN_PATH, start, end)
    st = data.st.drop_duplicates()

    n = len(st)

    if stations != None:
        st = st[st['Station'].isin(stations)]
    
    st.set_index(['Date','Station'], inplace=True)
    
    st = st[predictant + predictors]

    mean = st[predictors].mean().values
    var = st[predictors].var().values

    st[predictors] = (st[predictors] - mean) / np.sqrt(var)

    # st.reset_index(inplace=True)

    st = st.unstack(1).stack(1, dropna=False)
    st['X'] = st.groupby("Station")['X'].transform('mean')
    st['Y'] = st.groupby("Station")['Y'].transform('mean')
    
    return st, n, mean, var

class MultipleOptimizer(object):
    
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

def forward_backward_pass(inputs, labels, n, model, optimizer, q, f, x_ind, inducing_points=True, backward=True, f_marginal=True, n_samples=10, test_time=False, num_GP_dims=3, remove_from_inputs=True):
    
    inputs = inputs.float() # inputs [batch_size, num_predictors, num_stations]
    if not test_time: labels = labels.float() # labels [batch_size, num_stations]

    b = inputs.shape[0] # mini batch size

    # Build MV Normal Distribution
    q.build_normal()

    # GP inputs 
    x = inputs[0,:num_GP_dims,:].permute(1,0).float()

    if remove_from_inputs:  
        inputs = inputs[:,num_GP_dims:,:]

    # Repeat input tensor K (n_samples) times
    inputs = inputs.unsqueeze(-1).repeat(1,1,1,n_samples).permute(0,2,3,1) # inputs shape: (b, num_stations, n_samples, num_predictors)
    if not test_time: labels = labels.unsqueeze(-1).repeat(1,1,n_samples) # labels shape: (b, num_stations, n_samples, 1)

    sample_q_once = False
    # Compute KL
    if inducing_points:
        if sample_q_once:
            q_sample =  q.sample()
        else:
            q_sample = B.transpose(q.sample(b * n_samples))[:, :, None]
        
        f_post = f | (f(x_ind),q_sample) # MEMORY LEAK HERE
        
        kl = q.kl(f(x_ind))
        
        # Sample z and concatenate to inputs
        if f_marginal:
            f_sample = Normal(f_post.mean(x),
                              Diagonal(f_post.kernel.elwise(x)[:, 0])
                             ).sample(b*n_samples) # sample from f_post    
        else:
            if sample_q_once:
                f_sample = f_post(x).sample(b*n_samples) # sample from f_post
            else:
                f_sample = f_post(x).sample().squeeze().permute(1,0) # sample from f_post
            
                
        f_sample = f_sample.permute(1,0).unsqueeze(1).reshape(b,-1,x.shape[0],n_samples).permute(0,2,3,1) # reshape f_sample to match inputs shape
        inputs = torch.cat([f_sample, inputs], dim=3)  # concatenate f_sample to inputs
    
    else:
        f_x = f(x) # output of the GP model
        kl = q.kl(f_x) # KL divergence between q and p

        q_sample = q.sample(b*n_samples) # sample from q
        q_sample = q_sample.permute(1,0).unsqueeze(1).reshape(b,-1,x.shape[0],n_samples).permute(0,2,3,1) # reshape q_sample to match inputs shape
        inputs = torch.cat([q_sample, inputs], dim=3) # concatenate q_sample to inputs

    # Masking for missing data
    # inputs = inputs.permute(0,1,3,2)

    # if nan in inputs, then pdb.set_trace
    if inputs.isnan().sum() > 0:
        pdb.set_trace()

    mask = ~torch.any(inputs.isnan(),dim=3)

    k = mask.any(dim=0).any(dim=1).sum()

    # Forward pass
    outputs = model(inputs.float()) # outputs: (b, num_stations, n_samples, num_output_dims)

    # assert inputs.isnan().sum() == 0
    # assert outputs.isnan().sum() == 0
    # if not test_time: assert labels.isnan().sum() == 0

    if not test_time: 
        device = labels.device
        logp = torch.zeros_like(labels, device=device)
    
        b_mask = labels == 0 # shape (b, n_samples, k)
        g_mask = labels > 0

        if model.likelihood == 'bgmm':
            pi = outputs[...,0] # shape (b, num_stations, n_samples)
            alpha = outputs[...,1] # shape (b, num_stations, n_samples)
            beta = outputs[...,2] # shape (b, num_stations, n_samples)

            # Computing log probabilities for gamma distribution where obs > 0
            gamma_dist = Gamma(concentration=alpha[g_mask], rate=beta[g_mask])
            logp[g_mask] = torch.log((1 - pi[g_mask])) + gamma_dist.log_prob(labels[g_mask])
            
            # Computing log probabilities for Bernoulli distribution where obs == 0
            logp[b_mask] = torch.log(pi[b_mask])

        elif model.likelihood == 'bernoulli':
            pi = outputs.squeeze()
            # print(f"pi: {pi.min().item()}, {pi.max().item()}")
            logp = torch.log(pi) * labels + torch.log(1 - pi) * (1 - labels)

        if mask is not None:
            logp = logp * mask

        # dimensions of logp : (b, num_stations, n_samples)
        recon = logp.sum(1).mean() # sum over stations, then average across samples, then average over batch size
        
        # sum logp over stations - resulting tensor dims: (b, n_samples)
        sum_logp_N = logp.sum(1)

        # Use torch.logsumexp for the final computation over Monte Carlo samples. 
        sum_logp_NM = torch.logsumexp(sum_logp_N, dim=1) - torch.log(torch.tensor(n_samples, dtype=torch.float, device=device))

        # Reconstruction term, averaging the term sum_logp_NM over the batch size
        neg_sum_logp_NM = -sum_logp_NM.mean()
        nll = neg_sum_logp_NM

        # Use torch.logsumexp for computing the expectation over the Monte Carlo samples without summing over stations
        sum_logp_M = torch.logsumexp(logp, dim=2) - torch.log(torch.tensor(n_samples, dtype=torch.float, device=device)) # dims: (b, num_stations)
        sum_logp_M_stations = sum_logp_M.sum(1) # dims: (b)
        sum_logp_M_stations_batchmean = sum_logp_M_stations.mean() # dims: (1)
        
        # ELBO
        elbo = recon/(k) - kl/(b*k) # ORIGINALLY: elbo = recon/(b*k) - kl/n # OR (n/(b*k)*recon - kl)/n
        
        # print(f"ELBO: {elbo.item():.4f}, Recon: {(recon).item():.4f}, KL: {(kl/n).item():.4f}, K: {k.item()}")

        # Backward pass and optimizer step
        if backward:
            (-elbo).backward()
            nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'], 10) # Bit of regularisation
            optimizer.step()
            optimizer.zero_grad()
        
        return elbo, recon, kl, k, nll
    
    else:
        return inputs, outputs

if __name__ == "__main__":
    # Parameters
    start="2010-01-01"
    end="2020-12-31"
    TRAIN_PATH = r"data/norris/enriched_obs/enriched_langtang_obs_norris_ready.pkl"

    predictant = ['Prec']
    predictors = [
              #'Date',
              #'Station',
              #'Prec',
              #'Corrected Station Name', 
              'X', 'Y',
              #'Altitude (m)', 
              'Z', 
              'precip_norris', #'wrf_bc_prcp', 
              #'elev_hr', 
              #'aspect','slope', 
              #'doy', 
              'doy_sin', 'doy_cos', 
              #'wrf_prcp_-1', 'wrf_prcp_-2','wrf_prcp_1', 'wrf_prcp_2', 
              #'Basin', 'lon', 'lat', 
#               'era5_u', 'era5_v',
              #'era5_u_-2', 'era5_u_-1', 'era5_u_1', 'era5_u_2', 
              #'era5_v_-2', 'era5_v_-1', 'era5_v_1', 'era5_v_2'
             ]

    ds_dataset = UpperIndusDataset(TRAIN_PATH, start, end, predictant, predictors, stations=None)
    

