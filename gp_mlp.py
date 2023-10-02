import numpy as np
import pandas as pd
import xarray as xr
import os, shutil, time, pdb, random, pickle
import scipy.stats as stats 

from IPython.display import clear_output

from math import pi
from datetime import datetime
from collections import OrderedDict

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

from importlib import reload
from models import *
from utils import *
from runmanager import *
from experiment import *
from plot_utils import *
from preprocessing_utils import *
from elbo import *

# from shapely.geometry import box, mapping
# import descartes

from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt

pd.options.display.max_columns = None

np.random.seed(4)

import matplotlib
matplotlib.rc_file_defaults()


from plum import dispatch
from varz.torch import Vars
import torch.nn as nn
from stheno.torch import B, GP, EQ, Normal, Measure
from matrix import Diagonal

# Detect device.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

__all__ = ['UpperIndusGridDataset', 'UpperIndusDataset', 'prepare_data', 'MultipleOptimizer', 'forward_backward_pass']

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

class UpperIndusDataset(Dataset):
    """Station data for the Beas and Sutlej basins of the Upper Indus region"""
    def __init__(self, TRAIN_PATH, start, end, predictant, predictors, stations=None):
        st, n, mean, var = prepare_data(TRAIN_PATH, start, end, predictant, predictors, stations=stations)
        self.stations = stations
        self.st = st
        self.ds = st.to_xarray()
        self.n = n
        
        self.mean = mean
        self.var = var
        
    def __len__(self):
        return len(self.ds.Date)

    def __getitem__(self,idx):
        arr = self.ds.isel(Date=idx).to_array().values
        return arr[1:,:], arr[0,:]

def prepare_data(TRAIN_PATH, start, end, predictant, predictors, stations=None):
    
    st = (import_dataframe(TRAIN_PATH)
    .pipe(drop_df_NaNs, series='Prec') # Drop NaNs
    .pipe(clip_time_period, start, end) # Clip data temporally
    )
    
    st[st['Basin']=='Beas']
    
    n = len(st)

    if stations != None:
        st = st[st['Station'].isin(stations)]
    st.set_index(['Date','Station'], inplace=True)
    st = st[predictant + predictors]

    mean = st[predictors].mean().values
    var = st[predictors].var().values

    st[predictors] = (st[predictors] - mean) / np.sqrt(var)

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

def forward_backward_pass(inputs, labels, n, model, optimizer, q, f, x_ind, inducing_points=True, backward=True, f_marginal=True, n_samples=10):
    
    b = inputs.shape[0] # mini batch size

    # Build MV Normal Distribution
    q.build_normal()

    # GP inputs 
    x = inputs[0,:2,:].permute(1,0).float()

    # Compute KL
    if inducing_points:
        f_post = f | (f(x_ind), q.sample())
        kl = q.kl(f(x_ind))
    else:
        f_x = f(x)
        kl = q.kl(f_x)
    
    # Repeat input tensor K (n_samples) times
    inputs = inputs[:,2:,:].unsqueeze(-1).repeat(1,1,1,n_samples).permute(0,2,3,1)
    labels = labels.unsqueeze(-1).repeat(1,1,n_samples)
    
    # Sample z and concatenate to inputs
    if inducing_points:
        if f_marginal:
            f_sample = Normal(f_post.mean(x),
                              Diagonal(f_post.kernel.elwise(x)[:, 0])
                             ).sample(b*n_samples)     
        else:
            f_sample = f_post(x).sample(b*n_samples)
                
        f_sample = f_sample.permute(1,0).unsqueeze(1).reshape(b,-1,x.shape[0],n_samples).permute(0,2,3,1)
        inputs = torch.cat([f_sample, inputs], dim=3)

    else:
        q_sample = q.sample(b).permute(1,0).unsqueeze(1)
        inputs = torch.cat([q_sample, inputs], dim=1)

    # Masking for missing data
    # inputs = inputs.permute(0,1,3,2)
    mask = ~torch.any(inputs.isnan(),dim=3)
    
    k = mask.sum()

    # Forward pass
    outputs = model(inputs[mask].float())
    
    # Reconstruction term  
    recon = -loss_fn(outputs, labels[mask], inputs, model, reduction='sum') # Check if this is biased

    # ELBO
    elbo = recon/(b*k) - kl/n # OR (n/(b*k)*recon - kl)/n

    # Backward pass and optimizer step
    if backward:
        (-elbo).backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return elbo, recon, kl, k