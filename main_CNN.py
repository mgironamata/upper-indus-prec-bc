import numpy as np 
import pandas as pd 
import xarray as xr 
import os 

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.gamma import Gamma

from sklearn.preprocessing import StandardScaler
from datetime import datetime 
import time

from experiment import *
from models import *
from utils import RunningAverage

import pdb

torch.autograd.detect_anomaly = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_stations = ['Pandoh', 'Banjar', 'Bhuntar', 'Larji', 'Sainj', 'Janjehl',
                'Hamirpur', 'Nadaun', 'Sujanpur', 'Dehra', 'Kangra', 'Palampur',
                'Sadar-Mandi', 'Jogindernagar', 'Sarkaghat']

train_stations = ['Sundernagar', 'BanjarIMD', 'Bharmaur', 'Churah', 'Kalatop',
                    'Salooni', 'Rohru', 'Jubbal', 'Kothai', 'Nahan', 'Paonta Sahib',
                    'Rakuna', 'Pachhad', 'Dadahu', 'Dhaula Kuan', 'Kandaghat', 'Suni',
                    'Rampur', 'Kasol', 'Bhakra', 'Sadar-Bilarspur', 'Ghumarwin',
                    'Bhoranj', 'Karsog', 'Theog', 'Kumarsain', 'Mashobra', 'Arki',
                    'RampurIMD', 'SuniIMD', 'Berthin', 'Bharatgarh', 'Daslehra',
                    'Ganguwal', 'Ghanauli', 'Kotata', 'Lohard', 'Naina Devi', 'Nangal',
                    'Olinda', 'Swaghat', 'Kalpa', 'Kaza']

wrf_path = '..\..\..\Google Drive (mg963)/PhD/data/wrf/precipitation.nc'
stations_path = 'C:\Google Drive (mg963)\PhD\data\pickle\df_stations_all_nonzero_extended_filtered.pkl'
stations_val_path = 'C:\Google Drive (mg963)\PhD\data\pickle\df_stations_val_all_nonzero_extended.pkl'
srtm_path = '..\..\..\Google Drive (mg963)/PhD/gis/srtm_dem.tif'

train_stations = pd.read_pickle(stations_path)['Station'].unique()
val_stations = pd.read_pickle(stations_val_path)['Station'].unique()

class Dataset_SimpleCNN(torch.utils.data.Dataset):

    def __init__(self,wrf_path,stations_path,stations_list):

        self.ds = xr.open_dataset(wrf_path)
        self.wrf_mean = np.mean(self.ds.model_precipitation)
        self.wrf_std = np.std(self.ds.model_precipitation)
        self.ds['model_precipitation'] = (self.ds.model_precipitation - self.wrf_mean) / self.wrf_std

        self.srtm = xr.open_rasterio(srtm_path).interp(y=self.ds['projection_y_coordinate'],x=self.ds['projection_x_coordinate'])
        self.srtm_mean = np.mean(self.srtm)
        self.srtm_std = np.std(self.srtm)
        self.srtm = (self.srtm - self.srtm_mean) / self.srtm_std

        self.stations = pd.read_pickle(stations_path)
        self.val_stations = pd.read_pickle(stations)
        self.stations = self.stations[self.stations['Station'].isin(stations_list)]
        self.stations[['X_std','Y_std','Z_std','P_std']] = self.stations[['X','Y','Z','wrf_prcp']]
        self.stations[['X_std','Y_std','Z_std','doy_sin','doy_cos','P_std']] = StandardScaler().fit_transform(self.stations[['X','Y','Z','doy_sin','doy_cos','P_std']])
        
    def __len__(self):
        return len(self.stations)

    def __getitem__(self, idx):

        point = self.stations.iloc[idx]
        time = datetime.strftime(point['Date'], format='%d/%m/%Y')

        grid_point = self.ds.sel(time=time, 
                            projection_x_coordinate = point.X, 
                            projection_y_coordinate = point.Y,
                            method='nearest'
                            )

        grid_point_time = grid_point.time.values
        grid_point_X = grid_point.projection_x_coordinate.values
        grid_point_Y = grid_point.projection_y_coordinate.values

        index_time = int(np.where(self.ds.time.values == grid_point_time)[0])
        index_X = int(np.where(self.ds.projection_x_coordinate.values == grid_point_X)[0])
        index_Y = int(np.where(self.ds.projection_y_coordinate.values == grid_point_Y)[0])

        dh = 32
        dt = 10

        p_grid = self.ds.isel(
                            time = slice(index_time - int(dt/2), index_time + int(dt/2)),
                            projection_x_coordinate = slice(index_X - int(dh/2), index_X + int(dh/2)), 
                            projection_y_coordinate = slice(index_Y - int(dh/2), index_Y + int(dh/2))
                            )
        
        z_grid = self.srtm.isel(
                            projection_x_coordinate = slice(index_X - int(dh/2), index_X + int(dh/2)), 
                            projection_y_coordinate = slice(index_Y - int(dh/2), index_Y + int(dh/2))
                            )
        
        channels = torch.cat(
                                (
                                    torch.tensor(p_grid.model_precipitation.values,dtype=torch.float32),
                                    torch.tensor(z_grid.values,dtype=torch.float32)
                                ),  
                                dim=0
                            )

        features = point[[#'X_std','Y_std','Z_std',
                         'doy_sin','doy_cos',
                         #'wrf_prcp'
                        ]].to_numpy(dtype=np.float32)
        
        labels = point['Prec']

        return (features, channels, labels)

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


    obs = obs.flatten()
    logp = torch.zeros(obs.shape).to(device)
    
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

def train(batch_size=16, num_workers=24, prints_per_epoch=5, num_epochs=10, in_channels=11, in_features=2, out_features=3):
    """
    Train model and display training and validations loss.

    Parameters
    ----------
        batch_size : int
            Batch size.
        
        num_workers : int
            Number of CPU threads loading data in parallel during training and validation.

    Returns [TO DO]

    """

    experiment_name = 'GeoStatCNN'
    wd = WorkingDirectory(generate_root(experiment_name))

    train_losses = []
    val_losses = []

    train_dataset = Dataset_SimpleCNN(wrf_path, stations_path, train_stations)
    val_dataset = Dataset_SimpleCNN(wrf_path, stations_val_path, val_stations)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #network = SimpleCNN().to(device)
    network = GeoStatCNN(in_channels=in_channels, in_features=in_features, out_features=out_features).to(device)
    print("Number of parameters is: ", sum(p.numel() for p in network.parameters() if p.requires_grad))
    
    optimizer = torch.optim.Adam(network.parameters(), lr=.0001)

    start_time = time.time()
        
    for epoch in range(num_epochs):

        train_loss = RunningAverage()
        
        print_frequency = int(train_dataloader.__len__()/prints_per_epoch)
        print_index = 0
        print_time = time.time()

        for index, (point, grid, label) in enumerate(train_dataloader):
            
            grid = grid.to(device)
            point = point.to(device)
            label = label.to(device)

            with torch.enable_grad():

                optimizer.zero_grad()
                label = label.to(dtype=torch.float32)

                output = network(grid, point)#.flatten()

                loss = -bgmm_logpdf(label, pi=output[:,0], alpha=output[:,1], beta=output[:,2], reduction='mean')
                #loss = F.mse_loss(output,label)

                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                train_loss.update(loss)

                if index % print_frequency == 0:
                    
                    print('Epoch: ', epoch,
                            '\t', print_index,"/",prints_per_epoch, 
                            #'\t', index, 
                            #'\t Batch loss: ', float(loss) , 
                            '\t Average loss:', "{:.2f}".format(float(train_loss.avg)),
                            '\t Elapsed time: ', "{:.2f}".format(time.time() - print_time)
                            )
                    print_time = time.time() 
                    print_index += 1 

        print_time = time.time() 
        val_loss = RunningAverage()    
        for index, (point, grid, label) in enumerate(val_dataloader):

            grid = grid.to(device)
            point = point.to(device)
            label = label.to(device)
            
            with torch.no_grad():    

                label = label.to(dtype=torch.float32)
                
                output = network(grid, point)#.flatten()

                loss = -bgmm_logpdf(label, pi=output[:,0], alpha=output[:,1], beta=output[:,2], reduction='mean')
                #loss = F.mse_loss(output,label)
                val_loss.update(loss)
    
        val_losses.append(float(val_loss.avg))
        
          
        save_as_best = True if val_loss == min(val_losses) else False
        save_checkpoint(wd,network.state_dict(),is_best=save_as_best)

        print('Epoch: ', epoch, 
              '\t Val loss:', "{:.2f}".format(float(val_loss.avg)),
              '\t Elapsed time: ', "{:.2f}".format(time.time() - print_time)
               )

        PATH = os.path.join(wd.root,'e_%s_loss_%.3f.pth.tar' % (epoch, float(val_loss.avg)))
        torch.save(network.state_dict(), PATH)
            
if __name__ == '__main__':
    train(batch_size=16, num_workers=24, prints_per_epoch=5, num_epochs=10, 
          in_channels=11, in_features=2, out_features=3)
