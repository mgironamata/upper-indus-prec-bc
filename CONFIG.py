# Imports
from collections import OrderedDict
from dataclasses import dataclass
from typing import List
import torch

# @dataclass
# class Parameters:
#     start : str
#     end : str
#     TRAIN_PATH : str
#     predictors : List[str]
#     predictand : List[str]
#     params : OrderedDict
#     epochs : int
#     seasons : List[str]
#     n_samples: int

# Device to perform computations on.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Start and end date
start="1998-01-01"
end="2015-12-31"

# Data paths
TRAIN_PATH = '../../data/norris/langtang/observations_with_WRF_norris.pkl'

# Features 
predictors = [ 
                'doy_sin',
                'doy_cos',
                'Z',
                'X',
                'Y',
                #'aspect',
                #'slope',
                'year',
                'CWV_norris', 
                'RH2_norris', 'RH500_norris', 
                'T2_norris', 'T2max_norris', 'T2min_norris', 'Td2_norris', 
                'precip_norris', 'rain_norris', 
                'u10_norris', 'u500_norris', 'v10_norris', 'v500_norris',
              ]

predictand = ['Prec']

# Multirun parameters
params = OrderedDict(
    lr = [0.005]
    ,batch_size = [32]
    ,likelihood_fn = ['bgmm','bernoulli_loggaussian'] # 'bernoulli_loggaussian', 'b2gmm']
    ,hidden_channels = [[50]] #[[10],[30],[50],[100],[10,10],[30,30],[50,50],[100,100]]
    ,dropout_rate = [0]
    ,linear_model = [False, True]
    ,k = list(range(10))
)

epochs = 20

# Seasons
seasons = ['JFM', 'AM', 'JJAS','OND']

# Number of samples
n_samples = 10
