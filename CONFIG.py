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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Start and end date
start="1970-01-01"
end="2015-12-31"

# Data paths

DATA_PATHS = {'WAPDA' : '../../data/norris/enriched_obs/enriched_wapda_obs_norris_ready.pkl',
              'ICIMOD' : '../../data/norris/enriched_obs/enriched_langtang_obs_norris_ready.pkl',
              'SUSHIWAT' : '../../data/norris/enriched_obs/enriched_sushiwat_obs_norris_ready.pkl',
              'COMBINED' : '../../data/norris/enriched_obs/enriched_combined_obs_norris_ready.pkl'}

REGION = 'SUSHIWAT'

RUN_NAME = f'{REGION}_BGMM_21_AUG_2023_EXP_4'

TRAIN_PATH = DATA_PATHS[REGION]

K_FOLD = 10

# Features 
predictors = [ 
                'doy_sin',
                'doy_cos',
                'Z',
                'X',
                'Y',
                # 'aspect',
                # 'slope',
                'year',
                'CWV_norris', 
                'RH2_norris', 'RH500_norris', 
                'T2_norris', 'T2max_norris', 'T2min_norris', 'Td2_norris', 
                'precip_norris', 'rain_norris', 
                'u500_norris', 'v500_norris',
                'u10_norris', 'v10_norris', 
                'cape_norris', 
                'u250_norris','v250_norris', 
                'w250_norris', 'w500_norris', 
                'hgt_norris', 'lu_index_norris', 
                # 'xland_norris'
              ]

predictand = ['Prec']

sort_by_quantile = False

split_by = 'station' # by "region" or "station"

# Multirun parameters
params = OrderedDict(
    lr = [0.001]
    ,batch_size = [32]
    ,likelihood_fn = ['bgmm'] #,'bernoulli_loggaussian', 'b2gmm']
    ,dropout_rate = [0]
    ,random_noise = [0]
    ,k = list(range(K_FOLD)) 
    ,model_arch = [
                   ('VGLM',[]),
                   ('MLP',[10]),
                   # ('SimpleRNN',[10]),
                   ('MLP',[50,50]),
                   # ('SimpleRNN',[50,50])
                  ]
)

epochs = 10

# Seasons
seasons = ['JFM', 'AM', 'JJAS','OND']

# Number of samples
n_samples = 10