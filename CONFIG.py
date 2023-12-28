# Imports
from collections import OrderedDict

# Device to perform computations on.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Start and end date
start="1970-01-01"
end="2015-12-31"

# Data paths\

EXPERIMENT = 2 # 1, 2, 3 or 4

REGION = 'ICIMOD'

# Sort by elevation
if EXPERIMENT == 4:
  SORT_BY_ELEVATION = True
else:
  SORT_BY_ELEVATION = False

# Split by region or station
if EXPERIMENT == 3:
  split_by = 'region'
else:
  split_by = 'station'

# Use combined data or not
if (EXPERIMENT == 2) or (EXPERIMENT == 3):
  REGION = 'COMBINED'

# Data paths
DATA_PATHS = {'WAPDA' : '../../data/norris/enriched_obs/enriched_wapda_obs_norris_ready.pkl',
              'ICIMOD' : '../../data/norris/enriched_obs/enriched_langtang_obs_norris_ready.pkl',
              'SUSHIWAT' : '../../data/norris/enriched_obs/enriched_sushiwat_obs_norris_ready.pkl',
              'COMBINED' : '../../data/norris/enriched_obs/enriched_combined_obs_norris_ready.pkl'}

# Run name
# RUN_NAME = f'{REGION}_ALLMODELS_19_NOV_2023_EXP_2'
RUN_NAME = f'{REGION}_RECURRENT_2_DEC_2023_EXP_{EXPERIMENT}'

ADD_PREVIOUS_DAY = False

# Input data path
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

# Multirun parameters
params = OrderedDict(
    lr = [0.001]
    ,batch_size = [32]
    ,likelihood_fn = [
                      # 'gaussian',
                      # 'gamma',
                      # 'lognormal',
                      # 'gumbel',
                      # 'halfnormal',
                      # ' bgmm',
                      #'b2gmm',
                      'bernoulli_lognormal',
                      # 'bernoulli_gaussian',
                      # 'bernoulli_gumbel',
                      # 'bernoulli_halfnormal',
                      ]
    ,dropout_rate = [0]
    ,random_noise = [0]
    ,k = list(range(K_FOLD)) 
    ,model_arch = [
                  #  ('VGLM',[]),
                  #  ('MLP',[10]),
                  #  ('SimpleRNN',[10]),
                   ('MLP',[50,50]),
                  #  ('MLP',[100,100,100,100]),
                   ('GRU',[50,50]),
                    ('LSTM',[50,50]),
                   ('SimpleRNN',[50,50])
                  ]
)

epochs = 5

# Seasons
seasons = ['JFM', 'AM', 'JJAS','OND']

# Number of samples
n_samples = 10