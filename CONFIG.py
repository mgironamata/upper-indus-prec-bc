# Imports
from collections import OrderedDict


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
    ,batch_size = [128]
    ,likelihood_fn = ['bgmm','bernoulli_loggaussian'] #, 'bernoulli_loggaussian', 'b2gmm'] #['bernoulli_loggaussian']
    ,hidden_channels = [[50]] #[[10],[30],[50],[100],[10,10],[30,30],[50,50],[100,100]]
    ,dropout_rate = [0]
    ,linear_model = [False, True] #['True','False']
    #,k = [0]
    ,k = list(range(10))
)

epochs = 5