import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from datetime import datetime
import geopandas
import scipy.stats as stats 
import pdb

from utils import mixture_percentile

__all__ = [ 'print_summary_of_results',
            'plot_timeseries',
            'build_geodataframe',
            'plot_parameter_histograms',
            'plot_sample_distribution',
            'plot_squared_errors',
            'season_apply']

def build_geodataframe(df, x, y):
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[x], df[y]))

def print_summary_of_results(st_test, likelihood_fn, alldays=True, drydays=True, wetdays=True, wet_threshold=0):

    for rand_test_st in st_test['Station'].unique():

        st_test_r = st_test[st_test['Station']==rand_test_st].copy()
        st_test_r.set_index('Date', inplace=True)
        
        zero_mask = st_test_r['Prec']==0
        nonzero_mask = st_test_r['Prec']>wet_threshold

        if alldays:
            print("All Days ¦ Station : %s ¦ WRF : %.1f ¦ REG : %.1f ¦ MLP : %.1f ¦ Imp(REG) : %.2f%% ¦ Imp(MLP) : %.2f%% ¦ Imp(MLP_median) : %.2f%% ¦ Imp(MLP_median_gamma) : %.2f%% ¦ Imp(MLP_sample) : %.2f%%"   %
                  (rand_test_st,
                   st_test_r['se_wrf'].mean(),
                   st_test_r['se_bcp'].mean(),
                   st_test_r['se_mlp'].mean(),
                   100 - st_test_r['se_bcp'].mean()/st_test_r['se_wrf'].mean()*100,
                   100 - st_test_r['se_mlp'].mean()/st_test_r['se_wrf'].mean()*100,
                   100 - st_test_r['se_mlp_median'].mean()/st_test_r['se_wrf'].mean()*100,
                   100 - st_test_r['se_mlp_median_gamma'].mean()/st_test_r['se_wrf'].mean()*100,
                   100 - st_test_r['se_mlp_sample'].mean()/st_test_r['se_wrf'].mean()*100,
                  )
                 )

        if drydays:
            print("Dry Days ¦ Station : %s ¦ WRF : %.1f ¦ REG : %.1f ¦ MLP : %.1f ¦ Imp(REG) : %.2f%% ¦ Imp(MLP) : %.2f%% ¦ Imp(MLP_median) : %.2f%% ¦ Imp(MLP_median_gamma) : %.2f%% ¦ Imp(MLP_sample) : %.2f%%"   %
                  (rand_test_st,
                   st_test_r['se_wrf'][zero_mask].mean(),
                   st_test_r['se_bcp'][zero_mask].mean(),
                   st_test_r['se_mlp'][zero_mask].mean(),
                   100 - st_test_r['se_bcp'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,
                   100 - st_test_r['se_mlp'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,
                   100 - st_test_r['se_mlp_median'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,   
                   100 - st_test_r['se_mlp_median_gamma'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,  
                   100 - st_test_r['se_mlp_sample'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,  
                  )
                 )
        
        if wetdays:
            print("Wet Days ¦ Station : %s ¦ WRF : %.1f ¦ REG : %.1f ¦ MLP : %.1f ¦ Imp(REG) : %.2f%% ¦ Imp(MLP) : %.2f%% ¦ Imp(MLP_median) : %.2f%% ¦ Imp(MLP_median_gamma) : %.2f%% ¦ Imp(MLP_sample) : %.2f%%"   %
                  (rand_test_st,
                   st_test_r['se_wrf'][nonzero_mask].mean(),
                   st_test_r['se_bcp'][nonzero_mask].mean(),
                   st_test_r['se_mlp'][nonzero_mask].mean(),
                   100 - st_test_r['se_bcp'][nonzero_mask].mean()/st_test_r['se_wrf'][nonzero_mask].mean()*100,
                   100 - st_test_r['se_mlp'][nonzero_mask].mean()/st_test_r['se_wrf'][nonzero_mask].mean()*100,
                   100 - st_test_r['se_mlp_median'][nonzero_mask].mean()/st_test_r['se_wrf'][nonzero_mask].mean()*100,  
                   100 - st_test_r['se_mlp_median_gamma'][nonzero_mask].mean()/st_test_r['se_wrf'][nonzero_mask].mean()*100,    
                   100 - st_test_r['se_mlp_sample'][nonzero_mask].mean()/st_test_r['se_wrf'][nonzero_mask].mean()*100,       
                  )
                 )
            
        if wetdays or drydays:
            print("-------------------------------------------------------------------------------")

def plot_timeseries(st_test_r, likelihood_fn, xmin, xmax, p=0.05, 
                    show_obs=True, show_raw_model=True, show_bc_baseline=True,
                    show_mean=True, show_median=True, show_median_gamma=True, show_confidence=True, show_sample=True,
                    complete_title=True, show_legend=True, ax=None):
    
    df = st_test_r.set_index(['Date'])
    df = df.loc[xmin:xmax].copy()

    date = df.index.values

    # rand_test_st = df['Station'].unique()[0]

    # st_test_r['low_ci'] = st_test_r.apply(mixture_percentile, axis=1, args=(p, likelihood_fn))
    # st_test_r['high_ci'] = st_test_r.apply(mixture_percentile, axis=1, args=(1-p, likelihood_fn))

    marker = 'o'
    msize = 3
    lwidth = 0.5

    if show_raw_model:
        df['wrf_prcp'].plot(
                    marker=marker, markersize=msize,linewidth=lwidth, label='Raw WRF', color='red', ax=ax)

    if show_bc_baseline:
        df['wrf_bc_prcp'].plot(
                marker=marker, markersize=msize,linewidth=lwidth, label='Bannister et al. (2019)', color='orange', ax=ax)
    if show_obs:
        df['Prec'].plot(
                    marker=marker, markersize=msize,linewidth=lwidth, label='Observation', color='green', ax=ax)

    if show_mean:
        df['mean'].plot(
                    marker=marker, markersize=msize,linewidth=lwidth, label='MLP (mean)', color='blue', ax=ax)
    if show_median:
        df['median'].plot(
                    marker=marker, markersize=msize,linewidth=lwidth, label='MLP (median)', color='black', ax=ax)
    if show_median_gamma:
        df['median_gamma'].plot(
                    marker=marker, markersize=msize,linewidth=lwidth, linestyle='--', label='MLP (median gamma)', color='black', ax=ax)
    
    if show_sample:
        df['sample'].plot(
                    marker=marker, markersize=msize,linewidth=lwidth, label='MLP (sample)', color='blue', ax=ax)

    if show_confidence:
        ax.fill_between(x=date,
                        y1=df['low_ci'].astype('float64'), 
                        y2=df['high_ci'].astype('float64'),
                        alpha=0.4)



    # if complete_title:
    #     plt.title(f"Test station : {rand_test_st}   (MLP likelihood : {likelihood_fn})", fontsize=15)
    # else:
    #     plt.title(f"Test station : {rand_test_st}")
    
    ax.set_xlim([datetime.strptime(xmin,"%Y-%m-%d"),          
              datetime.strptime(xmax,"%Y-%m-%d")])

    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_xlabel('')
    
    ax.grid(False)
    
    if show_legend:
        ax.legend()


def plot_parameter_histograms(model, outputs):

    n = outputs.shape[1]
    
    if model.likelihood=='gamma':
        variable = ['alpha', 'beta']

    elif model.likelihood=='gaussian':
        variable = ['mu','sigma']
        
    elif model.likelihood=='ggmm': 
        variable = ['alpha1', 'alpha2', 'beta1', 'beta2', 'q']
        
    elif model.likelihood=='bgmm':
        variable = ['pi', 'alpha', 'beta']
        
    elif model.likelihood=='b2gmm': 
        variable = ['pi', 'alpha1', 'alpha2', 'beta1', 'beta2', 'q']
        
    elif model.likelihood=='b2sgmm': 
        variable = ['pi', 'alpha1', 'alpha2', 'beta1', 'beta2', 'q', 't']

    elif model.likelihood=='btgmm':
        variable = ['pi', 'alpha', 'beta', 't']
        
    fig, ax = plt.subplots(1,n, figsize=(6*n, 5), sharex=False, sharey=False)
        
    for i in range(n):

        if (i==0) and (model.likelihood=='bgmm'):
            ax[i].hist(1-outputs[:,i].numpy(), bins=50, density=False)
        else:
            ax[i].hist(outputs[:,i].numpy(), bins=50, density=False)
        
        ax[i].title.set_text(f'{variable[i].capitalize()}')
        # ax[i].title.set_text('%s (mean: %.2f, median: %.2f)' % (variable[i].capitalize(), 
        #                                                     outputs[:,i].mean(), 
        #                                                     outputs[:,i].median()))
        ax[i].set_yticks([])

def plot_sample_distribution(model, outputs, test_dataset, force_non_zero=True):

    if force_non_zero:
        obs=0
        while obs==0:
            r = np.random.randint(len(outputs))
            obs = test_dataset.tensors[1][r]
    else:
        r = np.random.randint(len(outputs))
        obs = test_dataset.tensors[r]
        
    x = np.linspace (0, obs*2, 100) 

    fig = plt.figure(figsize=(8,5))

    plt.plot(obs,0,'x', label='observation')

    if model.likelihood=='gamma':
        alpha = outputs[r,0]
        beta = outputs[r,1]

    if model.likelihood=='bgmm':
        pi = outputs[r,0]
        alpha = outputs[r,1]
        beta = outputs[r,2]

    if model.likelihood=='ggmm':
        alpha1 = outputs[r,0]
        alpha2 = outputs[r,1]
        beta1 = outputs[r,2]
        beta2 = outputs[r,3]
        q = outputs[r,4]
        
    if model.likelihood=='b2gmm':
        pi = outputs[r,0]
        alpha1 = outputs[r,1]
        alpha2 = outputs[r,2]
        beta1 = outputs[r,3]
        beta2 = outputs[r,4]
        q = outputs[r,5]

    if model.likelihood=='b2sgmm':
        pi = outputs[r,0]
        alpha1 = outputs[r,1]
        alpha2 = outputs[r,2]
        beta1 = outputs[r,3]
        beta2 = outputs[r,4]
        q = outputs[r,5]
        t = outputs[r,6]

    if model.likelihood=='gamma':
        y1 = stats.gamma.pdf(x, a=alpha, scale=1/beta)
        plt.plot(x, y1, label='Gamma')
        
    if model.likelihood=='bgmm':
        plt.plot(0, pi, 'o', label='pi')
        y1 = stats.gamma.cdf(x, a=alpha, scale=1/beta)
        y1_med = stats.gamma.median(a=alpha, scale=1/beta)
        plt.plot(x, y1*(1-pi).numpy()+pi.numpy(), label='Gamma')
        plt.plot(y1_med, 0, 'o')

    if model.likelihood=='ggmm':
        y1 = stats.gamma.pdf(x, a=alpha1, scale=1/beta1)
        plt.plot(x, y1*q.numpy(), label='Gamma1')
        y2 = stats.gamma.pdf(x, a=alpha2, scale=1/beta2)
        plt.plot(x, y2*(1-q).numpy(), label='Gamma2')

    if model.likelihood=='b2gmm':
        plt.plot(0, pi, 'o', label='pi')
        y1 = stats.gamma.pdf(x, a=alpha1, scale=1/beta1)
        plt.plot(x, y1*(1-pi).numpy()*q.numpy(), label='Gamma1')
        y2 = stats.gamma.pdf(x, a=alpha2, scale=1/beta2)
        plt.plot(x, y2*(1-pi).numpy()*(1-q).numpy(), label='Gamma2')

    if model.likelihood=='b2sgmm':
        plt.plot(0, pi, 'o', label='pi')
        plt.plot(t,0,'x', label='t')

    #TODO: ADD BTGMM LIKELIHOOD
        
    plt.legend()
    plt.title(model.likelihood)
    plt.ylim(-0.1,1)
    print(outputs[r])
    plt.show()

def plot_squared_errors(st_test):
    
    fig = plt.figure(figsize=(10,5))

    columns = ['se_mlp','se_wrf','se_bcp','se_mlp_median']

    for idx, col in enumerate(columns):
        x = st_test['Prec']
        y = st_test[col]
        plt.scatter(x,y,label=col,s=3)
        
    plt.ylabel('Squared error')
    plt.ylabel('Precipitation (mm)')
    plt.legend()
    plt.show()

def season_apply(df):
    # Premonsoon: Apr-May
    if (df['month'] >= 4) and (df['month'] <= 5):
        return 'Premonsoon (AM)'
    # Monsoon: June-Jul-Aug-Sept
    elif (df['month'] >= 6) and (df['month'] <= 9):
        return 'Monsoon (JJAS)'
    # Postmonsoon: Oct-Nov-Dec
    elif (df['month'] >= 10) and (df['month'] <= 12):
        return 'Postmonsoon (OND)'
    # Winter: Jan-Feb-Mar
    elif (df['month'] >= 1) or (df['month'] <= 3):
        return 'Winter (JFM)'