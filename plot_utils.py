import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from datetime import datetime
import geopandas
import scipy.stats as stats 
import seaborn as sns
import time
from preprocessing_utils import clip_time_period
from IPython.display import clear_output
import pdb
import torch

import rasterio
from rasterio.plot import show

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

from utils import mixture_percentile, gmm_fn

__all__ = [ 'print_summary_of_results',
            'plot_timeseries',
            'build_geodataframe',
            'plot_parameter_histograms',
            'plot_sample_distribution',
            'plot_squared_errors',
            'plot_bgmm_params_annual_timeseries',
            'plot_loglik_epochs_multirun',
            'plot_sample_from_bernoulli_gamma_mixture_model',
            'plot_acf_for_random_station',
            'plot_sample_from_2gamma_mixure_model',
            'plot_seasonal_timeseries_for_station_year',
            'plot_map_stations_cv(st_test',
            'table_of_predictions',
            'print_ks_scores',
            'plot_seasonal_boxplot_per_station'
            ]

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

def plot_bgmm_params_annual_timeseries(st_test, station='Pandoh', start_time='2003-01-01', end_time='2003-12-31'):

    """ Creates a figure with 3 subplots plotting annual evolution of the BGMM parameters"""

    df = st_test[st_test['Station']==station]
    df = clip_time_period(df, start_time, end_time)

    fig,[ax1,ax2] = plt.subplots(2,1,figsize=(15,10))
    ax1.plot(df['pi'],label='pi')
    ax1.plot(df['alpha'],label='alpha')
    ax1.plot(df['beta'],label='beta')
    ax1.legend()
    ax2.plot(df['wrf_prcp'],label='wrf')
    ax2.plot(df['Prec'],label='obs')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def plot_loglik_epochs_multirun(df):

    sns.set_theme(context='paper',style='white',font_scale=1.4)
    plt.figure(figsize=(10,5))

    linecolors = {"0" : "tab:blue",
                  "1" : "tab:orange",
                  "2" : "tab:green",
                  "3" : "tab:red",
                  "4" : "tab:purple",
                  "5" : "tab:brown",
                  "6" : "tab:pink",
                  "7" : "tab:grey",
                  "8" : "tab:olive",
                  "9" : "tab:cyan"}

    linestyles = {"bgmm" : "solid",
                 "b2gmm" : "dashed",
                 "gamma" : "solid"}

    markerstyles = {"0" : "o",
                    "0.2" : "x",
                    "0.3" : "^",
                    "0.4" : ">"}

    style_dict = {"linestyles" : linestyles,
                 "linecolors" : linecolors}

    for a in df.likelihood_fn.unique():
        for b in df.k.unique():
            for d in df.dropout_rate.unique(): 
                c = df[(df.likelihood_fn == a) & (df.k == b) & (df.dropout_rate == d)].copy()
                plt.plot(c.epoch, 
                         c.valid_loss, 
                         label=f'{a}_k{b}_d{d}', 
                         linestyle=linestyles[a],
                         color = linecolors[str(b)],
                         marker = markerstyles[str(d)],
                         linewidth = 1,
                         markersize = 5
                        )

    plt.legend(loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('-loglik')
    plt.tight_layout
    plt.show()

def plot_sample_from_bernoulli_gamma_mixture_model(inputs, outputs, tensor_y):

    interval=5

    flag = 0
    if flag:

        while True:

            x = np.linspace (0, 100, 200) 
            r = np.random.randint(len(inputs))
            print(r)

            pi = outputs[r,0]
            a = outputs[r,1]
            rate = outputs[r,2]

            y1 = stats.gamma.cdf(x, a=a, scale=1/rate) #a is alpha, loc is beta???

            mu = stats.gamma.mean(a=a, scale=1/rate)
            median = stats.gamma.median(a=a, scale=1/rate)
            mixture_median = stats.gamma.ppf((0.5 - pi) / (1 - pi), a=a, scale=1/rate) if pi < 0.5 else 0
            mixture_mean = (1-pi) * mu

            print("mean: %.2f" % mu)
            print("median: %.2f" % median)
            print("mixture median: %.2f" % mixture_median)
            print("mixture mean: %.2f" % mixture_mean)

            modelled = inputs[r,0]*x_std[0] + x_mean[0]
            observed = tensor_y[r,0]

            if modelled > 0:

                #mu = a/rate if pi<0.5 else 0

                plt.plot(0,pi,"oy", label="pi = %.2f" % pi)

                plt.plot(mu, 0, "or", mfc="white", label="mu = %.2f" % mu)
                plt.plot(median, 0, "xr", label="median = %.2f" % median)  
                plt.plot(mixture_median, 0, "ok", mfc="white", label="mixture median = %.2f" % mixture_median)  
                plt.plot(mixture_mean, 0, "^k", label="mixture median = %.2f" % mixture_mean)  

                plt.plot(modelled, 0, "x", label="WRF = %.2f" % modelled)

                plt.plot(observed, 0, "xg", label="observed = %.2f" % observed)
                plt.plot(x, pi + (1-pi)*y1, "y-", label=(r'$\alpha=%.2f, \beta=%.2f$') % (a, rate))

                plt.ylim([-0.05,1])
                plt.xlim([-0.2, max([1,modelled,observed,mu])*1.20])
                plt.legend(loc="upper right")
                plt.show()

                time.sleep(interval)

            else:
                pass

            clear_output(wait=True)

def plot_acf_for_random_station(st_test, seasons):
    acf_dict = {}

    stations = st_test['Station'].unique()
    random_station = np.random.randint(len(stations))
    st_test_station = st_test[st_test['Station']==stations[random_station]]
    print(st_test_station['Station'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(20,10))

    for index, ax in enumerate(axes.flatten()):

        for col in list(np.array(columns)[[0,2,3]]):
            
            x = st_test_station[st_test_station['season']==seasons[index]][col]
            
            if seasons[index] not in acf_dict:
                acf_dict[seasons[index]] = {}
                
            y = acf_dict[seasons[index]][col] = sm.tsa.pacf(x,nlags=30)
            
            ax.plot(y[1:], 'o--', label="col")
            
    #         plot_pacf(x,
    #                 ax=ax,
    #                 use_vlines=False,
    #                 alpha=1,
    #                 zero=False,
    #                 title=seasons[index])
            
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Autocorrelation factor")
        ax.set_title(seasons[index])

        if index==0:
            pass
            ax.legend(["Observations","Bias corrected WRF","MLP"])
            #ax.legend(columns[0:3])

    #plt.suptitle(f"Autocorrelation factors by season (test station: {st_test_station['Station'].unique()}) ")

    plt.tight_layout()
    plt.show()

def plot_sample_from_2gamma_mixure_model(outputs, test_tensor_y, k=1, bins=1000, hist_max=50):
    
    r = np.random.randint(outputs.shape[0], size=k)
    print(r)
    
    dist = gmm_fn(pi=outputs[r,0],
                alpha1=outputs[r,1],
                alpha2=outputs[r,2],
                beta1=outputs[r,3],
                beta2=outputs[r,4],
                q=outputs[r,5],
                )

    plt.figure(figsize=(15,5))
    plt.plot(np.linspace(0, hist_max,bins),(torch.histc(dist.sample([1000000]), bins=bins, min=0, max=hist_max)))
    plt.plot(dist.mean,np.zeros(k),'or')
    plt.plot(test_tensor_y[r],np.zeros(k),'xk',ms=10)
    plt.show()

def plot_seasonal_timeseries_for_station_year(st_test, station_name='Rampur', year=2000):

    sns.set_theme(context='paper', style='white', font_scale=1.4)

    #fig, axes = plt.subplots(4,1,figsize=(20,13))

    # year = 2000

    # test_station_names = st_test['Station'].unique()
    # random_index = np.random.randint(len(test_station_names))
    # random_station = test_station_names[random_index]

    # random_station = 'Rampur'
    # print(random_station)

    st_test_r = st_test[st_test['Station']==station_name].copy()
    st_test_r['sample'] = st_test_r['sample_0']

    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(20)

    ax1 = plt.subplot2grid(shape=(4, 122), loc=(0, 0), colspan=90)
    ax2 = plt.subplot2grid(shape=(4, 122), loc=(1, 0), colspan=61)
    ax3 = plt.subplot2grid(shape=(4, 122), loc=(2, 0), colspan=122)
    ax4 = plt.subplot2grid(shape=(4, 122), loc=(3, 0), colspan=92)

    for index, ax in enumerate([ax1,ax2,ax3,ax4]):
            
        if index == 0:
            xmin = f"{year}-01-01"
            xmax = f"{year}-04-01" 
        elif index == 1:
            xmin = f"{year}-04-01"
            xmax = f"{year}-06-01" 
        elif index == 2:
            xmin = f"{year}-06-01"
            xmax = f"{year}-10-01"
        elif index == 3:
            xmin = f"{year}-10-01"
            xmax = f"{year+1}-01-01" 
        
        show_legend = True if index == 0 else False

        plot_timeseries(st_test_r, 
                        likelihood_fn='b2gmm',
                        xmin = xmin,
                        xmax = xmax,
                        show_mean=False,
                        show_median=False, 
                        show_median_gamma=False,
                        show_confidence=True,
                        show_sample=True,
                        complete_title=False,
                        show_legend=show_legend,
                        ax=ax
                        )
        
        if show_legend:
            leg = ax.get_legend()
            new_labels = ['Raw WRF (B19)', 'Corr WRF (B19)','Observations','BG2MM$_{[100]}$']
            leg.set_title('')
            for t, l in zip(leg.texts, new_labels): t.set_text(l)

        x_pos = [0.01, 0.015, 0.0075, 0.01]
        ax.text(x_pos[index], 0.85, f'{seasons[index]}', fontweight="bold", transform=ax.transAxes)

    plt.tight_layout(w_pad=-0.5)
    # plt.savefig(f"figures/timeseries_seasons_{random_station}_{year}.png", dpi=300)
    plt.show()

def plot_map_stations_cv(st_test):

    path = '../../PhD/gis/exports/beas_watershed.shp'
    beas_watershed = geopandas.read_file(path)

    path = '../../PhD/gis/exports/sutlej_watershed.shp'
    sutlej_watershed = geopandas.read_file(path)

    out_fp_masked_lcc = r'/Users/marron31/Google Drive/PhD/srtm/mosaic_masked_lcc.tif'
    dem_masked_lcc = rasterio.open(out_fp_masked_lcc)

    gdf = build_geodataframe(st_test.groupby('Station').mean(), x='X', y='Y')

    fig, ax = plt.subplots(1, 1, figsize=(10,5), constrained_layout=False)
    gdf.plot(ax=ax, column='k_fold', legend=False, cmap='Set1', markersize=5, marker="o", linewidth=3)
    show(dem_masked_lcc, cmap='terrain', ax=ax, alpha=0.25)
    beas_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=1)
    sutlej_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=1)
    ax.set_axis_off()
    ax.set_aspect(1)
    plt.tight_layout()
    plt.savefig('figures/kfold_cv_map.png',dpi=300)
    plt.show()

def table_of_predictions(predictions, seasons):

    table = []
    headers = ['Model']

    for i, (k, v) in enumerate(predictions.items()):
        if i==0:
            row_a = ['WRF']
            row_b = ['BC WRF']
        row = [k]
        st_test = v['k_all']
        for season in seasons:
            if i==0:
                headers.append(f'{season} mean')
                headers.append(f'{season} median')
            
            ks_lista, ks_listb = [], []
            ks_list = []

            for s in st_test['Station'].unique():
                df = st_test[st_test['Station']==s].copy()
                rvs = df[df['season']==season]['Prec']
                
                cdfa = df[df['season']==season]['wrf_prcp']
                cdfb = df[df['season']==season]['wrf_bc_prcp']
                kstesta = scipy.stats.ks_2samp(rvs, cdfa, alternative='two-sided', mode='auto')
                kstestb = scipy.stats.ks_2samp(rvs, cdfb, alternative='two-sided', mode='auto')
                ks_lista.append(kstesta)
                ks_listb.append(kstestb)

                for sample in sample_cols:
                    cdf = np.array([])
                    cdf_s = np.array(df[df['season']==season][sample])
                    cdf = np.concatenate((cdf,cdf_s), axis=None)
                    
                kstest = scipy.stats.ks_2samp(rvs, cdf, alternative='two-sided', mode='auto')
                ks_list.append(kstest)
            
            if i ==0:
                row_a.append(f'{np.mean(ks_lista):.4f}')
                row_a.append(f'{np.median(ks_lista):.4f}')
                row_b.append(f'{np.mean(ks_listb):.4f}')
                row_b.append(f'{np.median(ks_listb):.4f}')
            
            row.append(f'{np.mean(ks_list):.4f}')
            row.append(f'{np.median(ks_list):.4f}')
            
        if i== 0:
            table.append(row_a)
            table.append(row_b)
        print(k)
            
        table.append(row)
        
    return table
    
def print_ks_scores(st_test, seasons, columns):

    for season in seasons:
        print(f"--- {season} ---")
        for col in columns[1:]:
            rvs = st_test[st_test['season']==season]['Prec']
            cdf = st_test[st_test['season']==season][col]
            #kstest = scipy.stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='auto')
            kstest = scipy.stats.ks_2samp(rvs, cdf, alternative='two-sided', mode='auto')
            print(f"{col} : {kstest.statistic:.4f}")

def plot_seasonal_boxplot_per_station(data1, st_test, yaxislabel, new_labels, basins, seasons, filter_by_basin_flag, y_limits=None):

    fig, axes = plt.subplots(4,len(basins),figsize=(16,10))

    for i, season in enumerate(seasons):
        
        data2 = data1[data1['season']==season].copy()
        
        for j, basin in enumerate(basins):
            
            ax = axes[i, j] if not(filter_by_basin_flag) else axes[i]

            st_basin = st_test[st_test['Basin']==basin]

            sorted_stations = st_basin.groupby(['Station']).mean().sort_values('Z').reset_index()['Station'].unique()
            sorted_elevations = [int(a) for a in st_basin.groupby(['Station']).mean().sort_values('Z').reset_index()['Z'].unique()]

            sorted_labels = [f'{b} ({a})' for a,b in zip(sorted_elevations,sorted_stations)]

            data3 = data2[data2['Station'].isin(sorted_stations)].copy()
    #         data3['value'] = abs(data3['value']).copy()

            sns.boxplot(data=data3, x='Station',y='value',hue='variable', ax=ax, order=sorted_stations, width=0.65, palette='pastel')
            
            if len(y_limits)==2: ax.set_ylim(y_limits[0], y_limits[1])
            elif len(y_limits)==4: ax.set_ylim(y_limits[i][0], y_limits[i][1])

            ax.grid()
            
            if i == 0:
                pass #ax.set_title(basin)
            if i == 3:
                ax.set_xlabel('Station (sorted by elevation in m.a.s.l.)')
                ax.set_xticklabels(sorted_labels,rotation=90, horizontalalignment='center')
            else:
                ax.set_xlabel(None)
                ax.set_xticklabels([])
                
            if j == 0:
                ax.set_ylabel(yaxislabel)
                pass #ax.set_ylabel(f'{season}')
            else:
                ax.set_yticklabels([])
                ax.set_ylabel(None)
            
            ax.text(0.01, 0.9, f'{season} in {basin}', fontweight="bold", transform=ax.transAxes, size='small')
            
            # ax.set_ylim(-400,400)
            # plt.ylabel('Correlation factor')
            # plt.xlabel('Lag (days)')
            
            if (i == j == 0):
                ax.legend(loc='best')
                leg = ax.get_legend()
                leg.set_title('')
                for t, l in zip(leg.texts, new_labels): t.set_text(l)
            else:
                ax.get_legend().remove()

    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    # plt.savefig('figures/seasonal-boxplot-smape.png',dpi=300)
    plt.show()