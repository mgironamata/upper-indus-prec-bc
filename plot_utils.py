import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from datetime import datetime
import geopandas

from utils import mixture_percentile

__all__ = [ 'print_summary_of_results',
            'plot_timeseries',
            'build_geodataframe']

def build_geodataframe(df, x, y):
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[x], df[y]))

def print_summary_of_results(st_test, likelihood_fn, alldays=True, drydays=True, wetdays=True, wet_threshold=0):

    for rand_test_st in st_test['Station'].unique():

        st_test_r = st_test[st_test['Station']==rand_test_st].copy()
        st_test_r.set_index('Date', inplace=True)
        
        zero_mask = st_test_r['Prec']==0
        nonzero_mask = st_test_r['Prec']>wet_threshold

        if alldays:
            print("All Days ¦ Station : %s ¦ WRF : %.1f ¦ REG : %.1f ¦ MLP : %.1f ¦ Imp(REG) : %.2f%% ¦ Imp(MLP) : %.2f%% ¦ Imp(MLP_median) : %.2f%% ¦ Imp(MLP_median_gamma) : %.2f%%"  %
                  (rand_test_st,
                   st_test_r['se_wrf'].mean(),
                   st_test_r['se_bcp'].mean(),
                   st_test_r['se_mlp'].mean(),
                   100 - st_test_r['se_bcp'].mean()/st_test_r['se_wrf'].mean()*100,
                   100 - st_test_r['se_mlp'].mean()/st_test_r['se_wrf'].mean()*100,
                   100 - st_test_r['se_mlp_median'].mean()/st_test_r['se_wrf'].mean()*100,
                   100 - st_test_r['se_mlp_median_gamma'].mean()/st_test_r['se_wrf'].mean()*100,
                  )
                 )

        if drydays:
            print("Dry Days ¦ Station : %s ¦ WRF : %.1f ¦ REG : %.1f ¦ MLP : %.1f ¦ Imp(REG) : %.2f%% ¦ Imp(MLP) : %.2f%% ¦ Imp(MLP_median) : %.2f%% ¦ Imp(MLP_median_gamma) : %.2f%%"  %
                  (rand_test_st,
                   st_test_r['se_wrf'][zero_mask].mean(),
                   st_test_r['se_bcp'][zero_mask].mean(),
                   st_test_r['se_mlp'][zero_mask].mean(),
                   100 - st_test_r['se_bcp'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,
                   100 - st_test_r['se_mlp'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,
                   100 - st_test_r['se_mlp_median'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,   
                   100 - st_test_r['se_mlp_median_gamma'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,  
        
                  )
                 )
        
        if wetdays:
            print("Wet Days ¦ Station : %s ¦ WRF : %.1f ¦ REG : %.1f ¦ MLP : %.1f ¦ Imp(REG) : %.2f%% ¦ Imp(MLP) : %.2f%% ¦ Imp(MLP_median) : %.2f%% ¦ Imp(MLP_median_gamma) : %.2f%%"  %
                  (rand_test_st,
                   st_test_r['se_wrf'][nonzero_mask].mean(),
                   st_test_r['se_bcp'][nonzero_mask].mean(),
                   st_test_r['se_mlp'][nonzero_mask].mean(),
                   100 - st_test_r['se_bcp'][nonzero_mask].mean()/st_test_r['se_wrf'][nonzero_mask].mean()*100,
                   100 - st_test_r['se_mlp'][nonzero_mask].mean()/st_test_r['se_wrf'][nonzero_mask].mean()*100,
                   100 - st_test_r['se_mlp_median'][nonzero_mask].mean()/st_test_r['se_wrf'][nonzero_mask].mean()*100,  
                   100 - st_test_r['se_mlp_median_gamma'][zero_mask].mean()/st_test_r['se_wrf'][zero_mask].mean()*100,           
                  )
                 )
            
        if wetdays or drydays:
            print("-------------------------------------------------------------------------------")

def plot_timeseries(st_test_r, likelihood_fn, xmin, xmax, figsize=(30,6), p=0.05):
    
    st_test_r.set_index(['Date'], inplace=True)
    st_test_r = st_test_r.loc[xmin:xmax].copy()

    date = st_test_r.index.values

    rand_test_st = st_test_r['Station'].unique()[0]

    st_test_r['low_ci'] = st_test_r.apply(mixture_percentile, axis=1, args=(p, likelihood_fn))
    st_test_r['high_ci'] = st_test_r.apply(mixture_percentile, axis=1, args=(1-p, likelihood_fn))

    msize = 5
    lwidth = 0.5
    
    plt.figure(figsize=figsize)

    st_test_r['mean'].plot(
                marker='x',markersize=msize,linewidth=lwidth+1, label='MLP (mean)', color='blue')

    st_test_r['median'].plot(
                marker='x',markersize=msize,linewidth=lwidth+1, label='MLP (median)', color='black')

    st_test_r['median_gamma'].plot(
                marker='x',markersize=msize,linewidth=lwidth+1, linestyle='--', label='MLP (median gamma)', color='black')

    plt.fill_between(x=date,
                    y1=st_test_r['low_ci'], 
                    y2=st_test_r['high_ci'],
                    alpha=0.1)

    st_test_r['model_precipitation'].plot(
                marker='x',markersize=msize,linewidth=lwidth, label='WRF', color='red')

    st_test_r['wrf_bcp'].plot(
                marker='x',markersize=msize,linewidth=lwidth, label='BC WRF', color='orange')

    st_test_r['Prec'].plot(
                marker='x',markersize=msize,linewidth=lwidth+1, label='Obs', color='green')


    plt.title (f"Test station : {rand_test_st}   (MLP likelihood : {likelihood_fn})", fontsize=15)

    plt.xlim([datetime.strptime(xmin,"%Y-%m-%d"),          
              datetime.strptime(xmax,"%Y-%m-%d")])
    
    plt.legend()
    plt.show()