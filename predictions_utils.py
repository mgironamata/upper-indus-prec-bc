import pandas as pd
from metrics import CRPS_apply, CRPS
from tqdm.notebook import tqdm
tqdm.pandas(desc="my bar!")
import matplotlib.pyplot as plt 
import numpy as np
from metrics import logprob, BS, ROC, AUC
from utils import cdf_apply
import copy
from sklearn.calibration import calibration_curve

__all__ = ['print_CRPS_results',
           'bias_correction',
           'calculate_variance_per_station',
           'compute_CRPS',
           'combine_run_predictions',
           'delete_uncomplete_runs_from_predictions',
           'create_k_all_predictions',
           'custom_preprocessing_for_predictors_run',
           'plot_cv_split_vs_elevation',
           'calculate_skill_scores',
           'print_dict_structure',
           'bias_correction_by_month',
           'Predictions'
]

import pickle
import importlib.util
import os, sys

def load_config(config_path, alias=None):
    if alias is None:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        alias = config_name
    spec = importlib.util.spec_from_file_location(alias, config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    sys.modules[alias] = config_module  # Add the module to sys.modules with the alias
    return config_module

class Predictions:

    def __init__(self
                ,path='/data/hpcdata/users/marron31/_experiments'
                ,configs = []
                ,bc_methods = []
                ):

        self.path = path
        self.configs = configs
        self.bc_methods = bc_methods
        self.data = {}
        self.results = {}
        self.use_gamma_likelihood = False
        
        self.predictions = self.load_predictions()

    def load_predictions(self):

        self.uber_preds = {}

        for config in self.configs:
            
            load_config(config, alias='C')
            
            config = sys.modules['C']

            run_prefix = config.RUN_NAME

            if run_prefix not in self.uber_preds.keys():
                self.uber_preds[run_prefix] = {}

            with open(f'{self.path}/{run_prefix}/predictions.pkl', 'rb') as handle:
                predictions = pickle.load(handle)

            if len(self.configs) == 0:
                raise ValueError('No configs in the list')
            else:
                if len(self.uber_preds[run_prefix].keys()) < 1:
                    self.uber_preds[run_prefix] = predictions
                else:
                    for k,v in predictions.items():
                        self.uber_preds[run_prefix][k] = v
            
    def rearrange_uber_dict(self):
        for k,v in self.uber_preds.items():
            for kk,vv in v.items():
                # rename keys (kk) to include the run prefix
                # v[f'{k}_{kk}'] = v.pop(kk)
                self.data[f'{k}_{kk}'] = vv

    def gamma_component_only(self):
        for k,v in self.data.items():
            df = v['k_all']
            self.data[k]['k_all'] = df.loc[df['Prec']>0].reset_index()

        #  set self.use_gamma_likelihood to True
        self.use_gamma_likelihood = True
        
    # to do: implement delete incomplete runs

    def calculate_MOS_baselines(self):

        for idx, (k,v) in enumerate(self.data.items()):
    
            if idx > 0: continue
            
            df = self.data[k]['k_all']

            # Scaling factor
            if 'sf' in self.bc_methods:
                df = bias_correction(df,ignore_zeros=False, dry_day_correction=False, bc_mode='sf', output_series='sf')
            
            # Mean and variance 
            if 'mav' in self.bc_methods:
                df = bias_correction(df,ignore_zeros=False, dry_day_correction=False, bc_mode='mav', output_series='mav')
            
            # Scaling factor with dry day correction 
            if 'sfdd' in self.bc_methods:
                df = bias_correction(df,ignore_zeros=False, dry_day_correction=True, bc_mode='sf', output_series='sfdd')
            
            # Mean and variance with dry day correction  
            if 'mavdd' in self.bc_methods:
                f = bias_correction(df,ignore_zeros=False, dry_day_correction=True, bc_mode='mav', output_series='mavdd')
            
            # Quantile mapping
            if 'qm' in self.bc_methods:
                df = bias_correction(df,ignore_zeros=False, dry_day_correction=False, bc_mode='qm', output_series='qm')
            
            # Scaling factor by month
            if 'sfm' in self.bc_methods:
                df = bias_correction_by_month(df,ignore_zeros=False, dry_day_correction=False, bc_mode='sf', output_series='sfm')
            
            # Mean and variance by month
            if 'mavm' in self.bc_methods:
                df = bias_correction_by_month(df,ignore_zeros=False, dry_day_correction=False, bc_mode='mav', output_series='mavm')
            
            # Scaling factor with dry day correction by month
            if 'sfddm' in self.bc_methods:
                df = bias_correction_by_month(df,ignore_zeros=False, dry_day_correction=True, bc_mode='sf', output_series='sfddm')
            
            # Mean and variance with dry day correction by month
            if 'mavddm' in self.bc_methods:
                df = bias_correction_by_month(df,ignore_zeros=False, dry_day_correction=True, bc_mode='mav', output_series='mavddm')
            
            # Quantile mapping by month
            if 'qmm' in self.bc_methods:
                df = bias_correction_by_month(df,ignore_zeros=False, dry_day_correction=False, bc_mode='qm', output_series='qmm')
            
        #     df.loc[:,:] = calculate_variance_per_station(v['k_all'])

            self.data[k]['k_all'] = df

    def NLL(self):
        """
        Calculate the negative log likelihood for each model.
        
        """

        self.results['nll'] = {}

        for (k, v) in self.data.items():

            df = v['k_all']
            
            # skip if RNN
            if ('SimpleRNN' in k) or ('LSTM' in k) or ('GRU' in k):
                continue
            
            if self.use_gamma_likelihood:
                likelihood = 'gamma'
            else:
                likelihood = k.split(']_')[1].split('_B')[0]
            
            result_key = f"{k}"
            
            if likelihood == 'None':
                continue

            self.results['nll'][result_key] = (-logprob(df, dist=likelihood).item())

    def compute_skill_scores(self):
        self.skill_scores = calculate_skill_scores(self.results)

    def CRPS(self, k_fold = 'k_all'): 

        self.results['crps'] = {}

        for idx, (k,v) in enumerate(self.data.items()):
            print(f'Calculating CRPS for {k}')
                
            # s = time.time()
            df = v[k_fold]
            
            if self.use_gamma_likelihood:
                likelihood = 'gamma'
            else:
                likelihood = k.split(']_')[1].split('_B')[0]
            
            # if idx == 0: p['CRPS_clim'] = p.apply(CRPS_clim_apply, axis=1, args=(p['Prec'].to_numpy()))
            if likelihood == 'None':
                df['MAE_mod'] = abs(df['y'] - df['Prec'])
                self.results['crps'][k] = v[k_fold]['MAE_mod'].mean()
            else:
                df['CRPS_mod'] = df.apply(CRPS_apply, dist=likelihood, axis=1)
                self.results['crps'][k] = v[k_fold]['CRPS_mod'].mean()
            
            # p['CRPS_bc'] = p.apply(CRPS_apply, axis=1, args=('gaussian','naive_bc'))
            
            # e = time.time() - s            
            # print(f'Completed model for {k} in {e:.2f}')

            if idx == 0:
    
                df['MAE_sim'] = abs(df['precip_norris'] - df['Prec'])
                self.results['crps']['N19-MAE'] = v[k_fold]['MAE_sim'].mean()
                
                for s in self.bc_methods:
                    df[f'MAE_{s}'] = abs(df[s] - df['Prec'])         
                    self.results['crps'][f'{s}_MAE'] = v[k_fold][f'MAE_{s}'].mean()
                
        self.results['crps']['N19'] = copy.deepcopy(self.results['crps']['N19-MAE'])  # copy the N19-MAE value to N19

    def BSS(self,
            simulated = 'precip_norris',  
            observed = 'Prec', 
            threshold = 0,
            k_fold = 'k_all'):
        
        if 'bs' not in self.results: self.results['bs'] = {}
        if str(threshold) not in self.results['bs']: self.results['bs'][str(threshold)] = {}
        
        for idx, run in enumerate(self.data.keys()):

            print(f'Calculating Brier Score for {run}')
            
            if idx == 0:
                    
                # Occurrence
                self.data[run][k_fold].loc[:,'simulated_occurrence'] = (self.data[run][k_fold][simulated]>threshold).astype('int')
                self.data[run][k_fold].loc[:,'observed_occurrence'] = (self.data[run][k_fold][observed]>threshold).astype('int')
            
                for s in self.bc_methods:
                    self.data[run][k_fold].loc[:,f'{s}_occurrence'] = (self.data[run][k_fold][s]>threshold).astype('int')

    #             self.data[run][k_fold].loc[:,'climatology_occurrence'] = self.data[run][k_fold]['observed_occurrence'].mean()
    #             self.data[run][k_fold].loc[:,'climatology_occurrence_station'] = self.data[run][k_fold].groupby('Station')['observed_occurrence'].transform('mean')    
            
                # Brier Score
                for s in self.bc_methods:
                    self.data[run][k_fold].loc[:,f'BS_{s}_{threshold}'] = self.data[run][k_fold].apply(BS, axis=1, args=(f'{s}_occurrence','Prec',threshold))

                self.data[run][k_fold].loc[:,f'BS_sim_{threshold}'] = self.data[run][k_fold].apply(BS, axis=1, args=('simulated_occurrence','Prec',threshold))
    #             self.data[run][k_fold].loc[:,f'BS_obs_{threshold}'] = self.data[run][k_fold].apply(BS, axis=1, args=('climatology_occurrence','Prec',threshold))
    #             self.data[run][k_fold].loc[:,f'BS_obs_station_{threshold}'] = self.data[run][k_fold].apply(BS, axis=1, args=('climatology_occurrence_station','Prec',threshold))
            
                # Results dictionary
                for s in self.bc_methods:
                    self.results['bs'][str(threshold)][s] = self.data[run][k_fold].loc[:,f'BS_{s}_{threshold}'].mean()

                self.results['bs'][str(threshold)]['N19'] = self.data[run][k_fold].loc[:,f'BS_sim_{threshold}'].mean()
    #             self.results['bs'][str(threshold)]['clim'] = self.data[run][k_fold].loc[:,f'BS_obs_{threshold}'].mean()
    #             self.results['bs'][str(threshold)]['clim-st'] = self.data[run][k_fold].loc[:,f'BS_obs_station_{threshold}'].mean()
            
            if self.use_gamma_likelihood:
                likelihood = 'gamma'
                print('gamma')
            else:
                likelihood = run.split(']_')[1].split('_B')[0]

            print(likelihood)

            if likelihood == 'b2gmm': 
                continue
            elif likelihood == 'None':
                self.data[run][k_fold].loc[:,'modelled_occurrence'] = (self.data[run][k_fold]['y']>threshold).astype('int')
            else:
                if threshold == 0:
                    self.data[run][k_fold].loc[:,'modelled_occurrence'] = (1-self.data[run][k_fold]['pi'])
                else:
                    self.data[run][k_fold].loc[:,'modelled_occurrence'] = 1 - self.data[run][k_fold].apply(cdf_apply, axis=1, args=(likelihood,'10000',threshold))
            
            # Brier Score
            self.data[run][k_fold].loc[:,f'BS_{threshold}'] = self.data[run][k_fold].apply(BS, axis=1, args=('modelled_occurrence','Prec',threshold))
            
            self.results['bs'][str(threshold)][run] = self.data[run][k_fold].loc[:,f'BS_{threshold}'].mean()

    def ROC(self, k_fold = 'k_all', thresholds = [0.1,1,10,30], only_wet_days = True):

        quantiles = np.concatenate(([0.00001,0.0001,0.001],np.linspace(0,0.99,100),[.999,.9999,.99999,.999999,.9999999,.99999999,.999999999]))

        self.results['roc'] = {}

        for idx, wet_threshold in enumerate(thresholds):
            
            self.results['roc'][str(wet_threshold)] = {}
            
            for i, (k,v) in enumerate(self.data.items()):
                
                likelihood = k.split(']_')[1].split('_B')[0]

                self.results['roc'][str(wet_threshold)][k] = {}
                
                TPRs, FPRs = [], []
                
                for q in quantiles:
                    TPR, FPR = ROC(v['k_all'], obs='Prec', wet_threshold=wet_threshold, quantile=q, only_wet_days=only_wet_days, likelihood=likelihood)
                    TPRs.append(round(TPR,3))
                    FPRs.append(round(FPR,3))
                auc = AUC(FPRs, TPRs) 
                
                self.results['roc'][str(wet_threshold)][k]['tpr'] = TPRs
                self.results['roc'][str(wet_threshold)][k]['fpr'] = FPRs
                self.results['roc'][str(wet_threshold)][k]['auc'] = auc 

                if i == 0: 
                    TPR_sim, FPR_sim = ROC(v['k_all'],obs='Prec',sim='precip_norris',wet_threshold=wet_threshold, quantile=0, only_wet_days=only_wet_days)
                    TPR_bc, FPR_bc = ROC(v['k_all'],obs='Prec',sim='sf',wet_threshold=wet_threshold, quantile=0, only_wet_days=only_wet_days)
            
                    self.results['roc'][str(wet_threshold)]['WRF'] = {}
                    self.results['roc'][str(wet_threshold)]['WRF-BC'] = {}
                    
                    self.results['roc'][str(wet_threshold)]['WRF']['tpr'] = TPR_sim
                    self.results['roc'][str(wet_threshold)]['WRF']['fpr'] = FPR_sim
                    self.results['roc'][str(wet_threshold)]['WRF-BC']['tpr'] = TPR_bc
                    self.results['roc'][str(wet_threshold)]['WRF-BC']['fpr'] = FPR_bc
            
            print(f'Wet threshold = {wet_threshold} -- Hit rate: {round(TPR_sim,3)}; False alarm rate: {round(FPR_sim,3)}')

            
    def plot_ROC(self, save=False, thresholds = [0.1,1,10,30], filepath = 'figures/roc.png'):

        import seaborn as sns
        sns.set_style('whitegrid')
        # paper style
        sns.set_context("paper", font_scale=1.5)

        color_dict = {
        0: '#E69F00',  # Orange
        1: '#56B4E9',  # Sky Blue
        2: '#1B02A3',  # Dark Blue
        }
        
        # quantiles = np.concatenate(([0.00001,0.0001,0.001],np.linspace(0,0.99,100),[.999,.9999,.99999,.999999,.9999999,.99999999,.999999999]))

        _, axes = plt.subplots(nrows=1,ncols=4,figsize=(22,5))

        for idx, wet_threshold in enumerate(thresholds):
            
            ax = axes.reshape(-1)[idx]

            subpanels = ['a','b','c','d']
            
            for i, (k,v) in enumerate(self.data.items()):

                entries = ['$VGLM$','$MLP_{S}$', '$MLP_{L}$']
                
                FPRs = self.results['roc'][str(wet_threshold)][k]['fpr']
                TPRs = self.results['roc'][str(wet_threshold)][k]['tpr']
                auc = self.results['roc'][str(wet_threshold)][k]['auc']

                color = color_dict.get(i, '#000000')
                ax.plot(FPRs, TPRs,'--',label=entries[i], color=color)
                # add annotation 'a)' to the left upper coner of the axes, 
            
            ax.text(-0, 1.07, f'{subpanels[idx]})', transform=ax.transAxes, verticalalignment='top')
                
            ax.plot([0,1],[0,1],'k')
            
            
            ax.set_xlabel('False alarm rate')
            
            if idx == 0: ax.set_ylabel('Hit rate')
            
            ax.set_title(f'Events exceeding {str(wet_threshold)} mm/day')

            TPR_sim = self.results['roc'][str(wet_threshold)]['WRF']['tpr']
            FPR_sim = self.results['roc'][str(wet_threshold)]['WRF']['fpr']
            TPR_bc = self.results['roc'][str(wet_threshold)]['WRF-BC']['tpr']
            FPR_bc = self.results['roc'][str(wet_threshold)]['WRF-BC']['fpr']

            ax.plot(FPR_sim, TPR_sim, 'ok', label = '$WRF$', fillstyle='none')
            ax.plot(FPR_bc, TPR_bc, 'og', label = '$WRF_{BC}$', fillstyle='none')
            
            if idx == 0: ax.legend()
            # ax.grid()

        if save:
            plt.savefig(filepath, dpi=400)
        plt.show()

    def reliability_diagram(self, MODEL, threshold = 0, num_bins = 20):

        if 'reliability' not in self.results.keys():
            self.results['reliability'] = {}
        if MODEL not in self.results['reliability'].keys():
            self.results['reliability'][MODEL] = {}
        if str(threshold) not in self.results['reliability'][MODEL].keys():
            self.results['reliability'][MODEL][str(threshold)] = {}

        # simulated = 'precip_norris'
        observed = 'Prec'

        if threshold == 0:
            y_pred = 1 - self.data[MODEL]['k_all']['pi']
        else:
            y_pred = 1 - self.data[MODEL]['k_all'].apply(cdf_apply, axis=1, args=('bgmm','10000',threshold))

        y_true = (self.data[MODEL]['k_all'][observed]>threshold).astype('int')    

        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=num_bins, strategy='uniform')

        self.results['reliability'][MODEL][str(threshold)]['prob_true'] = prob_true
        self.results['reliability'][MODEL][str(threshold)]['prob_pred'] = prob_pred
        self.results['reliability'][MODEL][str(threshold)]['y_pred'] = y_pred
        self.results['reliability'][MODEL][str(threshold)]['y_true'] = y_true

    def plot_RDs(self, MODEL, thresholds = [0,1,10,30], num_bins = 20):
        
        import seaborn as sns
        sns.set_style('whitegrid')
        # paper style
        sns.set_context("paper", font_scale=1.5)

        fig, axes = plt.subplots(2,4, figsize=(20,7))

        for idx, threshold in enumerate(thresholds):
            # self.reliability_diagram(MODEL, threshold, num_bins)

            subpanels = ['a','b','c','d','e','f','g','h']

            ax1 = axes[0, idx]
            ax2 = axes[1, idx]

            prob_pred = self.results['reliability'][MODEL][str(threshold)]['prob_pred']
            prob_true = self.results['reliability'][MODEL][str(threshold)]['prob_true']
            y_pred = self.results['reliability'][MODEL][str(threshold)]['y_pred']

            ax1.plot(prob_pred, prob_true, 'o--', color='#1B02A3')
            ax1.plot([0,1],[0,1],'k')
            

            ax1.set_xlabel('Predicted probability')
            if idx == 0:
                ax1.set_ylabel('Observed frequency')
            # else:
            #     ax1.set_yticklabels([])

            ax1.set_xlim(0,1)
            ax1.set_ylim(0,1)
            # ax1.set_xticklabels([])
            # ax1.grid()

            ax2.hist(y_pred, bins=num_bins, rwidth=0.9, color='#1B02A3')
            ax2.set_yscale('log')
            ax2.set_xlim(0,1)
            if idx == 0:
                ax2.set_ylabel('Count')
            # else:
            #     ax2.set_yticklabels([])
            
            ax1.text(-0, 1.11, f'{subpanels[idx]})', transform=ax1.transAxes, verticalalignment='top')
            ax2.text(-0, 1.11, f'{subpanels[idx+4]})', transform=ax2.transAxes, verticalalignment='top')
            



            ax2.set_ylim(1,1e6)
            
            ax2.set_xlabel('Predicted probability')
            # ax1.set_title(f'Threshold: {threshold} mm')
            ax1.set_title(f'Events exceeding {threshold} mm/day')
            ax2.set_title(f'Events exceeding {threshold} mm/day')
        
        plt.tight_layout()
        plt.savefig('figures/reliability_diagrams.png', dpi=400)
        plt.show()

    def print_dict_structure(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print('    ' * indent + str(key))
                print_dict_structure(value, indent + 1)
            else: 
                print('    ' * indent + str(key.split('_')[0]) + '       ' + f'{value:.3f}')

def calculate_skill_scores(data):
    skill_scores = {}
    
    for metric, values in data.items():
        skill_scores[metric] = {}
        
        if metric == 'nll':
            continue
        elif metric == 'bs':
            for threshold, sub_values in values.items():
                ref_value = sub_values['N19']
                skill_scores[metric][threshold] = {}
                for key, value in sub_values.items():
                    skill_scores[metric][threshold][key] = 1 - (value / ref_value)
        else:
#             pass
            ref_value = values['N19']
            for key, value in values.items():
                if key != 'N19':
                    skill_scores[metric][key] = 1 - (value / ref_value)
    
    return skill_scores

def plot_cv_split_vs_elevation(predictions, MODEL, CONFIG):

    df = predictions[MODEL]['k_all'].groupby('Station').mean(numeric_only=True).reset_index()

    x = df.k_fold
    y = df.Z

    plt.figure(figsize=(10,5))
    plt.grid()
    plt.scatter(x,y, s=100)
    plt.xticks([i for i in range(len(set(x)))])
    plt.ylabel('Elevation (m.a.s.l.)')
    plt.xlabel('k_fold')
    plt.title(f'Station elevation split for {CONFIG.RUN_NAME} experiment ')
    plt.show()

def custom_preprocessing_for_predictors_run(predictions):
    pred = predictions.copy()
    predictions = {}
    for run,v in pred.items():
        for k in v.keys():
            k_all_key = run[:-4]
            pred[run][k][f'k_fold'] = int(k[-1])
            if k_all_key not in predictions:

                predictions[k_all_key] = {}

                predictions[k_all_key][f'k_{run[-1]}'] = pred[run][k]
                predictions[k_all_key]['k_all'] = pred[run][k]

            else:
                predictions[k_all_key][f'k_{run[-1]}'] = pred[run][k]
                predictions[k_all_key]['k_all'] = pd.concat([predictions[k_all_key]['k_all'], pred[run][k]])
    
    return predictions

def create_k_all_predictions(predictions, CONFIG):
    # Create predictions for k_all
    for run in predictions.keys():
        for i in range(len(CONFIG.params['k'])):
            predictions[run][f'k{i}']['k_fold'] = i
            if i == 0:
                predictions[run]['k_all'] = predictions[run][f'k{i}']
            else:
                predictions[run]['k_all'] = pd.concat([predictions[run]['k_all'],predictions[run][f'k{i}']])

def delete_uncomplete_runs_from_predictions(predictions, CONFIG, delete=False):
    for k,run in predictions.items():
        if len(run.keys()) < len(CONFIG.params['k']):
            if delete:
                del run
                print(f'deleted run {k}')
            else:
                pass
                print(f'would delete run {k}')

def combine_run_predictions(list_of_configs):
    for idx, C in enumerate(list_of_configs):
        run_prefix = C.RUN_NAME
        print(run_prefix)
        with open(f'/data/hpcdata/users/marron31/_experiments/{run_prefix}/predictions.pkl', 'rb') as handle:
            predictions = pickle.load(handle)
        
        if idx==0:
            uber_predictions = predictions.copy()
        else:
            for k,v in predictions.items():
                uber_predictions[k] = v

    return uber_predictions

def calculate_variance_per_station(df):

    df['std'] = -999

    stations = df.Station.unique()

    for s in stations:
        df_s = df[df['Station']==s].copy()
        df_s['std'] = df_s['Prec'].std()
        if s == stations[0]:
            dfc = df_s
        else:
            dfc = pd.concat([dfc, df_s])
    
    return dfc

def bias_correction(df, ignore_zeros=False, dry_day_correction=False, bc_mode='sf', output_series='bc'):
    df[output_series] = -999  # initialize column

    k_fold = df['k_fold'].unique()  # get unique k_fold values

    for k in k_fold:
        dfa = df.loc[df['k_fold'] != k].copy()  # get all data except k
        dfb = df.loc[df['k_fold'] == k].copy()  # get data for k

        dd_obs = dfa.loc[dfa['Prec'] == 0].shape[0]
        dd_sim = dfa.loc[dfa['precip_norris'] == 0].shape[0]

        # ddf = dd_obs / dd_sim
        ddf = dd_obs / dfa.shape[0]

        mean_obs = dfa['Prec'].mean()
        mean_sim = dfa['precip_norris'].mean()
        sigma_obs = dfa['Prec'].std()
        sigma_mean = dfa['precip_norris'].std()

        if ignore_zeros:
            dfa_nonzeros = dfa.loc[dfa['Prec'] > 0]  # select only non-zero values
            factor = dfa_nonzeros['Prec'].mean() / dfa_nonzeros['precip_norris'].mean()  # calculate factor for non-zero values
        else:
            factor = dfa['Prec'].mean() / dfa['precip_norris'].mean()  # calculate factor for all values
            # std_factor = dfa['Prec'].std() / dfa['precip_norris'].std()

        # zero_factor = dfa.loc[dfa['Prec'] == 0].shape[0] / dfa.shape[0]

        if dry_day_correction:
            dfb = replace_lowest_percent_by_group(dfb, ddf=ddf, output_series=output_series).reset_index(drop=True)
        else:
            dfb[output_series] = dfb['precip_norris']

        if bc_mode == 'sf':
            dfb.loc[:, output_series] = dfb[output_series] * factor

        elif bc_mode == 'mav':
            dfb.loc[:, output_series] = (dfb[output_series] - mean_sim) * (sigma_obs / sigma_mean) + mean_obs

        elif bc_mode == 'qm':
            # Implement quantile mapping
            def empirical_cdf(data):
                """ Return the empirical CDF for a data set. """
                n = len(data)
                x = np.sort(data)
                y = np.arange(1, n + 1) / n
                return x, y

            # Calculate empirical CDF for observed and simulated data
            obs_sorted, obs_cdf = empirical_cdf(dfa['Prec'])
            sim_sorted, sim_cdf = empirical_cdf(dfa['precip_norris'])

            dfb[output_series] = dfb[output_series].apply(quantile_mapping, args=(sim_sorted, obs_sorted, sim_cdf, obs_cdf))

        if k == k_fold[0]:
            dfc = dfb
        else:
            dfc = pd.concat([dfc, dfb])

    return dfc

def bias_correction_by_month(df, ignore_zeros=False, dry_day_correction=False, bc_mode='sf', output_series='bc'):
    
    df[output_series] = -999  # initialize column
    df['month'] = df['Date'].dt.month  # assuming the Date column is in datetime format
    
    k_fold = df['k_fold'].unique()  # get unique k_fold values

    for k in k_fold:
        dfa = df.loc[df['k_fold'] != k].copy()  # get all data except k
        dfb = df.loc[df['k_fold'] == k].copy()  # get data for k

        # Calculate correction factors by month
        def calculate_factors(dfa, ignore_zeros):
            """ Calculate mean factors by month."""
            factors = {}
            for month in range(1, 13):
                dfa_month = dfa.loc[dfa['month'] == month]
                if ignore_zeros:
                    dfa_nonzeros = dfa_month.loc[dfa_month['Prec'] > 0]  # select only non-zero values
                    factor = dfa_nonzeros['Prec'].mean() / dfa_nonzeros['precip_norris'].mean()  # calculate factor for non-zero values
                else:
                    factor = dfa_month['Prec'].mean() / dfa_month['precip_norris'].mean()  # calculate factor for all values
                factors[month] = factor
            return factors

        def calculate_std_factors(dfa):
            """ Calculate standard deviation factors by month."""
            std_factors = {}
            for month in range(1, 13):
                dfa_month = dfa.loc[dfa['month'] == month]
                std_factor = dfa_month['Prec'].std() / dfa_month['precip_norris'].std()
                std_factors[month] = std_factor
            return std_factors

        def calculate_empirical_cdfs(dfa):
            """ Calculate empirical CDFs for observed and simulated data by month."""
            cdfs = {}
            for month in range(1, 13):
                dfa_month = dfa.loc[dfa['month'] == month]
                obs_sorted, obs_cdf = empirical_cdf(dfa_month['Prec'])
                sim_sorted, sim_cdf = empirical_cdf(dfa_month['precip_norris'])
                cdfs[month] = (obs_sorted, obs_cdf, sim_sorted, sim_cdf)
            return cdfs
        
        def empirical_cdf(data):
            """ Return the empirical CDF for a data set. """
            n = len(data)
            x = np.sort(data)
            y = np.arange(1, n + 1) / n
            return x, y

        # Calculate correction factors
        factors = calculate_factors(dfa, ignore_zeros)
        std_factors = calculate_std_factors(dfa)
        cdfs = calculate_empirical_cdfs(dfa)

        for month in range(1, 13):
            dfb_month = dfb.loc[dfb['month'] == month]

            if dry_day_correction:
                # implement monthly ddf
                dd_obs = dfa.loc[(dfa['month'] == month) & (dfa['Prec'] == 0)].shape[0]
                dd_sim = dfa.loc[(dfa['month'] == month) & (dfa['precip_norris'] == 0)].shape[0]

                # ddf = dd_obs / dd_sim
                ddf = dd_obs / dfa.shape[0]

                dfb_month = replace_lowest_percent_by_group(dfb_month, ddf=ddf, output_series=output_series).reset_index(drop=True)
            
            else:
                dfb_month[output_series] = dfb_month['precip_norris']
            
            # Bias correction
            if bc_mode == 'sf':
                dfb_month.loc[:, output_series] = dfb_month[output_series] * factors[month]
                
            elif bc_mode == 'mav':
                mean_obs = dfa.loc[dfa['month'] == month, 'Prec'].mean()
                mean_sim = dfa.loc[dfa['month'] == month, 'precip_norris'].mean()
                sigma_obs = dfa.loc[dfa['month'] == month, 'Prec'].std()
                sigma_mean = dfa.loc[dfa['month'] == month, 'precip_norris'].std()

                dfb_month.loc[:, output_series] = (dfb_month[output_series] - mean_sim) * (sigma_obs / sigma_mean) + mean_obs
            
            elif bc_mode == 'qm':
                obs_sorted, obs_cdf, sim_sorted, sim_cdf = cdfs[month]
                dfb_month[output_series] = dfb_month[output_series].apply(quantile_mapping, args=(sim_sorted, obs_sorted, sim_cdf, obs_cdf))

            if month == 1:
                dfc = dfb_month
            else:
                dfc = pd.concat([dfc, dfb_month])

        if k == k_fold[0]:
            final_dfc = dfc
        else:
            final_dfc = pd.concat([final_dfc, dfc])

    return final_dfc

    
def quantile_mapping(value, sim_sorted, obs_sorted, sim_cdf, obs_cdf):
    sim_percentile = np.interp(value, sim_sorted, sim_cdf)
    mapped_value = np.interp(sim_percentile, obs_cdf, obs_sorted)
    return mapped_value

def replace_lowest_percent_by_group(df, ddf, output_series='bc'):

    def replace_lowest_percent(group, ddf):
        # dd_ratio = group.loc[group['Prec'] == 0].shape[0] / group.shape[0
        threshold = group['precip_norris'].quantile(ddf)
        group.loc[group['precip_norris'] <= threshold, output_series] = 0
        return group

    df = df.groupby('Station').apply(replace_lowest_percent, ddf=ddf)
    return df

def print_CRPS_results(predictions):
    for idx, (k,v) in enumerate(predictions.items()):
        p = v['k_all']
        if idx < 1:
            print(f"CRPS_clim: {p['CRPS_clim'].mean():.3f}")
            print(f"CRPS_sim: {p['CRPS_sim'].mean():.3f}")
            print(f"MAE_clim: {p['MAE_clim'].mean():.3f}")
            print(f"MAE_sim: {p['MAE_sim'].mean():.3f}")
        
        print(f"CRPS_mod for {k}: {p['CRPS_mod'].mean():.3f}")
        
def compute_CRPS(predictions, per_station = False, limit = None): 
    for idx, (k,v) in enumerate(predictions.items()):
        
        p = v['k_all']
        
        if per_station:
            stations = p['Station'].unique()

            for s in stations:

                p_s = p[p['Station']==s].copy()
                x_clim = p_s['Prec'].to_numpy()
                x_sim = p_s['precip_norris'].to_numpy()
            
                p_s['CRPS_clim'] = CRPS(p_s, ensemble=x_clim)
                p_s['CRPS_sim'] = CRPS(p_s, ensemble=x_sim)
                p_s['CRPS_mod'] = CRPS(p_s, limit=limit)
                p_s['MAE_clim'] = abs(p_s['Prec'] - p_s['Prec'].mean()).mean()
                p_s['MAE_sim'] = abs(p_s['Prec'] - p_s['precip_norris']).mean()

                print(s)
                print(f"CRPS_clim: {p_s['CRPS_clim'].mean():.3f}")
                print(f"CRPS_sim: {p_s['CRPS_sim'].mean():.3f}")
                print(f"MAE_clim: {p_s['MAE_clim'].mean():.3f}")
                print(f"MAE_sim: {p_s['MAE_sim'].mean():.3f}")
                print(f"CRPS_mod for {k}: {p_s['CRPS_mod'].mean():.3f}")
                print("--------------------")
     
        
        else:
            if idx == 0:
                x_clim = p['Prec'].to_numpy()
                x_sim = p['precip_norris'].to_numpy()
            
                p['CRPS_clim'] = CRPS(p, ensemble=x_clim)
                p['CRPS_sim'] = CRPS(p, ensemble=x_sim)

            p['CRPS_mod'] = CRPS(p, limit=limit)
            print(f'Completed CRPS_mod for {k}') 
            
            p['MAE_clim'] = abs(p['Prec'] - p['Prec'].mean()).mean()
            p['MAE_sim'] = abs(p['Prec'] - p['precip_norris']).mean()
        
    print_CRPS_results(predictions)

