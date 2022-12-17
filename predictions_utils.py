import pandas as pd
from metrics import CRPS_apply, CRPS
from tqdm.notebook import tqdm
tqdm.pandas(desc="my bar!")


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

