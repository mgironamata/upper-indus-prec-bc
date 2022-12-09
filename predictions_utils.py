import pandas as pd
from metrics import CRPS_apply


def print_CRPS_results(predictions):
    for idx, (k,v) in enumerate(predictions.items()):
        p = v['k_all']
        if idx < 1:
            print(f"CRPS_clim: {p['CRPS_clim'].mean():.3f}")
            print(f"CRPS_sim: {p['CRPS_sim'].mean():.3f}")

            print(f"MAE_clim: {p['MAE_clim'].mean():.3f}")
            print(f"MAE_sim: {p['MAE_sim'].mean():.3f}")
        
        print(f"CRPS_mod for {k}: {p['CRPS_mod'].mean():.3f}")

def compute_CRPS(predictions): 
    for idx, (k,v) in enumerate(predictions.items()):
        
        p = v['k_all']
        
        if idx == 0:
            x_clim = p['Prec'].to_numpy()
            x_sim = p['precip_norris'].to_numpy()
        
            p['CRPS_clim'] = p.apply(CRPS_apply, axis=1, args=(x_clim,))
            print(f'Completed CRPS_clim for {k}')
            p['CRPS_sim'] = p.apply(CRPS_apply, axis=1, args=(x_sim,))
            print(f'Completed CRPS_sim for {k}')

        p['CRPS_mod'] = p.apply(CRPS_apply, axis=1)
        print(f'Completed CRPS_mod for {k}') 
        
        p['MAE_clim'] = abs(p['Prec'] - p['Prec'].mean()).mean()
        p['MAE_sim'] = abs(p['Prec'] - p['precip_norris']).mean()

    print_CRPS_results(predictions)

