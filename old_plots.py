
# --- MAP_PLOT ---
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(x=map_results['X'],y=map_results['Y'],c=map_results['Z'], cmap='Blues', marker='s')
beas_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=1)
sutlej_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=1)
ax.set_yticks([]), ax.set_xticks([])
ax.set_frame_on(False)
#     ax.scatter(x=st.X, y=st.Y, s=5)
#     scatter = ax.scatter(x=st_test.X, y=st_test.Y, c=st_test['k_fold'],cmap='Paired',s=50, marker='p')
    
#     legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="CV class")
#     ax.add_artist(legend1)
    
plt.show()
    #plt.tight_layout()

# --- MAP PLOT ---
fig, ax = plt.subplots(1, 1, figsize=(10,10))
gdf.plot(ax=ax, column='k_fold', legend=False, cmap='Set1', markersize=20, marker="o", linewidth=3)
beas_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=0.5)
sutlej_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=0.5)
ax.set_axis_off()
plt.savefig(f'exports/k_fold_{likelihood_fn}_{np.random.randint(1000)}.png')
plt.show()


# --- MAP PLOT --- 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
plt.style.use('seaborn-dark-palette')

title = ['$\mu$ Neural Network', '$median$ Neural Network', '$median$ $\Gamma$ Neural Network', 'Bannister et al. (2019)']
var = ['se_mlp_ratio', 'se_mlp_median_ratio', 'se_mlp_median_gamma_ratio', 'se_reg_ratio']

fig = plt.figure(figsize=(20, 20))

color_map = plt.cm.get_cmap('seismic').reversed()

grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                #aspect=False,
                axes_pad=0,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.05,
                cbar_size="2.5%"
                )

for i, ax in enumerate(grid):
    ax.set_axis_off()
    ax.set_yticks([])
    ax.set_xticks([])
    
    #ax.set_xlim(gdf.X.min() - margin*(gdf.X.max() - gdf.X.min()), gdf.X.max() + margin*(gdf.X.max() - gdf.X.min()))
    #ax.set_ylim(gdf.Y.min() - margin*(gdf.Y.max() - gdf.Y.min()), gdf.Y.max() + margin*(gdf.Y.max() - gdf.Y.min()))
    
    beas_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=0.5)
    sutlej_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=0.5)
    
    if i==0:
        pass
        #gdf_train.plot(ax=ax, markersize=15, color='green')
        #gdf_val.plot(ax=ax, markersize=15, color='orange')
    
    sc = ax.scatter(gdf.X, gdf.Y, 
                    c=gdf['k_fold'],#[var[i]], 
                    s=100, marker='p', cmap=color_map, vmin=-1, vmax=1, edgecolor='k')
    ax.set_title(title[i], fontsize=15)
    
# when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]
cbar = grid.cbar_axes[0].colorbar(sc)
cbar.ax.set_yticks([-1,0,1])
cbar.ax.tick_params(labelsize=15)
#plt.savefig(f'exports/{likelihood_fn}_{np.random.randint(1000)}.png')
plt.show()

# --- MAP PLOT --- 
fig, axes = plt.subplots(1, 2, figsize=(15,10), constrained_layout=True)
#color_map = plt.cm.get_cmap('seismic').reversed()
margin = 0.25

for i, ax in enumerate(axes):
    gdf.plot(ax=ax, column='k_fold', legend=False, cmap='Set1', markersize=20, marker="o", linewidth=3)
    beas_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=0.5)
    sutlej_watershed.plot(ax=ax, edgecolor='k', color='None', linewidth=0.5)
    #if i==len(axes)-1:
    #    plt.colorbar(plt.cm.ScalarMappable(cmap='Set1'), ax=ax, shrink=0.6)
    #ax.set_xlim(gdf.X.min() - margin*(gdf.X.max() - gdf.X.min()), gdf.X.max() + margin*(gdf.X.max() - gdf.X.min()))
    #ax.set_ylim(gdf.Y.min() - margin*(gdf.Y.max() - gdf.Y.min()), gdf.Y.max() + margin*(gdf.Y.max() - gdf.Y.min()))
    #ax.set_ylabel(" ")
    #ax.set_xlabel(" ")
    ax.set_axis_off()
    ax.set_aspect(1)

#axes[0].scatter(gdf.X, gdf.Y, c=gdf.se_mlp_ratio, cmap=color_map)
#gdf.plot(ax=axes[0], column='se_mlp_ratio',legend=True, edgecolor = 'white',  cmap=color_map,  markersize=100, vmin=-1, vmax=1)
axes[0].set_title('MLP', fontsize=16)
#gdf.plot(ax=axes[1], column='se_reg_ratio',legend=True, edgecolor = 'white', cmap=color_map,  markersize=100, vmin=-1, vmax=1)
#axes[1].set_title('Bannister et al. (2019)', fontsize=16)

#plt.legend()
#plt.tight_layout()
plt.show()

#plt.savefig(likelihood_fn+'.png');


# --- PLOT ---
plt.figure(figsize=(10,10))

plt.plot(st_test_r['Prec'], st_test_r['se_mlp'],'o',ms=4,label="MLP")
plt.plot(st_test_r['Prec'], st_test_r['se_wrf'],'x',ms=4,label="WRF")
plt.legend()
# plt.ylim([-0,20000])
plt.show()


# --- PLOT ---
plt.figure(figsize=(8,8))
plt.style.use('seaborn-dark-palette')
plt.plot(st_test_summary['Z'],st_test_summary['se_mlp_ratio'],'og', label='$\mu$ - Neural Network (Bernoully Gamma mixture model)')
plt.plot(st_test_summary['Z'],st_test_summary['se_mlp_median_ratio'],'xb', label='$median$ - Neural Network (Bernoully Gamma mixture model)')
plt.plot(st_test_summary['Z'],st_test_summary['se_mlp_median_gamma_ratio'],'>k', label='$median$ $\Gamma$ - Neural Network (Bernoully Gamma mixture model)')

plt.plot(st_test_summary['Z'],st_test_summary['se_reg_ratio'],'or', label='Bannister et al. (2019)')

plt.xlabel("Elevation (masl)")
plt.ylabel("MSE reduction ratio")
plt.ylim([-1, 1])
plt.legend()
#plt.savefig(f"exports/MSE_plot_{day_filter}.png")
plt.show()



# --- PLOT ---
df = st_test[st_test['Station']=='Pandoh']
df = clip_time_period(df,'2003-01-01','2003-12-31')

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