import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
import geopandas as gpd

def create_basin_maps_figure(df):
    
    dfs = df.groupby('Station').mean().reset_index()
    dfs['CRPS_mod'] = dfs['CRPS_mod'].round(2) - dfs['CRPS_mod'].min() / dfs['CRPS_mod'].max() - dfs['CRPS_mod'].min()
    # dfs['CRPS_mod'] = dfs['CRPS_mod'].astype(str)
    # create a figure using folium
    fig = folium.Figure(width=800, height=500)
    # use folium to create a map of the region
    m = folium.Map(location=[35.5, 75.5], zoom_start=5, tiles=None)
    # add a marker for each station
    # for i in range(len(dfs)):
    lat = dfs['Y']
    lon = dfs['X']
    station = dfs['Station']
    crps = dfs['CRPS_mod']
    # plot the stations on the map with round markers and colored by the CRPS value from a color map and add a popup with the station name and CRPS value, and with no border
    for i in range(len(dfs)):
        folium.CircleMarker(location=[lat[i], lon[i]], radius=5, color='black', fill=True, fill_color=plt.cm.get_cmap('viridis')(crps[i]), popup=f'{station[i]}: {crps[i]}', fill_opacity=0.7).add_to(m)
    # add the map to the figure
    m.add_to(fig)
    return fig
    

# fig, ax = plt.subplots(1, 1, figsize=(10,5), constrained_layout=False)
# # use hexagonal markers
# gdf.plot(ax=ax, column='k_fold', legend=False, cmap='Set1', markersize=5, marker="o", linewidth=3)
# plt.show()




        

