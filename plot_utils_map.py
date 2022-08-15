import rasterio
import geopandas
from matplotlib import pyplot as plt
from rasterio.plot import show

__all__ = [build_geodataframe, 
           plot_map_stations_cv]


def build_geodataframe(df, x, y):
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[x], df[y]))

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
