import os
import warnings
import matplotlib
matplotlib.use('Agg')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore')

lon_min, lon_max = 10.0, 14.0
lat_min, lat_max = 41.0, 44.2

grid_lon = np.linspace(lon_min, lon_max, 250)
grid_lat = np.linspace(lat_min, lat_max, 250)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

lat_mean = (lat_min + lat_max) / 2.0
k_huglin = 1.0 + (lat_mean - 40) * 0.006

elevation_approx = np.clip(np.sin((lat_grid - 41) * 2) * np.cos((lon_grid - 11) * 2) * 800 + 200, 0, 1800)
t_mean_season = 21.5 - (elevation_approx * 0.0065)
t_max_season = 27.0 - (elevation_approx * 0.0070)

gdd_daily = np.maximum(0, t_mean_season - 10)
winkler_grid = gdd_daily * 183

huglin_daily = (np.maximum(0, t_mean_season - 10) + np.maximum(0, t_max_season - 10)) / 2.0
huglin_grid = huglin_daily * 183 * k_huglin

provinces_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', facecolor='none')

fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=150, subplot_kw={'projection': ccrs.PlateCarree()})

def format_map_base(ax, title):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#fbfbfb', zorder=1)
    ax.add_feature(provinces_feature, edgecolor='#555555', linewidth=0.7, linestyle='--', alpha=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=1.2, edgecolor='black', zorder=6)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#e6f2ff', zorder=7)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.1, edgecolor='black', zorder=8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

format_map_base(axes[0], 'INDICE DI HUGLIN (HI) - ZONAZIONE VITICOLA')
levels_h = [1200, 1500, 1800, 2100, 2400, 2700, 3000]
cmap_h = mcolors.ListedColormap(['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c', '#a50026'])
norm_h = mcolors.BoundaryNorm(levels_h, cmap_h.N)
cf1 = axes[0].contourf(lon_grid, lat_grid, huglin_grid, levels=levels_h, cmap=cmap_h, norm=norm_h, alpha=0.85, zorder=2)
plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.06, shrink=0.85)

format_map_base(axes[1], 'INDICE DI WINKLER (WI / GDD) - REGIONI CLIMATICHE')
levels_w = [800, 1110, 1390, 1670, 1940, 2220, 2600]
cmap_w = mcolors.ListedColormap(['#edf8fb', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'])
norm_w = mcolors.BoundaryNorm(levels_w, cmap_w.N)
cf2 = axes[1].contourf(lon_grid, lat_grid, winkler_grid, levels=levels_w, cmap=cmap_w, norm=norm_w, alpha=0.85, zorder=2)
plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.06, shrink=0.85)

plt.suptitle('ANALISI AGRO-CLIMATICA - ETRURIA', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('mappe_bioclimatiche_huglin_winkler_era5.png', bbox_inches='tight', dpi=150)
plt.close()
