import os
import warnings
import matplotlib
matplotlib.use('Agg')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie

warnings.filterwarnings('ignore')
xr.set_options(use_new_combine_kwarg_defaults=True)

lon_min, lon_max = 10.0, 14.0
lat_min, lat_max = 41.0, 44.2

run_date_dt = pd.Timestamp.now(tz='UTC').floor('D')
run_date = run_date_dt.strftime('%Y-%m-%d 00:00')

grid_lon = np.linspace(lon_min, lon_max, 350)
grid_lat = np.linspace(lat_min, lat_max, 350)

def standardize_coords(da):
    rename_dict = {}
    for col in da.coords:
        if col in ['lon', 'long', 'x', 'gridlon']: rename_dict[col] = 'longitude'
        elif col in ['lat', 'y', 'gridlat']: rename_dict[col] = 'latitude'
    if rename_dict: da = da.rename(rename_dict)
    if 'longitude' in da.coords and (da.longitude.values > 180).any():
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby('longitude')
    return da

try:
    H_gfs_24 = Herbie(date=run_date, fxx=24, model='gfs', product='pgrb2.0p25')

    pcp_total = None
    for f_hour in [6, 12, 18, 24]:
        H_temp = Herbie(date=run_date, fxx=f_hour, model='gfs', product='pgrb2.0p25')
        ds_p = H_temp.xarray('APCP')
        da_p = standardize_coords(ds_p['apcp'] if 'apcp' in ds_p else ds_p[list(ds_p.data_vars)[0]])
        p_interp = da_p.interp(longitude=grid_lon, latitude=grid_lat).values
        pcp_total = np.nan_to_num(p_interp) if pcp_total is None else pcp_total + np.nan_to_num(p_interp)

    ds_tmp = H_gfs_24.xarray('TMP:2 m above ground')
    da_tmp = standardize_coords(ds_tmp['t2m'] if 't2m' in ds_tmp else ds_tmp[list(ds_tmp.data_vars)[0]]) - 273.15
    tmp_interp = da_tmp.interp(longitude=grid_lon, latitude=grid_lat).values

    ds_rh = H_gfs_24.xarray('RH:2 m above ground')
    da_rh = standardize_coords(ds_rh['r2'] if 'r2' in ds_rh else ds_rh[list(ds_rh.data_vars)[0]])
    rh_interp = da_rh.interp(longitude=grid_lon, latitude=grid_lat).values

    ds_u = H_gfs_24.xarray('UGRD:10 m above ground')
    ds_v = H_gfs_24.xarray('VGRD:10 m above ground')
    da_u = standardize_coords(ds_u['u10'] if 'u10' in ds_u else ds_u[list(ds_u.data_vars)[0]])
    da_v = standardize_coords(ds_v['v10'] if 'v10' in ds_v else ds_v[list(ds_v.data_vars)[0]])
    wind_interp = np.sqrt(da_u.values**2 + da_v.values**)

    fungal_risk = np.clip((rh_interp - 50) * 2.2, 0, 100)
    fungal_risk = np.where((tmp_interp < 10) | (tmp_interp > 32), 0, fungal_risk)

    et0_grid = np.clip(0.16 * (tmp_interp + 10) * np.sqrt(np.clip(wind_interp, 0.5, 12)), 0, 10)
    gdd_grid = np.maximum(0, tmp_interp - 10)

    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

    provinces_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', facecolor='none')

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=150, subplot_kw={'projection': ccrs.PlateCarree()})

    def format_map_base(ax, title):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#ffffff', zorder=1)
        ax.add_feature(provinces_feature, edgecolor='#444444', linewidth=0.7, linestyle='--', alpha=0.85, zorder=5)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=1.2, edgecolor='black', zorder=6)
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#e6f2ff', zorder=7)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.1, edgecolor='black', zorder=8)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    format_map_base(axes[0], '1. RISCHIO FUNGINO + PIOGGIA CUMULATA (24h)')
    levels_f = [0, 20, 40, 60, 80, 100]
    cmap_f = mcolors.ListedColormap(['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f', '#006d2c'])
    cf1 = axes[0].contourf(lon_grid, lat_grid, fungal_risk, levels=levels_f, cmap=cmap_f, alpha=0.85, zorder=2)
    plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.06, shrink=0.85)

    format_map_base(axes[1], '2. EVAPOTRASPIRAZIONE ET0 + GRADI GIORNO GDD')
    cf2 = axes[1].contourf(lon_grid, lat_grid, et0_grid, levels=12, cmap='YlOrRd', alpha=0.85, zorder=2)
    plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.06, shrink=0.85)

    plt.suptitle(f'QUADRO AGROMETEO - CENTRO ITALIA | Run: {run_date} UTC', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('mappe_sovrapposte_24h.png', bbox_inches='tight', dpi=150)
    plt.close()

except Exception as e:
    print(f"Errore generazione agrometeo: {e}")
