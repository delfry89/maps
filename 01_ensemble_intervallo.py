import os
import warnings
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie

xr.set_options(use_new_combine_kwarg_defaults=True)
warnings.filterwarnings('ignore')

w_ecm, w_gefs, w_icon = 0.50, 0.25, 0.25
ore_target, finestra_ore = 24, 24
ore_start = max(0, ore_target - finestra_ore)

run_date_dt = pd.Timestamp.now(tz='UTC').floor('D')
run_date = run_date_dt.strftime('%Y-%m-%d 00:00')
date_str_dwd = run_date_dt.strftime('%Y%m%d00')

lon_min, lon_max = 6.0, 19.0
lat_min, lat_max = 35.5, 47.5
grid_lon = np.linspace(lon_min, lon_max, 350)
grid_lat = np.linspace(lat_min, lat_max, 350)

def get_precip_var(ds):
    for var in ['tp', 'apcp', 'precip', 'APCP', 'tot_prec', 'TOT_PREC', 'unknown', 'tp_acc']:
        if var in ds.data_vars:
            return ds[var]
    return ds[list(ds.data_vars.keys())[0]]

def standardize_coords(da):
    rename_dict = {}
    for col in da.coords:
        if col in ['lon', 'long', 'x', 'gridlon']:
            rename_dict[col] = 'longitude'
        elif col in ['lat', 'y', 'gridlat']:
            rename_dict[col] = 'latitude'
    if rename_dict:
        da = da.rename(rename_dict)
    if 'longitude' in da.coords and (da.longitude.values > 180).any():
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby('longitude')
    return da

def fetch_ensemble_data(fxx):
    models_out = {}
    try:
        H_ecm = Herbie(date=run_date, fxx=fxx, model='ifs', product='enfo', member='mean')
        ds_ecm = H_ecm.xarray('tp')
        tp_ecm = standardize_coords(get_precip_var(ds_ecm))
        if 'number' in tp_ecm.dims: tp_ecm = tp_ecm.mean(dim='number')
        if tp_ecm.max() < 2.0: tp_ecm *= 1000.0
        ecm_interp = tp_ecm.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
        models_out['ECMWF'] = np.clip(np.nan_to_num(ecm_interp.values, nan=0.0), 0, None)
    except Exception as e:
        print(f"Error ECMWF: {e}")

    try:
        H_gefs = Herbie(date=run_date, fxx=fxx, model='gefs', product='atmos.25', member='mean')
        ds_gefs = H_gefs.xarray('APCP')
        tp_gefs = standardize_coords(get_precip_var(ds_gefs))
        if 'number' in tp_gefs.dims: tp_gefs = tp_gefs.mean(dim='number')
        gefs_interp = tp_gefs.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
        models_out['GEFS'] = np.clip(np.nan_to_num(gefs_interp.values, nan=0.0), 0, None)
    except Exception as e:
        print(f"Error GEFS: {e}")

    return models_out

data_target = fetch_ensemble_data(ore_target)
data_start = fetch_ensemble_data(ore_start) if ore_start > 0 else {m: np.zeros((350, 350)) for m in data_target}

models_interval = {}
active_weights = {}

for m in data_target:
    if m in data_start:
        models_interval[m] = np.clip(data_target[m] - data_start[m], 0, None)
        active_weights[m] = w_ecm if m == 'ECMWF' else w_gefs

if active_weights:
    tot_w = sum(active_weights.values())
    norm_w = {m: active_weights[m]/tot_w for m in active_weights}
    precip_weighted_ens = sum(norm_w[m] * models_interval[m] for m in models_interval)

    levels = [0, 0.2, 1, 3, 5, 8, 10, 15, 25, 40, 60, 100, 150]
    colors = ['#ffffff', '#e0f7fa', '#80deea', '#29b6f6', '#0288d1', '#1565c0', '#00c832', '#ffff00', '#ff9600', '#ff0000', '#c80032', '#a00064']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    fig = plt.figure(figsize=(11, 11), dpi=120)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8)
    ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', facecolor='none'), edgecolor='gray', linewidth=0.5, linestyle=':')

    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
    cf = ax.contourf(lon_grid, lat_grid, precip_weighted_ens, levels=levels, cmap=cmap, norm=norm, extend='max', transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85, ticks=levels, aspect=30)
    cbar.set_label(f'Precipitazione Media Ensemble 24h (+{ore_start+1}h -> +{ore_target}h) [mm]', fontsize=10, fontweight='bold')
    
    plt.title(f'MULTI-MODEL ENSEMBLE MEAN - ITALIA\nRun: {run_date} UTC', fontsize=10, fontweight='bold', pad=12)
    ax.text(0.99, 0.01, 'Elab. & grafica Delfry', transform=ax.transAxes, fontsize=8, fontweight='bold', color='#333333', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='none'))

    plt.savefig(f'mappa_ensemble_intervallo_{ore_start+1}_{ore_target}h.png', bbox_inches='tight', dpi=150)
    plt.close()
