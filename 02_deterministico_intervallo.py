import bz2
from datetime import datetime
import os
import tempfile
import urllib.request
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Silenzia warning
xr.set_options(use_new_combine_kwarg_defaults=True)
warnings.filterwarnings('ignore')

# =========================================================
# 1. CONFIGURAZIONE PARAMETRI E PESI
# =========================================================
peso_ECMWF = 0.50
peso_GFS = 0.25
peso_ICON = 0.25

ore_target = 24       # Scadenza finale (es. 24, 48, 72)
finestra_ore = 24     # Ampiezza intervallo in ore
ore_start = max(0, ore_target - finestra_ore)

# Run automatico 00:00 UTC di oggi
run_date_dt = pd.Timestamp.now(tz='UTC').floor('D')
run_date = run_date_dt.strftime('%Y-%m-%d 00:00')
date_str_dwd = run_date_dt.strftime('%Y%m%d00')

# Dominio Italia
lon_min, lon_max = 6.0, 19.0
lat_min, lat_max = 35.5, 47.5

grid_lon = np.linspace(lon_min, lon_max, 350)
grid_lat = np.linspace(lat_min, lat_max, 350)


# =========================================================
# FUNZIONI UTILI
# =========================================================
def get_precip_var(ds):
    for var in ['tp', 'apcp', 'precip', 'APCP', 'tot_prec', 'TOT_PREC', 'unknown', 'tp_acc']:
        if var in ds.data_vars:
            return ds[var]
    keys = list(ds.data_vars.keys())
    return ds[keys[0]]


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


def fetch_models(fxx):
    data = {}

    # 1. ECMWF IFS
    try:
        H_ecm = Herbie(date=run_date, fxx=fxx, model='ifs', product='oper')
        ds_ecm = H_ecm.xarray('tp')
        tp_ecm = standardize_coords(get_precip_var(ds_ecm))
        if tp_ecm.max() < 2.0:
            tp_ecm = tp_ecm * 1000.0
        ecm_interp = tp_ecm.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
        data['ECMWF'] = np.clip(np.nan_to_num(ecm_interp.values, nan=0.0), 0, None)
    except Exception as e:
        print(f"  ⚠️ Errore ECMWF (+{fxx}h): {e}")

    # 2. GFS
    try:
        H_gfs = Herbie(date=run_date, fxx=fxx, model='gfs', product='pgrb2.0p25')
        ds_gfs = H_gfs.xarray('APCP')
        tp_gfs = standardize_coords(get_precip_var(ds_gfs))
        gfs_interp = tp_gfs.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
        data['GFS'] = np.clip(np.nan_to_num(gfs_interp.values, nan=0.0), 0, None)
    except Exception as e:
        print(f"  ⚠️ Errore GFS (+{fxx}h): {e}")

    # 3. ICON (Herbie con Fallback su OpenData DWD)
    icon_loaded = False
    try:
        H_icon = Herbie(date=run_date, fxx=fxx, model='icon', product='global')
        ds_icon = H_icon.xarray('TOT_PREC')
        tp_icon = standardize_coords(get_precip_var(ds_icon))
        icon_interp = tp_icon.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
        data['ICON'] = np.clip(np.nan_to_num(icon_interp.values, nan=0.0), 0, None)
        icon_loaded = True
    except Exception:
        pass

    # Fallback DWD OpenData se Herbie non recupera ICON
    if not icon_loaded:
        fxx_str = f"{fxx:03d}"
        filename_bz2 = f"icon-eu_europe_regular-lat-lon_single-level_{date_str_dwd}_{fxx_str}_TOT_PREC.grib2.bz2"
        icon_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/tot_prec/{filename_bz2}"

        tmp_bz2 = os.path.join(tempfile.gettempdir(), f"icon_{fxx}h.grib2.bz2")
        tmp_grib = os.path.join(tempfile.gettempdir(), f"icon_{fxx}h.grib2")

        try:
            urllib.request.urlretrieve(icon_url, tmp_bz2)
            with bz2.open(tmp_bz2, 'rb') as f_in:
                with open(tmp_grib, 'wb') as f_out:
                    f_out.write(f_in.read())

            ds_icon = xr.open_dataset(tmp_grib, engine='cfgrib', backend_kwargs={'filter_by_keys': {}})
            tp_icon = standardize_coords(get_precip_var(ds_icon))
            icon_interp = tp_icon.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
            data['ICON'] = np.clip(np.nan_to_num(icon_interp.values, nan=0.0), 0, None)
        except Exception as e:
            print(f"  ⚠️ Errore ICON DWD (+{fxx}h): {e}")
        finally:
            for f in [tmp_bz2, tmp_grib]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    return data


# =========================================================
# 2. DOWNLOAD E SOTTRAZIONE INTERVALLO
# =========================================================
print(f"--- RUN {run_date} UTC | Calcolo Intervallo: +{ore_start+1}h -> +{ore_target}h ---")

data_target = fetch_models(ore_target)
data_start = fetch_models(ore_start) if ore_start > 0 else {m: np.zeros((350, 350)) for m in data_target}

models_interval_data = {}
weights_map = {'ECMWF': peso_ECMWF, 'GFS': peso_GFS, 'ICON': peso_ICON}
active_weights = {}

for m in data_target:
    if m in data_start:
        models_interval_data[m] = np.clip(data_target[m] - data_start[m], 0, None)
        if m in weights_map:
            active_weights[m] = weights_map[m]

if not models_interval_data:
    raise RuntimeError("Impossibile scaricare o calcolare l'intervallo per nessun modello.")

# =========================================================
# 3. PONDERAZIONE MULTI-MODEL
# =========================================================
tot_w = sum(active_weights.values())
precip_weighted = sum((active_weights[m] / tot_w) * models_interval_data[m] for m in models_interval_data)

info_pesi = [f"{m} ({int((active_weights[m]/tot_w)*100)}%)" for m in models_interval_data]
title_pesi = " | ".join(info_pesi)

# =========================================================
# 4. ELABORAZIONE GRAFICA
# =========================================================
levels = [0, 0.2, 1, 3, 5, 8, 10, 15, 25, 40, 60, 100, 150]
colors = ['#ffffff', '#e0f7fa', '#80deea', '#29b6f6', '#0288d1', '#1565c0', '#00c832', '#ffff00', '#ff9600', '#ff0000', '#c80032', '#a00064']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(levels, cmap.N)

fig = plt.figure(figsize=(11, 11), dpi=120)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8)
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=0.3)
ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', facecolor='none'), edgecolor='gray', linewidth=0.5, linestyle=':')

lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
cf = ax.contourf(lon_grid, lat_grid, precip_weighted, levels=levels, cmap=cmap, norm=norm, extend='max', transform=ccrs.PlateCarree())

cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85, ticks=levels, aspect=30)
cbar.set_label(f'Precipitazione Cumulata 24h (+{ore_start+1}h ➔ +{ore_target}h) [mm]', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

plt.title(f'MULTI-MODEL DETERMINISTICO - ITALIA\nRun: {run_date} UTC | Pesi: {title_pesi}', fontsize=10, fontweight='bold', pad=12)
ax.text(0.99, 0.01, 'Elab. & grafica Delfry', transform=ax.transAxes, fontsize=8, fontweight='bold', color='black', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

output_path = f'mappa_intervallo_{ore_start+1}_{ore_target}h.png'
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()

print(f"🖼️ Mappa salvata con successo come: {output_path}")
