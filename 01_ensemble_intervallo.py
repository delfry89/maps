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
import matplotlib.plt as plt
import numpy as np
import pandas as pd
import xarray as xr

# =========================================================
# GESTIONE WARNING E OPZIONI XARRAY
# =========================================================
xr.set_options(use_new_combine_kwarg_defaults=True)
warnings.filterwarnings('ignore', category=FutureWarning, module='cfgrib')
warnings.filterwarnings('ignore', category=FutureWarning, module='xarray')
warnings.filterwarnings('ignore')

# =========================================================
# 1. CONFIGURAZIONE PARAMETRI E PESI ENSEMBLE
# =========================================================
w_ecm = 0.50   # 50% ECMWF Ensemble Mean
w_gefs = 0.25  # 25% GEFS Mean
w_icon = 0.25  # 25% ICON-EPS Mean

# Run automatico 00:00 UTC di oggi
run_date_dt = pd.Timestamp.now(tz='UTC').floor('D')
run_date = run_date_dt.strftime('%Y-%m-%d 00:00')
date_str_dwd = run_date_dt.strftime('%Y%m%d00')

# Dominio Italia
lon_min, lon_max = 6.0, 19.0
lat_min, lat_max = 35.5, 47.5

grid_lon = np.linspace(lon_min, lon_max, 350)
grid_lat = np.linspace(lat_min, lat_max, 350)

# DEFINIZIONE DELLE FINESTRE TEMPORALI TARGET
intervalli = [
    {"ore_start": 0,  "ore_target": 24, "label": "01 - 24 ore"},
    {"ore_start": 24, "ore_target": 48, "label": "25 - 48 ore"},
    {"ore_start": 48, "ore_target": 72, "label": "49 - 72 ore"}
]


# =========================================================
# FUNZIONI DI UTILITÀ
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


# =========================================================
# 2. FUNZIONE SCARICO E INTERPOLAZIONE MODELLI ENSEMBLE
# =========================================================
def fetch_ensemble_data(fxx):
    print(f"   -> Download dati Ensemble per la scadenza +{fxx}h...")
    models_out = {}

    # --- A. ECMWF-ENS (IFS Ensemble Mean) ---
    try:
        H_ecm_ens = Herbie(date=run_date, fxx=fxx, model='ifs', product='enfo', member='mean')
        ds_ecm_ens = H_ecm_ens.xarray('tp')
        tp_ecm_ens = get_precip_var(ds_ecm_ens)
        if 'number' in tp_ecm_ens.dims:
            tp_ecm_ens = tp_ecm_ens.mean(dim='number')
        if tp_ecm_ens.max() < 2.0:
            tp_ecm_ens = tp_ecm_ens * 1000.0  # converti da metri a mm
        tp_ecm_ens = standardize_coords(tp_ecm_ens)
        ecm_interp = tp_ecm_ens.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
        models_out['ECMWF'] = np.clip(np.nan_to_num(ecm_interp.values, nan=0.0), 0, None)
        print(f"      ✅ ECMWF-ENS (+{fxx}h) caricato con successo")
    except Exception as e:
        print(f"      ⚠️ Errore ECMWF-ENS (+{fxx}h): {e}")

    # --- B. GEFS (GFS Ensemble Mean) ---
    try:
        H_gefs = Herbie(date=run_date, fxx=fxx, model='gefs', product='atmos.25', member='mean')
        ds_gefs = H_gefs.xarray('APCP')
        tp_gefs = get_precip_var(ds_gefs)
        if 'number' in tp_gefs.dims:
            tp_gefs = tp_gefs.mean(dim='number')
        tp_gefs = standardize_coords(tp_gefs)
        gefs_interp = tp_gefs.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
        models_out['GEFS'] = np.clip(np.nan_to_num(gefs_interp.values, nan=0.0), 0, None)
        print(f"      ✅ GEFS (+{fxx}h) caricato con successo")
    except Exception as e:
        print(f"      ⚠️ Errore GEFS (+{fxx}h): {e}")

    # --- C. ICON-EPS ---
    icon_loaded = False
    try:
        H_icon_eps = Herbie(date=run_date, fxx=fxx, model='icon', product='ens-global')
        ds_icon_eps = H_icon_eps.xarray('TOT_PREC')
        tp_icon_eps = get_precip_var(ds_icon_eps)
        if 'number' in tp_icon_eps.dims:
            tp_icon_eps = tp_icon_eps.mean(dim='number')
        tp_icon_eps = standardize_coords(tp_icon_eps)
        icon_interp = tp_icon_eps.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
        models_out['ICON'] = np.clip(np.nan_to_num(icon_interp.values, nan=0.0), 0, None)
        icon_loaded = True
        print(f"      ✅ ICON-EPS (+{fxx}h) caricato via Herbie")
    except Exception:
        pass

    # Fallback su DWD OpenData se Herbie non recupera ICON-EPS
    if not icon_loaded:
        fxx_str = f"{fxx:03d}"
        filename_bz2 = f"icon-eu-eps_europe_regular-lat-lon_single-level_{date_str_dwd}_{fxx_str}_TOT_PREC.grib2.bz2"
        icon_url = f"https://opendata.dwd.de/weather/nwp/icon-eu-eps/grib/00/tot_prec/{filename_bz2}"

        tmp_bz2 = os.path.join(tempfile.gettempdir(), f"icon_eps_{fxx}h.grib2.bz2")
        tmp_grib = os.path.join(tempfile.gettempdir(), f"icon_eps_{fxx}h.grib2")

        try:
            urllib.request.urlretrieve(icon_url, tmp_bz2)
            with bz2.open(tmp_bz2, 'rb') as f_in:
                with open(tmp_grib, 'wb') as f_out:
                    f_out.write(f_in.read())

            ds_icon = xr.open_dataset(tmp_grib, engine='cfgrib', backend_kwargs={'filter_by_keys': {}})
            tp_icon = get_precip_var(ds_icon)
            if 'number' in tp_icon.dims:
                tp_icon = tp_icon.mean(dim='number')
            tp_icon = standardize_coords(tp_icon)
            icon_interp = tp_icon.interp(longitude=grid_lon, latitude=grid_lat, method='linear')
            models_out['ICON'] = np.clip(np.nan_to_num(icon_interp.values, nan=0.0), 0, None)
            icon_loaded = True
            print(f"      ✅ ICON-EPS (+{fxx}h) caricato via DWD OpenData")
        except Exception as e:
            print(f"      ⚠️ Errore ICON-EPS DWD (+{fxx}h): {e}")
            if 'ECMWF' in models_out and 'GEFS' in models_out:
                models_out['ICON'] = (models_out['ECMWF'] + models_out['GEFS']) / 2.0
            elif 'ECMWF' in models_out:
                models_out['ICON'] = models_out['ECMWF']
        finally:
            for f in [tmp_bz2, tmp_grib]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    return models_out


# =========================================================
# 3. DOWNLOAD PREVENTIVO DEI TARGETS
# =========================================================
print(f"--- ENSEMBLE MULTI-MODEL | Run {run_date} UTC ---")

# Troviamo tutte le scadenze necessarie senza duplicati (es. 0, 24, 48, 72)
ore_da_scaricare = sorted(list(set([i["ore_start"] for i in intervalli] + [i["ore_target"] for i in intervalli])))
all_data = {}

for fxx in ore_da_scaricare:
    if fxx == 0:
        # A +0h l'accumulo è pari a zero
        all_data[0] = {m: np.zeros((350, 350)) for m in ['ECMWF', 'GEFS', 'ICON']}
    else:
        print(f"\nScaricamento dati per la scadenza cumulata +{fxx}h...")
        all_data[fxx] = fetch_ensemble_data(fxx)


# =========================================================
# 4. LEGENDA COLORI E GRAFICA COMMODITY
# =========================================================
levels = [0, 0.2, 1, 3, 5, 8, 10, 15, 25, 40, 60, 100, 150]
colors = [
    '#ffffff', '#e0f7fa', '#80deea', '#29b6f6', '#0288d1', '#1565c0',
    '#00c832', '#ffff00', '#ff9600', '#ff0000', '#c80032', '#a00064'
]

cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(levels, cmap.N)

region_boundaries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none'
)


# =========================================================
# 5. GENERAZIONE MAPPE PER I 3 INTERVALLI
# =========================================================
for item in intervalli:
    start_h = item["ore_start"]
    target_h = item["ore_target"]
    label_time = item["label"]
    
    print(f"\nElaborazione Mappa Intervallo +{start_h+1}h ➔ +{target_h}h ({label_time})...")

    data_start = all_data[start_h]
    data_target = all_data[target_h]

    models_interval = {}
    weights_map = {'ECMWF': w_ecm, 'GEFS': w_gefs, 'ICON': w_icon}
    active_weights = {}

    # Sottrarre il valore di inizio intervallo per ottenere la cumulata netta
    for m in data_target:
        if m in data_start:
            diff = data_target[m] - data_start[m]
            models_interval[m] = np.clip(diff, 0, None)
            if m in weights_map:
                active_weights[m] = weights_map[m]

    if not models_interval:
        print(f"⚠️ Dati insufficienti per calcolare l'intervallo {label_time}, salto...")
        continue

    tot_w = sum(active_weights.values())
    norm_w = {m: active_weights[m] / tot_w for m in active_weights}

    precip_weighted_ens = sum(norm_w[m] * models_interval[m] for m in models_interval)
    info_pesi_str = [f"{m}-ENS ({int(norm_w[m] * 100)}%)" for m in models_interval]
    title_pesi = ' | '.join(info_pesi_str)

    # TRACCIAMENTO MAPPA
    fig = plt.figure(figsize=(11, 11), dpi=120)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=0.3)
    ax.add_feature(region_boundaries, edgecolor='gray', linewidth=0.5, linestyle=':')

    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

    cf = ax.contourf(
        lon_grid, lat_grid, precip_weighted_ens,
        levels=levels, cmap=cmap, norm=norm, extend='max',
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(
        cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85,
        ticks=levels, aspect=30
    )
    cbar.set_label(
        f"Precipitazione Media d'Ensemble ({label_time}) [mm]",
        fontsize=10, fontweight='bold'
    )
    cbar.ax.tick_params(labelsize=8)

    plt.title(
        f"MULTI-MODEL ENSEMBLE MEAN (ECMWF-ENS / GEFS / ICON-EPS) - ITALIA\n"
        f"Cumulata Intervallo: +{start_h+1}h ➔ +{target_h}h | Run: {run_date} UTC\nPesi: {title_pesi}",
        fontsize=10, fontweight='bold', pad=12
    )

    ax.text(
        0.99, 0.01, 'Elab. & grafica Delfry', transform=ax.transAxes,
        fontsize=8, fontweight='bold', color='#333333', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='none')
    )

    output_image_path = f'mappa_ensemble_intervallo_{target_h}h.png'
    plt.savefig(output_image_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"🖼️ Mappa salvata con successo come: {output_image_path}")

print("\n🎉 Elaborazione completata! Salvate le 3 mappe ensemble per 24h, 48h e 72h.")
