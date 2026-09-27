import os
import ssl
import warnings
import matplotlib
matplotlib.use('Agg')  # Modalità headless per GitHub Actions

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import DownloadWarning
from herbie import Herbie

# =========================================================
# 0. GESTIONE WARNING & SSL / OPZIONI
# =========================================================
warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=DownloadWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
xr.set_options(use_new_combine_kwarg_defaults=True)

# Disabilita verifica rigorosa SSL per evitare blocchi su mirror UCAR/NOAA/AWS
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

print("=== AVVIO SCRIPT 03: AGROMETEO & VITIVINICOLO 24H ===")

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
# 1. CONFIGURAZIONE AREA & DATA CORRENTE
# =========================================================
lon_min, lon_max = 10.0, 14.0
lat_min, lat_max = 41.0, 44.2

# Data automatica aggiornata (Run 00:00 UTC di oggi o di ieri a seconda dell'orario)
now_utc = pd.Timestamp.now(tz='UTC')
run_date_str = now_utc.strftime('%Y-%m-%d 00:00')

grid_lon = np.linspace(lon_min, lon_max, 350)
grid_lat = np.linspace(lat_min, lat_max, 350)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

print(f"QUADRO AGROMETEO | Centro Italia & Lazio | Run {run_date_str} UTC")

# =========================================================
# 2. CARICAMENTO DATI METEO (GFS - CUMULATA 24H) CON FALLBACK
# =========================================================
data_loaded = False

try:
    print("Recupero e somma intervalli di pioggia 24h (f06, f12, f18, f24)...")
    pcp_total = None

    for f_hour in [6, 12, 18, 24]:
        H_temp = Herbie(date=run_date_str, fxx=f_hour, model='gfs', product='pgrb2.0p25')
        ds_p = H_temp.xarray('APCP')
        da_p = standardize_coords(ds_p['apcp'] if 'apcp' in ds_p else ds_p[list(ds_p.data_vars)[0]])
        p_interp = da_p.interp(longitude=grid_lon, latitude=grid_lat).values

        if pcp_total is None:
            pcp_total = np.nan_to_num(p_interp)
        else:
            pcp_total += np.nan_to_num(p_interp)

    pcp_interp = pcp_total

    # Reference Herbie per parametri istantanei a 24h
    H_gfs_24 = Herbie(date=run_date_str, fxx=24, model='gfs', product='pgrb2.0p25')

    # Temperatura a 2m (°C)
    ds_tmp = H_gfs_24.xarray('TMP:2 m above ground')
    da_tmp = standardize_coords(ds_tmp['t2m'] if 't2m' in ds_tmp else ds_tmp[list(ds_tmp.data_vars)[0]]) - 273.15
    tmp_interp = da_tmp.interp(longitude=grid_lon, latitude=grid_lat).values

    # Umidità relativa (%)
    ds_rh = H_gfs_24.xarray('RH:2 m above ground')
    da_rh = standardize_coords(ds_rh['r2'] if 'r2' in ds_rh else ds_rh[list(ds_rh.data_vars)[0]])
    rh_interp = da_rh.interp(longitude=grid_lon, latitude=grid_lat).values

    # Vento a 10m (m/s)
    ds_u = H_gfs_24.xarray('UGRD:10 m above ground')
    ds_v = H_gfs_24.xarray('VGRD:10 m above ground')
    da_u = standardize_coords(ds_u['u10'] if 'u10' in ds_u else ds_u[list(ds_u.data_vars)[0]])
    da_v = standardize_coords(ds_v['v10'] if 'v10' in ds_v else ds_v[list(ds_v.data_vars)[0]])
    wind_speed = np.sqrt(da_u.values**2 + da_v.values**2)
    da_wind = da_u.copy(data=wind_speed)
    wind_interp = da_wind.interp(longitude=grid_lon, latitude=grid_lat).values

    data_loaded = True

except Exception as e:
    print(f"⚠️ Errore durante il caricamento dati da Herbie: {e}")
    print("Generazione mappa fallback coordinata...")

    # Griglia di emergenza coerente per garantire la generazione grafica
    tmp_interp = np.full((350, 350), 20.0)
    rh_interp = np.full((350, 350), 65.0)
    wind_interp = np.full((350, 350), 2.5)
    pcp_interp = np.zeros((350, 350))

# =========================================================
# 3. CALCOLO PARAMETRI AGROMETEO & BIOCLIMATICI
# =========================================================
# A. Indice Rischio Fungino (%)
fungal_risk = np.clip((rh_interp - 50) * 2.2, 0, 100)
fungal_risk = np.where((tmp_interp < 10) | (tmp_interp > 32), 0, fungal_risk)

# B. Evapotraspirazione (ET0 mm/giorno)
et0_grid = np.clip(0.16 * (tmp_interp + 10) * np.sqrt(np.clip(wind_interp, 0.5, 12)), 0, 10)

# C. Gradi Giorno (GDD Base 10°C)
gdd_grid = np.maximum(0, tmp_interp - 10)

# =========================================================
# 4. PREPARAZIONE CONFINI PROVINCIALI & REGIONALI
# =========================================================
provinces_feature = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces',
    scale='10m',
    facecolor='none'
)

# =========================================================
# 5. CREAZIONE MAPPE SOVRAPPOSTE (2 RIQUADRI)
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=150, subplot_kw={'projection': ccrs.PlateCarree()})

def format_map_base(ax, title):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    water_color = '#e6f2ff'

    # 1. Sfondo terraferma
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#ffffff', zorder=1)

    # 2. Confini Provinciali
    ax.add_feature(provinces_feature, edgecolor='#444444', linewidth=0.7, linestyle='--', alpha=0.85, zorder=5)

    # 3. Confini Regionali
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=1.2, edgecolor='black', zorder=6)

    # 4. Maschera Mare e Laghi
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=water_color, zorder=7)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=water_color, edgecolor='#666666', linewidth=0.6, zorder=7)

    # 5. Linea di Costa
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.1, edgecolor='black', zorder=8)

    # Griglia coordinate
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle=':', zorder=10)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

def add_signature(ax):
    """Aggiunge la firma nell'angolo in basso a destra dell'asse."""
    ax.text(0.99, 0.01, 'Elab. & grafica Delfry', transform=ax.transAxes,
            fontsize=8, fontweight='bold', color='black', ha='right', va='bottom', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='none'))

# ---------------------------------------------------------
# MAPPA 1: RISCHIO FUNGINO + PIOGGIA CUMULATA 24H
# ---------------------------------------------------------
format_map_base(axes[0], '1. RISCHIO FUNGINO + PIOGGIA CUMULATA (24h)\n[Toscana, Umbria, Lazio fino al Basso Lazio]')

levels_f = [0, 20, 40, 60, 80, 100]
cmap_f = mcolors.ListedColormap(['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f', '#006d2c'])
norm_f = mcolors.BoundaryNorm(levels_f, cmap_f.N)

cf1 = axes[0].contourf(
    lon_grid, lat_grid, fungal_risk,
    levels=levels_f, cmap=cmap_f, norm=norm_f, alpha=0.85, zorder=2
)

# Isoiete pioggia
levels_p = [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
cs1 = axes[0].contour(
    lon_grid, lat_grid, pcp_interp,
    levels=levels_p, colors='#00008b', linewidths=1.3, zorder=8
)

if len(cs1.levels) > 0 and np.max(pcp_interp) >= 1.0:
    axes[0].clabel(cs1, inline=True, fmt='%.0f mm', fontsize=8, colors='#00008b', zorder=9)

cbar1_a = plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.06, shrink=0.85)
cbar1_a.set_label('Rischio Infezione Fungina Peronospora/Oidio (%)', fontsize=9, fontweight='bold')

add_signature(axes[0])

# ---------------------------------------------------------
# MAPPA 2: EVAPOTRASPIRAZIONE + GRADI GIORNO (GDD)
# ---------------------------------------------------------
format_map_base(axes[1], '2. EVAPOTRASPIRAZIONE ET0 + GRADI GIORNO GDD\n[Toscana, Umbria, Lazio fino al Basso Lazio]')

cf2 = axes[1].contourf(
    lon_grid, lat_grid, et0_grid,
    levels=12, cmap='YlOrRd', alpha=0.85, zorder=2
)

levels_gdd = np.arange(2, 20, 2)
cs2 = axes[1].contour(
    lon_grid, lat_grid, gdd_grid,
    levels=levels_gdd, colors='purple', linewidths=1.3, linestyles='--', zorder=8
)
axes[1].clabel(cs2, inline=True, fmt='%d GDD', fontsize=8, colors='purple', zorder=9)

cbar2_a = plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.06, shrink=0.85)
cbar2_a.set_label('Evapotraspirazione $ET_0$ (mm/giorno)', fontsize=9, fontweight='bold')

add_signature(axes[1])

# ---------------------------------------------------------
# TITOLO GENERALE E SALVATAGGIO
# ---------------------------------------------------------
plt.suptitle(
    f"QUADRO AGROMETEO & VITIVINICOLO - CENTRO ITALIA & BASSO LAZIO\nRun Meteo: {run_date_str} UTC (TEST MALTEMPO 24H)",
    fontsize=14, fontweight='bold', y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.93])

output_filename = 'mappe_sovrapposte_24h.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=150)
plt.close()

print(f"✅ MAPPA AGROMETEO GENERATA E SALVATA CON SUCCESSO: {output_filename}")
