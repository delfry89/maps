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
# 0. CONFIGURAZIONE E SSL
# =========================================================
warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=DownloadWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
xr.set_options(use_new_combine_kwarg_defaults=True)

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

print("=== AVVIO SCRIPT 05: RISCHIO FITOSANITARIO (OIDIO & BAGNATURA FOGLIARE) ===")

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

# Area geografica: Centro Italia (Focus Siena/Toscana/Umbria/Lazio)
lon_min, lon_max = 10.0, 14.0
lat_min, lat_max = 41.0, 44.2

now_utc = pd.Timestamp.now(tz='UTC')
run_date_str = now_utc.strftime('%Y-%m-%d 00:00')

grid_lon = np.linspace(lon_min, lon_max, 350)
grid_lat = np.linspace(lat_min, lat_max, 350)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

# =========================================================
# 1. CARICAMENTO DATI METEO GFS
# =========================================================
try:
    print("Download dati GFS per modellistica fitosanitaria...")
    H_gfs = Herbie(date=run_date_str, fxx=24, model='gfs', product='pgrb2.0p25')

    ds_tmp = H_gfs.xarray('TMP:2 m above ground')
    da_tmp = standardize_coords(ds_tmp['t2m'] if 't2m' in ds_tmp else ds_tmp[list(ds_tmp.data_vars)[0]]) - 273.15
    tmp_interp = da_tmp.interp(longitude=grid_lon, latitude=grid_lat).values

    ds_rh = H_gfs.xarray('RH:2 m above ground')
    da_rh = standardize_coords(ds_rh['r2'] if 'r2' in ds_rh else ds_rh[list(ds_rh.data_vars)[0]])
    rh_interp = da_rh.interp(longitude=grid_lon, latitude=grid_lat).values

    # Stima ore di bagnatura fogliare accumulando gli intervalli f06-f24 con RH > 85%
    lwd_total = np.zeros((350, 350))
    for f_hour in [6, 12, 18, 24]:
        H_t = Herbie(date=run_date_str, fxx=f_hour, model='gfs', product='pgrb2.0p25')
        ds_r = H_t.xarray('RH:2 m above ground')
        da_r = standardize_coords(ds_r['r2'] if 'r2' in ds_r else ds_r[list(ds_r.data_vars)[0]])
        r_val = da_r.interp(longitude=grid_lon, latitude=grid_lat).values
        # Se RH > 85%, accumula 6 ore di probabile bagnatura
        lwd_total += np.where(r_val >= 85, 6.0, 0.0)

except Exception as e:
    print(f"⚠️ Errore download GFS: {e}. Attivazione fallback...")
    tmp_interp = np.full((350, 350), 22.0)
    rh_interp = np.full((350, 350), 70.0)
    lwd_total = np.full((350, 350), 12.0)

# =========================================================
# 2. CALCOLO INDICI FITOSANITARI
# =========================================================
# Indice Oidio (%): massimo nella finestra 20-27 °C con umidità moderata/alta
oidium_risk = np.where(
    (tmp_interp >= 15) & (tmp_interp <= 32),
    np.clip((rh_interp - 40) * 1.5 * (1 - np.abs(tmp_interp - 23.5) / 10.0), 0, 100),
    0.0
)

# =========================================================
# 3. GRAFICA E MAPPATURA
# =========================================================
provinces_feature = cfeature.NaturalEarthFeature(
    category='cultural', name='admin_1_states_provinces', scale='10m', facecolor='none'
)

fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=150, subplot_kw={'projection': ccrs.PlateCarree()})

def format_map_base(ax, title):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    water_color = '#e6f2ff'
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#ffffff', zorder=1)
    ax.add_feature(provinces_feature, edgecolor='#444444', linewidth=0.7, linestyle='--', alpha=0.85, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=1.2, edgecolor='black', zorder=6)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=water_color, zorder=7)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=water_color, edgecolor='#666666', linewidth=0.6, zorder=7)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.1, edgecolor='black', zorder=8)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle=':', zorder=10)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

def add_signature(ax):
    ax.text(0.99, 0.01, 'Elab. & grafica Delfry', transform=ax.transAxes,
            fontsize=8, fontweight='bold', color='black', ha='right', va='bottom', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='none'))

# Riquadro 1: Indice Oidio
format_map_base(axes[0], '1. INDICE DI RISCHIO OIDIO (Uncinula necator)\n[Toscana, Umbria, Lazio]')
cf1 = axes[0].contourf(lon_grid, lat_grid, oidium_risk, levels=np.linspace(0, 100, 11), cmap='YlOrBr', alpha=0.85, zorder=2)
cbar1 = plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.06, shrink=0.85)
cbar1.set_label('Indice Potenziale Rischio Oidio (%)', fontsize=9, fontweight='bold')
add_signature(axes[0])

# Riquadro 2: Bagnatura Fogliare
format_map_base(axes[1], '2. ORE DI BAGNATURA FOGLIARE STIMATE (24h)\n[Infezione Peronospora / Botrite]')
cf2 = axes[1].contourf(lon_grid, lat_grid, lwd_total, levels=np.arange(0, 25, 2), cmap='YlGnBu', alpha=0.85, zorder=2)
cs2 = axes[1].contour(lon_grid, lat_grid, lwd_total, levels=[6, 12, 18], colors='darkblue', linewidths=1.0, zorder=8)
axes[1].clabel(cs2, inline=True, fmt='%d h', fontsize=8, colors='darkblue', zorder=9)
cbar2 = plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.06, shrink=0.85)
cbar2.set_label('Ore di Bagnatura Fogliare Cumulate (Ore/Giorno)', fontsize=9, fontweight='bold')
add_signature(axes[1])

plt.suptitle(f"MONITORAGGIO FITOSANITARIO VITIVINICOLO - CENTRO ITALIA\nRun Meteo: {run_date_str} UTC", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])

output_filename = 'mappe_rischio_fitosanitario.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=150)
plt.close()
print(f"✅ MAPPA FITOSANITARIA GENERATA: {output_filename}")
