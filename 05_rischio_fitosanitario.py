import os
import ssl
import warnings
import matplotlib
matplotlib.use('Agg')  # Modalità headless per GitHub Actions / Server

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

print("=== AVVIO SCRIPT 05: RISCHIO FITOSANITARIO E BILANCIO IDRICO (GFS) ===")

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
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))

    # CRUCIALE: Ordina sempre lat e lon in senso crescente per evitare problemi di interpolazione
    if 'latitude' in da.coords:
        da = da.sortby('latitude')
    if 'longitude' in da.coords:
        da = da.sortby('longitude')
        
    return da

# Area geografica: Centro Italia (Focus Toscana, Umbria, Lazio, Marche, Abruzzo)
lon_min, lon_max = 10.0, 14.0
lat_min, lat_max = 41.0, 44.2

now_utc = pd.Timestamp.now(tz='UTC')
run_date_str = now_utc.strftime('%Y-%m-%d 00:00')

grid_lon = np.linspace(lon_min, lon_max, 350)
grid_lat = np.linspace(lat_min, lat_max, 350)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

# =========================================================
# 1. CARICAMENTO DATI METEO GFS CON HERBIE
# =========================================================
try:
    print("Download dati GFS per modellistica agrometeorologica...")
    H_gfs = Herbie(date=run_date_str, fxx=24, model='gfs', product='pgrb2.0p25')

    # Temperatura 2m (°C)
    ds_tmp = H_gfs.xarray('TMP:2 m above ground')
    da_tmp = standardize_coords(ds_tmp['t2m'] if 't2m' in ds_tmp else ds_tmp[list(ds_tmp.data_vars)[0]]) - 273.15
    tmp_interp = da_tmp.interp(longitude=grid_lon, latitude=grid_lat).values

    # Umidità Relativa 2m (%)
    ds_rh = H_gfs.xarray('RH:2 m above ground')
    da_rh = standardize_coords(ds_rh['r2'] if 'r2' in ds_rh else ds_rh[list(ds_rh.data_vars)[0]])
    rh_interp = da_rh.interp(longitude=grid_lon, latitude=grid_lat).values

    # Precipitazione Cumulata 24h (mm)
    ds_pr = H_gfs.xarray('APCP:surface')
    da_pr = standardize_coords(ds_pr['tp'] if 'tp' in ds_pr else ds_pr[list(ds_pr.data_vars)[0]])
    pr_interp = da_pr.interp(longitude=grid_lon, latitude=grid_lat).values
    pr_interp = np.clip(pr_interp, 0, None)  # Evita valori negativi dovuti all'interpolazione

    # Accumulo ore di bagnatura fogliare (f06 -> f24)
    lwd_total = np.zeros((350, 350))
    for f_hour in [6, 12, 18, 24]:
        H_t = Herbie(date=run_date_str, fxx=f_hour, model='gfs', product='pgrb2.0p25')
        ds_r = H_t.xarray('RH:2 m above ground')
        da_r = standardize_coords(ds_r['r2'] if 'r2' in ds_r else ds_r[list(ds_r.data_vars)[0]])
        r_val = da_r.interp(longitude=grid_lon, latitude=grid_lat).values
        lwd_total += np.where(r_val >= 85, 6.0, 0.0)

except Exception as e:
    print(f" Errore download GFS: {e}. Attivazione dati fallback...")
    tmp_interp = np.full((350, 350), 22.0)
    rh_interp = np.full((350, 350), 70.0)
    pr_interp = np.full((350, 350), 2.0)
    lwd_total = np.full((350, 350), 12.0)

# =========================================================
# 2. CALCOLO INDICI AGRO-CLIMATICI
# =========================================================
# Indice Rischio Oidio (%)
oidium_risk = np.where(
    (tmp_interp >= 15) & (tmp_interp <= 32),
    np.clip((rh_interp - 40) * 1.5 * (1 - np.abs(tmp_interp - 23.5) / 10.0), 0, 100),
    0.0
)

# Stima Evapotraspirazione di Riferimento ETo (Formula semplificata Hargreaves-Samani in mm/giorno)
eto_grid = np.clip(0.0023 * (tmp_interp + 17.8) * np.sqrt(np.maximum(1.0, 12.0)) * 15.0, 1.0, 8.0)

# Bilancio Idrico Cumulato 24h (P - ETo)
water_balance = pr_interp - eto_grid

# =========================================================
# 3. GRAFICA E MAPPATURA CARTOPY
# =========================================================
provinces_feature = cfeature.NaturalEarthFeature(
    category='cultural', name='admin_1_states_provinces', scale='10m', facecolor='none'
)

fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=150, subplot_kw={'projection': ccrs.PlateCarree()})

def format_map_base(ax, title):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    water_color = '#e6f2ff'
    
    # Sfondo Terra (zorder 1)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#ffffff', zorder=1)
    
    # Grigliata coordinate (zorder 10)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle=':', zorder=10)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    # Confini, Mare, Laghi e Coste sopra i dati colorati (zorder da 12 a 15)
    ax.add_feature(provinces_feature, edgecolor='#444444', linewidth=0.7, linestyle='--', alpha=0.85, zorder=12)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=1.2, edgecolor='black', zorder=13)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=water_color, zorder=14)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=water_color, edgecolor='#666666', linewidth=0.6, zorder=14)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.1, edgecolor='black', zorder=15)
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

def add_signature(ax):
    ax.text(0.99, 0.01, 'Elab. & grafica Delfry', transform=ax.transAxes,
            fontsize=8, fontweight='bold', color='black', ha='right', va='bottom', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='none'))

# ---------------------------------------------------------
# MAPPA 1: BILANCIO IDRICO CUMULATO 24H (P - ETo)
# ---------------------------------------------------------
format_map_base(axes[0], f'1. BILANCIO IDRICO CUMULATO 24H (P - ET0)\n[Deficit / Surplus Idrico in mm]')

# Configurazione Mappa Colori con Gestione del Fuori Scala
cmap_bal = plt.cm.get_cmap('RdYlBu').copy()
cmap_bal.set_over('#000066')   # Blu scuro per piogge/surplus > +8 mm
cmap_bal.set_under('#800000')  # Rosso scuro per deficit > -8 mm

levels_bal = np.linspace(-8, 8, 17)
norm_bal = mcolors.BoundaryNorm(levels_bal, cmap_bal.N)

cf1 = axes[0].contourf(
    lon_grid, lat_grid, water_balance,
    levels=levels_bal,
    cmap=cmap_bal,
    norm=norm_bal,
    extend='both',  # Evita aree bianche per accumuli di pioggia superiori a +8 mm
    alpha=0.85,
    zorder=3
)

cs1 = axes[0].contour(
    lon_grid, lat_grid, water_balance,
    levels=[0], colors='black', linewidths=1.0, linestyles='--', zorder=8
)

cbar1 = plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.06, shrink=0.85, extend='both', ticks=np.arange(-8, 9, 2))
cbar1.set_label('Bilancio Idrico (mm/giorno) [Rosso = Deficit | Blu = Surplus]', fontsize=9, fontweight='bold')
add_signature(axes[0])

# ---------------------------------------------------------
# MAPPA 2: INDICE RISCHIO OIDIO
# ---------------------------------------------------------
format_map_base(axes[1], f'2. INDICE DI RISCHIO OIDIO (Uncinula necator)\n[Toscana, Umbria, Lazio]')

cmap_oid = plt.cm.get_cmap('YlOrBr').copy()
cmap_oid.set_over('#400000')

levels_oid = np.linspace(0, 100, 11)
cf2 = axes[1].contourf(
    lon_grid, lat_grid, oidium_risk,
    levels=levels_oid,
    cmap=cmap_oid,
    extend='max',
    alpha=0.85,
    zorder=3
)

cbar2 = plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.06, shrink=0.85, extend='max')
cbar2.set_label('Indice Potenziale Rischio Oidio (%)', fontsize=9, fontweight='bold')
add_signature(axes[1])

# Layout globale e salvataggio
plt.suptitle(f"MONITORAGGIO AGROMETEO & FITOSANITARIO - CENTRO ITALIA\nRun Meteo: {run_date_str} UTC", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])

output_filename = 'mappe_bilancio_idrico_fitosanitario.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=150)
plt.close()

print(f" MAPPA SALVATA CON SUCCESSO SENZA AREA FUORI SCALA: {output_filename}")
