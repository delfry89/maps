import os
import ssl
import warnings
import matplotlib
matplotlib.use('Agg')

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

print("=== AVVIO SCRIPT 06: BILANCIO IDRICO & STRESS TRASPIRATIVO (VPD) ===")

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

lon_min, lon_max = 10.0, 14.0
lat_min, lat_max = 41.0, 44.2

now_utc = pd.Timestamp.now(tz='UTC')
run_date_str = now_utc.strftime('%Y-%m-%d 00:00')

grid_lon = np.linspace(lon_min, lon_max, 350)
grid_lat = np.linspace(lat_min, lat_max, 350)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

# =========================================================
# 1. CARICAMENTO DATI GFS
# =========================================================
try:
    print("Download dati GFS per bilancio idrico e VPD...")
    # Precipitazione cumulata 24h
    pcp_total = None
    for f_hour in [6, 12, 18, 24]:
        H_t = Herbie(date=run_date_str, fxx=f_hour, model='gfs', product='pgrb2.0p25')
        ds_p = H_t.xarray('APCP')
        da_p = standardize_coords(ds_p['apcp'] if 'apcp' in ds_p else ds_p[list(ds_p.data_vars)[0]])
        p_val = da_p.interp(longitude=grid_lon, latitude=grid_lat).values
        pcp_total = np.nan_to_num(p_val) if pcp_total is None else pcp_total + np.nan_to_num(p_val)

    # Parametri per ET0 e VPD
    H_gfs = Herbie(date=run_date_str, fxx=24, model='gfs', product='pgrb2.0p25')
    ds_tmp = H_gfs.xarray('TMP:2 m above ground')
    da_tmp = standardize_coords(ds_tmp['t2m'] if 't2m' in ds_tmp else ds_tmp[list(ds_tmp.data_vars)[0]]) - 273.15
    tmp_interp = da_tmp.interp(longitude=grid_lon, latitude=grid_lat).values

    ds_rh = H_gfs.xarray('RH:2 m above ground')
    da_rh = standardize_coords(ds_rh['r2'] if 'r2' in ds_rh else ds_rh[list(ds_rh.data_vars)[0]])
    rh_interp = da_rh.interp(longitude=grid_lon, latitude=grid_lat).values

    ds_u = H_gfs.xarray('UGRD:10 m above ground')
    ds_v = H_gfs.xarray('VGRD:10 m above ground')
    da_u = standardize_coords(ds_u['u10'] if 'u10' in ds_u else ds_u[list(ds_u.data_vars)[0]])
    da_v = standardize_coords(ds_v['v10'] if 'v10' in ds_v else ds_v[list(ds_v.data_vars)[0]])
    wind_interp = np.sqrt(da_u.interp(longitude=grid_lon, latitude=grid_lat).values**2 + 
                          da_v.interp(longitude=grid_lon, latitude=grid_lat).values**2)

except Exception as e:
    print(f"⚠️ Errore download GFS: {e}. Attivazione fallback...")
    pcp_total = np.zeros((350, 350))
    tmp_interp = np.full((350, 350), 25.0)
    rh_interp = np.full((350, 350), 45.0)
    wind_interp = np.full((350, 350), 3.0)

# =========================================================
# 2. CALCOLO BILANCIO IDRICO ($P - ET_0$) E VPD
# =========================================================
# ET0 (Hargreaves/Samani)
et0_grid = np.clip(0.16 * (tmp_interp + 10) * np.sqrt(np.clip(wind_interp, 0.5, 12)), 0, 10)

# Bilancio Idrico 24h
water_balance = pcp_total - et0_grid

# Deficit Pressione Vapore (VPD in kPa)
# Tensione di vapore saturo (es) e attuale (ea)
es = 0.61078 * np.exp((17.27 * tmp_interp) / (tmp_interp + 237.3))
ea = es * (rh_interp / 100.0)
vpd_grid = np.clip(es - ea, 0, 5.0)

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

# Riquadro 1: Bilancio Idrico
format_map_base(axes[0], '1. BILANCIO IDRICO CUMULATO 24H (P - ET0)\n[Deficit / Surplus Idrico in mm]')
cf1 = axes[0].contourf(lon_grid, lat_grid, water_balance, levels=np.linspace(-8, 8, 17), cmap='RdYlBu', alpha=0.85, zorder=2)
cbar1 = plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.06, shrink=0.85)
cbar1.set_label('Bilancio Idrico (mm/giorno) [Rosso = Deficit | Blu = Surplus]', fontsize=9, fontweight='bold')
add_signature(axes[0])

# Riquadro 2: VPD Stress Traspirativo
format_map_base(axes[1], '2. DEFICIT PRESSIONE VAPORE (VPD)\n[Stress Stomatico e Traspirativo della Vite]')
cf2 = axes[1].contourf(lon_grid, lat_grid, vpd_grid, levels=np.linspace(0, 3.0, 13), cmap='YlOrRd', alpha=0.85, zorder=2)
cs2 = axes[1].contour(lon_grid, lat_grid, vpd_grid, levels=[1.5, 2.0, 2.5], colors='purple', linewidths=1.2, linestyles='--', zorder=8)
axes[1].clabel(cs2, inline=True, fmt='%.1f kPa', fontsize=8, colors='purple', zorder=9)
cbar2 = plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.06, shrink=0.85)
cbar2.set_label('VPD (kPa) [> 2.0 kPa = Chiusura Stomatica / Stress]', fontsize=9, fontweight='bold')
add_signature(axes[1])

plt.suptitle(f"BILANCIO IDRICO & STRESS FISIOLOGICO VITE - CENTRO ITALIA\nRun Meteo: {run_date_str} UTC", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])

output_filename = 'mappe_bilancio_idrico.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=150)
plt.close()
print(f"✅ MAPPA BILANCIO IDRICO GENERATA: {output_filename}")
