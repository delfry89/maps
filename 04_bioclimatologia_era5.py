import os
import sys
import zipfile
import datetime
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import DownloadWarning

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DownloadWarning)

print("=== AVVIO SCRIPT 04: AGRO-CLIMATOLOGIA DINAMICA ERA5 (4 MAPPE PERFECT) ===")

# =========================================================
# 1. CONFIGURAZIONE AREA DI STUDIO & DATE DINAMICHE
# =========================================================
lon_min, lon_max = 10.0, 14.0
lat_min, lat_max = 41.0, 44.2

grid_lon = np.linspace(lon_min, lon_max, 250)
grid_lat = np.linspace(lat_min, lat_max, 250)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

lat_mean = (lat_min + lat_max) / 2.0
k_huglin = 1.0 + (lat_mean - 40) * 0.006

now = datetime.datetime.now()
current_year = now.year
target_year = current_year if now.month >= 10 else current_year - 1

start_season = datetime.datetime(current_year, 4, 1)
if now < start_season:
    start_date_dyn = datetime.datetime(current_year - 1, 4, 1)
    end_date_dyn = datetime.datetime(current_year - 1, 9, 30)
    stato_stagione = f"Consuntivo Stagione {current_year - 1}"
else:
    start_date_dyn = start_season
    end_date_dyn = now
    stato_stagione = f"Accumulo dal 01/04 al {end_date_dyn.strftime('%d/%m/%Y')}"

giorni_stagione_completa = 183
giorni_trascorsi = max(1, (end_date_dyn - start_date_dyn).days + 1)

# =========================================================
# 2. CARICAMENTO DATI ERA5 REALI
# =========================================================
def get_climate_data(year):
    extract_dir = f"era5_extracted_{year}"
    os.makedirs(extract_dir, exist_ok=True)
    target_file = os.path.join(extract_dir, f"era5_land_{year}.nc")

    # Check se esiste già il file localmente
    if not os.path.exists(target_file):
        print(f" Richiesta dati ERA5 in formato NetCDF per {year} da Copernicus CDS...")
        try:
            import cdsapi
            cds_url = os.environ.get('CDS_URL')
            cds_key = os.environ.get('CDS_KEY')
            if cds_url and cds_key:
                with open(os.path.expanduser('~/.cdsapirc'), 'w') as f:
                    f.write(f"url: {cds_url}\nkey: {cds_key}\n")

            c = cdsapi.Client()
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': ['2m_temperature', 'maximum_2m_temperature_since_previous_post_processing'],
                    'year': str(year),
                    'month': ['04', '05', '06', '07', '08', '09'],
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': ['12:00'],
                    'area': [lat_max, lon_min, lat_min, lon_max],
                    'format': 'netcdf',  # <-- RICHIESTA IN FORMATO NETCDF
                },
                target_file
            )
        except Exception as e:
            print(f" ERRORE DOWNLOAD CDS API: {e}")
            sys.exit(1)

    print(f" Lettura dataset ERA5 reale: {target_file}")
    
    try:
        ds = xr.open_dataset(target_file)
    except Exception as e:
        print(f" Errore nell'apertura del file NetCDF: {e}")
        sys.exit(1)

    t2m_var = 't2m' if 't2m' in ds else ('2t' if '2t' in ds else list(ds.data_vars)[0])
    tmax_var = 'mx2t' if 'mx2t' in ds else ('mxt2m' if 'mxt2m' in ds else list(ds.data_vars)[0])

    t2m = ds[t2m_var] - 273.15 if ds[t2m_var].max() > 100 else ds[t2m_var]
    tmax = ds[tmax_var] - 273.15 if ds[tmax_var].max() > 100 else ds[tmax_var]

    lat_name = 'latitude' if 'latitude' in ds.coords else ('lat' if 'lat' in ds.coords else list(ds.coords)[0])
    lon_name = 'longitude' if 'longitude' in ds.coords else ('lon' if 'lon' in ds.coords else list(ds.coords)[1])

    ds = ds.sortby(lat_name).sortby(lon_name)

    t2m_mean = t2m.mean(dim='time') if 'time' in t2m.dims else t2m
    tmax_mean = tmax.mean(dim='time') if 'time' in tmax.dims else tmax

    t2m_interp = t2m_mean.interp({lon_name: grid_lon, lat_name: grid_lat}).values
    tmax_interp = tmax_mean.interp({lon_name: grid_lon, lat_name: grid_lat}).values

    return t2m_interp, tmax_interp

t2m_interp, tmax_interp = get_climate_data(target_year)

# CALCOLO INDICI
winkler_hist = np.maximum(0, t2m_interp - 10) * giorni_stagione_completa
huglin_daily = (np.maximum(0, t2m_interp - 10) + np.maximum(0, tmax_interp - 10)) / 2.0
huglin_hist = huglin_daily * giorni_stagione_completa * k_huglin

winkler_dyn = np.maximum(0, t2m_interp - 10) * giorni_trascorsi
huglin_dyn = huglin_daily * giorni_trascorsi * k_huglin

# =========================================================
# 3. GENERAZIONE GRAFICA 4 MAPPE CON BOUNDARY NORM
# =========================================================
provinces_feature = cfeature.NaturalEarthFeature(
    category='cultural', name='admin_1_states_provinces', scale='10m', facecolor='none'
)

fig, axes = plt.subplots(2, 2, figsize=(18, 18), dpi=150, subplot_kw={'projection': ccrs.PlateCarree()})

def format_map_base(ax, title):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    water_color = '#e6f2ff'
    
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#fbfbfb', zorder=1)
    ax.add_feature(provinces_feature, edgecolor='#555555', linewidth=0.7, linestyle='--', alpha=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=1.2, edgecolor='black', zorder=6)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=water_color, zorder=7)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=water_color, edgecolor='#666666', linewidth=0.6, zorder=7)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.1, edgecolor='black', zorder=8)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle=':', zorder=10)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

# --- MAPPA 1 (Alto SX): HUGLIN CONSUNTIVO ---
format_map_base(axes[0, 0], f'1. INDICE DI HUGLIN (HI) - CONSUNTIVO STAGIONALE ERA5 ({target_year})\n[Stagione Vegetativa Completa: 1 Apr - 30 Set]')
levels_h_hist = [1200, 1500, 1800, 2100, 2400, 2700, 3000]
cmap_h_hist = mcolors.ListedColormap(['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c', '#a50026'])
norm_h_hist = mcolors.BoundaryNorm(levels_h_hist, cmap_h_hist.N)

cf1 = axes[0, 0].contourf(lon_grid, lat_grid, huglin_hist, levels=levels_h_hist, cmap=cmap_h_hist, norm=norm_h_hist, extend='both', alpha=0.85, zorder=2)
cs1 = axes[0, 0].contour(lon_grid, lat_grid, huglin_hist, levels=levels_h_hist, colors='black', linewidths=0.6, zorder=4)
axes[0, 0].clabel(cs1, inline=True, fmt='%d', fontsize=7, colors='black', zorder=9)
cbar1 = plt.colorbar(cf1, ax=axes[0, 0], orientation='horizontal', pad=0.05, shrink=0.85, ticks=levels_h_hist)
cbar1.set_label('Indice di Huglin Totale (HI)', fontsize=8, fontweight='bold')

# --- MAPPA 2 (Alto DX): WINKLER CONSUNTIVO ---
format_map_base(axes[0, 1], f'2. INDICE DI WINKLER (WI / GDD) - CONSUNTIVO STAGIONALE ERA5 ({target_year})\n[Gradi Giorno Totali Stagione Completa]')
levels_w_hist = [800, 1110, 1390, 1670, 1940, 2220, 2600]
cmap_w_hist = mcolors.ListedColormap(['#edf8fb', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'])
norm_w_hist = mcolors.BoundaryNorm(levels_w_hist, cmap_w_hist.N)

cf2 = axes[0, 1].contourf(lon_grid, lat_grid, winkler_hist, levels=levels_w_hist, cmap=cmap_w_hist, norm=norm_w_hist, extend='both', alpha=0.85, zorder=2)
cs2 = axes[0, 1].contour(lon_grid, lat_grid, winkler_hist, levels=levels_w_hist, colors='#000055', linewidths=0.6, zorder=4)
axes[0, 1].clabel(cs2, inline=True, fmt='%d GDD', fontsize=7, colors='#000055', zorder=9)
cbar2 = plt.colorbar(cf2, ax=axes[0, 1], orientation='horizontal', pad=0.05, shrink=0.85, ticks=levels_w_hist)
cbar2.set_label('Gradi Giorno Totali (GDD Base 10°C)', fontsize=8, fontweight='bold')

# --- MAPPA 3 (Basso SX): HUGLIN PROGRESSIVO A OGGI ---
format_map_base(axes[1, 0], f'3. INDICE DI HUGLIN (HI) PROGRESSIVO - {current_year}\n[{stato_stagione}]')
levels_h_dyn = [0, 400, 800, 1200, 1600, 2000, 2400, 2800]
cmap_h_dyn = mcolors.ListedColormap(['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f', '#fee391', '#fec44f', '#fe9929'])
norm_h_dyn = mcolors.BoundaryNorm(levels_h_dyn, cmap_h_dyn.N)

cf3 = axes[1, 0].contourf(lon_grid, lat_grid, huglin_dyn, levels=levels_h_dyn, cmap=cmap_h_dyn, norm=norm_h_dyn, extend='both', alpha=0.85, zorder=2)
cs3 = axes[1, 0].contour(lon_grid, lat_grid, huglin_dyn, levels=levels_h_dyn, colors='black', linewidths=0.6, zorder=4)
axes[1, 0].clabel(cs3, inline=True, fmt='%d', fontsize=7, colors='black', zorder=9)
cbar3 = plt.colorbar(cf3, ax=axes[1, 0], orientation='horizontal', pad=0.05, shrink=0.85, ticks=levels_h_dyn)
cbar3.set_label('Indice di Huglin Accumulato ad Oggi (HI)', fontsize=8, fontweight='bold')

# --- MAPPA 4 (Basso DX): WINKLER PROGRESSIVO A OGGI ---
format_map_base(axes[1, 1], f'4. GRADI GIORNO (WI / GDD) PROGRESSIVI - {current_year}\n[{stato_stagione}]')
levels_w_dyn = [0, 300, 600, 900, 1200, 1500, 1800, 2200]
cmap_w_dyn = mcolors.ListedColormap(['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c'])
norm_w_dyn = mcolors.BoundaryNorm(levels_w_dyn, cmap_w_dyn.N)

cf4 = axes[1, 1].contourf(lon_grid, lat_grid, winkler_dyn, levels=levels_w_dyn, cmap=cmap_w_dyn, norm=norm_w_dyn, extend='both', alpha=0.85, zorder=2)
cs4 = axes[1, 1].contour(lon_grid, lat_grid, winkler_dyn, levels=levels_w_dyn, colors='#000055', linewidths=0.6, zorder=4)
axes[1, 1].clabel(cs4, inline=True, fmt='%d GDD', fontsize=7, colors='#000055', zorder=9)
cbar4 = plt.colorbar(cf4, ax=axes[1, 1], orientation='horizontal', pad=0.05, shrink=0.85, ticks=levels_w_dyn)
cbar4.set_label('Gradi Giorno Accumulati ad Oggi (GDD Base 10°C)', fontsize=8, fontweight='bold')

plt.suptitle(f'AGRO-CLIMATOLOGIA VITICOLA | ZONAZIONE CONSUNTIVA vs MONITORAGGIO REALE AD OGGI\nCENTRO ITALIA & LAZIO', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

output_filename = 'mappe_bioclimatiche_huglin_winkler_era5.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=150)
plt.close()

print(f" Mappe generate con successo per la data del {end_date_dyn.strftime('%d/%m/%Y')}!")
