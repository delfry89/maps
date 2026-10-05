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

warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=DownloadWarning)

print("=== AVVIO SCRIPT 04: AGRO-CLIMATOLOGIA ERA5 DINAMICO ===")

# =========================================================
# 1. CONFIGURAZIONE AREA DI STUDIO & ANNO DI RIFERIMENTO
# =========================================================
lon_min, lon_max = 10.0, 14.0
lat_min, lat_max = 41.0, 44.2

grid_lon = np.linspace(lon_min, lon_max, 250)
grid_lat = np.linspace(lat_min, lat_max, 250)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

lat_mean = (lat_min + lat_max) / 2.0
k_huglin = 1.0 + (lat_mean - 40) * 0.006

# Determina l'anno corrente o l'ultimo anno completo disponibile
current_year = datetime.datetime.now().year
target_year = current_year if datetime.datetime.now().month >= 10 else current_year - 1

# =========================================================
# 2. DOWNLOAD E CARICAMENTO DINAMICO DATI ERA5 REALI
# =========================================================
def get_era5_data(year):
    zip_path = f"era5_land_{year}_season.zip"
    extract_dir = f"era5_extracted_{year}"
    target_file = None

    # Check 1: Cerca se esistono già i file estratti in locale
    search_dirs = ['.', extract_dir, 'era5_extracted_2025']
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(('.nc', '.grib', '.grib2', '.bin')) or f == 'data.grib':
                    target_file = os.path.join(d, f)
                    break
        if target_file:
            break

    # Check 2: Se non ci sono file, scarica da Copernicus via CDS API
    if not target_file:
        print(f" Requesting ERA5-Land real data for year {year} from Copernicus CDS...")
        try:
            import cdsapi
            c = cdsapi.Client()
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': [
                        '2m_temperature',
                        'maximum_2m_temperature_since_previous_post_processing',
                    ],
                    'year': str(year),
                    'month': ['04', '05', '06', '07', '08', '09'],
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': ['12:00'],
                    'area': [lat_max, lon_min, lat_min, lon_max],
                    'format': 'zip',
                },
                zip_path
            )
            
            # Estrazione dello ZIP scaricato
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            for f in os.listdir(extract_dir):
                if f.endswith(('.nc', '.grib', '.grib2', '.bin')) or '.' not in f:
                    target_file = os.path.join(extract_dir, f)
                    break
        except Exception as e:
            print(f" ERRORE DOWNLOAD/CONFIGURAZIONE CDS API: {e}")
            print("Verifica di aver configurato correttamente le credenziali CDS_URL e CDS_KEY nei Secrets di GitHub.")
            sys.exit(1)

    if not target_file:
        print(" Nessun file dati valido trovato dopo il download.")
        sys.exit(1)

    print(f" Reading real ERA5 dataset: {target_file}")
    
    # Lettura dataset con xarray
    try:
        ds = xr.open_dataset(target_file, engine='cfgrib')
    except Exception:
        try:
            ds = xr.open_dataset(target_file, engine='netcdf4')
        except Exception:
            ds = xr.open_dataset(target_file)

    t2m_var = 't2m' if 't2m' in ds else ('2t' if '2t' in ds else list(ds.data_vars)[0])
    tmax_var = 'mx2t' if 'mx2t' in ds else ('mxt2m' if 'mxt2m' in ds else list(ds.data_vars)[0])

    t2m = ds[t2m_var] - 273.15 if ds[t2m_var].max() > 100 else ds[t2m_var]
    tmax = ds[tmax_var] - 273.15 if ds[tmax_var].max() > 100 else ds[tmax_var]

    t2m_mean = t2m.mean(dim='time')
    tmax_mean = tmax.mean(dim='time')

    lat_name = 'latitude' if 'latitude' in ds.coords else ('lat' if 'lat' in ds.coords else list(ds.coords)[0])
    lon_name = 'longitude' if 'longitude' in ds.coords else ('lon' if 'lon' in ds.coords else list(ds.coords)[1])

    t2m_interp = t2m_mean.interp({lon_name: grid_lon, lat_name: grid_lat}).values
    tmax_interp = tmax_mean.interp({lon_name: grid_lon, lat_name: grid_lat}).values

    winkler_grid = np.maximum(0, t2m_interp - 10) * 183
    huglin_daily = (np.maximum(0, t2m_interp - 10) + np.maximum(0, tmax_interp - 10)) / 2.0
    huglin_grid = huglin_daily * 183 * k_huglin

    print(" Indici Bioclimatici calcolati con successo dai dati reali ERA5!")
    return huglin_grid, winkler_grid

huglin_grid, winkler_grid = get_era5_data(target_year)

# =========================================================
# 3. GENERAZIONE GRAFICA REALE ED IDENTICA AL LOCALE
# =========================================================
provinces_feature = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces',
    scale='10m',
    facecolor='none'
)

fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=150, subplot_kw={'projection': ccrs.PlateCarree()})

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
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

# MAPPA 1: HUGLIN
format_map_base(axes[0], f'INDICE DI HUGLIN (HI) - ZONAZIONE VITICOLA ERA5 ({target_year})\n[Periodo Vegetativo: 1 Apr - 30 Set]')
levels_h = [1200, 1500, 1800, 2100, 2400, 2700, 3000]
cmap_h = mcolors.ListedColormap(['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c', '#a50026'])
norm_h = mcolors.BoundaryNorm(levels_h, cmap_h.N)

cf1 = axes[0].contourf(lon_grid, lat_grid, huglin_grid, levels=levels_h, cmap=cmap_h, norm=norm_h, alpha=0.85, zorder=2)
cs1 = axes[0].contour(lon_grid, lat_grid, huglin_grid, levels=levels_h, colors='black', linewidths=0.6, zorder=4)
axes[0].clabel(cs1, inline=True, fmt='%d', fontsize=7, colors='black', zorder=9)

cbar1 = plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.06, shrink=0.85, ticks=levels_h)
cbar1.set_label('Indice di Huglin (HI)\n[<1500: Pinot/Chardonnay | 1800-2100: Sangiovese/Merlot | >2400: Syrah/Primitivo]', fontsize=8, fontweight='bold')

# MAPPA 2: WINKLER
format_map_base(axes[1], f'INDICE DI WINKLER (WI / GDD) - REGIONI CLIMATICHE ({target_year})\n[Somma Gradi Giorno Base 10°C]')
levels_w = [800, 1110, 1390, 1670, 1940, 2220, 2600]
cmap_w = mcolors.ListedColormap(['#edf8fb', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'])
norm_w = mcolors.BoundaryNorm(levels_w, cmap_w.N)

cf2 = axes[1].contourf(lon_grid, lat_grid, winkler_grid, levels=levels_w, cmap=cmap_w, norm=norm_w, alpha=0.85, zorder=2)
cs2 = axes[1].contour(lon_grid, lat_grid, winkler_grid, levels=levels_w, colors='#000055', linewidths=0.6, zorder=4)
axes[1].clabel(cs2, inline=True, fmt='%d GDD', fontsize=7, colors='#000055', zorder=9)

cbar2 = plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.06, shrink=0.85, ticks=levels_w)
cbar2.set_label('Indice di Winkler (GDD Base 10°C)\n[Regione I: Fredda -> Regione V: Molto Calda]', fontsize=8, fontweight='bold')

plt.suptitle(f'AGRO-CLIMATOLOGIA VITICOLA | Calcolo Indici Bioclimatici Storici ERA5 ({target_year})\nCENTRO ITALIA & LAZIO', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])

output_filename = 'mappe_bioclimatiche_huglin_winkler_era5.png'
plt.savefig(output_filename, bbox_inches='tight', dpi=150)
plt.close()

print(f" Mappa salvata con successo in: {output_filename}")
