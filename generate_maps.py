import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configurazione cartella e stile
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def setup_map(title, subtitle):
    """Crea una figura base con mappa dell'Italia e stile coerente"""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Limiti geografici (Italia e mari circostanti)
    ax.set_extent([6.0, 19.0, 36.0, 47.5], crs=ccrs.PlateCarree())
    
    # Elementi geografici
    ax.add_feature(cfeature.LAND, facecolor='#f8f9fa')
    ax.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#2c3e50')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8, edgecolor='#7f8c8d')
    
    # Titolo e sottotitolo
    plt.title(f"{title}\n", fontsize=14, fontweight='bold', color='#1a5276', loc='left')
    plt.title(f"{subtitle}", fontsize=9, color='#555555', loc='right')
    
    return fig, ax

print("--> Generazione Mappa 1: Multi-Model Ensemble...")
fig, ax = setup_map("Multi-Model Ensemble Mean", "ECMWF-ENS + GEFS + ICON-EPS")
# Dati simulati per test/inizializzazione
lons = np.linspace(6, 19, 50)
lats = np.linspace(36, 47.5, 50)
lon_grid, lat_grid = np.meshgrid(lons, lats)
temp_ens = 15 + 10 * np.sin(np.radians(lat_grid)) - 5 * np.cos(np.radians(lon_grid))
cs = ax.contourf(lon_grid, lat_grid, temp_ens, cmap='YlOrRd', alpha=0.7, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
cbar.set_label('Temperatura Media prevista (°C)', fontsize=10)
plt.tight_layout()
plt.savefig('mappa_multi_ensemble.png', bbox_inches='tight')
plt.close()

print("--> Generazione Mappa 2: Multi-Model Deterministico...")
fig, ax = setup_map("Multi-Model Deterministico Ad Alta Risoluzione", "ECMWF IFS + GFS + ICON")
precip_det = np.random.exponential(scale=5.0, size=(50, 50))
cs2 = ax.contourf(lon_grid, lat_grid, precip_det, cmap='Blues', alpha=0.7, transform=ccrs.PlateCarree())
cbar2 = plt.colorbar(cs2, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
cbar2.set_label('Precipitazione Cumulata prevista (mm)', fontsize=10)
plt.tight_layout()
plt.savefig('mappa_multi_deterministico.png', bbox_inches='tight')
plt.close()

print("--> Generazione Mappa 3: Quadro Agro-Meteo...")
fig, ax = setup_map("Quadro Agro-Meteo Operativo (24h)", "Centro Italia - Risk Index & ET0")
# Zoom sul Centro Italia per agro-meteo
ax.set_extent([10.5, 14.5, 41.0, 44.0], crs=ccrs.PlateCarree())
risk_index = np.sin(lon_grid) * np.cos(lat_grid)
cs3 = ax.contourf(lon_grid, lat_grid, risk_index, cmap='YlGn', alpha=0.7, transform=ccrs.PlateCarree())
cbar3 = plt.colorbar(cs3, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
cbar3.set_label('Indice di Rischio Infezione Fungina / Favorabilità', fontsize=10)
plt.tight_layout()
plt.savefig('mappe_agrometeo_24h.png', bbox_inches='tight')
plt.close()

print("--> Generazione Mappa 4: Agro-Climatologia ERA5...")
fig, ax = setup_map("Zonazione Agro-Climatica Viticola", "ERA5 Historical - Indici HI e WI")
huglin_index = 1800 + 600 * np.cos(np.radians(lat_grid))
cs4 = ax.contourf(lon_grid, lat_grid, huglin_index, cmap='Spectral_r', alpha=0.7, transform=ccrs.PlateCarree())
cbar4 = plt.colorbar(cs4, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
cbar4.set_label('Indice di Huglin (HI)', fontsize=10)
plt.tight_layout()
plt.savefig('mappa_agro_clima_era5.png', bbox_inches='tight')
plt.close()

print("✅ Tutte le mappe sono state generate con successo!")
