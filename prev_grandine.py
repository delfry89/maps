import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import rgb_to_hsv

# Supporto Cartopy per il plotting geografico vettoriale
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# -------------------------------------------------------------------------
# 1. DOWNLOAD DELLE MAPPE ORARIE
# -------------------------------------------------------------------------
BASE_URL = "https://www.centrometeo.com/wrfmap/italia_sard/MTS{:02d}_d01.png"
IMAGES_DIR = "downloaded_maps"
os.makedirs(IMAGES_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://www.centrometeo.com/",
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
}

images = []
print("Scaricamento delle mappe orarie...")

session = requests.Session()
session.headers.update(HEADERS)

for i in range(1, 24):
    url = BASE_URL.format(i)
    file_path = os.path.join(IMAGES_DIR, f"MTS{i:02d}.png")
    
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(r.content)
            img = Image.open(file_path).convert("RGB")
            images.append(np.array(img))
            print(f"Scaricata con successo: MTS{i:02d}.png")
        else:
            print(f"Impossibile scaricare {url} (Status HTTP: {r.status_code})")
    except Exception as e:
        print(f"Errore download {url}: {e}")

if not images:
    raise RuntimeError("Nessuna immagine scaricata. Verifica connessione o URL.")

height, width, _ = images[0].shape

# -------------------------------------------------------------------------
# 2. ESTRAZIONE NUCLEI AD ALTO RISCHIO (GRANDINE + TEMPORALE INTENSO)
# -------------------------------------------------------------------------
def extract_phenomena_comprehensive(img_array):
    """
    Estrae tutti i nuclei attivi catturando le tonalità di:
    - Azzurro/Ciano/Blu Scuro/Indaco (Hail Index a sinistra)
    - Giallo/Arancione (Hail Index alto e Temporali intensi)
    - Rosso / Viola / Marrone (Temporali con elevata riflettività/MTS > 50)
    Esclude rigorosamente i toni del verde (territorio/MTS basso), mare e sfondi.
    """
    rgb_norm = img_array.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    
    h = hsv[:, :, 0]  # Hue (Tonalità)
    s = hsv[:, :, 1]  # Saturation
    v = hsv[:, :, 2]  # Value (Luminosità)
    
    val_map = np.zeros((height, width), dtype=np.float32)

    # 1. AZZURRO / BLU / INDACO (Hail Index 0.3 - 0.7)
    mask_blue_hail = (h >= 0.45) & (h <= 0.78) & (s >= 0.25) & (v >= 0.15)
    val_map[mask_blue_hail] = 0.60

    # 2. GIALLO / ARANCIONE (Hail Index 0.8 - 0.9 e Temporali Forti)
    mask_yellow_orange = (h >= 0.06) & (h <= 0.18) & (s >= 0.40) & (v >= 0.40)
    val_map[mask_yellow_orange] = 0.85

    # 3. ROSSO / VIOLA / BORDEAUX (Temporali Intensi MTS 60-90) -> Liguria, Emilia, Viterbese
    # Include sia il rosso (Hue vicina a 0/1) che i viola (Hue > 0.80)
    mask_red_purple = ((h < 0.06) | (h > 0.80)) & (s >= 0.35) & (v >= 0.20)
    val_map[mask_red_purple] = 0.95

    # Area geografica interna (taglia titoli e legende laterali)
    y1, y2 = int(height * 0.05), int(height * 0.94)
    x1, x2 = int(width * 0.10), int(width * 0.90)
    
    clean_map = np.zeros_like(val_map)
    clean_map[y1:y2, x1:x2] = val_map[y1:y2, x1:x2]
    
    return clean_map

# -------------------------------------------------------------------------
# 3. ACCUMULO SULLE 24 ORE
# -------------------------------------------------------------------------
accumulated_signal = np.zeros((height, width), dtype=np.float32)

print("Filtraggio cromatico globale ed estrazione accumulo 24h...")
for img in images:
    layer = extract_phenomena_comprehensive(img)
    accumulated_signal += layer

# -------------------------------------------------------------------------
# 4. PLOTTING GRAFICO EX-NOVO CON STILE "DELFRY"
# -------------------------------------------------------------------------
y1, y2 = int(height * 0.05), int(height * 0.94)
x1, x2 = int(width * 0.10), int(width * 0.90)
cropped_accum = accumulated_signal[y1:y2, x1:x2]

# Coordinate geografiche Italia
lon_min, lon_max = 6.0, 19.0
lat_min, lat_max = 35.5, 47.5
extent = [lon_min, lon_max, lat_min, lat_max]

# Colormap per l'accumulo complessivo (da trasparente ad azzurro, blu, giallo, rosso)
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    "accum_24h", 
    ["#ffffff00", "#38bdf8", "#1d4ed8", "#1e1b4b", "#eab308", "#dc2626"], 
    N=256
)

print("Generazione mappa riassuntiva stile Delfry...")
fig = plt.figure(figsize=(11, 11), dpi=150)

if HAS_CARTOPY:
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Cartografia vettoriale 10m
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f8fafc')
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#e0f2fe')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=0.3)
    
    # Confini provinciali/regionali tratteggiati
    ax.add_feature(
        cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', facecolor='none'),
        edgecolor='gray',
        linewidth=0.5,
        linestyle=':'
    )

    # Mascheriamo i valori nulli (< 0.2)
    masked_accum = np.ma.masked_where(cropped_accum < 0.2, cropped_accum)

    im = ax.imshow(
        masked_accum,
        extent=extent,
        origin='upper',
        cmap=cmap_custom,
        alpha=0.85,
        transform=ccrs.PlateCarree()
    )
else:
    ax = fig.add_subplot(111)
    masked_accum = np.ma.masked_where(cropped_accum < 0.2, cropped_accum)
    im = ax.imshow(masked_accum, cmap=cmap_custom, origin='upper')
    plt.axis('off')

# Legenda e Titoli
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85, aspect=30)
cbar.set_label('Accumulo Complessivo Fenomeni Intensi / Grandine (24h)', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

plt.title('MAPPA RIASSUNTIVA 24 ORE - ITALIA\nElaborazione dati accumulati MTS01 - MTS23', fontsize=11, fontweight='bold', pad=12)

# Firma in basso a destra
ax.text(
    0.99, 0.01, 
    'Elab. & grafica Delfry', 
    transform=ax.transAxes, 
    fontsize=8, 
    fontweight='bold', 
    color='black', 
    ha='right', 
    va='bottom', 
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
)

# Salvataggio Mappa Finale
output_path = "mappa_riassunto_24h.png"
plt.savefig(output_path, bbox_inches='tight', dpi=200)
plt.close()

print(f"🖼️ Mappa salvata con successo: {output_path}")
