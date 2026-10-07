import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import rgb_to_hsv

# Supporto Cartopy per il plotting geografico
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
        else:
            print(f"Impossibile scaricare {url} (Status: {r.status_code})")
    except Exception as e:
        print(f"Errore download {url}: {e}")

if not images:
    raise RuntimeError("Nessuna immagine scaricata.")

height, width, _ = images[0].shape

# -------------------------------------------------------------------------
# 2. SELEZIONE RIGIDA DELLE SOLLE GRANDINE (ESCLUZIONE FAKE)
# -------------------------------------------------------------------------
def extract_hail_strict(img_array):
    """
    Estrae l'indice di grandine basandosi sui colori esatti della scala di sinistra,
    inclusi i blu scuri/indaco della Liguria ed Emilia.
    """
    rgb_norm = img_array.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    
    h = hsv[:, :, 0]  # Hue
    s = hsv[:, :, 1]  # Saturation
    v = hsv[:, :, 2]  # Value
    
    hail_val = np.zeros((height, width), dtype=np.float32)

    # 1. AZZURRO CHIARO / CIANO (Soglia ~0.3 - 0.4)
    mask_cyan = (h >= 0.48) & (h <= 0.55) & (s >= 0.30) & (v >= 0.50)
    hail_val[mask_cyan] = 0.35

    # 2. BLU MEDIO / BLU SCURO / INDACO (Soglia ~0.5 - 0.7) -> Liguria / Emilia
    # Ammettiamo luminosità più bassa (v >= 0.15) per catturare i blu scuri
    mask_blue = (h > 0.55) & (h <= 0.78) & (s >= 0.35) & (v >= 0.15)
    hail_val[mask_blue] = 0.65

    # 3. GIALLO / ARANCIONE (Soglia ~0.8 - 0.9)
    mask_yellow = (h >= 0.05) & (h <= 0.18) & (s >= 0.50) & (v >= 0.50)
    hail_val[mask_yellow] = 0.88

    # Ritaglio del solo dominio geografico effettivo (esclude legende esterne)
    y1, y2 = int(height * 0.05), int(height * 0.94)
    x1, x2 = int(width * 0.10), int(width * 0.90)
    
    clean_map = np.zeros_like(hail_val)
    clean_map[y1:y2, x1:x2] = hail_val[y1:y2, x1:x2]
    
    return clean_map

# -------------------------------------------------------------------------
# 3. ACCUMULO DELLE 24 ORE
# -------------------------------------------------------------------------
accumulated_hail = np.zeros((height, width), dtype=np.float32)

print("Filtraggio cromatico e accumulo orario in corso...")
for img in images:
    hail_layer = extract_hail_strict(img)
    accumulated_hail += hail_layer

# -------------------------------------------------------------------------
# 4. PLOTTING MAPPA EX-NOVO
# -------------------------------------------------------------------------
# Ritagliamo la matrice per farla coincidere con i limiti della mappa
y1, y2 = int(height * 0.05), int(height * 0.94)
x1, x2 = int(width * 0.10), int(width * 0.90)
cropped_accum = accumulated_hail[y1:y2, x1:x2]

# Palette pulita per l'accumulo grandine
cmap_hail = mcolors.LinearSegmentedColormap.from_list(
    "hail_24h", 
    ["#ffffff00", "#38bdf8", "#1d4ed8", "#1e1b4b", "#eab308", "#ea580c"], 
    N=256
)

# Estensione geografica tarata sulla finestra ritagliata [LonMin, LonMax, LatMin, LatMax]
extent_crop = [5.5, 19.5, 36.2, 47.5]

print("Generazione output...")
fig = plt.figure(figsize=(10, 10), dpi=150)

if HAS_CARTOPY:
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent_crop, crs=ccrs.PlateCarree())

    # Elementi cartografici vettoriali di sfondo
    ax.add_feature(cfeature.LAND, facecolor='#f8fafc')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f2fe')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#1e293b')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6, edgecolor='#475569')

    # Mascheriamo i valori nulli (0) per renderli totalmente trasparenti
    masked_accum = np.ma.masked_where(cropped_accum < 0.2, cropped_accum)

    im = ax.imshow(
        masked_accum,
        extent=extent_crop,
        origin='upper',
        cmap=cmap_hail,
        alpha=0.85,
        transform=ccrs.PlateCarree()
    )
else:
    ax = fig.add_subplot(111)
    masked_accum = np.ma.masked_where(cropped_accum < 0.2, cropped_accum)
    im = ax.imshow(masked_accum, cmap=cmap_hail, origin='upper')
    plt.axis('off')

# Barra della legenda
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
cbar.set_label('Somma Indice Grandine (24h)', fontsize=11, fontweight='bold')

plt.title("Mappa Complessiva Grandine 24 Ore (Solo Rilevazioni Reali)", fontsize=13, fontweight='bold', pad=12)

output_path = "mappa_riassunto_24h.png"
plt.savefig(output_path, bbox_inches='tight', dpi=200)
plt.close()

print(f"Mappa generata correttamente: {output_path}")
