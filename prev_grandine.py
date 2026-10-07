import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import rgb_to_hsv

# Tenta l'importazione di Cartopy per generare la mappa geografica ex-novo
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

for i in range(1, 24):  # Da MTS01 a MTS23
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
        print(f"Errore durante il download di {url}: {e}")

if not images:
    raise RuntimeError("Nessuna immagine scaricata. Verifica gli URL o le regole del firewall.")

height, width, _ = images[0].shape

# -------------------------------------------------------------------------
# 2. RILEVAMENTO ESCLUSIVO DELLA SCALA GRANDINE (HSV)
# -------------------------------------------------------------------------
def extract_hail_only(img_array):
    """
    Estrae SOLO i pixel appartenenti alla scala della grandine:
    Azzurro chiaro, Azzurro, Blu chiaro, Blu, Blu scuro, Giallo e Arancione.
    Esclude sfondi, scritte, territori e la scala temporalesca.
    """
    # Normalizza RGB tra 0 e 1
    rgb_norm = img_array.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    
    h = hsv[:, :, 0]  # Tonalità (Hue) [0, 1]
    s = hsv[:, :, 1]  # Saturazione [0, 1]
    v = hsv[:, :, 2]  # Luminosità [0, 1]

    hail_intensity = np.zeros((height, width), dtype=np.float32)

    # Maschera 1: Ciano / Azzurro / Blu / Blu Scuro (Tonalità Hue indicativamente tra 0.50 e 0.75)
    mask_blue_cyan = (h >= 0.48) & (h <= 0.75) & (s > 0.25) & (v > 0.25)
    
    # Maschera 2: Giallo / Arancione (Tonalità Hue indicativamente tra 0.08 e 0.18)
    mask_yellow_orange = (h >= 0.07) & (h <= 0.18) & (s > 0.40) & (v > 0.40)

    # Assegnazione valori stimati dell'indice di grandine (da ~0.3 a ~1.0)
    hail_intensity[mask_blue_cyan] = 0.5  # Valore medio-basso (Azzurro/Blu)
    hail_intensity[mask_yellow_orange] = 0.9  # Valore alto (Giallo/Arancio)

    # Ritaglia solo la regione geografica interna (esclude le bande/legende laterali e superiori)
    # Coordinate stimata della mappa centrale: Y [8% - 92%], X [12% - 88%]
    y_min, y_max = int(height * 0.08), int(height * 0.92)
    x_min, x_max = int(width * 0.12), int(width * 0.88)
    
    clean_hail = np.zeros_like(hail_intensity)
    clean_hail[y_min:y_max, x_min:x_max] = hail_intensity[y_min:y_max, x_min:x_max]

    return clean_hail

# -------------------------------------------------------------------------
# 3. ACCUMULO DELLE 24 ORE
# -------------------------------------------------------------------------
accumulated_hail = np.zeros((height, width), dtype=np.float32)

print("Elaborazione ed estrazione isolata della sola grandine...")
for img in images:
    hail_layer = extract_hail_only(img)
    accumulated_hail += hail_layer

# -------------------------------------------------------------------------
# 4. CREAZIONE DI UNA NUOVA MAPPA EX-NOVO
# -------------------------------------------------------------------------
# Definizione della nuova Palette di Colori dedicata all'accumulo grandine
cmap_hail_sum = mcolors.LinearSegmentedColormap.from_list(
    "hail_cum", ["#ffffff00", "#7dd3fc", "#0284c7", "#1e3a8a", "#fde047", "#f97316"], N=256
)

# Estremi geografici approssimativi della domini della mappa Italia/Sardegna
extent = [5.0, 20.0, 36.0, 48.0]  # [lon_min, lon_max, lat_min, lat_max]

print("Generazione della nuova mappa...")

if HAS_CARTOPY:
    # Genera una mappa vettoriale geografica del tutto NUOVA
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Aggiungi dettagli cartografici di sfondo puliti
    ax.add_feature(cfeature.LAND, facecolor='#f8fafc')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f2fe')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#334155')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6, edgecolor='#64748b')

    # Traccia la matrice dell'accumulo grandine
    im = ax.imshow(
        accumulated_hail,
        extent=extent,
        origin='upper',
        cmap=cmap_hail_sum,
        alpha=0.85,
        vmin=0.1,
        transform=ccrs.PlateCarree()
    )
else:
    # Mappa pulita bidimensionale senza dipendenze C
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_facecolor('#f8fafc')
    im = ax.imshow(accumulated_hail, cmap=cmap_hail_sum, vmin=0.1)
    plt.axis('off')

# Barra della legenda personalizzata
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
cbar.set_label('Accumulo Complessivo Indice Grandine (24h)', fontsize=11, fontweight='bold')

plt.title("Mappa Accumulo Grandine 24 Ore (Ex-Novo)", fontsize=14, fontweight='bold', pad=15)

# Salvataggio Mappa Finale
output_path = "mappa_riassunto_24h.png"
plt.savefig(output_path, bbox_inches='tight', dpi=200)
plt.close()

print(f"Mappa ex-novo creata e salvata con successo in: {output_path}")
