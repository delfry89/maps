import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import rgb_to_hsv

# Cartopy per il tracciamento vettoriale pulito di confini regionali e provinciali
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
# 2. RICERCA ESTRAZIONE CROMATICA SENZA SFASAMENTO GEOGRAFICO
# -------------------------------------------------------------------------
def extract_hail_exact_pixels(img_array):
    """
    Estrae il segnale di grandine analizzando direttamente la matrice dei pixel.
    """
    rgb = img_array.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    rgb_norm = rgb / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hail_val = np.zeros((height, width), dtype=np.float32)

    # 1. AZZURRO / CIANO PURO (0.3 - 0.4)
    mask_cyan = (b > r + 10) & (b > 65) & (h >= 0.44) & (h <= 0.58)
    hail_val[mask_cyan] = 0.35

    # 2. BLU / INDACO / VIOLA SU FONDO ROSSO (0.5 - 0.7)
    mask_blue_blend = (b > 18) & (b > g + 3) & (s >= 0.12) & (v >= 0.06) & (h >= 0.48) & (h <= 0.88)
    hail_val[mask_blue_blend] = 0.65

    # 3. GIALLO / ARANCIONE PURO (0.8 - 0.9)
    mask_yellow_orange = (r > 130) & (g > 90) & (b < 120) & (h >= 0.06) & (h <= 0.20) & (s >= 0.30)
    hail_val[mask_yellow_orange] = 0.88

    # Ritaglio stringente per escludere cornici e scritte originali
    y1, y2 = int(height * 0.082), int(height * 0.915)
    x1, x2 = int(width * 0.122), int(width * 0.878)

    clean_map = np.zeros_like(hail_val)
    clean_map[y1:y2, x1:x2] = hail_val[y1:y2, x1:x2]

    return clean_map

# -------------------------------------------------------------------------
# 3. ACCUMULO SULLE 24 ORE
# -------------------------------------------------------------------------
accumulated_hail = np.zeros((height, width), dtype=np.float32)

print("Estrazione precisa sottoinsieme grandine...")
for img in images:
    hail_layer = extract_hail_exact_pixels(img)
    accumulated_hail += hail_layer

# -------------------------------------------------------------------------
# 4. PLOTTING MAPPA PULITA CON CONFINI PROVINCIALI E REGIONALI VETTORIALI
# -------------------------------------------------------------------------
# Ritaglio stringente dell'area geografica
y1, y2 = int(height * 0.082), int(height * 0.915)
x1, x2 = int(width * 0.122), int(width * 0.878)

cropped_accum = accumulated_hail[y1:y2, x1:x2]

# Estrazione dello sfondo pulito (Senza numeri e senza etichette)
base_bg = images[0][y1:y2, x1:x2].astype(np.float32)
r_bg, g_bg, b_bg = base_bg[:, :, 0], base_bg[:, :, 1], base_bg[:, :, 2]

# Creazione sfondo solido: Mare Azzurro chiaro, Terraferma Bianca/Grigio Chiarissimo
clean_background = np.ones_like(base_bg) * 255.0
is_sea = (r_bg > 120) & (g_bg > 180) & (b_bg > 200)
clean_background[is_sea] = [224, 242, 254]   # Mare #e0f2fe
clean_background[~is_sea] = [248, 250, 252]  # Terraferma #f8fafc

cmap_hail = mcolors.LinearSegmentedColormap.from_list(
    "hail_24h_clean", 
    ["#ffffff00", "#38bdf8", "#1d4ed8", "#1e1b4b", "#eab308", "#ea580c"], 
    N=256
)

# Coordinate di proiezione geografica allineate per i confini vettoriali
extent = [6.2, 18.6, 36.4, 47.0]

print("Generazione output vettoriale pulito con confini provinciali...")
fig = plt.figure(figsize=(10, 10), dpi=200)

if HAS_CARTOPY:
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 1. Sfondo pulito raster
    ax.imshow(clean_background.astype(np.uint8), extent=extent, origin='upper', transform=ccrs.PlateCarree())

    # 2. Accumulo Grandine
    masked_accum = np.ma.masked_where(cropped_accum < 0.2, cropped_accum)
    im = ax.imshow(masked_accum, extent=extent, cmap=cmap_hail, alpha=0.85, origin='upper', transform=ccrs.PlateCarree())

    # 3. Confini Vettoriali Puliti (Coste, Regioni e Province)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, edgecolor='#0f172a')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, edgecolor='#0f172a')
    
    # Confini Regionali e Provinciali tratteggiati puliti
    ax.add_feature(
        cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', facecolor='none'),
        edgecolor='#64748b',
        linewidth=0.5,
        linestyle=':'
    )
else:
    ax = fig.add_subplot(111)
    ax.imshow(clean_background.astype(np.uint8), origin='upper')
    masked_accum = np.ma.masked_where(cropped_accum < 0.2, cropped_accum)
    im = ax.imshow(masked_accum, cmap=cmap_hail, alpha=0.85, origin='upper')
    plt.axis('off')

# Legenda e Titoli
cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.03, shrink=0.85, aspect=30)
cbar.set_label('Accumulo Complessivo Indice Grandine Effettivo (24h)', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

plt.title('MAPPA RIASSUNTIVA GRANDINE 24 ORE - ITALIA\nElaborazione vettoriale pulita MTS01 - MTS23', fontsize=11, fontweight='bold', pad=12)

# Firma Delfry
ax.text(
    0.98, 0.02, 
    'Elab. & grafica Delfry', 
    transform=ax.transAxes, 
    fontsize=8, 
    fontweight='bold', 
    color='black', 
    ha='right', 
    va='bottom', 
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
)

# Salvataggio Mappa Finale
output_path = "mappa_riassunto_24h.png"
plt.savefig(output_path, bbox_inches='tight', dpi=200)
plt.close()

print(f"🖼️ Mappa vettoriale salvata con successo: {output_path}")
