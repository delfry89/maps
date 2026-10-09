import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage import maximum_filter

# Supporto Cartopy per il disegno dei soli confini provinciali
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
# 2. IDENTIFICAZIONE E MASCHERAMENTO CONTORNI LAGHI DALLA MAPPA BASE
# -------------------------------------------------------------------------
base_img = images[0].astype(np.float32)
r_b, g_b, b_b = base_img[:, :, 0], base_img[:, :, 1], base_bg_b = base_img[:, :, 2]

# I contorni dei laghi nella mappa base sono linee sottili arancioni/marroni costanti
mask_lake_borders = (r_b > 160) & (g_b > 80) & (g_b < 140) & (b_b < 40)
# Dilatiamo leggermente la maschera dei laghi per azzerare del tutto il loro contorno
mask_lake_borders = maximum_filter(mask_lake_borders, size=3)

# -------------------------------------------------------------------------
# 3. ESTRAZIONE PULITA SOTTOINSIEME GRANDINE (SENZA LAGHI E RUMORE)
# -------------------------------------------------------------------------
def extract_hail_clean(img_array):
    """
    Estrae solo la grandine effettiva ed esclude categoricamente i contorni dei laghi.
    """
    rgb = img_array.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    rgb_norm = rgb / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hail_val = np.zeros((height, width), dtype=np.float32)

    # 1. AZZURRO / CIANO PURO (Grandine 0.40)
    mask_cyan = (b > r + 30) & (b > 110) & (h >= 0.46) & (h <= 0.56) & (s >= 0.35)
    hail_val[mask_cyan] = 0.40

    # 2. BLU / INDACO / VIOLA SU FONDO ROSSO (Grandine 0.70)
    mask_blue_blend = (b > 25) & (b > g + 5) & (s >= 0.18) & (v >= 0.08) & (h >= 0.48) & (h <= 0.88)
    hail_val[mask_blue_blend] = 0.70

    # 3. GIALLO / ARANCIONE PURO (Grandine 0.95 - Esclude i laghi)
    mask_yellow_orange = (r > 140) & (g > 100) & (b < 100) & (h >= 0.06) & (h <= 0.20) & (s >= 0.40)
    # Rimuove espressamente i pixel dei contorni dei laghi
    mask_yellow_orange = mask_yellow_orange & (~mask_lake_borders)
    hail_val[mask_yellow_orange] = 0.95

    # DILATAZIONE CONTROLLATA (Rende i puntini più marcati ma non sfigura)
    hail_dilated = maximum_filter(hail_val, size=2)

    # Cutout rigoroso per eliminare cornici nere, scritte e numeri esterni
    y1, y2 = int(height * 0.082), int(height * 0.915)
    x1, x2 = int(width * 0.122), int(width * 0.878)

    clean_map = np.zeros_like(hail_dilated)
    clean_map[y1:y2, x1:x2] = hail_dilated[y1:y2, x1:x2]

    return clean_map

# -------------------------------------------------------------------------
# 4. ACCUMULO SULLE 24 ORE
# -------------------------------------------------------------------------
accumulated_hail = np.zeros((height, width), dtype=np.float32)

print("Estrazione pulita sottoinsieme grandine senza laghi...")
for img in images:
    hail_layer = extract_hail_clean(img)
    accumulated_hail += hail_layer

# -------------------------------------------------------------------------
# 5. PLOTTING FINALE CON SFONDO PULITO E CONFINI PROVINCIALI
# -------------------------------------------------------------------------
y1, y2 = int(height * 0.082), int(height * 0.915)
x1, x2 = int(width * 0.122), int(width * 0.878)

cropped_accum = accumulated_hail[y1:y2, x1:x2]

# Isolamento dello sfondo per creare un canvas totalmente pulito senza numeri (20, 30, 50)
base_bg = images[0][y1:y2, x1:x2].astype(np.float32)
r_bg, g_bg, b_bg = base_bg[:, :, 0], base_bg[:, :, 1], base_bg[:, :, 2]

clean_background = np.ones_like(base_bg) * 255.0
is_sea = (r_bg > 120) & (g_bg > 180) & (b_bg > 200)
clean_background[is_sea] = [224, 242, 254]   # Mare #e0f2fe
clean_background[~is_sea] = [250, 250, 252]  # Terraferma solida #f8fafc

cmap_vivid = mcolors.LinearSegmentedColormap.from_list(
    "hail_vivid_clean", 
    ["#ffffff00", "#0284c7", "#1d4ed8", "#312e81", "#eab308", "#ea580c"], 
    N=256
)

print("Generazione mappa riassuntiva vettoriale pulita...")
fig = plt.figure(figsize=(10, 10), dpi=200)

if HAS_CARTOPY:
    proj = ccrs.LambertConformal(central_longitude=12.7, central_latitude=41.9)
    ax = plt.axes(projection=proj)
    
    extent = [6.4, 18.6, 36.4, 47.0]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 1. Sfondo completamente pulito (Senza numeri 20, 30, 50 o scritte centrometeo)
    ax.imshow(clean_background.astype(np.uint8), extent=extent, origin='upper', transform=ccrs.PlateCarree())

    # 2. Accumulo grandine sovrapposto
    masked_accum = np.ma.masked_where(cropped_accum < 0.45, cropped_accum)
    im = ax.imshow(
        masked_accum, 
        extent=extent, 
        cmap=cmap_vivid, 
        alpha=0.92, 
        origin='upper', 
        transform=ccrs.PlateCarree()
    )

    # 3. Confini Vettoriali: Coste nere e Confini Regionali/Provinciali tratteggiati
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, edgecolor='#1e293b')
    ax.add_feature(
        cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', facecolor='none'),
        edgecolor='#64748b',
        linewidth=0.45,
        linestyle=':'
    )
else:
    ax = fig.add_subplot(111)
    ax.imshow(clean_background.astype(np.uint8), origin='upper')
    masked_accum = np.ma.masked_where(cropped_accum < 0.45, cropped_accum)
    im = ax.imshow(masked_accum, cmap=cmap_vivid, alpha=0.92, origin='upper')
    plt.axis('off')

# Legenda e Titolo
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

print(f"🖼️ Mappa pulita salvata con successo: {output_path}")
