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
# 2. RICERCA DEL SOTTOINSIEME GRANDINE (SENZA ISOIPSE E CON BLENDING REALE)
# -------------------------------------------------------------------------
def extract_hail_subset_only(img_array):
    """
    Estrae l'Indice Grandine in modo universale (su rosso chiaro, rosso scuro e bordeaux)
    senza perdere i pixel sfumati dall'antialiasing.
    """
    rgb = img_array.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    rgb_norm = rgb / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hail_val = np.zeros((height, width), dtype=np.float32)

    # 1. AZZURRO / CIANO PURO (0.3 - 0.4)
    mask_cyan = (b > r + 10) & (b > 70) & (h >= 0.44) & (h <= 0.58)
    hail_val[mask_cyan] = 0.35

    # 2. BLU / INDACO / VIOLA SU FONDO ROSSO (0.5 - 0.7) -> Viterbese, Frosinone, Liguria
    # Un pixel ha il segnale blu/indaco se il Blu è significativo rispetto al Verde e non è nero puro o grigio
    mask_blue_blend = (b > 20) & (b > g + 4) & (s >= 0.15) & (v >= 0.08) & (h >= 0.50) & (h <= 0.88)
    hail_val[mask_blue_blend] = 0.65

    # 3. GIALLO / ARANCIONE PURO (0.8 - 0.9)
    mask_yellow_orange = (r > 140) & (g > 100) & (b < 110) & (h >= 0.06) & (h <= 0.20) & (s >= 0.35)
    hail_val[mask_yellow_orange] = 0.88

    # Ritaglio preciso dell'area geografica della mappa sorgente
    y1, y2 = int(height * 0.06), int(height * 0.93)
    x1, x2 = int(width * 0.11), int(width * 0.89)

    clean_map = np.zeros_like(hail_val)
    clean_map[y1:y2, x1:x2] = hail_val[y1:y2, x1:x2]

    return clean_map
# -------------------------------------------------------------------------
# 3. ACCUMULO SULLE 24 ORE
# -------------------------------------------------------------------------
accumulated_hail = np.zeros((height, width), dtype=np.float32)

print("Estrazione corretta sottoinsieme grandine senza isoipse bianche...")
for img in images:
    hail_layer = extract_hail_subset_only(img)
    accumulated_hail += hail_layer

# -------------------------------------------------------------------------
# 4. PLOTTING MAPPA EX-NOVO CON STILE "DELFRY"
# -------------------------------------------------------------------------
# Ritaglio preciso dell'area geografica della matrice
y1, y2 = int(height * 0.06), int(height * 0.93)
x1, x2 = int(width * 0.11), int(width * 0.89)
cropped_accum = accumulated_hail[y1:y2, x1:x2]

# Coordinate geografiche ritarate al millimetro sulla griglia sorgente
lon_min, lon_max = 6.2, 18.6
lat_min, lat_max = 36.4, 47.0
extent = [lon_min, lon_max, lat_min, lat_max]

# Palette graduata per la grandine accumulata nelle 24h
cmap_hail = mcolors.LinearSegmentedColormap.from_list(
    "hail_24h_clean", 
    ["#ffffff00", "#38bdf8", "#1d4ed8", "#1e1b4b", "#eab308", "#ea580c"], 
    N=256
)

print("Generazione mappa riassuntiva ex-novo...")
fig = plt.figure(figsize=(11, 11), dpi=150)

if HAS_CARTOPY:
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Cartografia vettoriale ad alta risoluzione (10m)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f8fafc')
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#e0f2fe')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='none', edgecolor='black', linewidth=0.3)
    
    # Confini regionali e provinciali tratteggiati
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
        cmap=cmap_hail,
        alpha=0.85,
        transform=ccrs.PlateCarree()
    )
else:
    ax = fig.add_subplot(111)
    masked_accum = np.ma.masked_where(cropped_accum < 0.2, cropped_accum)
    im = ax.imshow(masked_accum, cmap=cmap_hail, origin='upper')
    plt.axis('off')

# Legenda e Titoli
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85, aspect=30)
cbar.set_label('Accumulo Complessivo Indice Grandine Effettivo (24h)', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

plt.title('MAPPA RIASSUNTIVA GRANDINE 24 ORE - ITALIA\nElaborazione sottoinsieme effettivo MTS01 - MTS23', fontsize=11, fontweight='bold', pad=12)

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

print(f"🖼️ Mappa riassuntiva salvata con successo: {output_path}")
