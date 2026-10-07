import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
# 2. COLOR MATCHING EUCLIDEO PERFETTO SULLA LEGENDA GRANDINE
# -------------------------------------------------------------------------
# RGB campionati direttamente dai 7 tasselli della scala grandine a sinistra:
# 0.3 (Azzurro chiaro), 0.4 (Azzurro), 0.5 (Blu medio), 0.6 (Blu scuro),
# 0.7 (Indaco / Blu profondo), 0.8 (Giallo), 0.9 (Arancione)
HAIL_PALETTE = np.array([
    [100, 225, 255],  # 0.3
    [  0, 180, 240],  # 0.4
    [  0,   0, 200],  # 0.5
    [  0,   0, 130],  # 0.6
    [  5,   0,  70],  # 0.7
    [255, 235,   0],  # 0.8
    [230, 130,   0]   # 0.9
], dtype=np.float32)

HAIL_VALUES = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)

def extract_hail_strict(img_array):
    """
    Calcola la distanza cromatica euclidea di ogni pixel rispetto ai colori della
    legenda della grandine, catturando anche i nuclei frastagliati, blu scuri e sottili.
    """
    img_float = img_array.astype(np.float32)
    h, w, _ = img_float.shape
    
    hail_map = np.zeros((h, w), dtype=np.float32)

    # Area interna geografica (esclude le legende laterali della mappa originale)
    x1, x2 = int(w * 0.11), int(w * 0.89)
    y1, y2 = int(h * 0.06), int(h * 0.93)
    crop_img = img_float[y1:y2, x1:x2]

    # Distanza euclidea RGB tridimensionale
    distances = np.linalg.norm(crop_img[:, :, None, :] - HAIL_PALETTE[None, None, :, :], axis=-1)
    
    min_dist = np.min(distances, axis=-1)
    closest_idx = np.argmin(distances, axis=-1)

    # Tolleranza cromatica < 45 per isolare i pixel veri da quelli trasparenti/sfondo
    valid_mask = min_dist < 45.0
    
    crop_hail = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    crop_hail[valid_mask] = HAIL_VALUES[closest_idx[valid_mask]]

    hail_map[y1:y2, x1:x2] = crop_hail
    return hail_map

# -------------------------------------------------------------------------
# 3. ACCUMULO SULLE 24 ORE
# -------------------------------------------------------------------------
accumulated_hail = np.zeros((height, width), dtype=np.float32)

print("Filtraggio cromatico ed estrazione accumulo in corso...")
for img in images:
    hail_layer = extract_hail_strict(img)
    accumulated_hail += hail_layer

# -------------------------------------------------------------------------
# 4. PLOTTING GRAFICO EX-NOVO CON STILE "DELFRY"
# -------------------------------------------------------------------------
y1, y2 = int(height * 0.06), int(height * 0.93)
x1, x2 = int(width * 0.11), int(width * 0.89)
cropped_accum = accumulated_hail[y1:y2, x1:x2]

# Coordinate geografiche Italia
lon_min, lon_max = 6.0, 19.0
lat_min, lat_max = 35.5, 47.5
extent = [lon_min, lon_max, lat_min, lat_max]

# Colormap per l'accumulo grandine
cmap_hail = mcolors.LinearSegmentedColormap.from_list(
    "hail_24h", 
    ["#ffffff00", "#38bdf8", "#1d4ed8", "#1e1b4b", "#eab308", "#ea580c"], 
    N=256
)

print("Generazione mappa stile Delfry...")
fig = plt.figure(figsize=(11, 11), dpi=150)

if HAS_CARTOPY:
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Dettagli vettoriali ad alta risoluzione (10m)
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
cbar.set_label('Accumulo Complessivo Indice Grandine (24h)', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

plt.title('MAPPA RIASSUNTIVA GRANDINE 24 ORE - ITALIA\nElaborazione dati accumulati MTS01 - MTS23', fontsize=11, fontweight='bold', pad=12)

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

print(f"🖼️ Mappa stile 'Delfry' salvata con successo: {output_path}")
