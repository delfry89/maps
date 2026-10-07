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
# 2. RICERCA DEL SOTTOINSIEME GRANDINE DENTRO L'INSIEME TEMPORALESCO
# -------------------------------------------------------------------------
def extract_hail_subset_only(img_array):
    """
    Individua l'area del temporale (Rosso/Viola), ma ESTRAE ed ISOLA solo il
    sottoinsieme di pixel che presentano la firma cromatica della GRANDINE
    (Azzurro, Blu, Indaco, Giallo, Arancio e vettori bianchi di contorno).
    """
    rgb = img_array.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    rgb_norm = rgb / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hail_val = np.zeros((height, width), dtype=np.float32)

    # A. AZZURRO / CIANO / BLU CHIARO (Grandine debole/media 0.3 - 0.5)
    # Riconosce i pixel dove la componente B o G supera R (oppure Hue tra azzurro e blu)
    mask_cyan_blue = (b > r + 20) & (b > 80) & (h >= 0.45) & (h <= 0.72)
    hail_val[mask_cyan_blue] = 0.45

    # B. BLU SCURO / INDACO / SFUMATURE VIOLA SU FONDO ROSSO (Grandine forte 0.6 - 0.7)
    # Quando l'indaco si sovrappone al rosso, R e B sono entrambi alti, ma B ha forte presenza
    mask_indaco_blend = (b > 60) & (r > 40) & (h >= 0.68) & (h <= 0.82) & (s > 0.35)
    hail_val[mask_indaco_blend] = 0.65

    # C. GIALLO / ARANCIONE (Grandine massima 0.8 - 0.9)
    # R e G entrambi alti, B basso
    mask_yellow_orange = (r > 180) & (g > 140) & (b < 100) & (h >= 0.07) & (h <= 0.18)
    hail_val[mask_yellow_orange] = 0.88

    # D. LINEE BIANCHE/ISOIPSE DI CONTORNO DENTRO IL NUCLEO (Contorno dell'Hail Index)
    # Pixel quasi bianchi (R,G,B alti e poca saturazione) situati dentro l'area del temporale
    mask_contour_lines = (r > 200) & (g > 200) & (b > 200) & (s < 0.15)
    # Consideriamo le linee bianche solo se sono vicine ad aree rosse/temporalesche
    mask_storm_background = ((r > 160) & (g < 100) & (b < 100)) | (h > 0.85) | (h < 0.05)
    
    # Applichiamo il filtro geografico interno (taglia legende esterne)
    y1, y2 = int(height * 0.05), int(height * 0.94)
    x1, x2 = int(width * 0.10), int(width * 0.90)

    clean_map = np.zeros_like(hail_val)
    clean_map[y1:y2, x1:x2] = hail_val[y1:y2, x1:x2]

    return clean_map

# -------------------------------------------------------------------------
# 3. ACCUMULO SULLE 24 ORE
# -------------------------------------------------------------------------
accumulated_hail = np.zeros((height, width), dtype=np.float32)

print("Estrazione precisa del sottoinsieme grandine in corso...")
for img in images:
    hail_layer = extract_hail_subset_only(img)
    accumulated_hail += hail_layer

# -------------------------------------------------------------------------
# 4. PLOTTING MAPPA EX-NOVO CON STILE "DELFRY"
# -------------------------------------------------------------------------
y1, y2 = int(height * 0.05), int(height * 0.94)
x1, x2 = int(width * 0.10), int(width * 0.90)
cropped_accum = accumulated_hail[y1:y2, x1:x2]

lon_min, lon_max = 6.0, 19.0
lat_min, lat_max = 35.5, 47.5
extent = [lon_min, lon_max, lat_min, lat_max]

# Colormap esclusiva della grandine accumulata nelle 24h
cmap_hail = mcolors.LinearSegmentedColormap.from_list(
    "hail_subset_24h", 
    ["#ffffff00", "#38bdf8", "#1d4ed8", "#1e1b4b", "#eab308", "#ea580c"], 
    N=256
)

print("Generazione mappa ex-novo stile Delfry...")
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

    # Mascheriamo i pixel dove l'accumulo è zero o trascurabile
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
