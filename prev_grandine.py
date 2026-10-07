import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -------------------------------------------------------------------------
# 1. DOWNLOAD DELLE MAPPE ORARIE
# -------------------------------------------------------------------------
BASE_URL = "https://www.centrometeo.com/wrfmap/italia_sard/MTS{:02d}_d01.png"
IMAGES_DIR = "downloaded_maps"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Intestazioni HTTP per bypassare il blocco 403 (simula un browser reale)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://www.centrometeo.com/",
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
}

images = []
print("Scaricamento delle mappe orarie...")

# Crea una sessione per riutilizzare la connessione ed evitare ulteriori blocchi
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
    raise RuntimeError("Nessuna immagine scaricata. Verifica gli URL o le regole del firewall del server.")

# Base e dimensioni dell'immagine
base_image = images[0]
height, width, _ = base_image.shape

# -------------------------------------------------------------------------
# 2. DEFINIZIONE DELLE SOGLIE DI COLORE (HSV / RGB)
# -------------------------------------------------------------------------
# Converte un'immagine RGB in un array di intensità per la scala Temporale e Grandine.
# Modifica i range di colore HSV/RGB in base ai valori esatti usati nei tuoi plot originali.

def extract_phenomena_layers(img_array):
    """
    Estrae separatamente l'intensità del temporale e della grandine dall'immagine.
    Restituisce due matrici 2D (height, width) con i valori stimati.
    """
    # Converte l'immagine PIL/Numpy in float per elaborazione
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Inizializza le mappe dei valori
    storm_val = np.zeros((height, width), dtype=np.float32)
    hail_val = np.zeros((height, width), dtype=np.float32)

    # --- SCALA TEMPORALE (da Verde a Rosso/Viola) ---
    # Esempio di mascheratura/soglie cromatiche
    # Verde (debole): R basso, G alto, B basso
    # Giallo/Arancio (medio): R alto, G alto, B basso
    # Rosso/Viola (forte): R alto, G basso, B medio/alto
    mask_storm = (g > 100) | (r > 150)  # Personalizzare in base all'esatta RGB
    storm_val[mask_storm] = (r[mask_storm].astype(float) + g[mask_storm].astype(float)) / 510.0

    # --- SCALA GRANDINE (da Azzurro a Giallo/Arancione) ---
    # Azzurro/Ciano (debole): R basso, G alto, B alto
    # Giallo/Arancione (forte): R alto, G alto, B basso
    mask_hail = (b > 150) & (g > 150)
    hail_val[mask_hail] = (g[mask_hail].astype(float) + b[mask_hail].astype(float)) / 510.0

    return storm_val, hail_val

# -------------------------------------------------------------------------
# 3. ACCUMULO (SOMMA DELLE 23 ORE)
# -------------------------------------------------------------------------
accumulated_storm = np.zeros((height, width), dtype=np.float32)
accumulated_hail = np.zeros((height, width), dtype=np.float32)

print("Elaborazione e somma dei fotogrammi orari...")
for img in images:
    storm_layer, hail_layer = extract_phenomena_layers(img)
    accumulated_storm += storm_layer
    accumulated_hail += hail_layer

# -------------------------------------------------------------------------
# 4. CREAZIONE MAPPA FINALE E SALVATAGGIO
# -------------------------------------------------------------------------
# Definiamo le Colormap personalizzate per il riassunto
cmap_storm = mcolors.LinearSegmentedColormap.from_list("storm_sum", ["none", "green", "yellow", "red", "purple"])
cmap_hail = mcolors.LinearSegmentedColormap.from_list("hail_sum", ["none", "cyan", "blue", "orange", "yellow"])

fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

# Disegna la mappa di sfondo (utilizziamo la prima immagine senza sovrapposizioni o una mappa geografica base)
ax.imshow(base_image)

# Sovrapponi l'accumulo temporalesco
if np.max(accumulated_storm) > 0:
    storm_overlay = ax.imshow(accumulated_storm, cmap=cmap_storm, alpha=0.6, vmin=0.1)
    cbar_storm = fig.colorbar(storm_overlay, ax=ax, fraction=0.035, pad=0.02)
    cbar_storm.set_label('Accumulo Temporali (24h)')

# Sovrapponi l'accumulo grandine
if np.max(accumulated_hail) > 0:
    hail_overlay = ax.imshow(accumulated_hail, cmap=cmap_hail, alpha=0.6, vmin=0.1)
    cbar_hail = fig.colorbar(hail_overlay, ax=ax, fraction=0.035, pad=0.08, location='left')
    cbar_hail.set_label('Accumulo Grandine (24h)')

plt.axis('off')
plt.title("Riassunto Giornaliero Complessivo (24h)", fontsize=14, pad=12)

# Salva l'output
output_path = "mappa_riassunto_24h.png"
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Mappa finale salvata con successo in: {output_path}")
