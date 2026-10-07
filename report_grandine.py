import os
import requests
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# ==========================================
# 1. ESTRAZIONE DATI DA ESWD
# ==========================================
def fetch_eswd_data():
    """
    Recupera i dati delle grandinate in Italia da ESWD.
    Nota: Se disponi di una API Key ESWD/ESSL, inseriscila nell'URL.
    In alternativa, lo script può leggere un file esportato o interrogare l'endpoint CSV.
    """
    print("Scaricamento dati da ESWD...")
    reports = []
    
    # Esempio di chiamata API / CSV Export ESWD per l'Italia (Country = ITA, Event = Hail)
    # Sostituire o configurare l'URL con l'API key se necessaria
    eswd_url = "https://eswd.eu/cgi-bin/eswd_export.cgi?country=ITA&event=hail&format=json"
    
    try:
        response = requests.get(eswd_url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            # Parsing della risposta JSON
            for item in data.get('events', []):
                reports.append({
                    'data_ora': item.get('datetime'),
                    'latitudine': float(item.get('latitude')),
                    'longitudine': float(item.get('longitude')),
                    'dimensione_cm': item.get('hail_size_cm', 'N/D'),
                    'localita': item.get('location', 'Italia'),
                    'descrizione': item.get('description', 'Segnalazione grandine ESWD'),
                    'fonte': 'ESWD'
                })
        else:
            print(f"ESWD API ha risposto con codice status: {response.status_code}")
    except Exception as e:
        print(f"Errore durante il download da ESWD: {e}")
        
    return pd.DataFrame(reports)


# ==========================================
# 2. ESTRAZIONE DATI DA METEONETWORK
# ==========================================
def fetch_meteonetwork_data():
    """
    Recupera i dati delle grandinate dal servizio Storm Report di MeteoNetwork.
    """
    print("Scaricamento dati da MeteoNetwork...")
    reports = []
    
    # Endpoint / API di MeteoNetwork per i report delle tempeste/grandine
    mn_url = "https://www.meteonetwork.it/api/stormreport/get_reports?event=grandine"
    
    try:
        response = requests.get(mn_url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('reports', []):
                reports.append({
                    'data_ora': item.get('datetime'),
                    'latitudine': float(item.get('lat')),
                    'longitudine': float(item.get('lon')),
                    'dimensione_cm': item.get('diametro_cm', 'N/D'),
                    'localita': f"{item.get('comune', '')} ({item.get('provincia', '')})",
                    'descrizione': item.get('note', 'Segnalazione MeteoNetwork'),
                    'fonte': 'MeteoNetwork'
                })
        else:
            print(f"MeteoNetwork API ha risposto con codice status: {response.status_code}")
    except Exception as e:
        print(f"Errore durante il download da MeteoNetwork: {e}")
        
    return pd.DataFrame(reports)


# ==========================================
# 3. UNIFICAZIONE E PULIZIA DATI
# ==========================================
def get_unified_data():
    df_eswd = fetch_eswd_data()
    df_mn = fetch_meteonetwork_data()
    
    # Unione dei due dataframe
    df = pd.concat([df_eswd, df_mn], ignore_index=True)
    
    if df.empty:
        print("Attenzione: Nessun dato scaricato. Creazione di un DataFrame di test.")
        # Dati simulati di fallback per verificare il funzionamento grafico
        df = pd.DataFrame([
            {
                'data_ora': '2025-06-15 16:30:00', 'latitudine': 45.4642, 'longitudine': 9.1900,
                'dimensione_cm': 3.5, 'localita': 'Milano (MI)', 
                'descrizione': 'Forte temporale con chicchi fino a 3.5 cm', 'fonte': 'ESWD'
            },
            {
                'data_ora': '2025-07-20 18:15:00', 'latitudine': 45.4384, 'longitudine': 10.9916,
                'dimensione_cm': 2.0, 'localita': 'Verona (VR)', 
                'descrizione': 'Grandinata intensa accumulo a terra', 'fonte': 'MeteoNetwork'
            }
        ])
    
    # Formattazione e parsing delle date
    df['data_ora'] = pd.to_datetime(df['data_ora'])
    df['mese'] = df['data_ora'].dt.month
    df['data_str'] = df['data_ora'].dt.strftime('%d/%m/%Y %H:%M')
    
    return df


# ==========================================
# 4. CREAZIONE MAPPA INTERATTIVA (FOLIUM)
# ==========================================
def generate_hail_map():
    df = get_unified_data()
    
    # Creazione della mappa base centrata sull'Italia
    m = folium.Map(
        location=[42.5000, 12.5000],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Mappa dei mesi dell'anno in italiano
    nomi_mesi = {
        1: "01 - Gennaio", 2: "02 - Febbraio", 3: "03 - Marzo",
        4: "04 - Aprile", 5: "05 - Maggio", 6: "06 - Giugno",
        7: "07 - Luglio", 8: "08 - Agosto", 9: "09 - Settembre",
        10: "10 - Ottobre", 11: "11 - Novembre", 12: "12 - Dicembre"
    }
    
    # Creazione di un FeatureGroup / Cluster per ogni mese
    for num_mese in range(1, 13):
        nome_layer = nomi_mesi[num_mese]
        data_mese = df[df['mese'] == num_mese]
        
        # Mostra per impostazione predefinita i mesi estivi (es. Giugno e Luglio)
        mostra_layer = True if num_mese in [6, 7] else False
        
        # Gruppo del mese
        layer_group = folium.FeatureGroup(name=nome_layer, show=mostra_layer)
        cluster = MarkerCluster().add_to(layer_group)
        
        for _, row in data_mese.iterrows():
            # Colore del marker in base alla fonte
            color_fonte = 'darkred' if row['fonte'] == 'ESWD' else 'orange'
            
            # Formattazione Popup HTML con stile grafico pulito
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 220px;">
                <h4 style="margin-bottom: 5px; color: #333;">{row['localita']}</h4>
                <hr style="margin: 5px 0;">
                <b>📅 Data & Ora:</b> {row['data_str']}<br>
                <b>🧊 Dimensione Chicco:</b> {row['dimensione_cm']} cm<br>
                <b>📍 Fonte Dati:</b> <span style="color: {color_fonte}; font-weight: bold;">{row['fonte']}</span><br>
                <p style="margin-top: 8px; font-size: 12px; color: #555;">
                    <b>Note:</b> {row['descrizione']}
                </p>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitudine'], row['longitudine']],
                radius=6,
                color=color_fonte,
                fill=True,
                fill_color=color_fonte,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(cluster)
            
        layer_group.add_to(m)
        
    # Aggiunta del selettore dei livelli (Layers per Mese)
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Salvataggio del file HTML finale
    m.save("index.html")
    print("Mappa aggiornata con successo e salvata in index.html!")

if __name__ == "__main__":
    generate_hail_map()
