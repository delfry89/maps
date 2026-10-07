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
    """
    print("Scaricamento dati da ESWD...")
    reports = []
    
    # Endpoint/API per ESWD (Event = Hail, Country = Italy)
    eswd_url = "https://eswd.eu/cgi-bin/eswd_export.cgi?country=ITA&event=hail&format=json"
    
    try:
        response = requests.get(eswd_url, timeout=15)
        if response.status_code == 200:
            data = response.json()
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
            print(f"ESWD ha risposto con codice di stato: {response.status_code}")
    except Exception as e:
        print(f"Errore durante il recupero dei dati da ESWD: {e}")
        
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
    
    # Endpoint/API per MeteoNetwork Storm Report
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
            print(f"MeteoNetwork ha risposto con codice di stato: {response.status_code}")
    except Exception as e:
        print(f"Errore durante il recupero dei dati da MeteoNetwork: {e}")
        
    return pd.DataFrame(reports)


# ==========================================
# 3. UNIFICAZIONE E PULIZIA DATI
# ==========================================
def get_unified_data():
    df_eswd = fetch_eswd_data()
    df_mn = fetch_meteonetwork_data()
    
    # Unione dei due dataset
    df = pd.concat([df_eswd, df_mn], ignore_index=True)
    
    # Se le API non restituiscono dati, usiamo un dataset di prova/fallback
    if df.empty:
        print("Nessun dato live disponibile al momento. Generazione dati dimostrativi...")
        df = pd.DataFrame([
            {
                'data_ora': '2025-05-12 15:45:00', 'latitudine': 45.0703, 'longitudine': 7.6869,
                'dimensione_cm': 2.5, 'localita': 'Torino (TO)', 
                'descrizione': 'Temporale intenso con grandine e forte vento', 'fonte': 'ESWD'
            },
            {
                'data_ora': '2025-06-15 16:30:00', 'latitudine': 45.4642, 'longitudine': 9.1900,
                'dimensione_cm': 3.5, 'localita': 'Milano (MI)', 
                'descrizione': 'Forte supercella con chicchi di grande dimensione', 'fonte': 'ESWD'
            },
            {
                'data_ora': '2025-07-20 18:15:00', 'latitudine': 45.4384, 'longitudine': 10.9916,
                'dimensione_cm': 2.0, 'localita': 'Verona (VR)', 
                'descrizione': 'Grandinata intensa con accumulo al suolo', 'fonte': 'MeteoNetwork'
            },
            {
                'data_ora': '2025-08-05 14:10:00', 'latitudine': 43.7696, 'longitudine': 11.2558,
                'dimensione_cm': 1.5, 'localita': 'Firenze (FI)', 
                'descrizione': 'Rovescio temporalesco accompagnato da grandine', 'fonte': 'MeteoNetwork'
            }
        ])
    
    # Conversione e gestione delle date
    df['data_ora'] = pd.to_datetime(df['data_ora'])
    df['mese'] = df['data_ora'].dt.month
    df['data_str'] = df['data_ora'].dt.strftime('%d/%m/%Y %H:%M')
    
    return df


# ==========================================
# 4. GENERAZIONE MAPPA FOLIUM
# ==========================================
def generate_hail_map():
    df = get_unified_data()
    
    # Mappa base centrata sull'Italia
    m = folium.Map(
        location=[42.5000, 12.5000],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    nomi_mesi = {
        1: "01 - Gennaio", 2: "02 - Febbraio", 3: "03 - Marzo",
        4: "04 - Aprile", 5: "05 - Maggio", 6: "06 - Giugno",
        7: "07 - Luglio", 8: "08 - Agosto", 9: "09 - Settembre",
        10: "10 - Ottobre", 11: "11 - Novembre", 12: "12 - Dicembre"
    }
    
    # Mesi attivi di default (Maggio, Giugno, Luglio, Agosto)
    mesi_attivi_default = [5, 6, 7, 8]
    
    for num_mese in range(1, 13):
        nome_layer = nomi_mesi[num_mese]
        data_mese = df[df['mese'] == num_mese]
        
        mostra_layer = num_mese in mesi_attivi_default
        
        layer_group = folium.FeatureGroup(name=nome_layer, show=mostra_layer)
        cluster = MarkerCluster().add_to(layer_group)
        
        for _, row in data_mese.iterrows():
            color_fonte = '#c0392b' if row['fonte'] == 'ESWD' else '#e67e22'
            
            popup_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; width: 230px;">
                <h4 style="margin: 0 0 5px 0; color: #1a5276; font-size: 15px;">{row['localita']}</h4>
                <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 5px 0 8px 0;">
                <div style="font-size: 13px; line-height: 1.5; color: #2c3e50;">
                    <b>📅 Data & Ora:</b> {row['data_str']}<br>
                    <b>🧊 Taglia Chicco:</b> {row['dimensione_cm']} cm<br>
                    <b>📍 Fonte Dati:</b> <span style="color: {color_fonte}; font-weight: bold;">{row['fonte']}</span>
                </div>
                <p style="margin: 8px 0 0 0; font-size: 12px; color: #666; background: #f8fafc; padding: 6px; border-radius: 4px;">
                    <b>Note:</b> {row['descrizione']}
                </p>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitudine'], row['longitudine']],
                radius=7,
                color=color_fonte,
                fill=True,
                fill_color=color_fonte,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(cluster)
            
        layer_group.add_to(m)
        
    # Controllo dei livelli (selettore mesi)
    folium.LayerControl(collapsed=False).add_to(m)
    
    # SALVATAGGIO IN mappa_grandine.html
    output_filename = "mappa_grandine.html"
    m.save(output_filename)
    print(f"Mappa generata con successo e salvata in: {output_filename}")

if __name__ == "__main__":
    generate_hail_map()
