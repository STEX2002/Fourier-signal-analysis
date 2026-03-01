# moduli/audio.py
import streamlit as st
import numpy as np

def render():
    st.title("🔊 Riproduci Segnale come Audio")
    
    # Controllo se ci sono segnali nello stato globale
    if not st.session_state.segnali_caricati:
        st.warning("Nessun segnale in memoria. Carica o elabora un segnale nelle altre schede.")
        return

    with st.expander("🎚️ PANNELLO DI RIPRODUZIONE", expanded=True):
        # Selezione del segnale
        nomi = list(st.session_state.segnali_caricati.keys())
        scelta = st.selectbox("Seleziona il segnale da ascoltare:", nomi, key="audio_select")
        
        data = st.session_state.segnali_caricati[scelta]
        
        # Recupero informazioni sulla frequenza di campionamento (fs)
        # Gestisce sia il caso in cui info sia un dizionario che un valore singolo
        info = st.session_state.info_segnali.get(scelta, 44100)
        if isinstance(info, dict):
            fs_salvata = info.get("fs", 44100)
        else:
            fs_salvata = info
            
        st.write(f"**Frequenza di campionamento rilevata:** {fs_salvata:.2f} Hz")
        
        # --- PRE-PROCESSING PER PLAYBACK ---
        # 1. Normalizzazione del picco a 1.0 (fondamentale per evitare distorsioni o silenzi)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            audio_data = data / max_val
        else:
            audio_data = data
            
        # 2. Rendering del player audio di Streamlit
        try:
            st.audio(audio_data, sample_rate=int(fs_salvata))
            st.info("Nota: Se il segnale ha una frequenza molto bassa (es. < 20Hz), potresti non sentire nulla a causa dei limiti fisici degli altoparlanti.")
        except Exception as e:
            st.error(f"Errore durante la riproduzione audio: {e}")

    # Visualizzazione opzionale della forma d'onda rapida
    if st.checkbox("Mostra anteprima forma d'onda"):
        st.line_chart(audio_data[:5000] if len(audio_data) > 5000 else audio_data)