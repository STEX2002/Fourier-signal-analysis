# main.py
import streamlit as st

# Importiamo i nostri moduli personalizzati
from moduli import creazione, statistica, filtri_iir, audio, engine




def main():
    st.set_page_config(page_title="Signal Suite Pro", layout="wide")

    # Inizializzazione Session State (deve stare nel main)
    if 'segnali_caricati' not in st.session_state:
        st.session_state.segnali_caricati = {}
    if 'info_segnali' not in st.session_state:
        st.session_state.info_segnali = {}
    if 'filtri' not in st.session_state:
        st.session_state.filtri = []  # <--- MANCAVA QUESTA RIGA
        
    if 'filtri_iir' not in st.session_state:
        st.session_state.filtri_iir = []



    st.sidebar.title("⚙️ Global Controls")
    if st.sidebar.button("🗑️ Reset Totale"):
        st.session_state.clear()
        st.rerun()

    # Definizione delle Tabs
    t1, t2, t3, t4, t5 = st.tabs([
        "📡 Creazione", 
        "📊 Statistica", 
        "📉 Filtri IIR", 
        "🔊 Audio", 
        "⚙️ Engine"
    ])

    # Esecuzione dei moduli nelle rispettive Tab
    with t1: creazione.render()
    
    with t2: statistica.render()
    
    with t3: filtri_iir.render()
    
    with t4: audio.render()
        
    with t5: engine.render()

if __name__ == "__main__":
    main()
