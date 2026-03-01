# moduli/statistica.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

def render():
    st.title("Analisi Statistica del segnale")
    
    # Controllo se ci sono segnali caricati nello stato globale
    if not st.session_state.segnali_caricati:
        st.warning("Nessun segnale in memoria. Carica o genera un segnale nella Tab Creazione.")
        return

    # --- SELEZIONE SEGNALE ---
    with st.expander("SELEZIONE SEGNALE", expanded=True):
        nomi = list(st.session_state.segnali_caricati.keys())
        scelta = st.selectbox("Scegli il segnale da analizzare:", nomi, key="stat_select")
        data = st.session_state.segnali_caricati[scelta]
    
    # --- METRICHE STATISTICHE ---
    st.subheader("Metriche Statistiche")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    media = np.mean(data)
    sigma = np.std(data)
    n_campioni = len(data)
    
    m1.metric("Massimo", f"{np.max(data):.4f}")
    m2.metric("Minimo", f"{np.min(data):.4f}")
    m3.metric("Media", f"{media:.4f}")
    m4.metric("Mediana", f"{np.median(data):.4f}")
    m5.metric("Sigma (σ)", f"{sigma:.4f}")
    m6.metric("N. Campioni", f"{n_campioni}")

    # --- CONTROLLI E ISTOGRAMMA ---
    col_ctrl, col_plot = st.columns([1, 3])
    
    with col_ctrl:
        st.write("**Parametri Istogramma**")
        tipo_bin = st.radio("Modalità divisioni:", ["Numero di divisioni", "Larghezza divisione"], key="bin_mode")
        
        if tipo_bin == "Numero di divisioni":
            n_bins = st.slider("Numero di bin:", 10, 500, 100, key="n_bins_slider")
            bin_size = None
        else:
            range_dati = np.max(data) - np.min(data)
            bin_size = st.number_input("Larghezza bin:", 
                                       value=float(range_dati/100) if range_dati > 0 else 0.1, 
                                       format="%.4f", 
                                       key="bin_size_input")
            n_bins = None
            
        normalizza = st.checkbox("Normalizza area a 1", value=False, key="norm_check")
        mostra_normale = st.checkbox("Confronta con Normale", 
                                     value=False, 
                                     disabled=not normalizza, 
                                     key="gauss_check")
    
    with col_plot:
        fig = go.Figure()
        
        # Aggiunta Istogramma
        fig.add_trace(go.Histogram(
            x=data, 
            nbinsx=n_bins, 
            xbins=dict(size=bin_size) if bin_size else None, 
            histnorm='probability density' if normalizza else None, 
            marker_color='#3498db', 
            opacity=0.7,
            name="Distribuzione"
        ))
        
        # Aggiunta Curva Gaussiana di confronto
        if normalizza and mostra_normale:
            x_n = np.linspace(np.min(data), np.max(data), 200)
            y_n = norm.pdf(x_n, media, sigma)
            fig.add_trace(go.Scatter(
                x=x_n, 
                y=y_n, 
                mode='lines', 
                line=dict(color='#e74c3c', width=3),
                name="Normale (Gauss)"
            ))
            
        fig.update_layout(
            template="plotly_dark", 
            height=500,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)