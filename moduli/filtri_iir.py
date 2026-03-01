# moduli/filtri_iir.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

# --- FUNZIONI DI SUPPORTO IIR ---
def aggiungi_filtro_iir(tipo, fs):
    nyq = fs / 2
    if "band" in tipo:
        freq_init = (float(nyq * 0.2), float(nyq * 0.5))
    else:
        freq_init = float(nyq * 0.3)
    
    if 'filtri_iir' not in st.session_state:
        st.session_state.filtri_iir = []
        
    st.session_state.filtri_iir.append({
        'tipo': tipo, 
        'metodo': 'butter', 
        'ordine': 4, 
        'freq': freq_init,
        'rp': 1.0  # Valore di default per il Ripple (dB)
    })

def rimuovi_filtro_iir(index):
    if 'filtri_iir' in st.session_state and 0 <= index < len(st.session_state.filtri_iir):
        st.session_state.filtri_iir.pop(index)

def render():
    st.title("📉 Filtraggio IIR e Diagramma di Bode")
    
    if not st.session_state.segnali_caricati:
        st.warning("Nessun segnale in memoria. Carica un segnale nella Tab Creazione.")
        return

    # 1. SELEZIONE SEGNALE
    with st.expander("📂 SELEZIONE SEGNALE", expanded=True):
        c1, c2 = st.columns([2, 1])
        nomi = list(st.session_state.segnali_caricati.keys())
        scelta = c1.selectbox("Segnale da filtrare:", nomi, key="sel_iir")
        unit = c2.text_input("Unità:", "V", key="unit_iir")
        info = st.session_state.info_segnali.get(scelta, {"fs": 1000.0, "durata": 1.0})
        fs = info["fs"]
        nyq = fs / 2

    # 2. AGGIUNTA FILTRI
    # moduli/filtri_iir.py (intorno alla riga 50)
    with st.expander("➕ AGGIUNGI FILTRO IIR", expanded=True):
        ca1, ca2, ca3, ca4 = st.columns(4)
        # Aggiungi key univoche qui:
        if ca1.button("➕ Passa-Basso", key="btn_iir_lp"): aggiungi_filtro_iir("lowpass", fs); st.rerun()
        if ca2.button("➕ Passa-Alto", key="btn_iir_hp"): aggiungi_filtro_iir("highpass", fs); st.rerun()
        if ca3.button("➕ Passa-Banda", key="btn_iir_bp"): aggiungi_filtro_iir("bandpass", fs); st.rerun()
        if ca4.button("➕ Arresta-Banda", key="btn_iir_bs"): aggiungi_filtro_iir("bandstop", fs); st.rerun()

    # 3. CONFIGURAZIONE E CALCOLO
    segnale_elaborato = st.session_state.segnali_caricati[scelta].copy()
    w_rad, h_total = None, np.ones(1024, dtype=complex)
    
    if 'filtri_iir' in st.session_state and st.session_state.filtri_iir:
        st.subheader("Filtri in Cascata")
        for i, f in enumerate(st.session_state.filtri_iir):
            with st.container():
                col_info, col_params, col_freq, col_del = st.columns([1.5, 2, 4, 0.5])
                col_info.info(f"**{i+1}. {f['tipo'].upper()}**")
                f['metodo'] = col_params.selectbox("Metodo:", ["butter", "cheby1", "bessel"], key=f"m_{i}")
                f['ordine'] = col_params.number_input("Ordine:", 1, 12, f['ordine'], key=f"o_{i}")
                if f['metodo'] == "cheby1":
                    f['rp'] = col_params.number_input("Ripple (dB):", 0.1, 10.0, f['rp'], key=f"rp_{i}")
                
                f['freq'] = col_freq.slider("Freq [Hz]", 0.1, float(nyq-0.1), f['freq'], key=f"f_{i}")
                if col_del.button("🗑️", key=f"del_{i}"):
                    rimuovi_filtro_iir(i); st.rerun()
                
                try:
                    kwargs = {'rp': f['rp']} if f['metodo'] == "cheby1" else {}
                    b, a = signal.iirfilter(f['ordine'], f['freq'], btype=f['tipo'], ftype=f['metodo'], fs=fs, **kwargs)
                    segnale_elaborato = signal.filtfilt(b, a, segnale_elaborato)
                    w, h = signal.freqz(b, a, worN=1024, fs=fs)
                    w_rad = 2 * np.pi * w 
                    h_total *= h 
                except Exception as e:
                    st.error(f"Errore nel calcolo del filtro: {e}")
            st.markdown("---")

    # 4. ANALISI E GRAFICI
    N = len(segnale_elaborato)
    t = np.linspace(0, info["durata"], N, endpoint=False)
    segnale_orig = st.session_state.segnali_caricati[scelta]
    
    freqs_r = np.fft.rfftfreq(N, d=1/fs)
    fft_orig = np.abs(np.fft.rfft(segnale_orig)) * (2.0 / N)
    fft_proc = np.abs(np.fft.rfft(segnale_elaborato)) * (2.0 / N)

    fig = make_subplots(
        rows=4, cols=1, 
        vertical_spacing=0.1, 
        subplot_titles=(
            "<b>ANALISI TEMPORALE</b>", 
            "<b>ANALISI SPETTRALE</b>", 
            "<b>BODE: MAGNITUDO [dB]</b>", 
            "<b>BODE: FASE [deg]</b>"
        )
    )

    fig.add_trace(go.Scatter(x=t, y=segnale_orig, name="Originale", line=dict(color='rgba(150,150,150,0.3)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=segnale_elaborato, name="Filtrato", line=dict(color='#f1c40f', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=freqs_r, y=fft_orig, name="Spettro Orig.", line=dict(color='rgba(150,150,150,0.3)', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=freqs_r, y=fft_proc, name="Spettro Filt.", line=dict(color='#e67e22', width=2)), row=2, col=1)

    if w_rad is not None:
        mag_db = 20 * np.log10(np.maximum(np.abs(h_total), 1e-4)) 
        fig.add_trace(go.Scatter(x=w_rad, y=mag_db, name="Guadagno", line=dict(color='#3498db', width=2.5)), row=3, col=1)
        fig.add_hline(y=-3, line_dash="dot", line_color="white", row=3, col=1)

        fase_deg = np.unwrap(np.angle(h_total, deg=True), period=360)
        fig.add_trace(go.Scatter(x=w_rad, y=fase_deg, name="Fase", line=dict(color='#9b59b6', width=2.5)), row=4, col=1)

    # Configurazione Assi
    fig.update_yaxes(title_text=f"Ampiezza [{unit}]", row=1, col=1)
    fig.update_yaxes(title_text="Magnitudo", row=2, col=1)
    fig.update_yaxes(title_text="dB", range=[np.max(mag_db)-60 if w_rad is not None else -60, np.max(mag_db)+5 if w_rad is not None else 5], row=3, col=1)
    fig.update_yaxes(title_text="Gradi [°]", row=4, col=1)
    
    fig.update_xaxes(title_text="Tempo [s]", row=1, col=1)
    fig.update_xaxes(title_text="Frequenza [Hz]", row=2, col=1)
    fig.update_xaxes(type="log", title_text="ω [rad/s]", row=3, col=1)
    fig.update_xaxes(type="log", title_text="ω [rad/s]", row=4, col=1)

    fig.update_layout(height=1200, template="plotly_dark", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)