import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from plotly.subplots import make_subplots
from pydub import AudioSegment
import io
from scipy import signal

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Signal Processing Suite", layout="wide")

# --- INIZIALIZZAZIONE STATO ---
if 'segnali_caricati' not in st.session_state:
    st.session_state.segnali_caricati = {}
if 'info_segnali' not in st.session_state:
    st.session_state.info_segnali = {}
if 'filtri' not in st.session_state:
    st.session_state.filtri = []

def aggiungi_filtro():
    st.session_state.filtri.append({'tipo': 'Passa-Basso', 'freq': 10.0})

def rimuovi_filtro():
    if len(st.session_state.filtri) > 0:
        st.session_state.filtri.pop()

# --- MODULO 1: CREAZIONE E PROCESSING ---
# --- FUNZIONI DI SUPPORTO FILTRI ---
def aggiungi_filtro_specifico(tipo, f_max):
    if "Banda" in tipo:
        freq_init = (float(f_max * 0.2), float(f_max * 0.5))
    else:
        freq_init = float(f_max * 0.3)
    st.session_state.filtri.append({'tipo': tipo, 'freq': freq_init})

def rimuovi_filtro_singolo(index):
    if 0 <= index < len(st.session_state.filtri):
        st.session_state.filtri.pop(index)

# --- MODULO 1: CREAZIONE E PROCESSING COMPLETO ---
def pagina_creazione():
    st.title("Creazione e Processing Segnale")
    
    with st.sidebar.expander("📂 CARICAMENTO FILE (TXT o MP3)", expanded=True):
        uploaded_files = st.file_uploader("Carica file", type=["txt", "mp3"], accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                if f.name not in st.session_state.segnali_caricati:
                    if f.name.endswith(".txt"):
                        content = f.read().decode("utf-8")
                        data = np.array([float(x) for x in content.split()])
                        st.session_state.segnali_caricati[f.name] = data
                        st.session_state.info_segnali[f.name] = {"fs": len(data)/30.0, "durata": 30.0}
                    elif f.name.endswith(".mp3"):
                        audio = AudioSegment.from_file(io.BytesIO(f.read()), format="mp3")
                        audio = audio.set_channels(1).set_frame_rate(22050)
                        data = np.array(audio.get_array_of_samples()).astype(np.float32)
                        if np.max(np.abs(data)) > 0: data /= np.max(np.abs(data))
                        st.session_state.segnali_caricati[f.name] = data
                        st.session_state.info_segnali[f.name] = {"fs": 22050.0, "durata": len(data)/22050.0}

    if not st.session_state.segnali_caricati:
        st.info("Carica un file .txt o .mp3 dalla sidebar per iniziare.")
        return

    # 1. CONFIGURAZIONE FILE
    with st.expander("1. CONFIGURAZIONE FILE", expanded=True):
        c1, c2 = st.columns([2, 1])
        nomi = list(st.session_state.segnali_caricati.keys())
        scelta = c1.selectbox("Seleziona segnale base:", nomi)
        unit = c2.text_input("Unità di misura:", "V", key="unit_crea")
        segnale_full = st.session_state.segnali_caricati[scelta]
        info_default = st.session_state.info_segnali.get(scelta, {"fs": 1000.0, "durata": 30.0})

    # 2. FINESTRA TEMPORALE
    with st.expander("2. CONFIGURAZIONE FINESTRA TEMPORALE", expanded=True):
        c_t0, c_t1, c_t2 = st.columns(3)
        T_totale = c_t0.number_input("Durata Totale (s)", value=float(info_default["durata"]), key=f"dur_{scelta}")
        t_start = c_t1.number_input("Inizio (s)", value=0.0, key=f"start_{scelta}")
        t_end = c_t2.number_input("Fine (s)", value=float(T_totale), key=f"end_{scelta}")
        
        N_tot = len(segnale_full)
        dt = T_totale / N_tot
        t_full = np.linspace(0, T_totale, N_tot, endpoint=False)
        mask_t = (t_full >= t_start) & (t_full <= t_end)
        segnale_orig = segnale_full[mask_t]
        t_orig = t_full[mask_t]
        if len(segnale_orig) % 2 == 0 and len(segnale_orig) > 0:
            segnale_orig, t_orig = segnale_orig[:-1], t_orig[:-1]
        N = len(segnale_orig)
        durata_finestra = t_end - t_start
        fs_reale = N / durata_finestra if durata_finestra > 0 else 0
        if N < 2: st.error("Seleziona una finestra valida."); return

    # FFT
    freqs = np.fft.fftfreq(N, d=dt)
    f_nyquist_val = float(np.max(np.abs(freqs)))
    fourier_coeffs = np.fft.fft(segnale_orig)
    magnitudo_norm = (2.0 / N) * np.abs(fourier_coeffs)
    magnitudo_norm[0] /= 2.0

    # 3. CONFIGURAZIONE NYQUIST
    with st.expander("3. CONFIGURAZIONE SOGLIA DI SEGNALE NULLO", expanded=True):
        abilita_nyq = st.checkbox("Abilita SOGLIA DI SEGNALE NULLO", value=False, key=f"nyq_en_{scelta}")
        metodo_banda = st.radio("Modalità di taglio:", ["Taglia ogni armonica sotto la soglia", "Mantieni tutto tra F_min e F_max (Banda Effettiva)"], index=1, disabled=not abilita_nyq, key=f"bm_{scelta}")
        col_n1, col_n2 = st.columns(2)
        metodo_nyq = col_n1.selectbox("Metodo calcolo soglia:", ["Soglia Assoluta", "Soglia Sigma (Statistica)"], disabled=not abilita_nyq, key=f"meth_{scelta}")
        soglia_calcolata = 0.0
        f_min_nyq, f_max_nyq = 0.0, 0.0
        if abilita_nyq:
            if metodo_nyq == "Soglia Assoluta":
                soglia_calcolata = col_n2.number_input("Valore soglia:", value=0.01, format="%.4f", key=f"val_{scelta}")
            else:
                n_sigma = col_n2.slider("Moltiplicatore Sigma (n-sigma):", 0.0, 10.0, 3.0, key=f"sig_{scelta}")
                mad = np.median(np.abs(magnitudo_norm - np.median(magnitudo_norm)))
                soglia_calcolata = np.median(magnitudo_norm) + (n_sigma * 1.4826 * mad)
            idx_v = np.where((magnitudo_norm >= soglia_calcolata) & (freqs > 1e-9))[0]
            if len(idx_v) > 0:
                f_min_nyq, f_max_nyq = float(freqs[idx_v[0]]), float(freqs[idx_v[-1]])

    # 4. CONFIGURAZIONE FILTRI MANUALI (VERSIONE BLOCCATA)
    with st.expander("4. CONFIGURAZIONE FILTRI IDEALI", expanded=True):
        st.write("**Aggiungi nuovo filtro:**")
        ca1, ca2, ca3, ca4 = st.columns(4)
        if ca1.button("➕ Passa-Basso"): aggiungi_filtro_specifico("Passa-Basso", f_nyquist_val); st.rerun()
        if ca2.button("➕ Passa-Alto"): aggiungi_filtro_specifico("Passa-Alto", f_nyquist_val); st.rerun()
        if ca3.button("➕ Passa-Banda"): aggiungi_filtro_specifico("Passa-Banda", f_nyquist_val); st.rerun()
        if ca4.button("➕ Arresta-Banda"): aggiungi_filtro_specifico("Arresta-Banda", f_nyquist_val); st.rerun()
        
        st.markdown("---")
        for i, f in enumerate(st.session_state.filtri):
            c_info, c_slider, c_del = st.columns([2, 5, 1])
            c_info.info(f"**{i+1}. {f['tipo']}**")
            label = "Range [Hz]" if "Banda" in f['tipo'] else "Taglio [Hz]"
            f['freq'] = c_slider.slider(label, 0.0, f_nyquist_val, value=f['freq'], key=f"f_{i}_{scelta}")
            if c_del.button("🗑️", key=f"del_{i}_{scelta}"):
                rimuovi_filtro_singolo(i); st.rerun()

    # LOGICA FILTRAGGIO
    abs_freqs = np.abs(freqs)
    m_nyq = np.ones(N, dtype=bool)
    if abilita_nyq:
        if metodo_banda == "Mantieni tutto tra F_min e F_max (Banda Effettiva)":
            m_nyq = (abs_freqs >= f_min_nyq) & (abs_freqs <= f_max_nyq) if f_max_nyq > 0 else np.zeros(N, bool)
        else: m_nyq = (magnitudo_norm >= soglia_calcolata)

    m_filt = np.ones(N, dtype=bool)
    for f in st.session_state.filtri:
        if f['tipo'] == "Passa-Basso": m_filt &= (abs_freqs <= f['freq'])
        elif f['tipo'] == "Passa-Alto": m_filt &= (abs_freqs >= f['freq'])
        elif f['tipo'] == "Passa-Banda": m_filt &= (abs_freqs >= f['freq'][0]) & (abs_freqs <= f['freq'][1])
        elif f['tipo'] == "Arresta-Banda": m_filt &= ~((abs_freqs >= f['freq'][0]) & (abs_freqs <= f['freq'][1]))

    m_tot = m_nyq & m_filt
    idx_effettivi = np.where(m_tot & (freqs > 1e-9))[0]
    banda_effettiva = (float(np.max(freqs[idx_effettivi])) - float(np.min(freqs[idx_effettivi]))) if len(idx_effettivi) > 0 else 0.0
    ricostruito = np.fft.ifft(np.where(m_tot, fourier_coeffs, 0)).real

    # INDICATORI
    st.write("### Parametri Segnale Originale")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Numero Campioni", N)
    o2.metric("Numero Armoniche", np.sum(magnitudo_norm > 1e-6))
    o3.metric("Banda Totale", f"{f_nyquist_val:.2f} Hz")
    o4.metric("Valore Massimo", f"{np.max(segnale_orig):.3f} {unit}")

    st.write("### Parametri Segnale Filtrato")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Numero Campioni", N)
    f2.metric("Armoniche Residue", np.sum(m_tot))
    f3.metric("Banda Effettiva", f"{banda_effettiva:.2f} Hz")
    f4.metric("Valore Massimo", f"{np.max(ricostruito):.3f} {unit}")

    # SALVATAGGIO
    with st.expander("💾 SALVA SEGNALE ELABORATO", expanded=True):
        cs1, cs2 = st.columns([2, 1])
        nome_n = cs1.text_input("Nome nuovo segnale:", value=f"{scelta}_proc")
        if cs2.button("Salva in Memoria"):
            st.session_state.segnali_caricati[nome_n] = ricostruito
            st.session_state.info_segnali[nome_n] = {"fs": fs_reale, "durata": durata_finestra}
            st.success(f"Salvato: {nome_n}"); st.rerun()
#grafici
    f_s, m_s, mask_s = np.fft.fftshift(freqs), np.fft.fftshift(magnitudo_norm), np.fft.fftshift(m_tot)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, subplot_titles=("DOMINIO DEL TEMPO", "DOMINIO DELLA FREQUENZA"))
    fig.add_trace(go.Scatter(x=t_orig, y=segnale_orig, name="Originale", line=dict(color='rgba(150,150,150,0.3)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_orig, y=ricostruito, name="Filtrato", line=dict(color='#2ecc71', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_s, y=np.where(~mask_s, m_s, np.nan), name="Tagliato", line=dict(color='rgba(150,150,150,0.5)', width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=f_s, y=np.where(mask_s, m_s, np.nan), name="Mantenuto", fill='tozeroy', line=dict(color='#FFFFFF', width=2.5), fillcolor='rgba(255,255,255,0.2)'), row=2, col=1)
    
    if abilita_nyq:
        fig.add_shape(type="line", x0=f_s[0], x1=f_s[-1], y0=soglia_calcolata, y1=soglia_calcolata, line=dict(color="#e74c3c", width=2, dash="dot"), row=2, col=1)
        if f_max_nyq > 0:
            for fl in [f_min_nyq, f_max_nyq, -f_min_nyq, -f_max_nyq]:
                if abs(fl) > 1e-9: fig.add_vline(x=fl, line=dict(color="#f39c12", width=1, dash="dash"), row=2, col=1)

    fig.update_layout(height=800, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
# --- MODULO 2: ANALISI STATISTICA ---
def pagina_statistica():
    st.title("Analisi Statistica del segnale")
    if not st.session_state.segnali_caricati:
        st.warning("Nessun segnale in memoria."); return
    with st.expander("SELEZIONE SEGNALE", expanded=True):
        nomi = list(st.session_state.segnali_caricati.keys())
        scelta = st.selectbox("Scegli il segnale:", nomi)
        data = st.session_state.segnali_caricati[scelta]
    
    st.subheader("Metriche Statistiche")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    media, sigma, n_campioni = np.mean(data), np.std(data), len(data)
    m1.metric("Massimo", f"{np.max(data):.4f}")
    m2.metric("Minimo", f"{np.min(data):.4f}")
    m3.metric("Media", f"{media:.4f}")
    m4.metric("Mediana", f"{np.median(data):.4f}")
    m5.metric("Sigma (σ)", f"{sigma:.4f}")
    m6.metric("N. Campioni", f"{n_campioni}")

    col_ctrl, col_plot = st.columns([1, 3])
    with col_ctrl:
        tipo_bin = st.radio("Modalità divisioni:", ["Numero di divisioni", "Larghezza divisione"])
        if tipo_bin == "Numero di divisioni":
            n_bins = st.slider("Numero di bin:", 10, 500, 100)
            bin_size = None
        else:
            range_dati = np.max(data) - np.min(data)
            bin_size = st.number_input("Larghezza bin:", value=float(range_dati/100) if range_dati > 0 else 0.1, format="%.4f")
            n_bins = None
        normalizza = st.checkbox("Normalizza area a 1", value=False)
        mostra_normale = st.checkbox("Confronta con Normale", value=False, disabled=not normalizza)
    with col_plot:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=n_bins, xbins=dict(size=bin_size) if bin_size else None, histnorm='probability density' if normalizza else None, marker_color='#3498db', opacity=0.7))
        if normalizza and mostra_normale:
            x_n = np.linspace(np.min(data), np.max(data), 200)
            fig.add_trace(go.Scatter(x=x_n, y=norm.pdf(x_n, media, sigma), mode='lines', line=dict(color='#e74c3c', width=3)))
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

# --- MODULO 3: AUDIO ---
def pagina_audio():
    st.title("Riproduci Segnale come Audio")
    if not st.session_state.segnali_caricati:
        st.warning("Nessun segnale in memoria."); return
    with st.expander("RIPRODUZIONE", expanded=True):
        nomi = list(st.session_state.segnali_caricati.keys())
        scelta = st.selectbox("Seleziona segnale:", nomi)
        data = st.session_state.segnali_caricati[scelta]
        info = st.session_state.info_segnali.get(scelta, 44100)
        fs_salvata = info.get("fs", 44100) if isinstance(info, dict) else info
        st.write(f"**Frequenza rilevata:** {fs_salvata:.2f} Hz")
        max_val = np.max(np.abs(data))
        audio_data = data / max_val if max_val > 0 else data
        st.audio(audio_data, sample_rate=int(fs_salvata))

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
        
# --- MODULO 3: FILTRAGGIO IIR CON BODE (MAGNITUDO E FASE LOG) ---
def pagina_filtraggio_iir():
    st.title("Filtraggio IIR e diagramma di Bode")
    
    if not st.session_state.segnali_caricati:
        st.warning("Nessun segnale in memoria."); return

    # 1. SELEZIONE SEGNALE
    with st.expander("📂 SELEZIONE SEGNALE", expanded=True):
        c1, c2 = st.columns([2, 1])
        nomi = list(st.session_state.segnali_caricati.keys())
        scelta = c1.selectbox("Segnale da filtrare:", nomi, key="sel_iir")
        unit = c2.text_input("Unità:", "V", key="unit_iir")
        info = st.session_state.info_segnali.get(scelta, {"fs": 1000.0, "durata": 1.0})
        fs = info["fs"]
        nyq = fs / 2

    # 2. AGGIUNTA FILTRI (Logica pulsanti invariata)
    with st.expander("➕ AGGIUNGI FILTRO IIR", expanded=True):
        ca1, ca2, ca3, ca4 = st.columns(4)
        if ca1.button("➕ Passa-Basso"): aggiungi_filtro_iir("lowpass", fs); st.rerun()
        if ca2.button("➕ Passa-Alto"): aggiungi_filtro_iir("highpass", fs); st.rerun()
        if ca3.button("➕ Passa-Banda"): aggiungi_filtro_iir("bandpass", fs); st.rerun()
        if ca4.button("➕ Arresta-Banda"): aggiungi_filtro_iir("bandstop", fs); st.rerun()

    # 3. CONFIGURAZIONE E CALCOLO
    segnale_elaborato = st.session_state.segnali_caricati[scelta].copy()
    w_rad, h_total = None, np.ones(1024, dtype=complex) # Aumentata risoluzione a 1024 punti
    
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
                    # Calcolo risposta in frequenza (w in rad/s se fs non specificato, ma qui usiamo Hz e convertiamo dopo)
                    w, h = signal.freqz(b, a, worN=1024, fs=fs)
                    w_rad = 2 * np.pi * w # Conversione Hz -> rad/s
                    h_total *= h 
                except Exception as e:
                    st.error(f"Errore: {e}")
            st.markdown("---")

    # --- 4. ANALISI E GRAFICI (VERSIONE CON AUTO-RANGE DINAMICO) ---
    N = len(segnale_elaborato)
    t = np.linspace(0, info["durata"], N, endpoint=False)
    segnale_orig = st.session_state.segnali_caricati[scelta]
    
    # Calcolo Spettri (RFFT)
    freqs_r = np.fft.rfftfreq(N, d=1/fs)
    fft_orig = np.abs(np.fft.rfft(segnale_orig)) * (2.0 / N)
    fft_proc = np.abs(np.fft.rfft(segnale_elaborato)) * (2.0 / N)

    fig = make_subplots(
        rows=4, cols=1, 
        vertical_spacing=0.12, 
        subplot_titles=(
            "<b>ANALISI TEMPORALE (Confronto Segnali)</b>", 
            "<b>ANALISI SPETTRALE (Confronto Magnitudo)</b>", 
            "<b>DIAGRAMMA DI BODE: MAGNITUDO (Guadagno)</b>", 
            "<b>DIAGRAMMA DI BODE: FASE (Sfasamento)</b>"
        )
    )

    # --- TRACCE (Invariate) ---
    fig.add_trace(go.Scatter(x=t, y=segnale_orig, name="Originale", line=dict(color='rgba(150,150,150,0.3)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=segnale_elaborato, name="Filtrato IIR", line=dict(color='#f1c40f', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=freqs_r, y=fft_orig, name="Spettro Orig.", line=dict(color='rgba(150,150,150,0.3)', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=freqs_r, y=fft_proc, name="Spettro IIR", line=dict(color='#e67e22', width=2)), row=2, col=1)

    if w_rad is not None:
        # Magnitudo: usiamo un limite inferiore di -80dB per evitare che il grafico diventi illeggibile
        mag_db = 20 * np.log10(np.maximum(np.abs(h_total), 1e-4)) 
        fig.add_trace(go.Scatter(x=w_rad, y=mag_db, name="Guadagno [dB]", line=dict(color='#3498db', width=2.5)), row=3, col=1)
        fig.add_hline(y=-3, line_dash="dot", line_color="white", annotation_text="-3dB", row=3, col=1)

        # Fase
        fase_deg = np.angle(h_total, deg=True)
        fase_unwrap = np.unwrap(fase_deg, period=360)
        fig.add_trace(go.Scatter(x=w_rad, y=fase_unwrap, name="Fase [deg]", line=dict(color='#9b59b6', width=2.5)), row=4, col=1)

    # --- CONFIGURAZIONE ASSI CON AUTO-RANGE ---
    # Tempo
    fig.update_yaxes(title_text=f"Ampiezza [{unit}]", autorange=True, row=1, col=1)
    
    # Spettro: partiamo da zero per la magnitudo
    fig.update_yaxes(title_text=f"Magnitudo [{unit}]", rangemode="tozero", autorange=True, row=2, col=1)
    
    # Bode Magnitudo: impostiamo un range intelligente per non "schiacciare" lo zero
    # Mostriamo da (Max + 5dB) fino a (Min o -60dB)
    if w_rad is not None:
        y_max = np.max(mag_db) + 5
        y_min = np.max([np.min(mag_db) - 5, -80]) # Non scendiamo sotto i -80dB per visibilità
        fig.update_yaxes(title_text="Guadagno [dB]", range=[y_min, y_max], row=3, col=1)
    
    # Bode Fase
    fig.update_yaxes(title_text="Fase [°]", autorange=True, row=4, col=1)

    # Assi X
    fig.update_xaxes(title_text="Tempo [s]", row=1, col=1)
    fig.update_xaxes(title_text="Frequenza [Hz]", row=2, col=1)
    fig.update_xaxes(type="log", title_text="Pulsazione ω [rad/s]", row=3, col=1)
    fig.update_xaxes(type="log", title_text="Pulsazione ω [rad/s]", row=4, col=1)

    fig.update_layout(
        height=1400, # Aumentato per gestire meglio i range variabili
        template="plotly_dark", 
        showlegend=True,
        margin=dict(t=100, b=50, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


# --- MAIN ---
def main():
    st.sidebar.title("Funzionalità")
    menu = st.sidebar.radio("Vai a:", ["1. Creazione Segnale", "2. Analisi Statistica", "3. Filtraggio IIR", "4. Audio"])
    
    if st.sidebar.button("🗑️ Svuota Tutto"):
        st.session_state.segnali_caricati = {}
        st.session_state.info_segnali = {}
        st.rerun()

    if menu == "1. Creazione Segnale": pagina_creazione()
    elif menu == "2. Analisi Statistica": pagina_statistica()
    elif menu == "3. Filtraggio IIR": pagina_filtraggio_iir() # <-- Nuova funzione
    else: pagina_audio()

if __name__ == "__main__": main()
