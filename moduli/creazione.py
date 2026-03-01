# moduli/creazione.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from scipy.io import wavfile

# --- FUNZIONI DI SUPPORTO FILTRI (Locali al modulo) ---
def aggiungi_filtro_specifico(tipo, f_max):
    if "Banda" in tipo:
        freq_init = (float(f_max * 0.2), float(f_max * 0.5))
    else:
        freq_init = float(f_max * 0.3)
    st.session_state.filtri.append({'tipo': tipo, 'freq': freq_init})

def rimuovi_filtro_singolo(index):
    if 0 <= index < len(st.session_state.filtri):
        st.session_state.filtri.pop(index)

def render():
    st.title("Creazione e Processing Segnale")
    
    # --- CARICAMENTO FILE ---
    with st.sidebar.expander("📂 CARICAMENTO FILE (TXT o WAV)", expanded=True):
        uploaded_files = st.file_uploader("Carica file", type=["txt", "wav"], accept_multiple_files=True)
        
        if uploaded_files:
            for f in uploaded_files:
                if f.name not in st.session_state.segnali_caricati:
                    if f.name.endswith(".txt"):
                        content = f.read().decode("utf-8")
                        data = np.array([float(x) for x in content.split()])
                        fs_estimated = len(data) / 30.0
                        st.session_state.segnali_caricati[f.name] = data
                        st.session_state.info_segnali[f.name] = {"fs": fs_estimated, "durata": 30.0}
                    
                    elif f.name.endswith(".wav"):
                        fs, data = wavfile.read(io.BytesIO(f.read()))
                        if data.dtype == np.int16: data = data.astype(np.float32) / 32768.0
                        elif data.dtype == np.int32: data = data.astype(np.float32) / 2147483648.0
                        elif data.dtype == np.uint8: data = (data.astype(np.float32) - 128.0) / 128.0
                        
                        if len(data.shape) > 1: data = np.mean(data, axis=1)
                        max_val = np.max(np.abs(data))
                        if max_val > 0: data = data / max_val
                            
                        durata = len(data) / float(fs)
                        st.session_state.segnali_caricati[f.name] = data
                        st.session_state.info_segnali[f.name] = {"fs": float(fs), "durata": durata}

    if not st.session_state.segnali_caricati:
        st.info("Carica un file .txt o .wav dalla sidebar per iniziare.")
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

    # 4. CONFIGURAZIONE FILTRI IDEALI
    # moduli/creazione.py (intorno alla riga 115)
    with st.expander("4. CONFIGURAZIONE FILTRI IDEALI", expanded=True):
        st.write("**Aggiungi nuovo filtro:**")
        ca1, ca2, ca3, ca4 = st.columns(4)
        # Aggiungi key univoche qui:
        if ca1.button("➕ Passa-Basso", key="btn_crea_lp"): aggiungi_filtro_specifico("Passa-Basso", f_nyquist_val); st.rerun()
        if ca2.button("➕ Passa-Alto", key="btn_crea_hp"): aggiungi_filtro_specifico("Passa-Alto", f_nyquist_val); st.rerun()
        if ca3.button("➕ Passa-Banda", key="btn_crea_bp"): aggiungi_filtro_specifico("Passa-Banda", f_nyquist_val); st.rerun()
        if ca4.button("➕ Arresta-Banda", key="btn_crea_bs"): aggiungi_filtro_specifico("Arresta-Banda", f_nyquist_val); st.rerun()
        
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
    st.write("### Parametri Segnale")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Campioni", N)
    o2.metric("Banda Totale", f"{f_nyquist_val:.2f} Hz")
    o3.metric("Banda Effettiva", f"{banda_effettiva:.2f} Hz")
    o4.metric("Max Filtrato", f"{np.max(ricostruito):.3f} {unit}")

    # SALVATAGGIO
    with st.expander("💾 SALVA SEGNALE ELABORATO", expanded=True):
        cs1, cs2 = st.columns([2, 1])
        nome_n = cs1.text_input("Nome nuovo segnale:", value=f"{scelta}_proc")
        if cs2.button("Salva in Memoria"):
            st.session_state.segnali_caricati[nome_n] = ricostruito
            st.session_state.info_segnali[nome_n] = {"fs": fs_reale, "durata": durata_finestra}
            st.success(f"Salvato: {nome_n}"); st.rerun()

    # GRAFICI
    f_s, m_s, mask_s = np.fft.fftshift(freqs), np.fft.fftshift(magnitudo_norm), np.fft.fftshift(m_tot)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, subplot_titles=("DOMINIO DEL TEMPO", "DOMINIO DELLA FREQUENZA"))
    fig.add_trace(go.Scatter(x=t_orig, y=segnale_orig, name="Originale", line=dict(color='rgba(150,150,150,0.3)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_orig, y=ricostruito, name="Filtrato", line=dict(color='#2ecc71', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_s, y=np.where(~mask_s, m_s, np.nan), name="Tagliato", line=dict(color='rgba(150,150,150,0.5)', width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=f_s, y=np.where(mask_s, m_s, np.nan), name="Mantenuto", fill='tozeroy', line=dict(color='#FFFFFF', width=2.5), fillcolor='rgba(255,255,255,0.2)'), row=2, col=1)
    
    if abilita_nyq:
        fig.add_shape(type="line", x0=f_s[0], x1=f_s[-1], y0=soglia_calcolata, y1=soglia_calcolata, line=dict(color="#e74c3c", width=2, dash="dot"), row=2, col=1)

    fig.update_layout(height=800, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)