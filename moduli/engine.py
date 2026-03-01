import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from .logic_registry import registry

# --- FUNZIONI HELPER ---
def get_node_type(node):
    if isinstance(node, dict): return node.get('type', node.get('node_type', 'default'))
    return getattr(node, 'node_type', getattr(node, 'type', 'default'))

def get_node_id(node):
    return node['id'] if isinstance(node, dict) else node.id

def get_node_data(node):
    return node['data'] if isinstance(node, dict) else node.data

def run_calculation(flow_state, src_mapping):
    G = nx.DiGraph()
    for edge in flow_state.edges:
        s = edge['source'] if isinstance(edge, dict) else edge.source
        t = edge['target'] if isinstance(edge, dict) else edge.target
        G.add_edge(s, t)
    
    for node in flow_state.nodes:
        n_id = get_node_id(node)
        if n_id not in G: G.add_node(n_id)

    try:
        execution_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return None, "Errore: Ciclo infinito rilevato!"

    node_results = {}
    nodes_dict = {get_node_id(n): n for n in flow_state.nodes}

    for node_id in execution_order:
        node = nodes_dict[node_id]
        temp_data = get_node_data(node).copy()
        l_type = temp_data.get('logic_type', 'default')
        
        incoming_edges = [e for e in flow_state.edges if (e['target'] if isinstance(e, dict) else e.target) == node_id]
        inputs = [node_results[e['source'] if isinstance(e, dict) else e.source] 
                  for e in incoming_edges if (e['source'] if isinstance(e, dict) else e.source) in node_results]

        if l_type == 'input':
            # 1. Recuperiamo il nome del segnale mappato per questo nodo
            sig_name = src_mapping.get(node_id)
            
            if sig_name and sig_name in st.session_state.segnali_caricati:
                # 2. Preleviamo il vettore numpy (i campioni)
                raw_signal = st.session_state.segnali_caricati[sig_name]
                
                # 3. Recuperiamo le info salvate (fs e durata)
                # Se mancano, usiamo 1.0s come durata di sicurezza per evitare divisioni per zero
                info = st.session_state.info_segnali.get(sig_name, {"durata": 1.0})
                durata_user = float(info.get('durata', 1.0))
                
                # 4. Calcoliamo la Fs reale basandoci sulla densità dei dati nel vettore
                # Fs = Numero Campioni / Durata Temporale
                fs_calc = len(raw_signal) / durata_user if durata_user > 0 else 1000.0
                
                # 5. Salviamo i dati nel nodo per i blocchi a valle (come il Filtro)
                # Passiamo sia il segnale che la frequenza di campionamento calcolata
                temp_data['signal_value'] = raw_signal
                temp_data['fs_calcolata'] = fs_calc
                
                # Debug opzionale (puoi commentarlo se non serve)
                # st.toast(f"Nodo {node_id}: Fs rilevata {fs_calc:.2f} Hz")
            else:
                # Fallback se il segnale non è selezionato o rimosso
                temp_data['signal_value'] = np.array([0.0])
                temp_data['fs_calcolata'] = 1000.0


            #temp_data['signal_value'] = st.session_state.segnali_caricati.get(sig_name, np.array([0.0]))

        node_results[node_id] = registry.execute(l_type, inputs, temp_data)
    
    return node_results, None

def render():
    # --- HEADER CON TITOLO E DURATA AFFIANCATI ---
    col_titolo, col_durata = st.columns([2, 1])
    
    with col_titolo:
        st.header("⚙️ Modular Signal Engine")
    
    with col_durata:
        # Inizializziamo la durata se non esiste
        if 'sim_durata' not in st.session_state:
            st.session_state.sim_durata = 10.0
            
        st.session_state.sim_durata = st.number_input(
            "⏱️ Durata Simulazione (s)", 
            min_value=0.001, 
            value=float(st.session_state.sim_durata), 
            step=0.1, 
            format="%.3f",
            help="Imposta la durata temporale totale per il calcolo delle frequenze (Hz)."
        )

    # ... resto del codice (controllo segnali_caricati, inizializzazione flow_state) ...

    
    if 'segnali_caricati' not in st.session_state or not st.session_state.segnali_caricati:
        st.warning("⚠️ Carica un segnale per iniziare.")
        return

    if 'engine_counters' not in st.session_state:
        st.session_state.engine_counters = {'input': 1, 'output': 1, 'sum': 1, 'prod': 1, 'gain': 1, 'butterworth': 1, 'abs': 1, 'butter_hp': 1}


    if 'flow_state' not in st.session_state:
        nodes = [
            StreamlitFlowNode('src_0', (50, 150), {'content': 'Sorgente 0', 'logic_type': 'input'}, 'input', 'right', 'left', deletable=True),
            StreamlitFlowNode('out_0', (750, 150), {'content': 'Scope 0', 'logic_type': 'output'}, 'output', 'right', 'left', deletable=True),
        ]
        edges = [StreamlitFlowEdge(id='edge_0', source='src_0', target='out_0', animated=True, deletable=True)]
        st.session_state.flow_state = StreamlitFlowState(nodes, edges)

    tab_canvas, tab_params = st.tabs(["🎨 Editor Grafico", "🔧 Parametri Precisi"])

    with tab_canvas:
        with st.sidebar:
            st.subheader("📦 Libreria")
            c1, c2 = st.columns(2)
            
            def add_node(label, l_type, kind='default', params={}):
                count = st.session_state.engine_counters.get(l_type, 0)
                new_id = f"{l_type}_{count}"
                data = {'content': f"{label} {count}", 'logic_type': l_type}
                data.update(params)
                new_node = StreamlitFlowNode(new_id, (300, 200), data, kind, 'right', 'left', deletable=True)
                st.session_state.flow_state.nodes.append(new_node)
                st.session_state.engine_counters[l_type] += 1
                st.rerun()

            if c1.button("➕ Sorgente"): add_node("Sorgente", "input", "input")
            if c2.button("➕ Scope"): add_node("Scope", "output", "output")
            if c1.button("➕ Somma"): add_node("Somma", "sum")
            if c2.button("➕ Gain"): add_node("Gain", "gain", params={'val': 1.0})
            
            # ABS e FILTRO nelle colonne per ordine
            if c1.button("➕ ABS"): add_node("ABS", "abs")
            # Sostituisci la riga del filtro esistente con queste due:
            if c1.button("➕ Filtro LP"): add_node("Filtro LP", "butterworth", params={'fc': 10.0, 'order': 4})
            if c2.button("➕ Filtro HP"): add_node("Filtro HP", "butter_hp", params={'fc': 50.0, 'order': 4})

            
            st.markdown("---")
            if st.button("🗑️ Reset Engine", type="primary", use_container_width=True):
                del st.session_state.flow_state
                del st.session_state.engine_counters
                st.rerun()


        result_state = streamlit_flow("main_canvas", st.session_state.flow_state, height=500, allow_new_edges=True, enable_node_menu=True, enable_edge_menu=True, hide_watermark=True)
        
        if result_state:
            changed = len(result_state.nodes) != len(st.session_state.flow_state.nodes) or len(result_state.edges) != len(st.session_state.flow_state.edges)
            for e in result_state.edges:
                if isinstance(e, dict): e['animated'], e['deletable'] = True, True
                else: e.animated, e.deletable = True, True
            st.session_state.flow_state = result_state
            if changed: st.rerun()

    with tab_params:
        st.subheader("🔧 Configurazione Numerica")
        param_nodes = [n for n in st.session_state.flow_state.nodes if get_node_type(n) not in ['output']]
        
        if not param_nodes:
            st.info("Aggiungi blocchi per configurare i parametri.")
        
        for n in param_nodes:
            n_id = get_node_id(n)
            n_data = get_node_data(n)
            l_type = n_data.get('logic_type')
            
            with st.expander(f"⚙️ {n_data['content']} ({n_id})", expanded=True):
                if l_type == 'input':
                    # 1. Recuperiamo la lista dei segnali caricati
                    src_list = list(st.session_state.segnali_caricati.keys())
                    sel_key = f"sel_{n_id}"
                    
                    # 2. Inizializziamo la selezione se non esiste
                    if sel_key not in st.session_state and src_list:
                        st.session_state[sel_key] = src_list[0]
                    
                    # 3. Widget di selezione
                    st.session_state[sel_key] = st.selectbox(
                        "Seleziona Sorgente", 
                        src_list, 
                        key=f"sb_{n_id}",
                        index=src_list.index(st.session_state[sel_key]) if st.session_state[sel_key] in src_list else 0
                    )
                    
                    # 4. Feedback visivo della Fs calcolata
                    nome_segnale = st.session_state[sel_key]
                    if nome_segnale:
                        vettore = st.session_state.segnali_caricati[nome_segnale]
                        durata_globale = st.session_state.sim_durata
                        fs_calc = len(vettore) / durata_globale if durata_globale > 0 else 0
                        st.metric("Frequenza Campionamento (Fs)", f"{fs_calc:.2f} Hz")
                
                elif l_type == 'gain':
                    n_data['val'] = st.number_input(f"Guadagno (K)", value=float(n_data.get('val', 1.0)), step=0.1, format="%.4f", key=f"num_gain_{n_id}")
                
                elif l_type == 'butterworth':
                    c1, c2 = st.columns(2)
                    
                    # --- CALCOLO NYQUIST DINAMICO ---
                    durata_globale = st.session_state.sim_durata
                    fs_riferimento = 1000.0
                    
                    # Cerchiamo il primo nodo input per dare un riferimento di Nyquist sensato
                    for node_check in st.session_state.flow_state.nodes:
                        if get_node_type(node_check) == 'input':
                            id_check = get_node_id(node_check)
                            sig_check = st.session_state.get(f"sel_{id_check}")
                            if sig_check in st.session_state.segnali_caricati:
                                fs_riferimento = len(st.session_state.segnali_caricati[sig_check]) / durata_globale
                                break
                    
                    nyquist = fs_riferimento / 2
                    
                    n_data['fc'] = c1.number_input(
                        "Freq. Taglio (Hz)", 
                        min_value=0.0, 
                        value=float(n_data.get('fc', 10.0)), 
                        key=f"num_fc_{n_id}"
                    )
                    
                    n_data['order'] = c2.number_input(
                        "Ordine", 1, 20, 
                        value=int(n_data.get('order', 4)), 
                        key=f"num_ord_{n_id}"
                    )

                    if n_data['fc'] >= nyquist:
                        st.warning(f"⚠️ Limite Nyquist: {nyquist:.2f} Hz. Il filtro saturerà.")
                    else:
                        st.info(f"✅ Limite Nyquist: {nyquist:.2f} Hz")

                elif l_type == 'butter_hp':
                    c1, c2 = st.columns(2)
                    durata_globale = st.session_state.sim_durata
                    fs_ref = 1000.0
                    
                    for node_check in st.session_state.flow_state.nodes:
                        if get_node_type(node_check) == 'input':
                            id_check = get_node_id(node_check)
                            sig_check = st.session_state.get(f"sel_{id_check}")
                            if sig_check in st.session_state.segnali_caricati:
                                fs_ref = len(st.session_state.segnali_caricati[sig_check]) / durata_globale
                                break
                    
                    nyquist = fs_ref / 2
                    
                    n_data['fc'] = c1.number_input(
                        "Taglio HP (Hz)", 
                        min_value=0.0, 
                        value=float(n_data.get('fc', 50.0)), 
                        key=f"fc_hp_{n_id}"
                    )
                    
                    n_data['order'] = c2.number_input(
                        "Ordine HP", 1, 20, 
                        value=int(n_data.get('order', 4)), 
                        key=f"ord_hp_{n_id}"
                    )

                    if n_data['fc'] >= nyquist:
                        st.error(f"⚠️ Errore: {n_data['fc']}Hz > Nyquist ({nyquist:.1f}Hz)")
                    else:
                        st.success(f"✅ Passa-Alto attivo (> {n_data['fc']}Hz)")





    # --- ESECUZIONE ---
    st.markdown("---")
    if st.button("🚀 CALCOLA TUTTO", use_container_width=True, type="primary"):
        src_mapping = {get_node_id(n): st.session_state.get(f"sel_{get_node_id(n)}") for n in st.session_state.flow_state.nodes if get_node_type(n) == 'input'}
        results, error = run_calculation(st.session_state.flow_state, src_mapping)
        if error: st.error(error)
        else:
            st.session_state.engine_results = results
            st.success("Simulazione completata!")

    # --- PLOT RISULTATI ---
    if 'engine_results' in st.session_state:
        out_nodes = [n for n in st.session_state.flow_state.nodes if get_node_type(n) == 'output']
        for out in out_nodes:
            oid = get_node_id(out)
            if oid in st.session_state.engine_results:
                y = st.session_state.engine_results[oid]
                with st.expander(f"📊 {get_node_data(out).get('content', oid)}", expanded=True):
                    fig = go.Figure(go.Scatter(y=y, line=dict(color='#00FFCC', width=1.5)))
                    fig.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True, key=f"res_{oid}")
