
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState

def add_node(label, node_type='default', num_in=1):
    import time
    
    # 1. GESTIONE ID E NOMI
    if node_type == 'input':
        num = st.session_state.counter_in
        st.session_state.counter_in += 1
        new_id = f"src_{num}"
        display_name = f"Sorgente {num}"
        official_type = 'input' # Tipo ammesso dalla libreria
    elif node_type == 'output':
        num = st.session_state.counter_out
        st.session_state.counter_out += 1
        new_id = f"out_{num}"
        display_name = f"Oscilloscopio {num}"
        official_type = 'output' # Tipo ammesso dalla libreria
    else:
        # Per Somma, Prodotto, ecc. usiamo 'default'
        suffix = int(time.time() * 1000) % 10000
        new_id = f"{node_type}_{suffix}"
        display_name = f"{label} ({num_in} in)" if num_in > 1 else label
        official_type = 'default' # <--- FIX: Forza 'default' per evitare l'errore

    # 2. CREAZIONE NODO
    new_node = StreamlitFlowNode(
        id=new_id, 
        pos=(350, 200), 
        data={
            'content': display_name,
            'num_inputs': num_in,
            'logic_type': node_type  # <--- Salviamo qui il vero tipo (sum, prod, ecc.)
        }, 
        node_type=official_type, # Passiamo solo valori legali: 'default', 'input', 'output'
        source_position='right', 
        target_position='left',
        deletable=True
    )

    if 'flow_state' in st.session_state:
        st.session_state.flow_state.nodes.append(new_node)
        st.rerun()

def render():
    st.title("🏗️ Modular Signal Processor")
    
    if not st.session_state.segnali_caricati:
        st.warning("Carica almeno un segnale nella Tab Creazione.")
        return

    # --- 1. INIZIALIZZAZIONE STATO ---
    if 'flow_state' not in st.session_state:
        st.session_state.counter_in = 1
        st.session_state.counter_out = 1
        nodes = [
            StreamlitFlowNode('src_0', (50, 150), {'content': 'Sorgente 0'}, 'input', 'right', 'left', deletable=True),
            StreamlitFlowNode('out_0', (750, 150), {'content': 'Oscilloscopio 0'}, 'output', 'right', 'left', deletable=True),
        ]
        edges = [StreamlitFlowEdge(id='edge_0', source='src_0', target='out_0', animated=True, deletable=True)]
        st.session_state.flow_state = StreamlitFlowState(nodes, edges)

    # --- 2. SIDEBAR (Invariata) ---
    # --- 2. SIDEBAR: LIBRERIA CATEGORIZZATA ---
    # Assicurati che questo blocco sia visibile
    with st.sidebar:
        st.header("📦 Libreria Blocchi")
        
        # Selettore universale per blocchi multi-ingresso
        st.subheader("Configurazione Nuovo Blocco")
        num_inputs = st.number_input("Numero Ingressi:", min_value=2, max_value=8, value=2)
        
        # --- MENU IN/OUT ---
        with st.expander("🔌 IN / OUT", expanded=False):
            if st.button("➕ Nuova Sorgente", key="btn_new_src"): 
                add_node("Sorgente", "input")
            if st.button("➕ Nuovo Oscilloscopio", key="btn_new_out"): 
                add_node("Oscilloscopio", "output")

        # --- OPERAZIONI ARITMETICHE ---
        with st.expander("➕ Operazioni Aritmetiche", expanded=True):
            if st.button("➕ Somma", key="btn_sum"): 
                add_node(f"Somma ({num_inputs} in)", "sum", num_in=num_inputs)
            if st.button("➕ Moltiplicazione", key="btn_prod"): 
                add_node(f"Prodotto ({num_inputs} in)", "prod", num_in=num_inputs)

        # --- FUNZIONI MATEMATICHE ---
        with st.expander("🧮 Funzioni Matematiche", expanded=False):
            if st.button("➕ Logaritmo", key="btn_log"): add_node("Log")
            if st.button("➕ Esponenziale", key="btn_exp"): add_node("Exp")

        # --- FILTRI IDEALI ---
        with st.expander("🌈 Filtri Ideali", expanded=False):
            if st.button("➕ Passa-Basso Ideale", key="btn_lpi"): add_node("LP Ideale")
            if st.button("➕ Passa-Alto Ideale", key="btn_hpi"): add_node("HP Ideale")

        # --- FILTRI REALI ---
        with st.expander("📉 Filtri Reali", expanded=False):
            if st.button("➕ Butterworth", key="btn_butt"): add_node("Butterworth")
            if st.button("➕ Chebyshev", key="btn_cheb"): add_node("Chebyshev")

        # --- FILTRI DIGITALI ---
        with st.expander("🔢 Filtri Digitali", expanded=False):
            if st.button("➕ FIR", key="btn_fir"): add_node("FIR")
            if st.button("➕ IIR", key="btn_iir"): add_node("IIR")

        st.markdown("---")
        if st.button("🗑️ Reset Canvas", type="primary", key="btn_reset"):
            del st.session_state.flow_state
            if 'multi_results' in st.session_state: del st.session_state.multi_results
            st.rerun()

    # --- 3. CONFIGURAZIONE INGRESSI (SPOSTATA SOPRA IL CANVAS) ---
    def get_node_type(node):
        return getattr(node, 'node_type', getattr(node, 'type', 'default'))

    src_nodes = [n for n in st.session_state.flow_state.nodes if get_node_type(n) == 'input']
    out_nodes = [n for n in st.session_state.flow_state.nodes if get_node_type(n) == 'output']
    
    src_mapping = {}
    if src_nodes:
        with st.container():
            st.write("### ⚙️ Configurazione Ingressi")
            cols = st.columns(len(src_nodes))
            for i, node in enumerate(src_nodes):
                display_name = node.data.get('content', node.id)
                src_mapping[node.id] = cols[i].selectbox(
                    f"Segnale per {display_name}:", 
                    list(st.session_state.segnali_caricati.keys()), 
                    key=f"map_{node.id}"
                )
        st.markdown("---")

    # --- 4. IL CANVAS ---
    updated_state = streamlit_flow(
        "modular_flow_canvas", 
        st.session_state.flow_state, 
        height=550, 
        allow_new_edges=True,
        show_minimap=True,
        hide_watermark=True,
        enable_node_menu=True,       
        enable_edge_menu=True,       
        fit_view=False
    )

    if updated_state:
        current_edges_count = len(st.session_state.flow_state.edges)
        if updated_state != st.session_state.flow_state:
            new_edges = []
            for i, edge in enumerate(updated_state.edges):
                new_edges.append(StreamlitFlowEdge(
                    id=f"e_{edge.source}_{edge.target}_{i}", 
                    source=edge.source, target=edge.target, 
                    animated=True, deletable=True,
                    style={'stroke': '#00FFCC', 'strokeWidth': 2}
                ))
            st.session_state.flow_state = StreamlitFlowState(updated_state.nodes, new_edges)
            if len(new_edges) != current_edges_count:
                st.rerun()

    # --- 5. MOTORE DI CALCOLO E RISULTATI (SOTTO IL CANVAS) ---

    # --- 4. MOTORE DI CALCOLO (Versione Vettoriale) ---
    if st.button("🚀 ESEGUI SIMULAZIONE", use_container_width=True, type="primary", key="btn_run"):
        node_outputs = {}
        
        # 1. Inizializziamo le Sorgenti
        for s_node in src_nodes:
            # Recuperiamo il segnale mappato dall'utente
            signal_name = src_mapping.get(s_node.id)
            if signal_name:
                node_outputs[s_node.id] = st.session_state.segnali_caricati[signal_name].copy()

        # 2. Propagazione del segnale (Algoritmo a passaggi)
        all_nodes = st.session_state.flow_state.nodes
        # Eseguiamo N passaggi per coprire tutta la profondità del grafo
        for _ in range(len(all_nodes)):
            for node in all_nodes:
                if node.id in node_outputs: continue
                
                # Troviamo chi entra in questo nodo
                incoming = [e for e in st.session_state.flow_state.edges if e.target == node.id]
                
                # Se tutti i predecessori hanno già un'uscita pronta
                if incoming and all(e.source in node_outputs for e in incoming):
                    # Lista dei segnali (array numpy) in ingresso
                    inputs = [node_outputs[e.source] for e in incoming]
                    
                    # --- LOGICA MATEMATICA BASATA SU logic_type ---
                    l_type = node.data.get('logic_type', 'default')
                    
                    if l_type == 'sum':
                        # Somma vettoriale: y_tot = y1 + y2 + ...
                        node_outputs[node.id] = np.sum(inputs, axis=0)
                    elif l_type == 'prod':
                        # Prodotto vettoriale: y_tot = y1 * y2 * ...
                        node_outputs[node.id] = np.prod(inputs, axis=0)
                    else:
                        # Per default (o oscilloscopio), passa il primo segnale che arriva
                        node_outputs[node.id] = inputs[0]

        # 3. Raccolta Risultati per gli Oscilloscopi
        results = {}
        for out_node in out_nodes:
            if out_node.id in node_outputs:
                y_final = node_outputs[out_node.id]
                # Recuperiamo info temporali (fs, durata) per il plot
                # Usiamo i dati del primo segnale caricato come riferimento
                first_sig = list(st.session_state.info_segnali.values())[0]
                t = np.linspace(0, first_sig["durata"], len(y_final), endpoint=False)
                
                results[out_node.id] = {
                    't': t, 
                    'y': y_final, 
                    'name': out_node.data.get('content', 'Output'),
                    'info': first_sig
                }
        
        st.session_state.multi_results = results

    # --- 5. RISULTATI E PLOT ---
    if 'multi_results' in st.session_state:
        for out_id, data in st.session_state.multi_results.items():
            # Titolo dell'expander basato sul nome del nodo (es. "Oscilloscopio 2")
            with st.expander(f"📊 Risultato: {data['name']}", expanded=True):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['t'], 
                    y=data['y'], 
                    name=data['name'],
                    line=dict(color='#00FFCC', width=2)
                ))
                
                fig.update_layout(
                    template="plotly_dark", 
                    height=300, 
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Tempo [s]",
                    yaxis_title="Ampiezza"
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{out_id}")
                
                # Opzione di salvataggio del segnale elaborato
                c1, c2 = st.columns([3, 1])
                default_save_name = f"Segnale_{data['name'].replace(' ', '_')}"
                s_name = c1.text_input("Nome salvataggio:", value=default_save_name, key=f"save_name_{out_id}")
                
                if c2.button("💾 Salva", key=f"save_btn_{out_id}"):
                    st.session_state.segnali_caricati[s_name] = data['y'].copy()
                    st.session_state.info_segnali[s_name] = data['info'].copy()
                    st.toast(f"Segnale '{s_name}' salvato con successo!")