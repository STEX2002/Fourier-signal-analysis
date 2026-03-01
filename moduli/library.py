import streamlit as st

def render_sidebar(add_node_func):
    """Disegna la libreria dei blocchi nella sidebar."""
    st.header("📦 Libreria Blocchi")
    
    # Selettore universale per blocchi multi-ingresso
    st.subheader("Configurazione Nuovo Blocco")
    num_inputs = st.number_input("Numero Ingressi:", min_value=2, max_value=8, value=2)
    
    # --- MENU IN/OUT ---
    with st.expander("🔌 IN / OUT", expanded=False):
        if st.button("➕ Nuova Sorgente", key="btn_new_src", use_container_width=True): 
            add_node_func("Sorgente", "input")
        if st.button("➕ Nuovo Oscilloscopio", key="btn_new_out", use_container_width=True): 
            add_node_func("Oscilloscopio", "output")
        st.divider()
        val_const = st.number_input("Valore Costante", value=1.0, step=0.1, key="const_val")
        if st.button("➕ Costante Reale", use_container_width=True):
        # Passiamo il valore come parametro extra
            add_node_func(f"Costante: {val_const}", "const", extra_data={'val': val_const})
        

    # --- OPERAZIONI ARITMETICHE ---
    with st.expander("➕ Operazioni Aritmetiche", expanded=True):
        if st.button("➕ Somma", key="btn_sum", use_container_width=True): 
            add_node_func(f"Somma", "sum", num_in=num_inputs)
        if st.button("➕ Moltiplicazione", key="btn_prod", use_container_width=True): 
            add_node_func(f"Prodotto", "prod", num_in=num_inputs)

    # --- FUNZIONI MATEMATICHE ---
    with st.expander("🧮 Funzioni Matematiche", expanded=False):
        if st.button("➕ Logaritmo", key="btn_log", use_container_width=True): add_node_func("Log", "log")
        if st.button("➕ Esponenziale", key="btn_exp", use_container_width=True): add_node_func("Exp", "exp")

    # --- FILTRI ---
    with st.expander("🌈 Filtri Ideali", expanded=False):
        if st.button("➕ Passa-Basso Ideale", key="btn_lpi", use_container_width=True): add_node_func("LP Ideale", "lpi")
    
    with st.expander("📉 Filtri Reali", expanded=False):
        if st.button("➕ Butterworth", key="btn_butt", use_container_width=True): add_node_func("Butterworth", "butt")

    st.markdown("---")
    if st.button("🗑️ Reset Canvas", type="primary", key="btn_reset", use_container_width=True):
        if 'flow_state' in st.session_state:
            del st.session_state.flow_state
            if 'multi_results' in st.session_state: del st.session_state.multi_results
            st.rerun()