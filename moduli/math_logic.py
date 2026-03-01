import numpy as np

def process_node(logic_type, inputs, node_data):
    """
    Esegue l'operazione matematica. 
    inputs: lista di array numpy dai nodi precedenti.
    node_data: dizionario con i parametri del nodo (es. 'val').
    """
    # Se il nodo è una costante, restituisce il suo valore interno
    if logic_type == 'const':
        return np.array([node_data.get('val', 1.0)])

    # Se non ci sono ingressi per gli altri blocchi, ritorna zero
    if not inputs:
        return np.array([0.0])

    # LOGICA DEI BLOCCHI
    try:
        if logic_type == 'sum':
            # Somma tutti gli ingressi (NumPy gestisce il broadcast se uno è scalare)
            res = inputs[0]
            for i in range(1, len(inputs)):
                res = np.add(res, inputs[i])
            return res
        
        elif logic_type == 'prod':
            res = inputs[0]
            for i in range(1, len(inputs)):
                res = np.multiply(res, inputs[i])
            return res

        elif logic_type == 'gain':
            return inputs[0] * node_data.get('val', 1.0)

        elif logic_type == 'abs':
            return np.abs(inputs[0])

        # Default: Pass-through (Oscilloscopio o blocchi ignoti)
        return inputs[0]
            
    except Exception as e:
        print(f"Errore nel blocco {logic_type}: {e}")
        return inputs[0]