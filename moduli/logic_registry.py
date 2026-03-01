import numpy as np
from scipy import signal

class SignalRegistry:
    def __init__(self):
        self.blocks = {}

    def register(self, name):
        def decorator(func):
            self.blocks[name] = func
            return func
        return decorator

    def execute(self, logic_type, inputs, node_data):
        if logic_type in self.blocks:
            return self.blocks[logic_type](inputs, node_data)
        # Default: Pass-through se c'è almeno un ingresso
        return inputs[0] if inputs else np.array([0.0])

registry = SignalRegistry()

# --- DEFINIZIONE BLOCCHI ---

@registry.register('input')
def handle_input(inputs, data):
    # 'data' qui è la copia temporanea che contiene l'array NumPy
    return data.get('signal_value', np.array([0.0]))


@registry.register('sum')
def handle_sum(inputs, data):
    if not inputs: return np.array([0.0])
    return np.sum(inputs, axis=0)

@registry.register('prod')
def handle_prod(inputs, data):
    if not inputs: return np.array([0.0])
    return np.prod(inputs, axis=0)

@registry.register('gain')
def handle_gain(inputs, data):
    k = data.get('val', 1.0) # Prende il valore dallo slider
    return inputs[0] * k if inputs else np.array([0.0])

@registry.register('butterworth')
def handle_butter(inputs, data):
    if not inputs or len(inputs[0]) == 0: 
        return np.array([0.0])
    
    signal_in = inputs[0]
    
    # 1. Recuperiamo la fs (passata dal motore di calcolo nel nodo input)
    # Se non disponibile (es. segnale generato internamente), usiamo 1000Hz
    fs = data.get('fs_calcolata', 1000.0)
    
    # 2. Parametri utente
    fc_user = data.get('fc', 10.0)
    order = data.get('order', 4)
    
    # 3. LIMITATORE AUTOMATICO (Nyquist)
    # Il filtro Butterworth richiede fc < fs/2. 
    # Se l'utente esagera, limitiamo a fs/2 - un piccolo margine.
    nyquist = fs / 2
    fc_safe = min(fc_user, nyquist - 0.001)
    
    # Se fc_safe è troppo vicino allo zero (o zero), non filtriamo
    if fc_safe <= 0:
        return signal_in

    try:
        b, a = signal.butter(order, fc_safe, btype='low', fs=fs)
        return signal.filtfilt(b, a, signal_in)
    except Exception as e:
        # Se qualcosa va storto (es. ordine troppo alto), restituiamo il segnale originale
        return signal_in


@registry.register('abs')
def handle_abs(inputs, data):
    """Calcola il valore assoluto del segnale in ingresso"""
    if not inputs or len(inputs[0]) == 0:
        return np.array([0.0])
    
    # Prende il primo ingresso e applica np.abs
    return np.abs(inputs[0])


@registry.register('butter_hp')
def handle_butter_hp(inputs, data):
    if not inputs or len(inputs[0]) == 0: 
        return np.array([0.0])
    
    signal_in = inputs[0]
    fs = data.get('fs_calcolata', 1000.0)
    fc_user = data.get('fc', 10.0)
    order = data.get('order', 4)
    
    # Limitatore Nyquist
    nyquist = fs / 2
    fc_safe = min(max(fc_user, 0.001), nyquist - 0.001)

    try:
        # btype='high' per il passa-alto
        b, a = signal.butter(order, fc_safe, btype='high', fs=fs)
        return signal.filtfilt(b, a, signal_in)
    except:
        return signal_in

