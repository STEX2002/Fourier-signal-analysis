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
    # Il segnale viene iniettato dall'engine recuperandolo da session_state
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
    gain = data.get('val', 1.0)
    return inputs[0] * gain if inputs else np.array([0.0])

@registry.register('butterworth')
def handle_butter(inputs, data):
    if not inputs: return np.array([0.0])
    # Parametri tipici: ordine e freq. taglio normalizzata (0-1)
    order = data.get('order', 4)
    wn = data.get('wn', 0.2) 
    b, a = signal.butter(order, wn, btype='low')
    return signal.filtfilt(b, a, inputs[0])

@registry.register('abs')
def handle_abs(inputs, data):
    return np.abs(inputs[0]) if inputs else np.array([0.0])
