import numpy as np

def normalize_segments(segments):
    max_vals = np.max(np.abs(segments), axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    return segments / max_vals