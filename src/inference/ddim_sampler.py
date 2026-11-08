"""Single-channel 3600-point DDIM inference using saved EMA weights."""
import numpy as np

from src.protocol3600 import denoise_raw, load_ema


class DDIMDenoiser:
    def __init__(self, config_path='configs/base.yaml',
                 checkpoint='output/fig05_mix7_3600/fig05_u4_3600_ema.pth', device=None):
        self.model, self.device = load_ema(config_path, checkpoint, device)

    def denoise(self, noisy_signal, ddim_steps=50, eta=0.0, num_shots=1):
        if ddim_steps != 50 or eta != 0 or num_shots != 1:
            raise ValueError('The reported paper protocol uses 50-step deterministic DDIM, one draw')
        signal = np.asarray(noisy_signal, dtype=np.float32).reshape(1, 3600)
        return denoise_raw(self.model, signal, self.device)
