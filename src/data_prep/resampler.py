import os
import wfdb
import numpy as np
from scipy import signal

def process_ludb(input_dir, output_dir, target_fs=500):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.hea'):
            record_name = file[:-4]
            rec = wfdb.rdrecord(os.path.join(input_dir, record_name))
            sig = rec.p_signal
            current_fs = rec.fs
            if current_fs == target_fs:
                np.save(os.path.join(output_dir, f'{record_name}.npy'), sig)
                continue
            num_samples = int(sig.shape[0] * target_fs / current_fs)
            x_old = np.linspace(0, sig.shape[0], sig.shape[0])
            x_new = np.linspace(0, sig.shape[0], num_samples)
            sig_resampled = np.zeros((num_samples, sig.shape[1]))
            for ch in range(sig.shape[1]):
                sig_resampled[:, ch] = np.interp(x_new, x_old, sig[:, ch])
            np.save(os.path.join(output_dir, f'{record_name}.npy'), sig_resampled)
    print(f"Processing completed. Output: {output_dir}")
