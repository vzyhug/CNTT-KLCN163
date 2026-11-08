import os
import wfdb
import numpy as np

def synthesize_noisy(clean_segments, noise_dir, output_dir, delta_range=(0.2, 2.0), seed=42):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách 3 loại nhiễu từ MIT-BIH NST
    noise_types = ['bw', 'em', 'ma']
    noises = {}
    
    # Đọc từng loại nhiễu
    for n_type in noise_types:
        path = os.path.join(noise_dir, n_type)
        try:
            # Lấy kênh đầu tiên
            rec = wfdb.rdrecord(path)
            sig = rec.p_signal[:, 0]
            current_fs = rec.fs
            if current_fs != 500:
                # Noi suy len 500Hz
                num_samples = int(len(sig) * 500 / current_fs)
                x_old = np.linspace(0, len(sig), len(sig))
                x_new = np.linspace(0, len(sig), num_samples)
                sig = np.interp(x_new, x_old, sig)
            noises[n_type] = sig
        except Exception as e:
            print(f"Warning: Could not load {path} - {e}")
            
    if not noises:
        raise ValueError(f"Không tìm thấy file nhiễu nào trong {noise_dir}. Cần tải bw.dat, em.dat, ma.dat")

    num_aug = 3 # Tạo 3 bản nhiễu khác nhau cho mỗi segment gốc
    noisy_list, clean_list, deltas = [], [], []
    for seg in clean_segments:
        for _ in range(num_aug):
            # Chọn ngẫu nhiên 1 loại nhiễu cho segment này
            selected_noise = np.random.choice(list(noises.keys()))
            noise_sig = noises[selected_noise]
            
            start = np.random.randint(0, len(noise_sig) - 512)
            noise_patch = noise_sig[start:start+512]
            
            # scale noise theo SNR
            power_clean = np.mean(seg**2)
            power_noise = np.mean(noise_patch**2)
            scale = np.sqrt(power_clean / power_noise) if power_noise > 0 else 1.0
            
            delta = np.random.uniform(*delta_range)
            noise_scaled = (noise_patch * scale * delta)
            
            # Expand dims for 12 channels broadcast if seg is multi-channel
            if seg.ndim == 2:
                noise_scaled = noise_scaled[:, np.newaxis]
                
            noisy = seg + noise_scaled
            
            noisy_list.append(noisy.astype(np.float32))
            clean_list.append(seg.astype(np.float32))
            deltas.append(delta)
        
    np.save(os.path.join(output_dir, 'noisy.npy'), np.array(noisy_list))
    np.save(os.path.join(output_dir, 'clean.npy'), np.array(clean_list))
    np.save(os.path.join(output_dir, 'deltas.npy'), np.array(deltas))
    print(f"Synthesized {len(noisy_list)} samples mixing {list(noises.keys())} noises.")