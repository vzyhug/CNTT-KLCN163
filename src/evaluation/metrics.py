import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def SSD(y,y_pred):
    return np.sum(np.square(y-y_pred),axis=1)

def MAD(y,y_pred):
    return np.max(np.abs(y-y_pred),axis=1)

def PRD(y,y_pred):
    N = np.sum(np.square(y_pred-y),axis=1)
    D = np.sum(np.square(y_pred-np.mean(y)),axis=1)

    PRD = np.sqrt(N/D) * 100

    return PRD


def COS_SIM(y, y_pred):
    cos_sim = []
    y = np.squeeze(y, axis=-1)
    y_pred = np.squeeze(y_pred, axis=-1)
    for idx in range(len(y)):
        kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
        cos_sim.append(kl_temp)

    cos_sim = np.array(cos_sim)
    return cos_sim


def SNR(y1, y2):
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)

    SNR = 10 * np.log10(N / D)

    return SNR


def SNR_improvement(y_in, y_out, y_clean):
    return SNR(y_clean, y_out) - SNR(y_clean, y_in)

def calculate_metrics(clean,denoised):
    # Đưa tín hiệu về mảng 2D (Batch, Chiều dài tín hiệu)
    clean = clean.view(clean.shape[0], -1)
    denoised = denoised.view(denoised.shape[0], -1)

    # 1. RMSE (Sai số toàn phương trung bình)
    mse = torch.mean((clean - denoised) ** 2, dim=1)
    rmse = torch.sqrt(mse).mean().item()

    # 2. SNR (Tỷ lệ tín hiệu trên nhiễu)
    signal_power = torch.sum(clean ** 2, dim=1)
    noise_power = torch.sum((clean - denoised) ** 2, dim=1)
    snr = (10 * torch.log10(signal_power / (noise_power + 1e-8))).mean().item()

    # 3. PRD (Tỷ lệ phần trăm sai khác)
    prd = (torch.sqrt(noise_power / (signal_power + 1e-8)) * 100).mean().item()

    # 4. Cosine Similarity (Độ tương đồng Cosin)
    cos_sim = F.cosine_similarity(clean, denoised, dim=1).mean().item()

    return rmse, snr, prd, cos_sim