import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, clean, noisy):
        self.clean = torch.FloatTensor(clean).permute(0, 2, 1)  # (N, 1, L)
        self.noisy = torch.FloatTensor(noisy).permute(0, 2, 1)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return self.clean[idx], self.noisy[idx]