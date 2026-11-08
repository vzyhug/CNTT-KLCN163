import torch
import torch.nn as nn
import math

class BridgeBlock(nn.Module):
    """FiLM‑based conditioning on noise level (sqrt(alpha_bar))."""
    def __init__(self, features, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.film = nn.Sequential(
            nn.Linear(emb_dim, features * 2),
            nn.SiLU()
        )

    def forward(self, x, alpha_bar):
        # alpha_bar: (B,) or scalar
        # Ép phẳng alpha_bar về dạng 1D (Batch,) để tránh bị dư chiều -- CỰC QUAN TRỌNG
        alpha_bar = alpha_bar.view(-1)
        emb = self.sinusoidal_embedding(alpha_bar)
        scale_shift = self.film(emb)   # (B, 2*features)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift

    def sinusoidal_embedding(self, x):
        if x.dim() == 0:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb