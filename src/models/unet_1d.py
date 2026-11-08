import torch
import torch.nn as nn
from .blocks import HNFBlock, BridgeBlock, SelfAttention1D

class UNet1D(nn.Module):
    """
    1D U‑Net với 4 levels, skip connections, HNF blocks,
    Bridge FiLM (timestep), và Self‑Attention tại bottleneck.
    """
    def __init__(self, in_channels=2, base_channels=80, emb_dim=128, out_channels=1):
        """
        Two input channels concatenate the diffusion state and single-lead ECG.
        """
        super().__init__()
        # ---------- ENCODER ----------
        # Level 1 (input -> 64)
        self.enc1 = HNFBlock(in_channels, base_channels)
        self.bridge1 = BridgeBlock(base_channels, emb_dim)
        self.down1 = nn.Conv1d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1)

        # Level 2 (64 -> 128)
        self.enc2 = HNFBlock(base_channels*2, base_channels*2)
        self.bridge2 = BridgeBlock(base_channels*2, emb_dim)
        self.down2 = nn.Conv1d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1)

        # Level 3 (128 -> 256)
        self.enc3 = HNFBlock(base_channels*4, base_channels*4)
        self.bridge3 = BridgeBlock(base_channels*4, emb_dim)
        self.down3 = nn.Conv1d(base_channels*4, base_channels*8, kernel_size=4, stride=2, padding=1)

        # Level 4 (256 -> 512) – bottleneck
        self.enc4 = HNFBlock(base_channels*8, base_channels*8)
        self.bridge4 = BridgeBlock(base_channels*8, emb_dim)
        # Self‑Attention tại bottleneck
        self.attn = SelfAttention1D(base_channels*8)

        # ---------- DECODER ----------
        # Level 4 -> 3
        self.up4 = nn.ConvTranspose1d(base_channels*8, base_channels*4, kernel_size=4, stride=2, padding=1)
        self.dec4 = HNFBlock(base_channels*8, base_channels*4)   # concat với skip từ enc3

        # Level 3 -> 2
        self.up3 = nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.dec3 = HNFBlock(base_channels*4, base_channels*2)   # concat với skip từ enc2

        # Level 2 -> 1
        self.up2 = nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = HNFBlock(base_channels*2, base_channels)     # concat với skip từ enc1

        # Output
        self.final = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, cond, noise_scale):
        """
        x: latent (x_t), cond: noisy observation, noise_scale: sqrt(alpha_bar)
        """
        # Concatenate x and cond along channel dimension
        inp = torch.cat([x, cond], dim=1)   # (B, 2, L)

        # ----- Encoder -----
        e1 = self.enc1(inp)
        e1 = self.bridge1(e1, noise_scale)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        e2 = self.bridge2(e2, noise_scale)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        e3 = self.bridge3(e3, noise_scale)
        d3 = self.down3(e3)

        e4 = self.enc4(d3)
        e4 = self.bridge4(e4, noise_scale)
        # Bottleneck self‑attention
        e4 = self.attn(e4)

        # ----- Decoder (with skip connections) -----
        u4 = self.up4(e4)
        u4 = torch.cat([u4, e3], dim=1)   # skip từ enc3
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, e2], dim=1)   # skip từ enc2
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, e1], dim=1)   # skip từ enc1
        d2 = self.dec2(u2)

        out = self.final(d2)
        return out
