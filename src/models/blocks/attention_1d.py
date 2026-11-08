import torch
import torch.nn as nn

class SelfAttention1D(nn.Module):
    """1D Self‑Attention (multi‑head) dùng tại bottleneck."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels phải chia hết cho num_heads"

        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # q, k, v: (B, num_heads, head_dim, L)
        attn = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, L)
        return self.proj(out)