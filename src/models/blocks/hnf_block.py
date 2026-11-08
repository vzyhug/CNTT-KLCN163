import torch
import torch.nn as nn

class HNFBlock(nn.Module):
    """
    Half Normalized Filter block.
    Multi-scale convs → concat → conv → half‑instance norm → residual.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 9, 15)):
        super().__init__()
        self.multi_convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes), k,
                      padding=k//2, padding_mode='reflect')
            for k in kernel_sizes
        ])
        self.agg_conv = nn.Conv1d(out_channels, out_channels, 1)
        # Half‑instance norm: chia channel làm 2 nửa
        self.half_inst_norm = nn.InstanceNorm1d(out_channels // 2)
        self.act = nn.ReLU(inplace=True)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        multi_out = [conv(x) for conv in self.multi_convs]
        out = torch.cat(multi_out, dim=1)
        out = self.agg_conv(out)
        half = out.shape[1] // 2
        out_norm = self.half_inst_norm(out[:, :half, :])
        out = torch.cat([out_norm, out[:, half:, :]], dim=1)
        out = self.act(out)
        return out + self.residual(x)