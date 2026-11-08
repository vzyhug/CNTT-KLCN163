"""Compatibility entry point for 3600-point paper evaluation."""
from src.protocol3600 import evaluate


def evaluate_and_plot(bundle='common_qtdb_mixed_bwem_ma_3600_ntype1.npz',
                      config='configs/base.yaml',
                      checkpoint='output/fig05_mix7_3600/fig05_u4_3600_ema.pth',
                      device=None):
    return evaluate(bundle, config, checkpoint, device)


if __name__ == '__main__':
    print(evaluate_and_plot())
