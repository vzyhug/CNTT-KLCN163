"""Training entry point for the QTDB/NSTDB 3600-point paper protocol."""
from src.protocol3600 import train


def train_model(config_path='configs/base.yaml', model_choice=None, device=None,
                bundle='common_qtdb_mixed_bwem_ma_3600_ntype1.npz',
                output='checkpoints/3600', epochs=None):
    if model_choice not in (None, '1', 1):
        raise ValueError('The paper protocol trains only HNF-FiLM 1D U-Net')
    return train(bundle, config_path, output, device, epochs)
