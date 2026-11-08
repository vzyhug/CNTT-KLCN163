"""Access the fixed QTDB/NSTDB 3600-point paper benchmark pairs."""
from pathlib import Path

from src.protocol3600 import load_bundle


def Data_Preparation(n_type=1, force_rebuild=False, test_records=None,
                     bundle='common_qtdb_mixed_bwem_ma_3600_ntype1.npz'):
    if n_type != 1:
        raise ValueError('The paper benchmark uses the fixed mixed-noise n_type=1 bundle')
    if force_rebuild:
        raise ValueError('Raw QTDB/NSTDB regeneration is not part of this source path; use the fixed benchmark NPZ')
    pairs = load_bundle(Path(bundle), ('train', 'test'))
    if test_records is not None and set(test_records) != set(pairs['test'][2]):
        raise ValueError('Requested test records differ from the fixed paper split')
    train_clean, train_noisy, _ = pairs['train']
    test_clean, test_noisy, _ = pairs['test']
    return train_clean, train_noisy, test_clean, test_noisy
