"""Run the 3600-point single-lead QTDB/NSTDB protocol without notebooks."""
import argparse
from pathlib import Path

from src.protocol3600 import denoise_raw, evaluate, load_bundle, load_ema, train

DEFAULT_BUNDLE = Path('common_qtdb_mixed_bwem_ma_3600_ntype1.npz')
DEFAULT_CONFIG = Path('configs/base.yaml')
DEFAULT_CHECKPOINT = Path('output/fig05_mix7_3600/fig05_u4_3600_ema.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preprocess', 'train', 'infer', 'eval'), required=True)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/3600'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--index', type=int, default=0, help='Held-out test example for infer mode')
    args = parser.parse_args()

    if args.mode == 'preprocess':
        # The fixed pair bundle already contains the published preprocessing and splits.
        data = load_bundle(args.bundle)
        for split, (clean, _, records) in data.items():
            print(f'{split}: {clean.shape}, {len(set(records))} source records')
    elif args.mode == 'train':
        print(train(args.bundle, args.config, args.output, args.device, args.epochs))
    elif args.mode == 'infer':
        model, device = load_ema(args.config, args.checkpoint, args.device)
        clean, noisy, _ = load_bundle(args.bundle, ('test',))['test']
        prediction = denoise_raw(model, noisy[args.index], device)
        import numpy as np
        args.output.mkdir(parents=True, exist_ok=True)
        target = args.output / f'inference_test_{args.index}_3600.npz'
        np.savez_compressed(target, clean=clean[args.index], noisy=noisy[args.index], prediction=prediction)
        print(target)
    else:
        print(evaluate(args.bundle, args.config, args.checkpoint, args.device))


if __name__ == '__main__':
    main()
