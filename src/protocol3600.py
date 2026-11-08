"""Runnable single-lead QTDB/NSTDB 3600-point protocol described in paper-eng.tex.

The fixed NPZ bundle is the canonical data artifact. This module does not
regenerate its QTDB preprocessing or NSTDB mixtures from raw records.
"""
from __future__ import annotations

import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.unet_1d import UNet1D

LENGTH = 3600
FS = 360
TEST_RECORDS = {
    'sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853',
    'sel16420', 'sel16795', 'sele0106', 'sele0121', 'sel32',
    'sel49', 'sel14046', 'sel15814',
}
COUNTS = {'train': 22804, 'val': 9774, 'test': 5012}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bundle(path: str | Path, splits=('train', 'val', 'test')) -> dict:
    """Load fixed pairs and reject a bundle with a different paper protocol."""
    with np.load(path, allow_pickle=False) as source:
        if int(source['sampling_rate']) != FS:
            raise ValueError(f'Expected {FS} Hz sampling rate')
        result = {}
        for split in splits:
            clean = np.asarray(source[f'clean_{split}'], dtype=np.float32)
            noisy = np.asarray(source[f'noisy_{split}'], dtype=np.float32)
            records = np.asarray(source[f'record_id_{split}']).astype(str)
            expected = (COUNTS[split], 1, LENGTH)
            if clean.shape != expected or noisy.shape != expected:
                raise ValueError(f'{split}: expected {expected}, got {clean.shape}/{noisy.shape}')
            if len(records) != expected[0] or not np.isfinite(clean).all() or not np.isfinite(noisy).all():
                raise ValueError(f'{split}: invalid records or nonfinite signal')
            if split == 'test' and set(records) != TEST_RECORDS:
                raise ValueError('Test record IDs differ from the paper protocol')
            if split != 'test' and set(records) & TEST_RECORDS:
                raise ValueError(f'{split}: held-out test record leakage')
            result[split] = (clean, noisy, records)
    return result


class ECGPairs(Dataset):
    def __init__(self, clean: np.ndarray, noisy: np.ndarray):
        self.clean, self.noisy = clean, noisy

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, index: int):
        x = self.clean[index]
        y = self.noisy[index]
        scale = max(float(np.max(np.abs(y))), 1e-8)
        return torch.from_numpy(x / scale), torch.from_numpy(y / scale), scale


class ConditionalDDPM(nn.Module):
    def __init__(self, beta_start=1e-4, beta_end=0.5, steps=50, channels=80):
        super().__init__()
        self.net = UNet1D(in_channels=2, base_channels=channels, emb_dim=128, out_channels=1)
        beta = torch.linspace(beta_start**0.5, beta_end**0.5, steps).square()
        alpha_bar = torch.cumprod(1.0 - beta, dim=0)
        self.register_buffer('alpha_bar', torch.cat((torch.ones(1), alpha_bar)))
        self.steps = steps

    def loss(self, x0: torch.Tensor, y: torch.Tensor, generator=None):
        if x0.shape != y.shape or x0.ndim != 3 or x0.shape[1:] != (1, LENGTH):
            raise ValueError('Expected paired B x 1 x 3600 ECG tensors')
        batch = x0.shape[0]
        t = torch.randint(1, self.steps + 1, (batch,), device=x0.device, generator=generator)
        hi = self.alpha_bar[t - 1].sqrt()
        lo = self.alpha_bar[t].sqrt()
        s = (lo + torch.rand(batch, device=x0.device, generator=generator) * (hi - lo)).view(-1, 1, 1)
        eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        xs = s * x0 + torch.sqrt((1.0 - s.square()).clamp_min(0)) * eps
        eps_hat = self.net(xs, y, s.flatten(1))
        l_eps = (eps_hat - eps).abs().mean(dim=(1, 2))
        x0_hat = (xs - torch.sqrt((1.0 - s.square()).clamp_min(0)) * eps_hat) / s.clamp_min(1e-6)
        pred_spec = torch.stft(x0_hat[:, 0], n_fft=128, hop_length=64,
                               window=torch.hann_window(128, device=x0.device),
                               center=True, pad_mode='reflect', return_complex=True).abs()
        ref_spec = torch.stft(x0[:, 0], n_fft=128, hop_length=64,
                              window=torch.hann_window(128, device=x0.device),
                              center=True, pad_mode='reflect', return_complex=True).abs()
        relative = (pred_spec - ref_spec).square().mean(dim=(1, 2)) / ref_spec.square().mean(dim=(1, 2)).clamp_min(1e-6)
        weight = (s.flatten().square() / (1 - s.flatten().square()).clamp_min(1e-6)).clamp(max=1)
        return (l_eps + 0.1 * weight * relative).mean()

    @torch.no_grad()
    def ddim(self, y: torch.Tensor, seed=2024, steps=50, generator=None,
             return_history=False, return_states=False, initial_noise=None):
        if y.ndim != 3 or y.shape[1:] != (1, LENGTH):
            raise ValueError('Expected B x 1 x 3600 noisy ECG')
        if not 1 <= steps <= self.steps:
            raise ValueError('DDIM steps must be between 1 and T')
        device = y.device
        rng = generator or torch.Generator(device=device).manual_seed(seed)
        if initial_noise is None:
            x = torch.randn(y.shape, device=device, dtype=y.dtype, generator=rng)
        else:
            if initial_noise.shape != y.shape:
                raise ValueError('initial_noise must have the same shape as y')
            x = initial_noise.to(device=device, dtype=y.dtype).clone()
        tau = [int(np.floor(self.steps * (j / steps)**2)) for j in range(steps + 1)]
        history = []
        states = [(0, tau[-1], x.detach().cpu().numpy().copy())] if return_states else []
        captures = {steps, max(1, steps * 4 // 5), max(1, steps * 3 // 5),
                    max(1, steps * 2 // 5), max(1, steps // 5), 1}
        for j in range(steps, 0, -1):
            t, previous = tau[j], tau[j - 1]
            at, ap = self.alpha_bar[t], self.alpha_bar[previous]
            eps_hat = self.net(x, y, at.sqrt().expand(len(y), 1))
            x0_hat = (x - (1 - at).sqrt() * eps_hat) / at.sqrt().clamp_min(1e-6)
            if return_history and j in captures:
                history.append((steps - j + 1, t, x0_hat.detach().cpu().numpy().copy()))
            x = ap.sqrt() * x0_hat + (1 - ap).sqrt() * eps_hat
            if return_states:
                states.append((steps - j + 1, previous, x.detach().cpu().numpy().copy()))
        if return_states:
            return x, history, states
        return (x, history) if return_history else x


def build_model(config: dict, device: torch.device) -> ConditionalDDPM:
    train_cfg, diff_cfg = config['train'], config['diffusion']
    return ConditionalDDPM(diff_cfg['beta_start'], diff_cfg['beta_end'],
                           diff_cfg['num_steps'], train_cfg['feats']).to(device)


def load_config(path: str | Path) -> dict:
    import yaml
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if config['train']['feats'] != 80 or config['diffusion']['num_steps'] != 50:
        raise ValueError('The paper protocol requires 80 base channels and T=50')
    experiment = config['experiment']
    if (experiment['dataset'], experiment['noise_dataset'], experiment['sample_rate'],
            experiment['signal_length'], experiment['signal_channels']) != ('QTDB', 'NSTDB', 360, 3600, 1):
        raise ValueError('The config does not describe the paper data protocol')
    return config


def train(bundle: str | Path, config_path: str | Path, output: str | Path,
          device: str | None = None, epochs: int | None = None) -> Path:
    config = load_config(config_path)
    seed_all(config['train']['seed'])
    pairs = load_bundle(bundle, ('train', 'val'))
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model(config, device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    micro = config['train']['micro_batch_size']
    accumulation = config['train']['gradient_accumulation_steps']
    if micro * accumulation != config['train']['batch_size']:
        raise ValueError('Effective batch size must equal 96')
    train_loader = DataLoader(ECGPairs(*pairs['train'][:2]), batch_size=micro, shuffle=True)
    val_loader = DataLoader(ECGPairs(*pairs['val'][:2]), batch_size=micro, shuffle=False)
    best = float('inf')
    total_epochs = epochs or config['train']['epochs']
    with (output / 'training_3600.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for epoch in range(1, total_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_total = 0.0
            updates = 0
            for index, (x, y, _) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                value = model.loss(x, y)
                (value / accumulation).backward()
                train_total += float(value.detach())
                if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['gradient_clip_norm'])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        for target, source in zip(ema.net.parameters(), model.net.parameters()):
                            target.lerp_(source, 1 - config['train']['ema_decay'])
                    updates += 1
            model.eval()
            val_rng = torch.Generator(device=device).manual_seed(config['train']['seed'])
            val_total = 0.0
            with torch.no_grad():
                for x, y, _ in val_loader:
                    val_total += float(model.loss(x.to(device), y.to(device), generator=val_rng))
            val_loss = val_total / len(val_loader)
            writer.writerow([epoch, train_total / len(train_loader), val_loss])
            handle.flush()
            state = {'epoch': epoch, 'model': model.net.state_dict(),
                     'ema': ema.net.state_dict(), 'optimizer': optimizer.state_dict(),
                     'validation_loss': val_loss, 'protocol': 'qtdb_nstdb_3600_v1'}
            torch.save(state, output / 'last_3600.pth')
            if val_loss < best:
                best = val_loss
                torch.save(state, output / 'best_3600.pth')
            print(f'epoch={epoch} train={train_total / len(train_loader):.6f} val={val_loss:.6f} updates={updates}')
    return output / 'best_3600.pth'


def load_ema(config_path: str | Path, checkpoint: str | Path, device: str | None = None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model(load_config(config_path), device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if 'ema' in state:
        weights = state['ema']
    elif 'ema_model_state_dict' in state:
        weights = {key.removeprefix('model.'): value
                   for key, value in state['ema_model_state_dict'].items()
                   if key.startswith('model.')}
    else:
        raise ValueError('Checkpoint must include EMA weights from the 3600-point protocol')
    model.net.load_state_dict(weights, strict=True)
    return model.eval(), device


def denoise_raw(model: ConditionalDDPM, raw: np.ndarray, device, seed=2024) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (1, LENGTH) or not np.isfinite(raw).all():
        raise ValueError('Expected one finite ECG of shape (1, 3600)')
    scale = max(float(np.max(np.abs(raw))), 1e-8)
    y = torch.from_numpy(raw[None] / scale).to(device)
    return (model.ddim(y, seed=seed)[0].cpu().numpy() * scale)


def metrics(clean: np.ndarray, noisy: np.ndarray, prediction: np.ndarray) -> dict:
    clean, noisy, prediction = map(lambda a: np.asarray(a, dtype=np.float64).reshape(-1),
                                   (clean, noisy, prediction))
    if not (len(clean) == len(noisy) == len(prediction) == LENGTH):
        raise ValueError('Metrics require aligned 3600-point signals')
    error = clean - prediction
    signal = np.square(clean).sum()
    residual = np.square(error).sum()
    input_error = np.square(clean - noisy).sum()
    centered = np.square(clean - clean.mean()).sum()
    predicted_energy = np.square(prediction).sum()
    if (min(signal, input_error, centered, predicted_energy, residual) <= 0 or
            not np.isfinite(clean).all() or not np.isfinite(noisy).all() or
            not np.isfinite(prediction).all()):
        raise ValueError('Zero-energy or nonfinite metric input')
    return {'SSD': residual, 'RMSE': np.sqrt(np.square(error).mean()),
            'MAD': np.abs(error).max(), 'PRD': 100 * np.sqrt(residual / centered),
            'Cosine': float(clean @ prediction / np.sqrt(signal * predicted_energy)),
            'SNR_in': 10 * np.log10(signal / input_error),
            'SNR_out': 10 * np.log10(signal / residual),
            'Delta_SNR': 10 * np.log10(input_error / residual)}


def evaluate(bundle: str | Path, config_path: str | Path, checkpoint: str | Path,
             device: str | None = None, seed=2024) -> dict:
    model, device = load_ema(config_path, checkpoint, device)
    clean, noisy, _ = load_bundle(bundle, ('test',))['test']
    results = []
    rng = torch.Generator(device=device).manual_seed(seed)
    batch_size = load_config(config_path)['experiment']['inference_batch_size']
    for start in range(0, len(clean), batch_size):
        stop = min(start + batch_size, len(clean))
        raw = noisy[start:stop]
        scale = np.maximum(np.max(np.abs(raw), axis=(1, 2), keepdims=True), 1e-8)
        y = torch.from_numpy(raw / scale).to(device)
        prediction = model.ddim(y, seed=seed, generator=rng).cpu().numpy() * scale
        results.extend(metrics(clean[i], noisy[i], prediction[i - start])
                       for i in range(start, stop))
    return {key: float(np.mean([row[key] for row in results])) for key in results[0]}
