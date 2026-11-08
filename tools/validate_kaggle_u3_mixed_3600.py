"""CPU synthetic preflight; does not train on or evaluate QTDB."""
from pathlib import Path
import ast
import base64
import copy
import io
import json
import sys
import random
import shutil
import tempfile
from unittest.mock import patch
import nbformat
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
nb = nbformat.read(ROOT/'notebooks/kaggle_train_u3_qtdb_mixed_3600_100ep.ipynb', 4)
old = nbformat.read(ROOT/'notebooks/kaggle_train_qtdb_bw_em_ma.ipynb', 4)
nbformat.validate(nb)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code' and not c.source.startswith('%pip'):
        compile(c.source, f'cell_{i}', 'exec')

def assignment(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))

payload = ast.literal_eval(assignment(nb.cells[6].source, 'EMBEDDED_SOURCE_B64').value)
previous = json.loads(ast.literal_eval(assignment(old.cells[6].source, 'EMBEDDED_SOURCE_B64').value.args[0]))
changed = [k for k in payload if payload[k] != previous[k]]
assert changed == ['src/models/unet_1d.py'], changed
snapshot = ROOT/'tmp/u3_3600_validation_sources'
for name, data in payload.items():
    p = snapshot/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(data))
sys.path.insert(0, str(snapshot))
from src.models.unet_1d import UNet1D
from src.models.main_model import DDPM, default, EMA

# Evaluate only pure configuration assignments, without Kaggle paths/GPU checks.
ns = {'torch': torch, 'np': np, 'F': F, 'DDPM': DDPM,
      'default': default, 'UNet1D': UNet1D, 'DEVICE': torch.device('cpu')}
names = ['TARGET_EPOCHS','MICRO_BATCH_SIZE','REFERENCE_BATCH_SIZE','GRAD_ACCUM_STEPS',
         'EMA_DECAY','SEED','SPLIT_SEED','MODEL_CONFIG']
for name in names:
    exec(compile(ast.Module(body=[assignment(nb.cells[4].source, name)], type_ignores=[]), '<config>', 'exec'), ns)
assert ns['MODEL_CONFIG']['experiment']['input_length'] == 3600
assert ns['TARGET_EPOCHS'] == 100
for n in ast.parse(nb.cells[12].source).body:
    if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
        exec(compile(ast.Module(body=[n], type_ignores=[]), '<model>', 'exec'), ns)
torch.set_num_threads(2)
torch.manual_seed(1234)
model = ns['build_model']()
assert isinstance(model.model.attn, torch.nn.Identity)
u4ns = {'__name__':'src.models.reference_u4', '__package__':'src.models'}
exec(base64.b64decode(previous['src/models/unet_1d.py']), u4ns)
u4 = u4ns['UNet1D'](in_channels=2, base_channels=80, emb_dim=128, out_channels=1)
u3_keys, u4_keys = set(model.model.state_dict()), set(u4.state_dict())
assert not u3_keys-u4_keys
assert u4_keys-u3_keys and all(k.startswith('attn.') for k in u4_keys-u3_keys)
assert all(model.model.state_dict()[k].shape == u4.state_dict()[k].shape for k in u3_keys)
del u4
signal = torch.sin(torch.linspace(0, 8*torch.pi, 3600)).reshape(1,1,3600)
condition = signal + .1*torch.randn_like(signal)
with torch.no_grad():
    output = model.model(signal, condition, torch.tensor([[.8]]))
assert output.shape == signal.shape and torch.isfinite(output).all()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss = model.loss_components(signal, condition)
assert all(torch.isfinite(v) for v in loss.values())
loss['total'].backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
buffer = io.BytesIO()
torch.save({'architecture_id':'U3','state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict()}, buffer)
buffer.seek(0)
restored = ns['build_model']()
saved = torch.load(buffer, map_location='cpu', weights_only=False)
restored.load_state_dict(saved['state_dict'], strict=True)
assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values()))
with torch.inference_mode():
    generated = restored.denoising(condition,use_ddim=True,ddim_steps=2,ddim_eta=0.,num_shots=1)
assert generated.shape == signal.shape and torch.isfinite(generated).all()

# Verify comparison gate accepts identical U4 settings and rejects different normalization.
ns['DATA_FINGERPRINTS'] = {'synthetic_only':True}
ns['ORIGINAL_U4_SOURCE_HASHES'] = {'synthetic_only':True}
fn = next(n for n in ast.parse(nb.cells[14].source).body if isinstance(n, ast.FunctionDef))
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<comparison>','exec'),ns)
ref = {'config':copy.deepcopy(ns['MODEL_CONFIG']), 'epoch':95,
       'data_fingerprints':ns['DATA_FINGERPRINTS'],
       'source_hashes':ns['ORIGINAL_U4_SOURCE_HASHES'],
       'state_dict':{'model.attn.qkv.weight':torch.zeros(1)}}
assert ns['compare_u4_reference'](ref)['settings_and_data_match']
ref['config']['experiment']['normalization'] = 'different'
try:
    ns['compare_u4_reference'](ref)
except RuntimeError:
    pass
else:
    raise AssertionError('Mismatched normalization was accepted')

# Exercise the notebook's actual resume cell, including EMA, optimizer, RNG and files.
with tempfile.TemporaryDirectory(prefix='u3_resume_') as directory:
    root = Path(directory)
    for folder in ('checkpoints','reports','input'):
        (root/folder).mkdir()
    resume_ns = dict(ns, model=model, EMA=EMA, random=random, shutil=shutil,
                     DataLoader=torch.utils.data.DataLoader, USE_AMP=False,
                     CHECKPOINT_DIR=root/'checkpoints', REPORT_DIR=root/'reports',
                     INPUT_ROOT=root/'input', START_FROM_SCRATCH=False,
                     REQUIRE_RESUME_CHECKPOINT=False, NUM_WORKERS=0,
                     parameter_count=sum(p.numel() for p in model.parameters()),
                     source_hashes={'synthetic_only':True})
    with patch.object(torch.cuda,'get_rng_state',torch.get_rng_state), \
         patch.object(torch.cuda,'set_rng_state',lambda state: None):
        exec(nb.cells[16].source, resume_ns)
        resume_ns['best_validation_loss'] = .75
        resume_ns['save_checkpoint'](2,is_best=True)
        resume_ns['save_checkpoint'](2)
        (root/'reports'/resume_ns['LOG_NAME']).write_text('epoch,loss\n3,0.75\n')
        exec(nb.cells[16].source, resume_ns)
        assert resume_ns['start_epoch'] == 3
        assert resume_ns['best_validation_loss'] == .75
        assert (root/'checkpoints'/resume_ns['BEST_NAME']).is_file()
        checkpoint_path = root/'checkpoints'/resume_ns['LAST_NAME']
        invalid = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        invalid['architecture_id'] = 'U4'
        torch.save(invalid,checkpoint_path)
        try:
            exec(nb.cells[16].source, resume_ns)
        except RuntimeError as exc:
            assert 'Only U3' in str(exc)
        else:
            raise AssertionError('U4 checkpoint accepted for U3 resume')
print(json.dumps({'validation':'passed','input_output_shape':list(output.shape),
                  'parameters':sum(p.numel() for p in model.parameters()),
                  'changed_embedded_sources':changed,
                  'finite_loss':{k:float(v.detach()) for k,v in loss.items()},
                  'backward_optimizer_checkpoint_roundtrip':True,
                  'actual_resume_cell_and_wrong_architecture_rejection':True,
                  'synthetic_ddim_steps':2,'full_qtdb_training_run':False}, indent=2))
