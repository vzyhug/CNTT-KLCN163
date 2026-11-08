"""Derive U3 from the exact embedded U4 trainer; never edit the U4 notebook."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import textwrap
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'notebooks/kaggle_train_qtdb_bw_em_ma.ipynb'
TARGET=ROOT/'notebooks/kaggle_train_u3_qtdb_mixed_3600_100ep.ipynb'
nb=nbf.read(SOURCE,as_version=4)
original=nbf.read(SOURCE,as_version=4)

def replace_once(s,a,b):
    if s.count(a)!=1: raise ValueError(f'Expected exactly one occurrence: {a[:100]!r}')
    return s.replace(a,b)

def md(s): return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s): return nbf.v4.new_code_cell(textwrap.dedent(s).strip())

nb.cells[0]=md('''
# U3: HNF–FiLM 1D U-Net without attention — QTDB mixed noise, 3600 samples

Train **U3 from scratch**, using the U4 trainer's 3600-point QTDB mixed-noise
protocol. The only functional network change is **bottleneck self-attention →
Identity**. HNF blocks, FiLM conditioning, channel widths, skips, diffusion,
research loss, normalization, optimizer, EMA, and model selection are retained.

The default budget is **100 epochs**, matching the currently reported U4 run.
This notebook does not convert or fine-tune a 512-point U3 checkpoint.
Outputs use unique `u3_...` names so U3 cannot resume from U4 by accident.
''')
nb.cells[1]=md('''
## 1. Kaggle inputs and running instructions

1. Enable **GPU** and Internet. Add the same
   `common_qtdb_mixed_bwem_ma_3600_ntype1.npz` used by U4.
2. The first run needs no U3 checkpoint. `START_FROM_SCRATCH=False` means
   **auto-resume if a U3 last checkpoint exists; otherwise start a new run**.
3. Defaults: 100 total epochs, up to 5 epochs per session. Increase
   `MAX_EPOCHS_THIS_SESSION` or set it to `None` only after checking runtime.
4. For another Kaggle session, attach the previous output containing **both last
   and best checkpoints plus the training log**, then Run All. Set
   `REQUIRE_RESUME_CHECKPOINT=True` if this session must continue an existing run.
5. Optionally attach the original U4 checkpoint and set `REFERENCE_U4_PATH`.
   A pre-training check compares data, normalization, loss, training settings,
   diffusion, and original U4 source. Without it, exact equivalence to that saved
   U4 run remains unverified, even though this notebook is derived from its trainer.

Train/validation artifacts are regenerated deterministically, exactly as in U4,
using NSTDB BW/EM/MA. The test pairs remain unchanged. Test metrics do not select
epochs, seeds, sampler settings, or checkpoints.
''')
nb.cells[2]=md('''
## 2. Fixed experiment settings

Single-channel 3600-point segments at 360 Hz. U3 uses HNF + FiLM without
bottleneck attention; 50 diffusion steps; the original SNR-weighted relative-STFT
research loss; Adam 1e-4; effective batch 96, micro-batch 4 with accumulation 24;
EMA 0.9; noisy-input max-absolute normalization; train/mixing seed 1234.

Architecture ablation holds data and training settings fixed, not parameter count:
removing attention reduces the number of parameters. Optional U4 comparison checks
the data and settings rather than claiming that fewer parameters alone explain results.
''')
cfg=nb.cells[4].source
cfg=replace_once(cfg,'START_FROM_SCRATCH = True','START_FROM_SCRATCH = False')
cfg=replace_once(cfg,'TARGET_EPOCHS = 400','TARGET_EPOCHS = 100')
cfg=cfg.replace('REQUIRE_RESUME_CHECKPOINT = False  # phiên tiếp tục từ checkpoint epoch 40; False nếu cố ý train mới',
                'REQUIRE_RESUME_CHECKPOINT = False  # True khi bắt buộc tiếp tục run U3 trước đó')
cfg=replace_once(cfg,"INFERENCE_BATCH_SIZE = 4", "INFERENCE_BATCH_SIZE = 4\nINFERENCE_SEED = 2024  # match the common 3600-point benchmark evaluator\nREFERENCE_U4_PATH = None  # optional absolute Kaggle path to the original U4 checkpoint")
cfg=replace_once(cfg,"Path('/kaggle/working/qtdb_mixed_bwem_ma_3600_fresh')",
                 "Path('/kaggle/working/u3_qtdb_mixed_bwem_ma_3600_100ep')")
cfg=replace_once(cfg,"if TARGET_EPOCHS != 400:\n    print('WARNING: compare.md mô tả 400 epoch; hiện tại:', TARGET_EPOCHS)",
                 "if TARGET_EPOCHS != 100:\n    print('NOTE: Match the U4 completed training budget before making an ablation comparison.')")
cfg=replace_once(cfg,"'name': 'qtdb_mixed_bwem_ma_3600_fresh_generated_train',",
                 "'name': 'u3_qtdb_mixed_bwem_ma_3600_generated_train',\n        'architecture_id': 'U3',\n        'attention': 'identity',")
nb.cells[4].source=cfg

tree=ast.parse(original.cells[6].source)
payload_node=next(n for n in tree.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='EMBEDDED_SOURCE_B64' for t in n.targets))
payload=json.loads(ast.literal_eval(payload_node.value.args[0]))
u4_sources={k:base64.b64decode(v) for k,v in payload.items()}
u4_hashes={k:hashlib.sha256(v).hexdigest() for k,v in u4_sources.items()}
network=u4_sources['src/models/unet_1d.py'].decode('utf-8')
network=replace_once(network,'self.attn = SelfAttention1D(base_channels*8)',
                     'self.attn = nn.Identity()  # U3: no bottleneck attention')
network=network.replace('và Self‑Attention tại bottleneck.', 'không dùng self-attention tại bottleneck (U3).')
network=network.replace('# Self‑Attention tại bottleneck', '# U3 bottleneck: Identity')
network=network.replace('# Bottleneck self‑attention', '# U3 bottleneck: no attention')
payload['src/models/unet_1d.py']=base64.b64encode(network.encode('utf-8')).decode()
source=nb.cells[6].source
source=source.replace(ast.get_source_segment(original.cells[6].source,payload_node),
                      'EMBEDDED_SOURCE_B64 = '+repr(payload))
source='ORIGINAL_U4_SOURCE_HASHES = '+repr(u4_hashes)+'\n'+source
# Avoid reusing imports from an earlier model in the same live kernel.
source=source.replace('sys.path.insert(0, str(SOURCE_SNAPSHOT))',
    "for key in list(sys.modules):\n    if key == 'src' or key.startswith('src.'):\n        del sys.modules[key]\nsys.path.insert(0, str(SOURCE_SNAPSHOT))")
source=source.replace("print('Imported byte-identical source embedded in this notebook')",
                     "print('Imported U3 embedded source: original U4 network with attention replaced by Identity')")
nb.cells[6].source=source
nb.cells[5]=md('## 3. Embedded U3 architecture — HNF + FiLM, Identity bottleneck')
nb.cells[11]=md('## 6. Check U3 input/output length, research loss, and strict weight loading')
nb.cells[12].source=nb.cells[12].source.replace('model = build_model()',
    "model = build_model()\nassert isinstance(model.model.attn, torch.nn.Identity)\nassert not any('attn.' in name for name,_ in model.named_parameters())",1)

reference=code('''
# Optional comparison with the exact U4 run; test tensors are never used for selection.
def compare_u4_reference(payload):
    ref = payload['config']
    differences = []
    for group, keys in {
        'train': ('feats','batch_size','micro_batch_size','gradient_accumulation_steps',
                  'lr','weight_decay','scheduler','gradient_clip_norm','lambda_time','lambda_freq'),
        'diffusion': ('beta_start','beta_end','num_steps','schedule'),
        'experiment': ('input_length','sampling_rate','in_channels','out_channels',
                       'normalization','stft_n_fft','stft_hop_length','ema_decay',
                       'seed','split_seed','data_protocol','n_type','mixed_train_seed',
                       'loss_version','research_loss_sha256'),
    }.items():
        for key in keys:
            if ref.get(group,{}).get(key) != MODEL_CONFIG[group].get(key):
                differences.append(f'{group}.{key}: U4={ref.get(group,{}).get(key)!r}, U3={MODEL_CONFIG[group].get(key)!r}')
    if payload.get('data_fingerprints') != DATA_FINGERPRINTS:
        differences.append('Training/validation/test tensors or their alignment differ')
    if payload.get('source_hashes') != ORIGINAL_U4_SOURCE_HASHES:
        differences.append('Reference U4 source differs from the original embedded U4 implementation')
    weights = payload.get('ema_model_state_dict',payload.get('state_dict',{}))
    if not any('attn.' in key for key in weights):
        differences.append('The supplied reference does not contain U4 attention weights')
    if differences: raise RuntimeError('U3/U4 controlled comparison failed:\\n'+'\\n'.join(differences))
    return {'settings_and_data_match':True, 'selected_epoch':int(payload['epoch'])+1,
            'u4_configured_epoch_budget':ref['train']['epochs'],
            'u3_epoch_budget':TARGET_EPOCHS,
            'note':'Selected epoch is not completed training budget. Confirm the U4 run completed 100 epochs independently.'}

reference_u4_receipt = {'settings_and_data_match':False,
                        'note':'No U4 reference checkpoint supplied; saved-run equivalence not verified.'}
if REFERENCE_U4_PATH is not None:
    reference_path = Path(REFERENCE_U4_PATH)
    if not reference_path.is_file(): raise FileNotFoundError(reference_path)
    reference_payload = torch.load(reference_path,map_location='cpu',weights_only=False)
    reference_u4_receipt = compare_u4_reference(reference_payload)
    reference_u4_receipt['file'] = str(reference_path)
    del reference_payload
(REPORT_DIR/'u4_comparison_check.json').write_text(json.dumps(reference_u4_receipt,indent=2),encoding='utf-8')
print(reference_u4_receipt)
''')

# Names are unique across training, resume, logs, exports, and zip packages.
for cell in nb.cells:
    for a,b in [
        ('last_qtdb_mixed_bwem_ma_3600_fresh.pth','last_u3_qtdb_mixed_3600_100ep.pth'),
        ('best_qtdb_mixed_bwem_ma_3600_fresh.pth','best_u3_qtdb_mixed_3600_100ep.pth'),
        ('training_qtdb_mixed_bwem_ma_3600_fresh.csv','training_u3_qtdb_mixed_3600_100ep.csv'),
        ('qtdb_mixed_bwem_ma_3600_fresh_seed1234.npz','u3_qtdb_mixed_3600_predictions.npz'),
        ('qtdb_mixed_bwem_ma_3600_fresh_outputs.zip','u3_qtdb_mixed_3600_100ep_outputs.zip'),
        ("'experiment': 'qtdb_mixed_bwem_ma_3600_fresh_generated_train'", "'experiment': 'u3_qtdb_mixed_bwem_ma_3600_generated_train'"),
    ]: cell.source=cell.source.replace(a,b)

resume=nb.cells[14].source
resume=resume.replace("    if payload['config'] != MODEL_CONFIG:",
    "    if payload.get('architecture_id') != 'U3':\n        raise RuntimeError('Only U3 checkpoints can resume this notebook')\n    if payload['config'] != MODEL_CONFIG:")
resume=resume.replace("        'state_dict': cpu_state_dict(model.state_dict()),",
                      "        'architecture_id': 'U3',\n        'state_dict': cpu_state_dict(model.state_dict()),")
# copy2(source, source) raises on a same-session Run All; preserve the local files.
resume=resume.replace("    if previous_log.is_file():\n        shutil.copy2(previous_log, REPORT_DIR / LOG_NAME)",
    "    if previous_log.is_file() and previous_log.resolve() != (REPORT_DIR/LOG_NAME).resolve():\n        shutil.copy2(previous_log, REPORT_DIR / LOG_NAME)")
resume=resume.replace("    if previous_best.is_file():\n        shutil.copy2(previous_best, CHECKPOINT_DIR / BEST_NAME)",
    "    if previous_best.is_file():\n        if previous_best.resolve() != (CHECKPOINT_DIR/BEST_NAME).resolve():\n            shutil.copy2(previous_best, CHECKPOINT_DIR / BEST_NAME)")
resume=resume.replace("        'cuda_rng_state': torch.cuda.get_rng_state(),",
    "        'cuda_rng_state': torch.cuda.get_rng_state(),\n        'numpy_rng_state': np.random.get_state(),\n        'python_rng_state': random.getstate(),")
resume=resume.replace("    torch.set_rng_state(payload['torch_rng_state'])",
    "    np.random.set_state(payload['numpy_rng_state'])\n    random.setstate(payload['python_rng_state'])\n    torch.set_rng_state(payload['torch_rng_state'])")
# Do not silently erase a previous best when resuming a finished run from last alone.
resume=resume.replace("        print('WARNING: Chỉ tìm thấy last checkpoint; thiếu best checkpoint trước đây. '\n              'Training vẫn tiếp tục, nhưng cần best gốc để giữ epoch tốt nhất trước resume.')",
    "        raise FileNotFoundError('Resume requires the earlier best checkpoint as well as last; attach both to preserve validation-based selection.')")
nb.cells[14].source=resume
export=nb.cells[22].source.replace('INFERENCE_MICRO_BATCH_SIZE','INFERENCE_BATCH_SIZE')
export=export.replace("    if best_payload['config'] != MODEL_CONFIG:",
    "    if best_payload.get('architecture_id') != 'U3':\n        raise RuntimeError('Expected U3 best checkpoint')\n    if best_payload['config'] != MODEL_CONFIG:")
export=export.replace('torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)',
                      'torch.manual_seed(INFERENCE_SEED); torch.cuda.manual_seed_all(INFERENCE_SEED)')
export=export.replace("'num_shots': NUM_SHOTS, 'seed': SEED,", "'num_shots': NUM_SHOTS, 'seed': INFERENCE_SEED,")
export=export.replace('normalization_scale=test_scale.reshape(-1))',
    "normalization_scale=test_scale.reshape(-1),\n                        data_sha256=actual_bundle_sha256,\n                        checkpoint_sha256=sha256_file(best_path),\n                        architecture_id=np.asarray('U3'), inference_seed=np.asarray(INFERENCE_SEED))")
nb.cells[22].source=export
nb.cells[24].source=nb.cells[24].source.replace("    'mode': MODE,", "    'architecture_id': 'U3', 'u4_comparison': reference_u4_receipt,\n    'mode': MODE,")
nb.cells[-1]=md('''
## Continue training and retrieve results

- **First run:** attach the common mixed-noise dataset, enable GPU and Internet,
  leave `START_FROM_SCRATCH=False`, and Run All. With no prior U3 checkpoint this
  starts at epoch 0; U4 or 512-point weights are never loaded into U3.
- **Resume:** attach the previous U3 output with last + best + log. Run All with
  the same configuration; the next session continues from the saved epoch.
  `REQUIRE_RESUME_CHECKPOINT=True` prevents accidental fresh training.
- **Budget:** default 100 total epochs and at most 5 per session. Changing only
  the per-session cap is safe. Changing the total budget changes the run contract.
- **Selection:** best is chosen by validation research loss. EMA weights from
  that selected epoch are used for inference, matching the original procedure.
- **Test export:** starts automatically only after 100 epochs. Outputs are
  `best_u3_qtdb_mixed_3600_100ep.pth`, `last_u3_qtdb_mixed_3600_100ep.pth`, and
  `u3_qtdb_mixed_3600_predictions.npz`. The inference seed is 2024 to match the
  common U4 benchmark evaluation; it is separate from training seed 1234.
- Compare U3 and U4 using the same 5012 test pairs, amplitude units, 50-step DDIM,
  1-shot sampling, evaluation seed/batch, and metric definitions. Do not compare
  the existing 512-point U3 result directly with this 3600-point experiment.

No numerical training or test outcome is supplied in this unexecuted Kaggle
training notebook. Local validation checks the embedded architecture, loss,
forward/backward computation, and checkpoint round-trip on synthetic inputs;
the full dataset training must run on Kaggle GPU.
''')
nb.cells.insert(13,md('## 6b. Optional verification against the saved U4 run'))
nb.cells.insert(14,reference)
for cell in nb.cells:
    if cell.cell_type=='code': cell.execution_count=None; cell.outputs=[]
nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nb.metadata['kaggle']={'accelerator':'gpu','isInternetEnabled':True,'language':'python','sourceType':'notebook'}
nbf.validate(nb)
for i,cell in enumerate(nb.cells):
    if cell.cell_type=='code' and not cell.source.startswith('%pip'):
        compile(cell.source,f'u3_cell_{i}','exec')
nbf.write(nb,TARGET)
print(TARGET)
print('Cells:',len(nb.cells),'| U3 source embedded | 100 epochs | no source notebook changes')
