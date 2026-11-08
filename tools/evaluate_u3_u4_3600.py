"""Recompute matched QTDB metrics from saved U3/U4 predictions; no inference."""
from pathlib import Path
import ast
import base64
import copy
import hashlib
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/u3_u4_3600'
OUT.mkdir(parents=True, exist_ok=True)
BUNDLE = ROOT/'common_qtdb_mixed_bwem_ma_3600_ntype1.npz'
U3ROOT = ROOT/'U3-3600'
U4ROOT = ROOT/'results (4)/tfcdiff_mixed_evaluation'
U4CKPT = ROOT/'checkpoints/mix/best_qtdb_mixed_bwem_ma_3600_fresh.pth'
METRICS = ['SSD','RMSE','MAD','PRD','CosSim','SNR_out','DeltaSNR']
BINS = ['0.2-0.6','0.6-1.0','1.0-1.5','1.5-2.0']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def digest_array(a):
    value=np.ascontiguousarray(a)
    h=hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(str(value.dtype).encode())
    h.update(value.view(np.uint8))
    return h.hexdigest()

u3run = json.loads((U3ROOT/'reports/run_info.json').read_text())
u3export = json.loads((U3ROOT/'reports/export_audit.json').read_text())
u4meta = json.loads((U4ROOT/'evaluation_metadata.json').read_text())
u3best_path = U3ROOT/'checkpoints/best_u3_qtdb_mixed_3600_100ep.pth'
u3last_path = U3ROOT/'checkpoints/last_u3_qtdb_mixed_3600_100ep.pth'
u3best = torch.load(u3best_path,map_location='cpu',weights_only=False)
u3last = torch.load(u3last_path,map_location='cpu',weights_only=False)
u4best = torch.load(U4CKPT,map_location='cpu',weights_only=False)
assert sha(u3best_path) == u3export['checkpoint_sha256']
assert sha(U4CKPT) == u4meta['proposed_sha256']
assert u3run['training_complete'] and u3run['completed_epochs'] == 100
assert u3last['epoch']+1 == 100 and u3best['epoch']+1 == u3export['selected_epoch']
assert u3best['architecture_id'] == 'U3'
assert not any('attn.' in k for k in u3best['state_dict'])
assert any('attn.' in k for k in u4best['state_dict'])
assert u3best['data_fingerprints'] == u4best['data_fingerprints'] == u3run['data_fingerprints']
c3,c4 = copy.deepcopy(u3best['config']),copy.deepcopy(u4best['config'])
budget3,budget4 = c3['train'].pop('epochs'),c4['train'].pop('epochs')
for cfg in (c3,c4):
    for key in ('name','architecture_id','attention'):
        cfg['experiment'].pop(key,None)
assert c3 == c4, 'Training settings differ beyond architecture/run name/epoch target'
for name, value in u3best['source_hashes'].items():
    if name != 'src/models/unet_1d.py':
        assert u4best['source_hashes'][name] == value
    assert sha(U3ROOT/'source_snapshot'/name) == value
u3code=(U3ROOT/'source_snapshot/src/models/unet_1d.py').read_text(encoding='utf-8')
u4code=(U4ROOT/'embedded_sources/src/models/unet_1d.py')
if u4code.is_file():
    assert sha(u4code) == u4best['source_hashes']['src/models/unet_1d.py']
    assert 'self.attn = nn.Identity()' in u3code
    class ArchitectureNormalize(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Attribute) and t.attr=='attn' for t in node.targets):
                node.value=ast.Constant(value='ATTENTION_VARIANT')
            return self.generic_visit(node)
    normalized=lambda source:ast.dump(ArchitectureNormalize().visit(ast.parse(source)))
    assert normalized(u3code)==normalized(u4code.read_text(encoding='utf-8'))

log=pd.read_csv(U3ROOT/'reports/training_u3_qtdb_mixed_3600_100ep.csv')
assert log.epoch.tolist() == list(range(1,101))
assert int(log.loc[log.validation_loss.idxmin(),'epoch']) == u3export['selected_epoch']
assert np.isclose(log.validation_loss.min(),u3best['best_validation_loss'])
bundle_sha = sha(BUNDLE)
assert bundle_sha == u3run['data_fingerprints']['bundle_file'] == u4meta['data_sha256']
with np.load(BUNDLE,allow_pickle=False) as z:
    clean,noisy=z['clean_test'],z['noisy_test']
    ids=z['sample_id_test'].astype(str)
    records=z['record_id_test'].astype(str)
    q=z['q_test'].astype(float)
assert clean.shape == noisy.shape == (5012,1,3600)
assert len(set(ids))==5012 and len(set(records))==14
assert digest_array(clean)==u3run['data_fingerprints']['clean_test']
assert digest_array(noisy)==u3run['data_fingerprints']['noisy_test']
manifest=pd.read_csv(U3ROOT/'reports/test_manifest_locked.csv',dtype={'sample_id':str})
assert manifest.sample_id.tolist()==ids.tolist()
assert manifest.record_id.tolist()==records.tolist()
assert np.allclose(manifest.q,q)
bin_idx=np.searchsorted([.6,1.,1.5],q,side='right')
assert ((q>=.2)&(q<=2.)).all()
counts=np.bincount(bin_idx,minlength=4).tolist()
assert counts==[1077,1172,1381,1382]

def metrics(x,y,p):
    x,y,p=[np.asarray(a,dtype=np.float64) for a in (x,y,p)]
    error=p-x
    ssd=np.sum(error**2,axis=1)
    energy=np.sum(x**2,axis=1)
    centered=np.sum((x-x.mean(axis=1,keepdims=True))**2,axis=1)
    pred_energy=np.sum(p**2,axis=1)
    noise_energy=np.sum((y-x)**2,axis=1)
    assert min(energy.min(),centered.min(),pred_energy.min(),noise_energy.min(),ssd.min())>0
    snrin=10*np.log10(energy/noise_energy)
    snrout=10*np.log10(energy/ssd)
    return pd.DataFrame({'SSD':ssd,'RMSE':np.sqrt(ssd/3600),
        'MAD':np.max(np.abs(error),axis=1),'PRD':100*np.sqrt(ssd/centered),
        'CosSim':np.sum(x*p,axis=1)/np.sqrt(energy*pred_energy),
        'SNR_out':snrout,'DeltaSNR':snrout-snrin})

specs={'U3':U3ROOT/'u3_qtdb_mixed_3600_predictions.npz',
       'U4':U4ROOT/'proposed_predictions.npz',
       'TCDAE':U4ROOT/'TCDAE_predictions.npz',
       'CBAM-Transformer':U4ROOT/'CBAM_Transformer_DAE_predictions.npz'}
details=[]
archive_receipts={}
for model,path in specs.items():
    # U3's own trainer saved sample IDs as an object array of strings.
    # These are supplied local artifacts; numerical predictions load without pickle.
    with np.load(path,allow_pickle=False) as z:
        prediction=z['prediction']
        assert str(z['data_sha256'].item())==bundle_sha
        scale=z['normalization_scale'] if 'normalization_scale' in z.files else None
        if model in ('U3','U4'):
            ckhash=z['checkpoint_sha256'].item()
            assert ckhash==(u3export['checkpoint_sha256'] if model=='U3' else u4meta['proposed_sha256'])
        if model=='U3': assert z['inference_seed'].item()==2024
    with np.load(path,allow_pickle=True) as z:
        prediction_ids=np.asarray(z['sample_id']).astype(str)
    assert len(set(prediction_ids))==5012 and set(prediction_ids)==set(ids)
    positions={sid:i for i,sid in enumerate(prediction_ids)}
    order=np.array([positions[sid] for sid in ids])
    prediction=prediction.reshape(5012,-1)[order]
    assert prediction.shape==(5012,3600) and np.isfinite(prediction).all()
    if scale is not None:
        assert np.allclose(scale[order].reshape(-1),np.max(np.abs(noisy),axis=(1,2)))
    frames=[]
    for start in range(0,5012,128):
        stop=min(start+128,5012)
        frames.append(metrics(clean[start:stop,0],noisy[start:stop,0],prediction[start:stop]))
    frame=pd.concat(frames,ignore_index=True)
    assert np.isfinite(frame.to_numpy()).all()
    frame.insert(0,'model',model)
    frame['sample_id'],frame['record_id'],frame['q']=ids,records,q
    frame['noise_bin']=np.array(BINS)[bin_idx]
    details.append(frame)
    archive_receipts[model]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),
                             'original_order_matches':bool(np.array_equal(order,np.arange(5012)))}

detail=pd.concat(details,ignore_index=True)
detail.to_csv(OUT/'per_segment_metrics.csv.gz',index=False,compression='gzip')
summary=[]
for model in specs:
    frame=detail[detail.model==model]
    for label in ['overall']+BINS:
        group=frame if label=='overall' else frame[frame.noise_bin==label]
        row={'model':model,'noise_bin':label,'N':len(group)}
        for metric in METRICS:
            row[metric+'_mean']=group[metric].mean()
            row[metric+'_sd']=group[metric].std(ddof=1)
        summary.append(row)
summary=pd.DataFrame(summary)
summary.to_csv(OUT/'summary_by_noise.csv',index=False)
original=pd.read_csv(U4ROOT/'summary_metrics_paper_protocol.csv').set_index('model')
for our,old in [('U4','Proposed'),('TCDAE','TCDAE'),('CBAM-Transformer','CBAM_Transformer_DAE')]:
    row=summary[(summary.model==our)&(summary.noise_bin=='overall')].iloc[0]
    for new,prev in {'SSD':'SSD','MAD':'MAD','PRD':'PRD_standard','CosSim':'COS_SIM','SNR_out':'SNR_out','DeltaSNR':'ImSNR'}.items():
        assert np.isclose(row[new+'_mean'],original.loc[old,prev],rtol=1e-7,atol=1e-8)

paired=[]
u3=detail[detail.model=='U3'].set_index('sample_id')
u4=detail[detail.model=='U4'].set_index('sample_id').loc[u3.index]
for label in ['overall']+BINS:
    selected=np.ones(len(u3),dtype=bool) if label=='overall' else (u3.noise_bin==label).to_numpy()
    for metric in METRICS:
        a,b=u3.loc[selected,metric],u4.loc[selected,metric]
        higher=metric in ('CosSim','SNR_out','DeltaSNR')
        paired.append({'noise_bin':label,'metric':metric,'N':len(a),
            'U3_mean':a.mean(),'U4_mean':b.mean(),'U3_minus_U4':(a-b).mean(),
            'U3_better_fraction':float(((a>b) if higher else (a<b)).mean()),
            'U3_relative_improvement_pct':float((a.mean()-b.mean())/abs(b.mean())*100*(1 if higher else -1))})
pd.DataFrame(paired).to_csv(OUT/'paired_u3_u4_comparison.csv',index=False)

receipt={'dataset_pairs':5012,'held_out_records':14,'samples_per_segment':3600,
    'noise_bin_counts':dict(zip(BINS,counts)),'bundle_sha256':bundle_sha,
    'identical_train_validation_test_fingerprints':True,
    'identical_training_settings_except_architecture_and_epoch_target':True,
    'U3_completed_epochs':100,'U3_selected_epoch':u3best['epoch']+1,
    'U4_selected_epoch':u4best['epoch']+1,'U4_completed_epochs_verified':False,
    'configured_epoch_targets':{'U3':budget3,'U4':budget4},
    'caveat':'U4 best checkpoint targets 400 epochs and was selected at epoch 92. A U4 last checkpoint or matching training log is absent; the stated completed budget of 100 epochs cannot be independently confirmed from best alone.',
    'sampler':{'steps':50,'eta':0,'shots':1,'seed':2024,'batch_size':4},
    'parameters':{'U3':u3run['parameter_count'],'U4':u4best['parameter_count']},
    'archives':archive_receipts,
    'interpretation':'Descriptive single-run comparisons; no independent-seed significance or causal claim.',
    'original_U4_and_peer_published_metrics_reproduced':True}
(receipt.update({'network_AST_identical_except_attention_assignment':True}))
(OUT/'comparison_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
overall=summary[summary.noise_bin=='overall'].set_index('model')
lines=['# U3 vs U4 — QTDB, 3600 samples','',
       'Mean ± sample SD over the same 5012 test pairs. PRD uses mean-centered reference energy.', '',
       '| Model | SSD | RMSE | MAD | PRD (%) | Cosine | SNR out (dB) | ΔSNR (dB) |',
       '|---|---|---|---|---|---|---|---|']
for model in specs:
    r=overall.loc[model]
    values=[f"{r[m+'_mean']:.4f} ± {r[m+'_sd']:.4f}" for m in METRICS]
    lines.append('| '+model+' | '+' | '.join(values)+' |')
lines+=['','## U3–U4 by noise intensity','']
for metric in METRICS:
    lines += ['### '+metric,'','| Model | '+' | '.join(BINS)+' |','|---|---|---|---|---|']
    for model in ('U3','U4'):
        part=summary[summary.model==model].set_index('noise_bin')
        values=[f"{part.loc[b,metric+'_mean']:.4f} ± {part.loc[b,metric+'_sd']:.4f}" for b in BINS]
        lines.append('| '+model+' | '+' | '.join(values)+' |')
    lines.append('')
lines += ['## Comparability','',
          '- U3 completed 100 epochs; best validation checkpoint: epoch 99.',
          '- U4 best validation checkpoint: epoch 92; its completed training budget is not verified by the supplied best checkpoint.',
          '- Train/validation/test fingerprints, normalization, optimizer, loss, EMA and diffusion settings match. Configured total epochs differ (100 vs 400).',
          '- U3 removes bottleneck attention; all other embedded network blocks have identical source hashes.',
          '- No statistical significance claim is made from these single-run averages.',
          '- Existing paper results have not been replaced.']
lines += ['','## Interpretation','',
          '- U4 has more favorable overall means on all seven metrics in this evaluated pair of runs.',
          '- In the highest noise interval, U3 has lower SSD (45.1470 vs 46.4034); U4 retains more favorable means on the other six metrics.',
          '- U3 uses 9,865,761 parameters vs 11,506,721 for U4, a 14.26% reduction.',
          '- These results favor retaining U4 as the reported model and adding the matched 3600-point U3/U4 comparison as an attention ablation, with the training-budget qualification above.',
          '- The 512-point ablation remains a separate experiment; its U3/U4 ordering does not determine the ordering at 3600 points.']
(OUT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(overall[[m+'_mean' for m in METRICS]].to_string())
print('\nU3-U4:')
print(pd.DataFrame(paired)[['noise_bin','metric','U3_relative_improvement_pct']].to_string(index=False))
print('\nSaved:',OUT)
