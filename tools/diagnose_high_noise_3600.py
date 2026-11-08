"""Paired, descriptive high-noise diagnostics from already saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/high_noise_diagnostics'
OUT.mkdir(parents=True,exist_ok=True)
MODELS=['U4','TCDAE','CBAM-Transformer']
METRICS=['SSD','RMSE','MAD','PRD','CosSim','SNR_out','DeltaSNR']
UP={'CosSim','SNR_out','DeltaSNR'}
detail=pd.read_csv(ROOT/'output/u3_u4_3600/per_segment_metrics.csv.gz',dtype={'sample_id':str})
high=detail[(detail.q>=1.5)&(detail.q<=2)&detail.model.isin(MODELS)].copy()
assert high.groupby('model').size().eq(1382).all()
tables={m:high[high.model==m].set_index('sample_id').sort_index() for m in MODELS}
ids=tables['U4'].index
assert all(t.index.equals(ids) for t in tables.values())
paired=tables['U4'][['record_id','q']].copy()
for m in MODELS:
    for metric in METRICS:
        paired[m+'_'+metric]=tables[m][metric]
for peer in MODELS[1:]:
    for metric in METRICS:
        # Positive gap always means U4 is worse.
        paired[f'gap_{peer}_{metric}']=(tables['U4'][metric]-tables[peer][metric])*(-1 if metric in UP else 1)
paired.to_csv(OUT/'paired_high_noise.csv')
summary=[]
for m in MODELS:
    for metric in METRICS:
        values=tables[m][metric]
        summary.append({'model':m,'metric':metric,'N':len(values),'mean':values.mean(),
            'sd':values.std(ddof=1),'p10':values.quantile(.1),'p25':values.quantile(.25),
            'median':values.median(),'p75':values.quantile(.75),'p90':values.quantile(.9),
            'p95':values.quantile(.95),'p99':values.quantile(.99)})
pd.DataFrame(summary).to_csv(OUT/'distribution_summary.csv',index=False)
win=[]
records=[]
tails=[]
for peer in MODELS[1:]:
    for metric in METRICS:
        gap=paired[f'gap_{peer}_{metric}']
        win.append({'peer':peer,'metric':metric,'N':len(gap),'u4_win_fraction':(gap<0).mean(),
            'u4_loss_fraction':(gap>0).mean(),'mean_gap_positive_is_worse':gap.mean(),
            'median_gap':gap.median(),'p90_gap':gap.quantile(.9),'p95_gap':gap.quantile(.95)})
    for record,g in paired.groupby('record_id'):
        row={'peer':peer,'record_id':record,'N':len(g),'PRD_gap':g[f'gap_{peer}_PRD'].mean(),
             'SSD_gap':g[f'gap_{peer}_SSD'].mean(),'PRD_u4_win_fraction':(g[f'gap_{peer}_PRD']<0).mean(),
             'SSD_gap_sum':g[f'gap_{peer}_SSD'].sum()}
        records.append(row)
    # Shared selection: exclude the same top 5% positive-gap cases from BOTH models.
    for metric in ('SSD','PRD'):
        key=f'gap_{peer}_{metric}'
        cutoff=paired[key].quantile(.95)
        retained=paired[paired[key]<=cutoff]
        positive=paired[key].clip(lower=0)
        worst=paired.nlargest(int(np.ceil(.05*len(paired))),key)
        tails.append({'peer':peer,'metric':metric,'all_mean_gap':paired[key].mean(),
            'shared_exclusion_N':len(paired)-len(retained),
            'remaining_mean_gap':retained[key].mean(),
            'top_5pct_positive_gap_share':worst[key].clip(lower=0).sum()/positive.sum()})
pd.DataFrame(win).to_csv(OUT/'paired_win_rates.csv',index=False)
records=pd.DataFrame(records)
records.to_csv(OUT/'by_record.csv',index=False)
pd.DataFrame(tails).to_csv(OUT/'tail_sensitivity.csv',index=False)

with np.load(ROOT/'common_qtdb_mixed_bwem_ma_3600_ntype1.npz',allow_pickle=False) as z:
    clean,noisy=z['clean_test'][:,0],z['noisy_test'][:,0]
    all_ids=z['sample_id_test'].astype(str)
    train_q,val_q,test_q=z['q_train'],z['q_val'],z['q_test']
row_index={sid:i for i,sid in enumerate(all_ids)}
indices=np.array([row_index[sid] for sid in ids])
x=clean[indices].astype(np.float64)
y=noisy[indices].astype(np.float64)
sources={'U4':'proposed_predictions.npz','TCDAE':'TCDAE_predictions.npz',
         'CBAM-Transformer':'CBAM_Transformer_DAE_predictions.npz'}
predictions={}
features=[]
components=[]
for m,file in sources.items():
    with np.load(ROOT/'results (4)/tfcdiff_mixed_evaluation'/file,allow_pickle=False) as z:
        pp=z['prediction'].reshape(len(all_ids),3600)
        ppids=z['sample_id'].astype(str)
        assert np.array_equal(ppids,all_ids)
        p=pp[indices].astype(np.float64)
    predictions[m]=p
    xm=x.mean(axis=1,keepdims=True)
    pm=p.mean(axis=1,keepdims=True)
    xc=x-xm
    denom=(xc**2).sum(axis=1)
    gain=(xc*(p-pm)).sum(axis=1)/denom
    bias=(pm-xm).reshape(-1)
    residual=p-(pm+gain[:,None]*xc)
    bias_ssd=3600*bias**2
    gain_ssd=denom*(gain-1)**2
    residual_ssd=(residual**2).sum(axis=1)
    ssd=((p-x)**2).sum(axis=1)
    assert np.allclose(ssd,bias_ssd+gain_ssd+residual_ssd,rtol=1e-10,atol=1e-10)
    assert np.allclose(ssd,tables[m].SSD.to_numpy(),rtol=1e-8)
    for j,sid in enumerate(ids):
        features.append({'model':m,'sample_id':sid,'record_id':paired.loc[sid,'record_id'],
            'q':paired.loc[sid,'q'],'gain':gain[j],'bias':bias[j],
            'bias_ssd':bias_ssd[j],'gain_ssd':gain_ssd[j],'orthogonal_residual_ssd':residual_ssd[j]})
    components.append({'model':m,'mean_bias_ssd':bias_ssd.mean(),
        'mean_gain_ssd':gain_ssd.mean(),'mean_orthogonal_residual_ssd':residual_ssd.mean(),
        'median_gain':np.median(gain),'median_abs_gain_error':np.median(abs(gain-1)),
        'median_abs_bias':np.median(abs(bias))})
pd.DataFrame(features).to_csv(OUT/'signal_error_components.csv',index=False)
pd.DataFrame(components).to_csv(OUT/'component_summary.csv',index=False)
bins=[.2,.6,1.,1.5,2.000001]
noise_distribution=[]
for split,qq in [('train',train_q),('validation',val_q),('test',test_q)]:
    count,_=np.histogram(qq,bins=bins)
    for k,n in enumerate(count):
        noise_distribution.append({'split':split,'bin':['0.2-0.6','0.6-1.0','1.0-1.5','1.5-2.0'][k],
            'N':int(n),'fraction':n/len(qq)})
pd.DataFrame(noise_distribution).to_csv(OUT/'noise_distribution.csv',index=False)

# Four deliberately selected diagnostic examples; these are not representative samples.
g=paired['gap_CBAM-Transformer_PRD']
choices=[('largest_PRD_loss',g.idxmax()),
         ('largest_SSD_loss',paired['gap_CBAM-Transformer_SSD'].idxmax()),
         ('median_PRD_gap',(g-g.median()).abs().idxmin()),
         ('largest_PRD_win',g.idxmin())]
colors={'Reference':'#222222','Noisy':'#999999','U4':'#2468A2','TCDAE':'#B77719','CBAM-Transformer':'#B14975'}
styles={'U4':'-','TCDAE':'--','CBAM-Transformer':':'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
selected=[]
for name,sid in choices:
    j=ids.get_loc(sid)
    # Largest local EXCESS squared error; not an annotated R peak.
    excess=(predictions['U4'][j]-x[j])**2-(predictions['CBAM-Transformer'][j]-x[j])**2
    window=np.convolve(excess,np.ones(90)/90,mode='same')
    center=int(window.argmax())
    lo=max(0,center-144); hi=min(3600,lo+288); lo=max(0,hi-288)
    t=np.arange(3600)/360
    fig=plt.figure(figsize=(13,7.5),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.2])
    full=fig.add_subplot(gs[0,:]); zoom=fig.add_subplot(gs[1,0]); err=fig.add_subplot(gs[1,1])
    for ax in (full,zoom):
        subset=slice(None) if ax is full else slice(lo,hi)
        if ax is full: ax.plot(t,y[j],color=colors['Noisy'],alpha=.5,lw=.8,label='Noisy')
        ax.plot(t[subset],x[j,subset],color=colors['Reference'],lw=1.2,label='Reference')
        for m in MODELS:
            ax.plot(t[subset],predictions[m][j,subset],color=colors[m],ls=styles[m],lw=1.1,label=m)
        ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude (bundle units)');ax.grid(alpha=.15)
    full.axvspan(t[lo],t[hi-1],color='#777777',alpha=.15)
    full.legend(loc='upper right',ncol=5,fontsize=9)
    zoom.set_title('Window with largest local excess squared error vs CBAM')
    for m in MODELS:
        err.plot(t[lo:hi],predictions[m][j,lo:hi]-x[j,lo:hi],color=colors[m],ls=styles[m],lw=1.1,label=m)
    err.axhline(0,color='#555555',lw=.7);err.set_xlabel('Time (s)');err.set_ylabel('Prediction - reference');err.grid(alpha=.15)
    metrics='; '.join(f'{m} PRD={tables[m].loc[sid,"PRD"]:.2f}%' for m in MODELS)
    fig.suptitle(f'{name}: sample {sid}, {paired.loc[sid,"record_id"]}, q={paired.loc[sid,"q"]:.2f}\n{metrics}',fontsize=12)
    fig.savefig(OUT/(name+'.png'),dpi=160)
    plt.close(fig)
    selected.append({'selection':name,'sample_id':sid,'record_id':paired.loc[sid,'record_id'],
        'q':paired.loc[sid,'q'],'zoom_start_s':lo/360,'zoom_end_s':hi/360,
        **{m+'_PRD':float(tables[m].loc[sid,'PRD']) for m in MODELS}})
pd.DataFrame(selected).to_csv(OUT/'selected_examples.csv',index=False)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
for m in MODELS:
    values=np.sort(tables[m].PRD.to_numpy())
    ax1.plot(values,np.arange(1,len(values)+1)/len(values),color=colors[m],ls=styles[m],label=m)
ax1.set_xlabel('PRD (%)');ax1.set_ylabel('Fraction of segments with PRD at or below x')
ax1.set_xlim(left=0);ax1.set_ylim(0,1);ax1.legend();ax1.grid(alpha=.15)
rr=records[records.peer=='CBAM-Transformer'].sort_values('SSD_gap')
ax2.barh(rr.record_id,rr.SSD_gap,color=colors['U4'])
ax2.axvline(0,color='#222222',lw=.8);ax2.set_xlabel('Mean SSD(U4) - SSD(CBAM)');ax2.grid(axis='x',alpha=.15)
fig.suptitle('High noise: 1.5 <= q <= 2.0; 1382 paired segments, 14 records')
fig.savefig(OUT/'distribution_and_records.png',dpi=160);plt.close(fig)
print('WIN RATES');print(pd.DataFrame(win).to_string(index=False))
print('\nRECORDS');print(records.to_string(index=False))
print('\nTAILS');print(pd.DataFrame(tails).to_string(index=False))
print('\nCOMPONENTS');print(pd.DataFrame(components).to_string(index=False))
print('\nNOISE DISTRIBUTION');print(pd.DataFrame(noise_distribution).to_string(index=False))
print('\nEXAMPLES');print(pd.DataFrame(selected).to_string(index=False))
