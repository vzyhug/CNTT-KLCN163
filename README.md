# ECG denoising ? 3600-point protocol

HNF-FiLM 1D U-Net conditional diffusion for single-lead ECG denoising.
Active protocol: **3600 samples at 360 Hz (10 seconds)**, QTDB/NSTDB,
fixed train/validation/test split of **22,804 / 9,774 / 5,012** pairs.
Mixed-noise benchmark: weighted BW + EM + MA, noise version 1.

## Download and run

Git LFS is required for datasets, checkpoints, prediction arrays and the comparative checkpoint ZIP.

```bash
git lfs install
git clone https://github.com/vzyhug/CNTT-KLCN163.git
cd CNTT-KLCN163
git lfs pull
python tools/restore_data.py
python -m venv .venv
# Activate .venv, then:
pip install -r requirements.txt
python run_workflow.py --mode preprocess
python run_workflow.py --mode infer --device cpu --index 0
streamlit run src/app/streamlit_app.py
```

`src/protocol3600.py` is the canonical implementation. See [PROTOCOL_3600.md](PROTOCOL_3600.md)
for training, inference, normalization, metrics and the fixed data split.
The demo uses `output/fig05_mix7_3600/fig05_u4_3600_ema.pth`.

## Research notebooks

Only 3600-point experiment notebooks are included; figure-only notebooks and
512-point training/evaluation notebooks are excluded. Notebook outputs are cleared.

- `kaggle_train_proposed_tfcdiff_mixed_3600.ipynb`: mixed-noise training.
- `kaggle_train_proposed_research_mixed_3600.ipynb`: research training variant.
- `kaggle_train_qtdb_bw_em_ma.ipynb`: fresh mixed-noise training pipeline.
- `kaggle_train_u3_qtdb_mixed_3600_100ep.ipynb`: no-attention comparison, also 3600 points.
- `evaluate-proposed-vs-tfcdiff-mixed-3600.ipynb`: matched baseline evaluation.
- `kaggle_table4_metrics_by_noise_intensity.ipynb`: metrics by noise-intensity interval.
- `kaggle_proposed_conditional_ddpm_3600.ipynb`: **BW-only legacy 3600-point experiment**;
  use its separate BW bundle/checkpoints, not the mixed-noise benchmark.

Upload the notebook-specific data/checkpoint files to Kaggle or Colab as indicated
inside each notebook. `comparative_checkpoints_bundle.zip` contains baseline weights
and the SHA-256 manifest expected by the comparison notebooks. The native 3584-point
baselines remain length-distinct as documented by those experiments.

## Checkpoints and artifacts

- `output/fig05_mix7_3600/`: compact EMA checkpoint used by the app.
- `checkpoints/mix/`: full mixed-noise training checkpoint.
- `U3-3600/checkpoints/`: best/last no-attention checkpoints.
- `results (4)/proposed_tfcdiff_mixed_3600*/checkpoints/`: mixed-noise training snapshots.
- `proposed_conditional_ddpm_3600_outputs/checkpoints/`: BW-only 3600-point snapshots.
- `checkpoints/model.pth`: TFCDiff comparator; not the proposed model.
- `results (4)/tfcdiff_mixed_evaluation/`: baseline weights, saved predictions and reports.
- `ARTIFACTS_3600.json`: original data/checkpoint paths, sizes and SHA-256 checksums.

Some support modules retain historical utilities; the entry point `run_workflow.py`
uses the strict 3600-point implementation. Raw QTDB/NSTDB distributions are not
included. Original third-party datasets and baseline models retain their licenses
and attribution requirements. This repository does not grant new rights over them.

The main mixed-noise dataset is stored as 64 MiB LFS parts in `data_parts/`.
Run `python tools/restore_data.py` once after `git lfs pull`; it validates every
part and the final dataset SHA-256 before installing the original NPZ filename.
