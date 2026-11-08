# Runnable 3600-point paper protocol

The active command-line entry point is `run_workflow.py`. It uses the fixed
`common_qtdb_mixed_bwem_ma_3600_ntype1.npz` bundle: QTDB single-lead ECG at
360 Hz, 3600 samples per pair, and the published 22,804 / 9,774 / 5,012
train / validation / held-out test split. The bundle already contains the
processed reference and mixed BW/EM/MA noisy pairs. `preprocess` validates this
artifact and its split; it does not regenerate QTDB/NSTDB signals from raw files.

```powershell
python run_workflow.py --mode preprocess
python run_workflow.py --mode train --device cuda
python run_workflow.py --mode infer --device cuda --index 0
python run_workflow.py --mode eval --device cuda
streamlit run src/app/streamlit_app.py
```

The Streamlit demo selects a fixed held-out QTDB test ECG or accepts an
uploaded PTB-XL CSV/NPY/NPZ export or matching WFDB `.hea`/`.dat` pair. Choose
the WFDB format in the sidebar and upload `.hea` and `.dat` in their separate
fields, using files with the same stem (for example, `00001_lr.hea` and
`00001_lr.dat`). Then choose one lead and the source sampling rate (100 or 500 Hz for NumPy/CSV); the app
resamples one 10-second window to 360 Hz. The main view shows the observed
ECG as the fixed condition and runs all 50 conditional DDIM calls, showing
the actual state after each call and the restored ECG. A separate button
illustrates Gaussian forward diffusion from t=0 through t=50; it is
a training-process example, not a corruption of the denoised output. The
user-selected k controls which frames are displayed, not how many calls the
model performs. It can run on CPU if CUDA is unavailable. Metrics are shown
only for QTDB test examples with a known reference. For an uploaded PTB-XL
signal, forward diffusion uses that signal as a visual starting point, and
the reverse model has no clean reference for numerical scoring.
For QTDB, the demo scales the stored mixed-artifact residual to the selected
q between 0.2 and 2.0. The default sample at its original q corresponds to
the locked benchmark pair; changing q creates a visual demonstration pair.
The MIX7 injection, training forward process and DDIM inference have separate
animations. Inference starts from an independent Gaussian draw and keeps the
observed ECG fixed as its condition.
If an uploaded record is shorter than 10 seconds, the app zero-pads it to
3600 points, shades the padded region in every waveform, and marks the result
as a visual demonstration rather than a protocol-comparable evaluation.

Training writes `checkpoints/3600/best_3600.pth` and `last_3600.pth`. Validation
selects the raw model's lowest loss and stores matching EMA weights. The default
inference checkpoint is the existing epoch-92 EMA export at
`output/fig05_mix7_3600/fig05_u4_3600_ema.pth`; pass `--checkpoint` to use another.
Inference normalizes by the noisy signal's maximum absolute amplitude and
restores the original scale. Evaluation uses 50-step DDIM with `eta=0`, one
initial draw, batch size 4, and seed 2024. The train objective uses mean L1
noise error plus 0.1 times the relative STFT magnitude error weighted by
diffusion SNR.

Some older standalone utilities in `src/data_prep`, `src/models/main_model.py`,
and `src/training/utils.py` are retained for historical experiments. The
published 3600-point workflow does not call them.
