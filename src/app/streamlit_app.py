"""Interactive 3600-point ECG forward diffusion and conditional DDIM demo."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import streamlit as st
import torch

from src.protocol3600 import FS, LENGTH, load_bundle, load_ema, metrics

BUNDLE = ROOT / 'common_qtdb_mixed_bwem_ma_3600_ntype1.npz'
CONFIG = ROOT / 'configs' / 'base.yaml'
CHECKPOINT = ROOT / 'output' / 'fig05_mix7_3600' / 'fig05_u4_3600_ema.pth'
PTBXL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


@st.cache_data(show_spinner='Đang đọc tập test...')
def test_data(bundle_path: str):
    clean, noisy, records = load_bundle(bundle_path, ('test',))['test']
    with np.load(bundle_path, allow_pickle=False) as bundle:
        return clean, noisy, records, np.asarray(bundle['q_test'])


@st.cache_resource(show_spinner='Đang nạp checkpoint EMA...')
def denoiser(config_path: str, checkpoint_path: str, device_name: str):
    return load_ema(config_path, checkpoint_path, device_name)


def read_uploaded(files):
    """Read one PTB-XL CSV/NumPy export or a matching WFDB .hea/.dat pair."""
    names = [Path(file.name).name for file in files]
    suffixes = {Path(name).suffix.lower() for name in names}
    if '.hea' in suffixes:
        import wfdb
        with tempfile.TemporaryDirectory() as folder:
            for file, name in zip(files, names):
                if Path(name).suffix.lower() in ('.hea', '.dat'):
                    (Path(folder) / name).write_bytes(file.getvalue())
            headers = [name for name in names if name.lower().endswith('.hea')]
            if len(headers) != 1:
                raise ValueError('Hãy tải đúng một cặp .hea và .dat cho một bản ghi.')
            header = Path(headers[0])
            header_path = Path(folder) / header.name
            header_text = header_path.read_bytes().decode('utf-8-sig', errors='replace')
            header_lines = [line.strip() for line in header_text.splitlines()]
            while header_lines and (not header_lines[0] or header_lines[0].startswith('#')):
                header_lines.pop(0)
            if not header_lines:
                raise ValueError('File .hea trống hoặc không có dòng định nghĩa bản ghi WFDB.')
            first_line = header_lines[0]
            if first_line.startswith(('<', '{')) or '<html' in first_line.lower():
                raise ValueError('File .hea có vẻ là trang HTML/JSON, không phải header WFDB. '
                                 'Hãy tải file raw từ thư mục records100 hoặc records500 của PTB-XL.')
            header_path.write_text('\n'.join(header_lines) + '\n', encoding='utf-8')
            try:
                wfdb.rdheader(str(Path(folder) / header.stem))
            except Exception as error:
                raise ValueError(f'Header WFDB không hợp lệ ({error}). Dòng đầu: {first_line[:100]!r}. '
                                 'Hãy kiểm tra đây là file .hea raw của PTB-XL.') from error
            data_names = {line.split()[0] for line in header_lines[1:]
                          if line.strip() and not line.startswith('#')
                          and line.split()[0].lower().endswith('.dat')}
            uploaded_data = {name for name in names if name.lower().endswith('.dat')}
            if data_names != uploaded_data:
                expected = ', '.join(sorted(data_names)) or f'{header.stem}.dat'
                raise ValueError(f'File .hea tham chiếu `{expected}`; hãy tải đúng file .dat đó.')
            record = wfdb.rdrecord(str(Path(folder) / header.stem))
            signal = record.p_signal
            fs = int(record.fs)
            leads = list(record.sig_name)
    elif len(files) == 1:
        file = files[0]
        suffix = Path(file.name).suffix.lower()
        if suffix == '.csv':
            frame = pd.read_csv(io.BytesIO(file.getvalue()))
            try:
                if all(np.isfinite(float(column)) for column in frame.columns):
                    frame = pd.read_csv(io.BytesIO(file.getvalue()), header=None)
            except (TypeError, ValueError):
                pass
            frame = frame.drop(columns=[c for c in frame.columns if str(c).lower() in ('time', 'timestamp')])
            frame = frame.apply(pd.to_numeric, errors='coerce')
            signal = frame.to_numpy(dtype=np.float32)
            leads = [str(column) for column in frame.columns]
        elif suffix == '.npy':
            signal = np.load(io.BytesIO(file.getvalue()), allow_pickle=False)
            leads = None
        elif suffix == '.npz':
            with np.load(io.BytesIO(file.getvalue()), allow_pickle=False) as archive:
                candidates = [key for key in archive.files if archive[key].ndim in (1, 2)
                              and np.issubdtype(archive[key].dtype, np.number)]
                if len(candidates) != 1:
                    raise ValueError('NPZ cần chứa đúng một mảng tín hiệu 1D hoặc 2D.')
                signal = archive[candidates[0]].copy()
            leads = None
        else:
            raise ValueError('Hỗ trợ CSV, NPY, NPZ hoặc cặp WFDB .hea/.dat.')
        fs = None
    else:
        raise ValueError('Chỉ tải một file tín hiệu hoặc một cặp WFDB .hea/.dat.')

    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2:
        raise ValueError('Tín hiệu phải có dạng thời gian × chuyển đạo.')
    if signal.shape[0] <= 16 and signal.shape[1] > 16:
        signal = signal.T
    if signal.shape[1] > 16 or not np.isfinite(signal).all():
        raise ValueError('Tối đa 16 chuyển đạo; dữ liệu không được chứa NaN/Inf.')
    if leads is None or len(leads) != signal.shape[1]:
        leads = PTBXL_LEADS if signal.shape[1] == 12 else [f'Kênh {i+1}' for i in range(signal.shape[1])]
    return signal, fs, leads


def signal_window(signal, fs: int, lead: int, start_seconds: float):
    count = fs * 10
    start = int(round(start_seconds * fs))
    if start >= len(signal):
        raise ValueError('Vị trí bắt đầu nằm ngoài tín hiệu.')
    segment = signal[start:min(start + count, len(signal)), lead]
    divisor = math.gcd(fs, FS)
    result = resample_poly(segment, FS // divisor, fs // divisor).astype(np.float32)
    if len(result) < LENGTH:
        result = np.pad(result, (0, LENGTH - len(result)))
    result = result[:LENGTH]
    if result.shape != (LENGTH,) or not np.isfinite(result).all():
        raise ValueError('Không thể tạo cửa sổ 3.600 mẫu ở 360 Hz.')
    return result


def waveform(series: dict[str, np.ndarray], left: float, right: float,
             normalized=False, padded_after=None):
    first, last = int(left * FS), min(int(right * FS) + 1, LENGTH)
    time_axis = np.arange(first, last) / FS
    fig, axis = plt.subplots(figsize=(11, 3.4))
    palette = {
        'ECG tham chiếu': '#28323d', 'x₀ dùng minh họa': '#28323d',
        'ECG quan sát': '#c98226', 'ECG điều kiện y': '#c98226',
        'ECG khôi phục': '#267ab0', 'Trạng thái xₜ': '#267ab0',
        'Mẫu khuếch tán thuận xₜ': '#735a9e',
    }
    for name, values in series.items():
        axis.plot(time_axis, np.asarray(values).reshape(-1)[first:last],
                  label=name, color=palette.get(name, '#267ab0'), linewidth=1.25)
    if padded_after is not None and padded_after < right:
        axis.axvspan(max(left, padded_after), right, color='#e7e9ed', alpha=0.55,
                     label='Vùng đệm, không phải ECG gốc')
    axis.set_xlim(left, right)
    axis.set(xlabel='Thời gian (giây)',
             ylabel='Biên độ chuẩn hóa' if normalized else 'Biên độ ECG')
    axis.grid(alpha=0.2)
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=len(series), frameon=False)
    fig.tight_layout()
    return fig


def show_waveform(target, series, left, right, normalized=False, padded_after=None):
    fig = waveform(series, left, right, normalized, padded_after)
    target.pyplot(fig, width='stretch')
    plt.close(fig)


def play_waveform(fixed, frames, labels, left, right, speed, fixed_label, moving_label,
                  normalized=False, padded_after=None):
    """Animate waveform frames in the browser without a Streamlit rerun per frame."""
    first, last = int(left * FS), min(int(right * FS) + 1, LENGTH)
    payload = json.dumps({
        'fixed': [np.asarray(row).reshape(-1)[first:last].tolist() for row in fixed],
        'frames': [np.asarray(frame).reshape(-1)[first:last].tolist() for frame in frames],
        'labels': labels,
        'padded': padded_after,
    }, separators=(',', ':')).replace('</', '<\\/')
    y_label = 'Biên độ chuẩn hóa' if normalized else 'Biên độ ECG'
    delay = int(float(speed) * 1000)
    markup = f'''<!doctype html><html><head><meta charset="utf-8"><style>
    body{{font:13px sans-serif;margin:0;color:#28323d}} .controls{{display:flex;gap:6px;align-items:center;margin:4px 0 5px}}
    .legend{{display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin:0 0 6px 4px;min-height:22px}}
    .legend span{{display:inline-flex;gap:6px;align-items:center}} .swatch{{width:20px;height:3px;display:inline-block}}
    #label{{margin-left:6px;white-space:nowrap;font-weight:600}}
    button{{padding:4px 8px;cursor:pointer;border:1px solid #d4d9df;border-radius:4px;background:#fff;color:#28323d}}
    canvas{{width:100%;height:340px;display:block}}
    </style></head><body><div class="controls"><button id="play">⏸ Tạm dừng</button>
    <button id="zoomout" title="Thu nhỏ">−</button><button id="zoomin" title="Phóng to">+</button>
    <button id="resetzoom" title="Đặt lại zoom">100%</button>
    <input id="seek" type="range" min="0" max="{len(frames)-1}" value="0" style="flex:1">
    </div><div class="legend"><span><i class="swatch" style="background:#c98226"></i>{fixed_label}</span>
    <span><i id="moving-swatch" class="swatch" style="background:#267ab0"></i>{moving_label}</span>
    <span id="label"></span></div><canvas id="plot"></canvas><script>
    const d={payload}, delay={delay}, xlabel='Thời gian (giây)', ylabel={json.dumps(y_label)};
    const fixedName={json.dumps(fixed_label)}, movingName={json.dumps(moving_label)};
    const movingColor=fixedName.startsWith('x₀')?'#735a9e':'#267ab0';
    document.getElementById('moving-swatch').style.backgroundColor=movingColor;
    const canvas=document.getElementById('plot'), ctx=canvas.getContext('2d'), seek=document.getElementById('seek');
    const label=document.getElementById('label'), button=document.getElementById('play'); let index=0, playing=true, timer, zoom=1;
    const split=d.labels.findIndex(x=>x.startsWith('Nghịch DDIM'));
    function bounds(start,end){{let lo=Infinity,hi=-Infinity;
      for(let i=start;i<end;i++){{for(const v of d.fixed[i]){{lo=Math.min(lo,v);hi=Math.max(hi,v)}}
        for(const v of d.frames[i]){{lo=Math.min(lo,v);hi=Math.max(hi,v)}}}}
      if(hi-lo<1e-8){{hi+=1;lo-=1}}const p=(hi-lo)*.025;return [lo-p,hi+p]}}
    const forwardBounds=bounds(0,split<0?d.frames.length:split);
    const reverseBounds=split<0?forwardBounds:bounds(split,d.frames.length);
    function draw(){{const w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;
      canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
      const L=58,R=16,T=16,B=42,pw=w-L-R,ph=h-T-B, a=d.frames[index];
      const [baseLo,baseHi]=split>=0&&index>=split?reverseBounds:forwardBounds;
      const mid=(baseHi+baseLo)/2,half=(baseHi-baseLo)/(2*zoom),ymin=mid-half,ymax=mid+half;
      ctx.font='12px sans-serif';ctx.lineWidth=1;ctx.strokeStyle='#dce1e7';ctx.fillStyle='#5d6875';
      for(let j=0;j<=4;j++){{let y=T+ph*j/4;ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(w-R,y);ctx.stroke();ctx.fillText((ymax-(ymax-ymin)*j/4).toFixed(2),5,y+4)}}
      if(d.padded!==null&&d.padded<({right})){{let x=L+Math.max(0,(d.padded-{left})/({right}-{left}))*pw;ctx.fillStyle='#e7e9ed';ctx.fillRect(x,T,w-R-x,ph)}}
      function line(v,color){{ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=1.5;for(let i=0;i<v.length;i++){{let x=L+pw*i/Math.max(1,v.length-1),y=T+ph*(ymax-v[i])/(ymax-ymin);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}}ctx.stroke()}}
      line(a,movingColor);line(d.fixed[index],'#c98226');ctx.fillStyle='#28323d';
      ctx.fillText(xlabel,L+pw/2-45,h-8);
      label.textContent=d.labels[index];seek.value=index;}}
    function step(){{if(!playing)return;index=(index+1)%d.frames.length;draw();timer=setTimeout(step,delay)}}
    button.onclick=()=>{{playing=!playing;button.textContent=playing?'⏸ Tạm dừng':'▶ Phát';if(playing)step();else clearTimeout(timer)}};
    seek.oninput=()=>{{index=Number(seek.value);draw()}};
    document.getElementById('zoomin').onclick=()=>{{zoom=Math.min(zoom*1.25,5);draw()}};
    document.getElementById('zoomout').onclick=()=>{{zoom=Math.max(zoom/1.25,1);draw()}};
    document.getElementById('resetzoom').onclick=()=>{{zoom=1;draw()}};
    window.addEventListener('resize',draw);draw();timer=setTimeout(step,delay);
    </script></body></html>'''
    st.components.v1.html(markup, height=400, scrolling=False)


def visible_steps(total: int, gap: int):
    return sorted(set(range(0, total + 1, gap)) | {total})


@torch.no_grad()
def ddim_from_initial_noise(model, y, initial_noise, steps=50):
    """Run the app trajectory from an explicit x_T, independent of stale imports."""
    if y.shape != initial_noise.shape:
        raise ValueError('Initial diffusion state must match the conditioning ECG shape.')
    x = initial_noise.to(device=y.device, dtype=y.dtype).clone()
    tau = [int(np.floor(model.steps * (j / steps) ** 2)) for j in range(steps + 1)]
    states = [(0, tau[-1], x.detach().cpu().numpy().copy())]
    for j in range(steps, 0, -1):
        t, previous = tau[j], tau[j - 1]
        at, ap = model.alpha_bar[t], model.alpha_bar[previous]
        eps_hat = model.net(x, y, at.sqrt().expand(len(y), 1))
        x0_hat = (x - (1 - at).sqrt() * eps_hat) / at.sqrt().clamp_min(1e-6)
        x = ap.sqrt() * x0_hat + (1 - ap).sqrt() * eps_hat
        states.append((steps - j + 1, previous, x.detach().cpu().numpy().copy()))
    return x, states


st.set_page_config(page_title='ECG Diffusion 3600', layout='wide')
st.title('Mô phỏng khuếch tán và khử nhiễu ECG')
st.caption('HNF-FiLM 1D U-Net · 3.600 mẫu · 360 Hz · DDIM 50 lượt · checkpoint EMA epoch 92')

if not CHECKPOINT.is_file():
    st.error('Không tìm thấy checkpoint EMA tại output/fig05_mix7_3600/.')
    st.stop()

with st.sidebar:
    st.header('Dữ liệu')
    source_kind = st.radio('Nguồn ECG', ['QTDB test', 'PTB-XL tải lên'])
    seed = st.number_input('Seed mô phỏng', min_value=0, value=2024, step=1)
    device_name = st.selectbox('Thiết bị', ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu'])
    gap = st.slider('Hiển thị mỗi k bước', 1, 50, 5)
    speed = st.slider('Thời gian mỗi khung (giây)', 0.1, 1.0, 0.3, 0.1)

reference = None
source_id = None
padded_after = None
composite_residual = None
if source_kind == 'QTDB test':
    if not BUNDLE.is_file():
        st.error('Không tìm thấy gói dữ liệu QTDB test.')
        st.stop()
    clean, noisy, records, q = test_data(str(BUNDLE))
    with st.sidebar:
        index = st.number_input('Chỉ số mẫu test', 0, len(clean) - 1, value=0, step=1)
        noise_level = st.slider('Mức nhiễu q', 0.2, 2.0, float(np.clip(q[index], 0.2, 2.0)),
                                0.1, format='%.1f')
        st.write(f'Bản ghi: **{records[index]}** · mức q gốc = **{float(q[index]):.2f}**')
    reference = clean[index, 0]
    source_q = max(float(q[index]), 1e-8)
    composite_residual = (noisy[index, 0] - reference) * (float(noise_level) / source_q)
    observation = reference + composite_residual
    source_id = f'qt:{index}:q{noise_level:.1f}'
    forward_base = reference
else:
    with st.sidebar:
        upload_format = st.radio('Định dạng PTB-XL', ['WFDB (.hea + .dat)', 'CSV / NPY / NPZ'])
        if upload_format.startswith('WFDB'):
            header_file = st.file_uploader('1. Tải file tiêu đề .hea', type=['hea'], key='ptbxl_header')
            data_file = st.file_uploader('2. Tải file tín hiệu .dat', type=['dat'], key='ptbxl_data')
            files = [header_file, data_file] if header_file and data_file else []
        else:
            signal_file = st.file_uploader('Tải một file ECG', type=['csv', 'npy', 'npz'], key='ptbxl_array')
            files = [signal_file] if signal_file else []
    if not files:
        if upload_format.startswith('WFDB'):
            st.info('Chọn đủ hai file cùng tên gốc, ví dụ `00001_lr.hea` và `00001_lr.dat`. '
                    'Có thể chọn lần lượt ở hai ô bên trái.')
        else:
            st.info('Tải một tín hiệu PTB-XL đã xuất ra CSV, NPY hoặc NPZ.')
        st.stop()
    try:
        matrix, file_fs, leads = read_uploaded(files)
        with st.sidebar:
            fs = file_fs or st.selectbox('Tần số lấy mẫu gốc (Hz)', [100, 500], index=1)
            lead = st.selectbox('Chuyển đạo ECG', range(len(leads)),
                                format_func=lambda i: leads[i])
            duration = len(matrix) / fs
            st.write(f'Thời lượng file: **{duration:.2f} giây** · {fs} Hz · {len(matrix)} mẫu')
            if duration > 10:
                start_seconds = st.number_input('Bắt đầu cửa sổ 10 giây (giây)',
                                                min_value=0.0, max_value=duration - 10,
                                                value=0.0, step=0.1)
            else:
                start_seconds = 0.0
        observation = signal_window(matrix, int(fs), int(lead), start_seconds)
        if duration < 10:
            padded_after = duration
        forward_base = observation
        digest = hashlib.sha256(b''.join(file.getvalue() for file in files)).hexdigest()[:16]
        source_id = f'upload:{digest}:{lead}:{start_seconds}:{fs}'
    except Exception as error:
        st.error(f'Không đọc được ECG tải lên: {error}')
        st.stop()
    st.warning('PTB-XL là dữ liệu ngoài tập huấn luyện QTDB. Không có ECG sạch tham chiếu cho file tải lên; '
               'mục khuếch tán thuận dùng chính tín hiệu tải lên làm x₀ để minh họa công thức. '
               'Nó không phải các bước xử lý đầu vào khi khử nhiễu.')
    if padded_after is not None:
        st.warning(f'File chỉ dài {padded_after:.2f} giây. Demo đệm số 0 đến đủ 10 giây/3.600 mẫu; '
                   'vùng tô xám trên đồ thị là phần đệm. Kết quả khử nhiễu của cửa sổ này chỉ mang tính minh họa. '
                   'Bản ghi PTB-XL chuẩn thường dài 10 giây; hãy kiểm tra file .hea/.dat nếu muốn đánh giá đúng giao thức.')

left, right = st.slider('Khoảng thời gian quan sát (giây)', 0.0, 10.0, (0.0, 2.0), step=0.1)
if right <= left:
    st.stop()
st.subheader('Tín hiệu đầu vào')
show_reference = reference is not None and st.checkbox('Hiện ECG tham chiếu', value=False)
if reference is not None:
    st.markdown(f'**Thí nghiệm QTDB · nhiễu tổ hợp MIX7 (BW + EM + MA), q={noise_level:.1f}**')
    st.caption('Điều chỉnh q bằng cách co giãn biên độ nhiễu tổng hợp của cặp tham chiếu đã chọn.')
else:
    st.markdown('**Tín hiệu ECG tải lên** · ECG màu cam')
input_series = {'ECG quan sát': observation}
if show_reference:
    input_series = {'ECG tham chiếu': reference, **input_series}
show_waveform(st, input_series, left, right, padded_after=padded_after)

if reference is not None:
    st.subheader('Bơm tổ hợp nhiễu MIX7')
    st.caption('Hoạt ảnh này minh họa riêng cách cộng phần nhiễu tổng hợp vào ECG sạch.')
    if st.button('▶ Phát quá trình bơm nhiễu MIX7'):
        mix_levels = visible_steps(50, int(gap))
        mix_frames = [reference + (level / 50) * composite_residual for level in mix_levels]
        play_waveform([reference] * len(mix_frames), mix_frames,
                      [f'Bơm MIX7 · {level*2}%' for level in mix_levels],
                      left, right, speed, 'ECG sạch x₀', 'ECG khi thêm nhiễu',
                      padded_after=padded_after)

model, device = denoiser(str(CONFIG), str(CHECKPOINT), device_name)
scale = max(float(np.max(np.abs(observation))), 1e-8)
x0 = forward_base / scale
forward_noise = np.random.default_rng(int(seed)).standard_normal(LENGTH).astype(np.float32)
xT = np.random.default_rng(int(seed) + 1).standard_normal(LENGTH).astype(np.float32)

st.subheader('1 · Khuếch tán thuận — minh họa huấn luyện')
st.caption('Bắt đầu từ ECG sạch x₀ và thêm nhiễu Gaussian theo q(xₜ|x₀). Hoạt ảnh này minh họa cách huấn luyện, '
           'không phải bước khử nhiễu lúc suy luận.')
if st.button('▶ Phát hoạt ảnh khuếch tán thuận'):
    forward_options = visible_steps(50, int(gap))
    forward_frames = []
    for t in forward_options:
        alpha = float(model.alpha_bar[t].cpu())
        forward_frames.append(np.sqrt(alpha) * x0 + np.sqrt(1 - alpha) * forward_noise)
    play_waveform([x0] * len(forward_frames), forward_frames,
                  [f'Thuận · t={t}/50' for t in forward_options],
                  left, right, speed, 'ECG sạch x₀', 'Trạng thái xₜ',
                  normalized=True, padded_after=padded_after)

st.subheader('2 · Khử nhiễu DDIM có điều kiện')
st.caption('Khử nhiễu bắt đầu từ Gaussian độc lập ε~N(0,I), không dùng ECG tham chiếu để khởi tạo. '
           'ECG nhiễu màu cam y được giữ làm điều kiện trong từng lượt.')
run = st.button('▶ Chạy hoạt ảnh khử nhiễu DDIM', type='primary')
if run:
    with st.spinner('Đang chạy mô hình; CPU có thể mất khoảng vài chục giây...'):
        y = torch.from_numpy((observation / scale)[None, None]).to(device)
        initial_noise = torch.from_numpy(xT[None, None]).to(device=device, dtype=y.dtype)
        final, states = ddim_from_initial_noise(model, y, initial_noise, steps=50)
        prediction = final[0, 0].cpu().numpy() * scale
        trajectory = [(call, t, array[0, 0].copy()) for call, t, array in states]
        st.session_state['demo_result'] = (source_id, int(seed), prediction, trajectory)

result = st.session_state.get('demo_result')
if result is not None and result[:2] == (source_id, int(seed)):
    _, _, prediction, trajectory = result
    reverse_options = visible_steps(50, int(gap))
    reverse_frames, fixed_frames, labels = [], [], []
    for call in reverse_options:
        _, t, state = trajectory[call]
        reverse_frames.append(state)
        fixed_frames.append(observation / scale)
        labels.append(f'Nghịch DDIM · lượt {call}/50 · t={t}')
    st.subheader('Hoạt ảnh khử nhiễu')
    play_waveform(fixed_frames, reverse_frames, labels, left, right, speed,
                  'ECG điều kiện y', 'Trạng thái DDIM',
                  normalized=True, padded_after=padded_after)

    st.subheader('ECG sau khử nhiễu')
    final_series = {'ECG quan sát': observation, 'ECG khôi phục': prediction}
    if show_reference:
        final_series = {'ECG tham chiếu': reference, **final_series}
    show_waveform(st, final_series, left, right, padded_after=padded_after)
    if reference is not None:
        noisy_score = metrics(reference, observation, observation)
        clean_score = metrics(reference, observation, prediction)
        rows = []
        for label, values in [('Đầu vào nhiễu', noisy_score), ('Sau khử nhiễu', clean_score)]:
            rows.append({'Tín hiệu': label, 'SSD ↓': values['SSD'], 'RMSE ↓': values['RMSE'],
                         'MAD ↓': values['MAD'], 'PRD (%) ↓': values['PRD'],
                         'Cosine ↑': values['Cosine'], 'SNR (dB) ↑': values['SNR_out'],
                         'ΔSNR (dB) ↑': values['Delta_SNR']})
        st.dataframe(pd.DataFrame(rows).style.format(precision=4), hide_index=True, width='stretch')
        st.caption('Chỉ số tính trên một ECG test được chọn, không phải trung bình tập test.')
    else:
        st.info('Không tính PRD, SSD, MAD, cosine hoặc SNR cho file PTB-XL tải lên vì không có ECG sạch tham chiếu.')
else:
    st.info('Đang chờ kích hoạt; hiện chỉ hiển thị ECG nhiễu và ECG tham chiếu tùy chọn.')
