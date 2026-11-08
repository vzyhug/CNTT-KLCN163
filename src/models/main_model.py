import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
from inspect import isfunction

# ----- Bắt buộc phải có torchaudio hoặc dùng torch.stft -----
# Nếu không có torchaudio, ta dùng torch.stft nguyên bản
def stft_loss(pred, target, n_fft=128, hop_length=64):
    """
    Tính MSE loss trên phổ STFT giữa pred và target.
    """
    # (B, C, L) -> (B*C, L)
    B, C, L = pred.shape
    pred = pred.view(-1, L)
    target = target.view(-1, L)
    
    # STFT: returns (B, freq_bins, time_frames, 2)
    spec_pred = torch.stft(pred, n_fft=n_fft, hop_length=hop_length, 
                           return_complex=True)
    spec_target = torch.stft(target, n_fft=n_fft, hop_length=hop_length,
                             return_complex=True)
    
    # Lấy magnitude (độ lớn)
    mag_pred = torch.abs(spec_pred)
    mag_target = torch.abs(spec_target)
    
    return F.mse_loss(mag_pred, mag_target)


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class DDPM(nn.Module):
    def __init__(self, base_model, config, device, conditional=True):
        super().__init__()
        self.device = device
        self.model = base_model
        self.config = config
        self.conditional = conditional
        
        # ----------------------
        # 👉 THÊM: Loss weights
        # ----------------------
        self.lambda_time = config['train'].get('lambda_time', 1.0)
        self.lambda_freq = config['train'].get('lambda_freq', 0.1)  # cân bằng
        
        self.loss_func_l1 = nn.L1Loss(reduction='sum').to(device)
        
        config_diff = config["diffusion"]
        self.num_steps = config_diff["num_steps"]
        self.set_new_noise_schedule(config_diff, device)
        
    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas
    
    def set_new_noise_schedule(self, config_diff, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        
        betas = self.make_beta_schedule(
            schedule=config_diff["schedule"], 
            n_timesteps=config_diff["num_steps"],
            start=config_diff["beta_start"], 
            end=config_diff["beta_end"]
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', 
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', 
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', 
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
    def predict_start_from_noise(self, x_t, t, noise):
        # Nắn lại chiều tensor hệ số thành (Batch, 1, 1) để nhân đúng với tensor tín hiệu 3D
        coef1 = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1)
        coef2 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)
        
        return coef1 * x_t - coef2 * noise
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + \
                         self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]
        ).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model(x, condition_x, noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model(x, noise_level))
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_steps//10))
        if not self.conditional:
            shape = x_in
            cur_x = torch.randn(shape, device=device)
            ret_x = cur_x
            for i in reversed(range(0, self.num_steps)):
                cur_x = self.p_sample(cur_x, i)
                if i % sample_inter == 0:
                    ret_x = torch.cat([ret_x, cur_x], dim=0)
        else:
            x = x_in
            shape = x.shape
            cur_x = torch.randn(shape, device=device)
            ret_x = [cur_x]
            for i in reversed(range(0, self.num_steps)):
                cur_x = self.p_sample(cur_x, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_x.append(cur_x)
        if continous:
            return ret_x
        else:
            return ret_x[-1]    
    
    @torch.no_grad()
    def sample(self, batch_size=1, shape=[1, 512], continous=False):
        return self.p_sample_loop((batch_size, shape[0], shape[1]), continous)
    
    # ==========================================================
    # 👉 DDIM SAMPLER (đã có sẵn trong source gốc)
    # ==========================================================
    @torch.no_grad()
    def ddim_sample_loop(self, x_in, ddim_timesteps=50, ddim_eta=0.0, num_shots=1):
        device = self.betas.device
        T = self.num_steps
        S = ddim_timesteps
        tau = [int(np.floor((T / (S ** 2)) * (i ** 2))) for i in range(S + 1)]
        
        alphas_cumprod_with_1 = torch.cat([
            torch.tensor([1.0], dtype=self.alphas_cumprod.dtype, device=device),
            self.alphas_cumprod
        ]).float()
        
        out_accum = 0.0
        for shot in range(num_shots):
            shape = x_in.shape
            x = torch.randn(shape, device=device)
            for i in reversed(range(1, S + 1)):
                t = tau[i]
                t_prev = tau[i - 1]
                noise_level = torch.FloatTensor(
                    [self.sqrt_alphas_cumprod_prev[t]]
                ).repeat(x.shape[0], 1).to(device)
                if not self.conditional:
                    eps = self.model(x, noise_level)
                else:
                    eps = self.model(x, x_in, noise_level)
                    
                a_t = alphas_cumprod_with_1[t]
                a_t_prev = alphas_cumprod_with_1[t_prev]
                x0_pred = (x - torch.sqrt(1. - a_t) * eps) / torch.sqrt(a_t)
                if t_prev == 0:
                    sigma_t = 0.0
                else:
                    sigma_t = ddim_eta * torch.sqrt((1. - a_t_prev) / (1. - a_t)) * \
                              torch.sqrt(torch.clamp(1. - a_t / a_t_prev, min=0.0))
                direction = torch.sqrt(torch.clamp(1. - a_t_prev - sigma_t**2, min=0.0)) * eps
                noise = sigma_t * torch.randn_like(x) if (ddim_eta > 0 and t_prev > 0) else 0.0
                x = torch.sqrt(a_t_prev) * x0_pred + direction + noise
            out_accum += x
        return out_accum / num_shots

    @torch.no_grad()
    def denoising(self, x_in, continous=False, use_ddim=False, ddim_steps=50, ddim_eta=0.0, num_shots=1):
        if use_ddim:
            return self.ddim_sample_loop(x_in, ddim_timesteps=ddim_steps, ddim_eta=ddim_eta, num_shots=num_shots)
        else:
            if num_shots > 1:
                out_accum = 0.0
                for _ in range(num_shots):
                    out_accum += self.p_sample_loop(x_in, continous)
                return out_accum / num_shots
            return self.p_sample_loop(x_in, continous)
    
    def q_sample_loop(self, x_start, continous=False):
        sample_inter = (1 | (self.num_steps//10))
        ret_x = [x_start]
        cur_x = x_start
        for t in range(1, self.num_steps+1):
            B,C,L = cur_x.shape
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t-1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=B
                )
            ).to(cur_x.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(B, -1)
            noise = torch.randn_like(cur_x)
            cur_x = self.q_sample(
                x_start=cur_x, 
                continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1), 
                noise=noise
            )
            if t % sample_inter == 0:
                ret_x.append(cur_x)
        if continous:
            return ret_x
        else:
            return ret_x[-1]
    
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return continuous_sqrt_alpha_cumprod * x_start + \
               (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
    
    # ==========================================================
    # 👉 HÀM LOSS CHÍNH (ĐÃ THÊM STFT)
    # ==========================================================
    def p_losses(self, x_in, y_in, noise=None):
        """
        Multi‑domain Loss:
        - L1 trên noise (miền thời gian)
        - STFT Loss trên tín hiệu dự đoán (miền tần số)
        """
        x_start = x_in
        B, C, L = x_start.shape

        # Lấy ngẫu nhiên các giá trị t khác nhau cho từng mẫu trong Batch, trực tiếp trên thiết bị (GPU)
        t = torch.randint(0, self.num_steps, (B,), device=x_start.device).long()
        # 1. Chuyển mảng NumPy sang PyTorch Tensor và đẩy lên cùng GPU với t
        sqrt_alphas_prev_tensor = torch.tensor(self.sqrt_alphas_cumprod_prev, dtype=torch.float32, device=t.device)
        # 2. Lấy giới hạn dưới và trên. 
        # (Lưu ý: Do t giờ chạy từ 0 đến num_steps-1, ta dùng chỉ số t và t+1 để thay thế cho t-1 và t nhằm tránh lỗi âm)
        lower = sqrt_alphas_prev_tensor[t]
        upper = sqrt_alphas_prev_tensor[t+1]
        # 3. Sinh số ngẫu nhiên uniform hoàn toàn bằng PyTorch trực tiếp trên GPU
        continuous_sqrt_alpha_cumprod = lower + torch.rand(B, device=t.device) * (upper - lower)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(B, -1)
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, 
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1), 
            noise=noise
        )

        # Dự đoán noise
        if not self.conditional:
            x_recon = self.model(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.model(x_noisy, y_in, continuous_sqrt_alpha_cumprod)

        # ---------- 1. L1 Loss (miền thời gian) ----------
        loss_l1 = self.loss_func_l1(noise, x_recon)  # (B*C*L)

        # ---------- 2. STFT Loss (miền tần số) ----------
        # Dự đoán x0 từ noise đã dự đoán
        x0_pred = self.predict_start_from_noise(x_noisy, t, x_recon)
        
        # Tính STFT loss giữa x0_pred và x_start (clean gốc)
        loss_stft = stft_loss(x0_pred, x_start, n_fft=128, hop_length=64)
        
        # ---------- 3. Tổng hợp ----------
        total_loss = self.lambda_time * loss_l1 + self.lambda_freq * loss_stft
        return total_loss
    
    def forward(self, x, y, *args, **kwargs):
        return self.p_losses(x, y, *args, **kwargs)


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
