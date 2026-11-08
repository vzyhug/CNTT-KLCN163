import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import src.evaluation.metrics as metrics
from src.models.main_model import EMA
from numpy.testing import verbose
import csv
import os
import src.evaluation.metrics as metrics
# Giả định bạn đã import Adam, EMA... ở trên

def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"])
    ema = EMA(0.9)
    ema.register(model)

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
        
        # 1. Khởi tạo file log CSV
        log_file = foldername + "/training_log.csv"
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'SNR', 'RMSE', 'PRD', 'Cosin_Sim'])
    else:
        log_file = None

    llr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=150, gamma=1.
    )
    best_valid_loss = 1e10

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, desc=f"Train Epoch {epoch_no}") as it:
            for batch_no, (clean_batch, noisy_patch) in enumerate(it, start=1):
                clean_batch, noisy_patch = clean_batch.to(device), noisy_patch.to(device)
                optimizer.zero_grad()
                loss = model(clean_batch, noisy_patch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                avg_loss += loss.item()
                ema.update(model)
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=True
                )
        llr_scheduler.step()
        train_epoch_loss = avg_loss / batch_no

        # Khởi tạo các giá trị rỗng phòng trường hợp epoch này không chạy Validation
        val_epoch_loss = "N/A"
        avg_snr = "N/A"
        avg_rmse = "N/A"
        avg_prd = "N/A"
        avg_cos = "N/A"

        # 2. Pha Đánh giá (Validation) & Tính Metrics
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            total_rmse, total_snr, total_prd, total_cos = 0.0, 0.0, 0.0, 0.0
            
            with torch.no_grad():
                with tqdm(valid_loader, desc=f"Valid Epoch {epoch_no}") as it:
                    for batch_no_val, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        
                        # Tính hàm mất mát
                        loss = model(clean_batch, noisy_batch)
                        avg_loss_valid += loss.item()

                        # Trích xuất tín hiệu sạch để đo các chỉ số vật lý
                        denoised_signal = model.denoising(noisy_batch,use_ddim=True,ddim_steps=15)
                        rmse, snr, prd, cos = metrics.calculate_metrics(clean_batch, denoised_signal)
                        
                        total_rmse += rmse
                        total_snr += snr
                        total_prd += prd
                        total_cos += cos

                        it.set_postfix(
                            ordered_dict={
                                "valid_loss": avg_loss_valid / batch_no_val,
                                "snr_db": total_snr / batch_no_val,
                            },
                            refresh=True,
                        )

            val_epoch_loss = avg_loss_valid / batch_no_val
            avg_snr = round(total_snr / batch_no_val, 2)
            avg_rmse = round(total_rmse / batch_no_val, 4)
            avg_prd = round(total_prd / batch_no_val, 2)
            avg_cos = round(total_cos / batch_no_val, 4)

            if best_valid_loss > val_epoch_loss:
                best_valid_loss = val_epoch_loss
                print("\n best loss is updated to", round(best_valid_loss, 4), "at epoch", epoch_no)
                if foldername != "":
                    torch.save(model.state_dict(), output_path)

        # 3. Ghi Log vào CSV ở cuối mỗi Epoch
        if log_file is not None:
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch_no, 
                    round(train_epoch_loss, 4), 
                    val_epoch_loss if isinstance(val_epoch_loss, str) else round(val_epoch_loss, 4), 
                    avg_snr, 
                    avg_rmse, 
                    avg_prd, 
                    avg_cos
                ])

    if foldername != "":
        torch.save(model.state_dict(), final_path)


def evaluate(model, test_loader, shots, device, foldername=""):
    ssd_total = 0
    mad_total = 0
    prd_total = 0
    cos_sim_total = 0
    snr_noise = 0
    snr_recon = 0
    snr_improvement = 0
    eval_points = 0

    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)

            if shots > 1:
                output = 0
                for i in range(shots):
                    output += model.denoising(noisy_batch)
                output /= shots
            else:
                output = model.denoising(noisy_batch)  # B,1,L
            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1)  # B,L,1
            out_numpy = output.cpu().detach().numpy()
            clean_numpy = clean_batch.cpu().detach().numpy()
            noisy_numpy = noisy_batch.cpu().detach().numpy()

            eval_points += len(output)
            ssd_total += np.sum(metrics.SSD(clean_numpy, out_numpy))
            mad_total += np.sum(metrics.MAD(clean_numpy, out_numpy))
            prd_total += np.sum(metrics.PRD(clean_numpy, out_numpy))
            cos_sim_total += np.sum(metrics.COS_SIM(clean_numpy, out_numpy))
            snr_noise += np.sum(metrics.SNR(clean_numpy, noisy_numpy))
            snr_recon += np.sum(metrics.SNR(clean_numpy, out_numpy))
            snr_improvement += np.sum(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))
            restored_sig.append(out_numpy)

            it.set_postfix(
                ordered_dict={
                    "ssd_total": ssd_total / eval_points,
                    "mad_total": mad_total / eval_points,
                    "prd_total": prd_total / eval_points,
                    "cos_sim_total": cos_sim_total / eval_points,
                    "snr_in": snr_noise / eval_points,
                    "snr_out": snr_recon / eval_points,
                    "snr_improve": snr_improvement / eval_points,
                },
                refresh=True,
            )

    restored_sig = np.concatenate(restored_sig)

    # np.save(foldername + '/denoised.npy', restored_sig)

    print("ssd_total: ", ssd_total / eval_points)
    print("mad_total: ", mad_total / eval_points, )
    print("prd_total: ", prd_total / eval_points, )
    print("cos_sim_total: ", cos_sim_total / eval_points, )
    print("snr_in: ", snr_noise / eval_points, )
    print("snr_out: ", snr_recon / eval_points, )
    print("snr_improve: ", snr_improvement / eval_points, )