

import argparse
import ast
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from nnunetv2.dataset_conversion.kaggle_2025_rsna.official_data_to_nnunet import (
    load_and_crop,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ==================================================================================
# 1. PHẦN BỔ SUNG: HÀM VẼ MIP NÂNG CAO (3 GÓC + LẬT TRỤC CHUẨN)
# ==================================================================================
def window_hu(vol, level=400.0, width=700.0):
    """Windowing chuẩn CTA: Level 400, Width 700."""
    low = level - width / 2.0
    high = level + width / 2.0
    vol = np.clip(vol, low, high)
    vol = (vol - low) / (high - low)
    return vol

def save_advanced_mip(volume, coords, label_name, prob, output_path):
    """
    Vẽ MIP 3 góc nhìn chuẩn Radiological (Axial, Coronal, Sagittal).
    Tự động lật trục để hiển thị đúng chiều giải phẫu (Đầu lên trên, Trước lên trên).
    """
    try:
        # 1. Windowing
        vol_windowed = window_hu(volume, level=400.0, width=700.0)
        
        z_peak, y_peak, x_peak = coords
        z_dim, y_dim, x_dim = vol_windowed.shape
        slab = 15 

        # --- CHUẨN BỊ 3 GÓC NHÌN ---
        
        # A. AXIAL (Nhìn từ trên xuống: Y dọc, X ngang)
        z_start, z_end = max(0, int(z_peak - slab)), min(z_dim, int(z_peak + slab))
        if z_end <= z_start: z_end = z_start + 1
        mip_axial = np.max(vol_windowed[z_start:z_end, :, :], axis=0)
        
        # !!! FIX TRỤC Y (AXIAL): Lật ngược để Anterior lên trên !!!
        mip_axial = np.flipud(mip_axial)
        # Tọa độ vẽ Axial: X giữ nguyên, Y đảo ngược
        pt_axial_x = x_peak
        pt_axial_y = y_dim - 1 - y_peak

        # B. CORONAL (Nhìn từ trước: Z dọc, X ngang)
        y_start, y_end = max(0, int(y_peak - slab)), min(y_dim, int(y_peak + slab))
        if y_end <= y_start: y_end = y_start + 1
        mip_coronal = np.max(vol_windowed[:, y_start:y_end, :], axis=1) 
        
        # !!! FIX TRỤC Z (CORONAL): Lật ngược để Head lên trên !!!
        mip_coronal = np.flipud(mip_coronal) 
        # Tọa độ vẽ Coronal: X giữ nguyên, Z đảo ngược
        pt_cor_x = x_peak
        pt_cor_y = z_dim - 1 - z_peak

        # C. SAGITTAL (Nhìn từ bên: Z dọc, Y ngang)
        x_start, x_end = max(0, int(x_peak - slab)), min(x_dim, int(x_peak + slab))
        if x_end <= x_start: x_end = x_start + 1
        mip_sagittal = np.max(vol_windowed[:, :, x_start:x_end], axis=2)
        
        # !!! FIX TRỤC Z (SAGITTAL): Lật ngược để Head lên trên !!!
        mip_sagittal = np.flipud(mip_sagittal)
        # Tọa độ vẽ Sagittal: Y giữ nguyên, Z đảo ngược
        pt_sag_x = y_peak
        pt_sag_y = z_dim - 1 - z_peak

        # --- HÀM VẼ HEATMAP BLOB ---
        def plot_blob_overlay(ax, img, point_x, point_y, title):
            # origin='lower' kết hợp với np.flipud sẽ hiển thị đúng chiều
            ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
            
            if point_x is not None and point_y is not None:
                # Vẽ điểm tâm (Đỏ đậm)
                ax.scatter(point_x, point_y, c="red", s=100, edgecolors="white", linewidth=1.5, zorder=5)
                
                # Vẽ Heatmap tỏa nhiệt (Gaussian Blob)
                h, w = img.shape
                yy, xx = np.ogrid[:h, :w]
                dist_sq = (xx - point_x)**2 + (yy - point_y)**2
                blob = np.exp(-dist_sq / (2 * 12**2))
                blob /= (blob.max() + 1e-8)
                ax.imshow(blob, cmap="hot", alpha=blob * 0.6, origin="lower")

            ax.set_title(title, color='white', fontsize=12, fontweight='bold')
            ax.axis('off')

        # --- PLOT ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
        fig.suptitle(f"{label_name} | Prob: {prob:.4f}", color='#00FF00', fontsize=16, fontweight='bold', y=0.98)

        # Plot 1: Axial (Đã lật Y)
        plot_blob_overlay(axes[0], mip_axial, pt_axial_x, pt_axial_y, f"Axial (Z={z_peak})")
        
        # Plot 2: Coronal (Đã lật Z)
        plot_blob_overlay(axes[1], mip_coronal, pt_cor_x, pt_cor_y, f"Coronal (Y={y_peak})")
        
        # Plot 3: Sagittal (Đã lật Z)
        plot_blob_overlay(axes[2], mip_sagittal, pt_sag_x, pt_sag_y, f"Sagittal (X={x_peak})")

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='black')
        plt.close(fig)

    except Exception as e:
        print(f"⚠️ Lỗi vẽ MIP: {e}")

# ==================================================================================
# 2. CODE CHÍNH (LOGIC INFERENCE GIỮ NGUYÊN)
# ==================================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-dir", type=Path, required=True, help="Path to directory with all DICOM data")
    p.add_argument("-o", "--output-path", type=Path, required=True, help="Where to store the resulting csv")
    p.add_argument("-m", "--model_folder", type=Path, required=True, help="Path to model checkpoint")
    p.add_argument("-c", "--chk", type=str, required=True, help="Name of the checkpoint")
    p.add_argument("--fold", type=ast.literal_eval, help="tuple of fold identifiers")
    p.add_argument("--step_size", type=float, required=False, default=0.5, help="Step size for sliding window prediction")
    p.add_argument("--disable_tta", action="store_true", required=False, default=False, help="Disable test time data augmentation")
    p.add_argument("--use_gaussian", action="store_true", required=False, default=False, help="Apply gaussian weighting")
    p.add_argument("--device", type=str, default="cuda", required=False, help="Device (cuda/cpu)")
    # Thêm tham số ngưỡng vẽ
    p.add_argument("--viz_threshold", type=float, default=0.2, help="Threshold to save MIP")

    return p.parse_args()


def main():
    args = parse_args()

    # --- SETUP FOLDER ẢNH MIP ---
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    mip_dir = args.output_path.parent / (args.output_path.stem + "_mips")
    mip_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=args.use_gaussian,
        use_mirroring=not args.disable_tta,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        [i if i == "all" else int(i) for i in args.fold],
        checkpoint_name=args.chk,
    )

    preprocessor = predictor.configuration_manager.preprocessor_class()

    # Label setup
    labels_dict = predictor.dataset_json["labels"]
    labels = ["SeriesInstanceUID"] + list(labels_dict.keys())[1:] + ["Aneurysm Present"]
    idx_to_label = {v: k for k, v in labels_dict.items() if v != 0} # Map cho vẽ ảnh

    # === 1. LOGIC WHITELIST (THEO YÊU CẦU) ===
    series_list = list(args.input_dir.iterdir())
    submission_path = Path(r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\analysis_results\ground_truth_coordinates.csv")
    if submission_path.exists():
        try:
            print(f"-> Đọc whitelist từ: {submission_path}")
            df_sub = pd.read_csv(submission_path)
            allowed_ids = set(df_sub["SeriesInstanceUID"].astype(str).str.strip())
            # Lọc list
            original_len = len(series_list)
            series_list = [s for s in series_list if s.name in allowed_ids]
            print(f"-> Đã lọc: {original_len} -> {len(series_list)} ca cần xử lý.")
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc {submission_path}: {e}")
            
    # === 2. LOGIC RESUME & INIT FILE ===
    processed_ids = set()
    if args.output_path.exists():
        try:
            df_done = pd.read_csv(args.output_path)
            if "SeriesInstanceUID" in df_done.columns:
                processed_ids = set(df_done["SeriesInstanceUID"].astype(str))
            print(f"-> Đã có {len(processed_ids)} ca xong. Sẽ bỏ qua.")
        except: pass
    else:
        # Ghi header lần đầu
        pd.DataFrame(columns=labels).to_csv(args.output_path, index=False)

    print(f"🚀 Bắt đầu Inference...")

    # === 3. VÒNG LẶP CHÍNH (LOGIC CŨ + BỔ SUNG VẼ) ===
    for series_dir in tqdm(series_list):
        if series_dir.name in processed_ids:
            continue

        try:
            # -------- START: LOGIC INFERENCE CHUẨN (KHÔNG ĐỔI) --------
            img, properties = load_and_crop(series_dir)
            img = np.flip(img, 1) # Flip chuẩn
            
            data, _, _ = preprocessor.run_case_npy(
                np.array([img]),
                None,
                properties,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.dataset_json,
            )
            logits = predictor.predict_logits_from_preprocessed_data(
                torch.from_numpy(data)
            ).cpu()
            probs = torch.sigmoid(logits)

            max_per_c = torch.amax(probs, dim=(1, 2, 3)).to(
                dtype=torch.float32, device="cpu"
            )
            # -------- END: LOGIC INFERENCE CHUẨN --------

            # === 4. LƯU CSV NGAY LẬP TỨC (APPEND MODE) ===
            res_row = [series_dir.name] + max_per_c.numpy().tolist()
            pd.DataFrame([res_row], columns=labels).to_csv(args.output_path, mode='a', header=False, index=False)

            # === 5. TÍNH TOÁN & VẼ ẢNH MIP (BỔ SUNG) ===
            # Phần này chỉ chạy để vẽ ảnh, không ảnh hưởng kết quả CSV
            fg_probs = max_per_c.numpy()[1:] # Bỏ background
            best_prob = np.max(fg_probs)

            if best_prob > args.viz_threshold:
                # Tìm tọa độ điểm max để vẽ
                best_cls_idx = np.argmax(fg_probs) + 1 # +1 do bỏ bg
                best_label_name = idx_to_label.get(best_cls_idx, "Unknown")
                
                # Lấy bản đồ xác suất của class đó
                prob_map = probs[best_cls_idx]
                peak_idx_flat = torch.argmax(prob_map).item()
                z, y, x = np.unravel_index(peak_idx_flat, prob_map.shape)
                
                # Vẽ ảnh MIP 3 góc xịn sò
                safe_name = best_label_name.replace(" ", "_").replace("/", "-")
                png_name = f"{series_dir.name}_{safe_name}_p{best_prob:.2f}.png"
                
                # Gọi hàm vẽ (truyền img đã flip để khớp tọa độ)
                save_advanced_mip(img, (z, y, x), best_label_name, best_prob, mip_dir / png_name)

        except Exception as e:
            print(f"❌ Lỗi ca {series_dir.name}: {e}")
            continue

    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()