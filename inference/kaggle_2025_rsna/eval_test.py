import argparse
import ast
import json
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk  # <--- Thêm thư viện đọc NIfTI

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ==================================================================================
# 1. HÀM HELPER & VISUALIZATION
# ==================================================================================

def window_hu(vol, level=400.0, width=700.0):
    """Windowing chuẩn cho mạch máu."""
    low = level - width / 2.0
    high = level + width / 2.0
    vol = np.clip(vol, low, high)
    vol = (vol - low) / (high - low)
    return vol

def save_heatmap_mip(volume, coords, label_name, prob, output_path):
    """
    Vẽ MIP axial đơn giản.
    """
    try:
        # Axial MIP (projection trục Z)
        mip = volume.max(axis=0) if volume.ndim == 3 else volume
        mip = (mip - mip.min()) / (mip.ptp() + 1e-8)

        z_peak, y_peak, x_peak = map(int, coords)
        h, w = mip.shape

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
        ax.imshow(mip, cmap="gray", origin="lower", vmin=0, vmax=1)

        # Blob heatmap quanh peak
        yy, xx = np.ogrid[:h, :w]
        dist_sq = (xx - x_peak) ** 2 + (yy - y_peak) ** 2
        blob = np.exp(-dist_sq / (2 * 12**2))
        blob /= blob.max() + 1e-8
        ax.imshow(blob, cmap="hot", alpha=blob * 0.75, origin="lower")

        ax.scatter(x_peak, y_peak, c="red", s=180, edgecolors="white", linewidth=2.5, zorder=5)
        ax.set_title(f"{label_name} | p={prob:.4f} | z={z_peak}", color="white", fontsize=14, pad=10)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#111111")
        plt.close(fig)

    except Exception as e:
        print(f"⚠️ Lỗi vẽ Heatmap: {e}")

def load_nifti_simple(nifti_path):
    """
    Đọc file .nii/.nii.gz và trả về array + properties cho nnU-Net.
    """
    # Đọc ảnh bằng SimpleITK
    img_itk = sitk.ReadImage(str(nifti_path))
    
    # SimpleITK trả về (X, Y, Z), convert sang numpy (Z, Y, X)
    img_npy = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    
    # Lấy spacing (X, Y, Z) và đảo ngược thành (Z, Y, X)
    spacing = np.array(img_itk.GetSpacing())[::-1]
    
    properties = {
        "sitk_stuff": None,
        "spacing": spacing,
        "shape_before_cropping": img_npy.shape,
        "bbox_used_for_cropping": None 
    }
    
    return img_npy, properties

def get_clean_id(filename):
    """
    Làm sạch tên file để khớp với value trong json.
    Ví dụ: 'iarsna_0001_0000.nii.gz' -> 'iarsna_0001'
    """
    name = filename
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
        
    # Bỏ suffix channel của nnU-Net (_0000)
    if name.endswith("_0000"):
        name = name[:-5]
        
    return name

# ==================================================================================
# 2. CODE CHÍNH
# ==================================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-dir", type=Path, required=True, help="Input directory (.nii files)")
    p.add_argument("-o", "--output-path", type=Path, required=True, help="Output CSV path")
    p.add_argument("-m", "--model_folder", type=Path, required=True, help="Model folder")
    p.add_argument("-c", "--chk", type=str, required=True, help="Checkpoint name")
    p.add_argument("--fold", type=ast.literal_eval, help="Fold tuple")
    p.add_argument("--mapping_json", type=Path, required=True, help="Path to ids_mapping.json")
    
    p.add_argument("--step_size", type=float, default=0.5)
    p.add_argument("--disable_tta", action="store_true", default=False)
    p.add_argument("--use_gaussian", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--viz_threshold", type=float, default=0.2, help="Threshold to save MIP")
    
    # Whitelist filters (nếu cần lọc, truyền path csv vào đây)
    p.add_argument("--whitelist-csv", type=Path, required=False)

    return p.parse_args()


def main():
    args = parse_args()

    # 1. Load Mapping JSON (SeriesUID <-> ShortID)
    print(f"Dataset mapping: Loading from {args.mapping_json}...")
    with open(args.mapping_json, 'r') as f:
        mapping_data = json.load(f)
    
    # Tạo dict ngược: { "iarsna_0001": "1.2.826..." }
    short_to_long_id = {v: k for k, v in mapping_data.items()}
    print(f"-> Loaded {len(short_to_long_id)} mapping entries.")

    # 2. Setup Output Folders
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    mip_dir = args.output_path.parent / (args.output_path.stem + "_heatmap_mips")
    mip_dir.mkdir(exist_ok=True, parents=True)

    # 3. Init Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=args.use_gaussian,
        use_mirroring=not args.disable_tta,
        device=device,
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        [i if i == "all" else int(i) for i in args.fold],
        checkpoint_name=args.chk,
    )

    preprocessor = predictor.configuration_manager.preprocessor_class()
    labels_dict = predictor.dataset_json["labels"]
    labels = ["SeriesInstanceUID"] + list(labels_dict.keys())[1:] + ["Aneurysm Present"]
    idx_to_label = {v: k for k, v in labels_dict.items() if v != 0}

    # 4. Get List of NIfTI Files
    all_files = sorted(list(args.input_dir.iterdir()))
    series_list = [f for f in all_files if f.name.endswith(('.nii', '.nii.gz'))]
    
    if len(series_list) == 0:
        print(f"⚠️ Không tìm thấy file .nii/.nii.gz nào trong {args.input_dir}")
        return

    # 5. Whitelist Logic (Optional)
    # Nếu có whitelist, chỉ chạy những UID nằm trong whitelist
    if args.whitelist_csv and args.whitelist_csv.exists():
        print(f"-> Filtering using whitelist: {args.whitelist_csv}")
        df_white = pd.read_csv(args.whitelist_csv)
        allowed_series_uids = set(df_white["SeriesInstanceUID"].astype(str).str.strip())
        
        filtered_list = []
        for f in series_list:
            short_id = get_clean_id(f.name)
            series_uid = short_to_long_id.get(short_id)
            if series_uid and series_uid in allowed_series_uids:
                filtered_list.append(f)
        
        print(f"-> Filtered: {len(series_list)} -> {len(filtered_list)} cases.")
        series_list = filtered_list

    # 6. Resume Logic
    processed_uids = set()
    if args.output_path.exists():
        try:
            df_done = pd.read_csv(args.output_path)
            processed_uids = set(df_done["SeriesInstanceUID"].astype(str))
        except: pass
    else:
        pd.DataFrame(columns=labels).to_csv(args.output_path, index=False)

    print(f"🚀 Bắt đầu Inference... (Ảnh lưu tại {mip_dir})")

    # =========================================================================
    # INFERENCE LOOP
    # =========================================================================
    for nifti_path in tqdm(series_list):
        # Lấy short ID từ tên file (vd: iarsna_0001)
        short_id = get_clean_id(nifti_path.name)
        
        # Mapping sang Long ID (SeriesInstanceUID)
        series_uid = short_to_long_id.get(short_id)
        
        if not series_uid:
            print(f"⚠️ Warning: Không tìm thấy mapping cho file {nifti_path.name}. Bỏ qua.")
            continue

        if series_uid in processed_uids:
            continue

        try:
            # 1. Load NIfTI (Thay vì load_and_crop)
            img, properties = load_nifti_simple(nifti_path)
            
            # 2. Add channel dimension: (Z, Y, X) -> (1, Z, Y, X)
            input_data = img[np.newaxis, ...]

            # 3. Predict
            # data: là ảnh sau khi resample/normalize (dùng để vẽ heatmap cho khớp)
            data, _, _ = preprocessor.run_case_npy(
                input_data, None, properties,
                predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json,
            )
            
            logits = predictor.predict_logits_from_preprocessed_data(
                torch.from_numpy(data)
            ).cpu()
            probs = torch.sigmoid(logits)

            # 4. Max pooling & Save CSV
            # Lưu ý: res_row dùng series_uid (Long ID)
            max_per_c = torch.amax(probs, dim=(1, 2, 3)).to(dtype=torch.float32, device="cpu")
            res_row = [series_uid] + max_per_c.numpy().tolist()
            pd.DataFrame([res_row], columns=labels).to_csv(args.output_path, mode='a', header=False, index=False)

            # 5. Visualization (Heatmap)
            fg_probs = max_per_c.numpy()[1:] # Bỏ bg
            best_prob = np.max(fg_probs) if len(fg_probs) > 0 else 0

            if best_prob > args.viz_threshold:
                best_cls_idx = np.argmax(fg_probs) + 1
                label_name = idx_to_label.get(best_cls_idx, "Unknown")
                
                prob_map = probs[best_cls_idx]
                peak_idx = torch.argmax(prob_map).item()
                z, y, x = np.unravel_index(peak_idx, prob_map.shape)
                
                # Tên file ảnh output dùng SeriesUID để dễ trace
                safe_name = label_name.replace(" ", "_").replace("/", "-")
                png_name = f"{series_uid}_{safe_name}_p{best_prob:.2f}.png"
                
                # Vẽ lên data[0] (ảnh đã preprocess) để khớp tọa độ heatmap
                save_heatmap_mip(data[0], (z, y, x), label_name, best_prob, mip_dir / png_name)

        except Exception as e:
            print(f"❌ Lỗi xử lý {nifti_path.name}: {e}")
            continue

    print(f"✅ Hoàn tất! Kết quả lưu tại: {args.output_path}")


if __name__ == "__main__":
    main()