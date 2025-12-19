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
# 1. HÀM VẼ HEATMAP BLOB (Style mới bạn muốn áp dụng)
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
    Vẽ MIP axial đơn giản (giống logic inference.py): max trục Z, không cộng bbox, origin='lower'.
    coords: (z, y, x)
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

# ==================================================================================
# 2. CODE CHÍNH
# ==================================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-dir", type=Path, required=True, help="Input directory")
    p.add_argument("-o", "--output-path", type=Path, required=True, help="Output CSV")
    p.add_argument("-m", "--model_folder", type=Path, required=True, help="Model folder")
    p.add_argument("-c", "--chk", type=str, required=True, help="Checkpoint")
    p.add_argument("--fold", type=ast.literal_eval, help="Fold tuple")
    p.add_argument("--step_size", type=float, default=0.5)
    p.add_argument("--disable_tta", action="store_true", default=False)
    p.add_argument("--use_gaussian", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--viz_threshold", type=float, default=0.2, help="Threshold to save MIP")
    
    # Cho phép truyền file whitelist khác nếu muốn, mặc định dùng file hardcode bên dưới
    p.add_argument("--whitelist-csv", type=Path, required=False)

    return p.parse_args()


def main():
    args = parse_args()

    # Tạo folder ảnh
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    mip_dir = args.output_path.parent / (args.output_path.stem + "_heatmap_mips")
    mip_dir.mkdir(exist_ok=True, parents=True)

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

    # =========================================================================
    # [QUAN TRỌNG] LOGIC LỌC WHITELIST TỪ FILE GROUND TRUTH CỦA BẠN
    # =========================================================================
    series_list = list(args.input_dir.iterdir())
    
    # 1. Xác định đường dẫn file CSV whitelist
    if args.whitelist_csv:
        target_csv = args.whitelist_csv
    else:
        # Đường dẫn cứng bạn yêu cầu
        target_csv = Path(r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\analysis_results\ground_truth_coordinates.csv")
    
    # 2. Thực hiện lọc
    if target_csv.exists():
        try:
            print(f"-> Đọc whitelist từ: {target_csv}")
            df_sub = pd.read_csv(target_csv)
            allowed_ids = set(df_sub["SeriesInstanceUID"].astype(str).str.strip())
            
            # Lọc danh sách file input
            original_len = len(series_list)
            series_list = [s for s in series_list if s.name in allowed_ids]
            print(f"-> Đã lọc: {original_len} -> {len(series_list)} ca cần xử lý.")
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc {target_csv}: {e}")
    else:
        print(f"⚠️ Không tìm thấy file whitelist: {target_csv}. Sẽ chạy TOÀN BỘ thư mục.")
    # =========================================================================

    # Resume logic
    processed_ids = set()
    if args.output_path.exists():
        try:
            processed_ids = set(pd.read_csv(args.output_path)["SeriesInstanceUID"].astype(str))
        except: pass
    else:
        pd.DataFrame(columns=labels).to_csv(args.output_path, index=False)

    print(f"🚀 Bắt đầu Inference... (Ảnh lưu tại {mip_dir})")

    for series_dir in tqdm(series_list):
        if series_dir.name in processed_ids:
            continue

        try:
            # 1. Load
            img, properties = load_and_crop(series_dir)
            # img = np.flip(img, 1) # Flip theo training
            
            # 2. Predict
            data, _, _ = preprocessor.run_case_npy(
                np.array([img]), None, properties,
                predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json,
            )
            logits = predictor.predict_logits_from_preprocessed_data(
                torch.from_numpy(data)
            ).cpu()
            probs = torch.sigmoid(logits)

            # 3. Max pooling & Save CSV
            max_per_c = torch.amax(probs, dim=(1, 2, 3)).to(dtype=torch.float32, device="cpu")
            res_row = [series_dir.name] + max_per_c.numpy().tolist()
            pd.DataFrame([res_row], columns=labels).to_csv(args.output_path, mode='a', header=False, index=False)

            # =================================================================
            # 4. ÁP DỤNG CÁCH VẼ HEATMAP BLOB (Chèn vào đây)
            # =================================================================
            fg_probs = max_per_c.numpy()[1:] # Bỏ bg
            best_prob = np.max(fg_probs)

            if best_prob > args.viz_threshold:
                # Tìm class và tọa độ
                best_cls_idx = np.argmax(fg_probs) + 1
                label_name = idx_to_label.get(best_cls_idx, "Unknown")
                
                prob_map = probs[best_cls_idx]
                peak_idx = torch.argmax(prob_map).item()
                z, y, x = np.unravel_index(peak_idx, prob_map.shape)
                
                # Tên file
                safe_name = label_name.replace(" ", "_").replace("/", "-")
                png_name = f"{series_dir.name}_{safe_name}_p{best_prob:.2f}.png"
                
                # Gọi hàm vẽ Heatmap mới
                save_heatmap_mip(img, (z, y, x), label_name, best_prob, mip_dir / png_name)
            # =================================================================

        except Exception as e:
            # print(f"Error {series_dir.name}: {e}")
            continue

    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()


# import argparse
# import ast
# from pathlib import Path
# import os
# import sys

# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# # --- 1. IMPORT NNUNET ---
# try:
#     from nnunetv2.dataset_conversion.kaggle_2025_rsna.official_data_to_nnunet import (
#         load_and_crop,
#     )
#     from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# except ImportError:
#     print("❌ Lỗi: Không tìm thấy thư viện nnUNetv2.")
#     sys.exit(1)

# # ==================================================================================
# # 2. HÀM VẼ HEATMAP CHỈ CHO AXIAL
# # ==================================================================================

# def window_hu(vol, level=400.0, width=700.0):
#     """Windowing chuẩn CTA: Level 400, Width 700."""
#     low = level - width / 2.0
#     high = level + width / 2.0
#     vol = np.clip(vol, low, high)
#     vol = (vol - low) / (high - low)
#     return vol

# def save_axial_heatmap(volume, coords, label_name, prob, output_path):
#     """
#     Chỉ vẽ MIP Axial (Z-projection) với Heatmap Blob.
#     """
#     try:
#         # 1. Windowing
#         vol_windowed = window_hu(volume, level=400.0, width=700.0)
        
#         z_peak, y_peak, x_peak = map(int, coords)
#         z_dim, y_dim, x_dim = vol_windowed.shape
#         slab = 15 # Độ dày lát cắt gộp

#         # --- TẠO ẢNH MIP AXIAL ---
#         # Trục Z là trục 0. Lấy max theo trục 0 để có ảnh (Y, X)
#         z_start = max(0, int(z_peak - slab))
#         z_end = min(z_dim, int(z_peak + slab))
#         if z_end <= z_start: z_end = z_start + 1
        
#         mip_axial = np.max(vol_windowed[z_start:z_end, :, :], axis=0)
        
#         # !!! QUAN TRỌNG: Lật trục Y (flipud) để Anterior (phía trước) nằm trên !!!
#         mip_axial = np.flipud(mip_axial)
        
#         # Tọa độ vẽ: X giữ nguyên, Y đảo ngược do flipud
#         pt_x = x_peak
#         pt_y = y_dim - 1 - y_peak

#         # --- VẼ HÌNH ---
#         fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        
#         # 1. Vẽ nền đen trắng
#         ax.imshow(mip_axial, cmap="gray", origin="lower", vmin=0, vmax=1)
        
#         # 2. Vẽ Heatmap & Điểm tâm
#         if pt_x is not None and pt_y is not None:
#             # Điểm tâm đỏ
#             ax.scatter(pt_x, pt_y, c="red", s=200, edgecolors="white", linewidth=3, zorder=5)
            
#             # Vòng tròn tỏa nhiệt (Gaussian Blob)
#             h, w = mip_axial.shape
#             yy, xx = np.ogrid[:h, :w]
#             dist_sq = (xx - pt_x)**2 + (yy - pt_y)**2
#             blob = np.exp(-dist_sq / (2 * 15**2)) # Sigma = 15 (to hơn chút cho dễ nhìn)
#             blob /= (blob.max() + 1e-8)
            
#             ax.imshow(blob, cmap="hot", alpha=blob * 0.6, origin="lower")

#         ax.set_title(f"{label_name}\nProb: {prob:.4f} | Z={z_peak}", color='#00FF00', fontsize=18, fontweight='bold', pad=15)
#         ax.axis('off')

#         plt.tight_layout()
#         plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
#         plt.close(fig)

#     except Exception as e:
#         print(f"⚠️ Lỗi vẽ Axial MIP: {e}")

# # ==================================================================================
# # 3. MAIN FUNCTION
# # ==================================================================================

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("-i", "--input-dir", type=Path, required=True, help="Input directory")
#     p.add_argument("-o", "--output-path", type=Path, required=True, help="Output CSV")
#     p.add_argument("-m", "--model_folder", type=Path, required=True, help="Model folder")
#     p.add_argument("-c", "--chk", type=str, required=True, help="Checkpoint name")
#     p.add_argument("--fold", type=ast.literal_eval, help="Fold tuple")
#     p.add_argument("--step_size", type=float, default=0.5)
#     p.add_argument("--disable_tta", action="store_true", default=False)
#     p.add_argument("--use_gaussian", action="store_true", default=False)
#     p.add_argument("--device", type=str, default="cuda")
#     p.add_argument("--viz_threshold", type=float, default=0.2, help="Threshold to save MIP")
    
#     # File whitelist mặc định
#     p.add_argument("--whitelist-csv", type=Path, 
#                    default=Path("output1/submission.csv"),
#                    help="Path to whitelist CSV")
#     return p.parse_args()

# def main():
#     args = parse_args()

#     # Tạo folder ảnh
#     args.output_path.parent.mkdir(parents=True, exist_ok=True)
#     mip_dir = args.output_path.parent / (args.output_path.stem + "_axial_mips")
#     mip_dir.mkdir(exist_ok=True, parents=True)

#     # --- LOAD MODEL ---
#     print(f"Loading model from {args.model_folder}...")
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     predictor = nnUNetPredictor(
#         tile_step_size=args.step_size,
#         use_gaussian=args.use_gaussian,
#         use_mirroring=not args.disable_tta,
#         device=device,
#         verbose=False, verbose_preprocessing=False, allow_tqdm=False,
#     )
#     predictor.initialize_from_trained_model_folder(
#         args.model_folder,
#         [i if i == "all" else int(i) for i in args.fold],
#         checkpoint_name=args.chk,
#     )
#     preprocessor = predictor.configuration_manager.preprocessor_class()

#     # --- XỬ LÝ LABELS ---
#     labels_dict = predictor.dataset_json["labels"]
#     idx_to_label = {int(v): k for k, v in labels_dict.items() if k != "background" and int(v) != 0}
#     ordered_indices = sorted(idx_to_label.keys()) 
#     label_names_ordered = [idx_to_label[i] for i in ordered_indices]
    
#     # Header CSV
#     csv_header = ["SeriesInstanceUID"] + label_names_ordered + ["Aneurysm Present", "peak_label", "coord_z", "coord_y", "coord_x"]

#     # --- LỌC WHITELIST ---
#     series_list = list(args.input_dir.iterdir())
    
#     if args.whitelist_csv.exists():
#         try:
#             print(f"-> Whitelist: {args.whitelist_csv}")
#             df_sub = pd.read_csv(args.whitelist_csv)
#             # Quan trọng: strip khoảng trắng ở ID
#             whitelist_uids = set(df_sub["SeriesInstanceUID"].astype(str).str.strip())
#             series_list = [s for s in series_list if s.name in whitelist_uids]
#             print(f"-> Đã lọc: Còn {len(series_list)} cases.")
#         except Exception as e:
#             print(f"⚠️ Lỗi whitelist: {e}")
#     else:
#         print("⚠️ Không thấy whitelist, chạy toàn bộ.")

#     # --- RESUME ---
#     processed_ids = set()
#     if args.output_path.exists():
#         try:
#             df_done = pd.read_csv(args.output_path)
#             if "SeriesInstanceUID" in df_done.columns:
#                 processed_ids = set(df_done["SeriesInstanceUID"].astype(str))
#             print(f"-> Resume: Đã xong {len(processed_ids)} cases.")
#         except: pass
#     else:
#         pd.DataFrame(columns=csv_header).to_csv(args.output_path, index=False)

#     print(f"🚀 Start Inference...")

#     # --- MAIN LOOP ---
#     for series_dir in tqdm(series_list):
#         uid = series_dir.name
        
#         if uid in processed_ids:
#             continue

#         try:
#             # 1. Load & Flip
#             img, properties = load_and_crop(series_dir)
#             img = np.flip(img, 1)

#             # 2. Predict
#             data, _, _ = preprocessor.run_case_npy(
#                 np.array([img]), None, properties,
#                 predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json,
#             )
#             logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
#             probs = torch.sigmoid(logits)

#             # 3. Post-process
#             max_probs_per_class = torch.amax(probs, dim=(1, 2, 3)).numpy()
#             fg_probs = max_probs_per_class[1:] 
#             aneurysm_present_prob = np.max(fg_probs)

#             # Find Peak
#             best_rel_idx = np.argmax(fg_probs)
#             best_abs_idx = best_rel_idx + 1
#             label_name = idx_to_label.get(best_abs_idx, "Unknown")
            
#             prob_map = probs[best_abs_idx]
#             peak_idx = torch.argmax(prob_map).item()
#             z, y, x = np.unravel_index(peak_idx, prob_map.shape)

#             # 4. Save CSV
#             row_data = [uid] + fg_probs.tolist() + [aneurysm_present_prob, label_name, z, y, x]
#             pd.DataFrame([row_data], columns=csv_header).to_csv(args.output_path, mode='a', header=False, index=False)

#             # 5. Save Axial MIP Only
#             if aneurysm_present_prob > args.viz_threshold:
#                 safe_name = label_name.replace(" ", "_").replace("/", "-")
#                 png_name = f"{uid}_{safe_name}_p{aneurysm_present_prob:.2f}.png"
                
#                 # Chỉ gọi hàm vẽ Axial
#                 save_axial_heatmap(img, (z, y, x), label_name, aneurysm_present_prob, mip_dir / png_name)

#         except Exception as e:
#             print(f"❌ Error {uid}: {e}")
#             continue

#     print(f"\n✅ Done! CSV: {args.output_path}")

# if __name__ == "__main__":
#     main()