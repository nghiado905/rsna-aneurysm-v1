# inference.py - RSNA 2025 - ONLY INFER CASES IN ground_truth_coordinates.csv
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from nnunetv2.dataset_conversion.kaggle_2025_rsna.official_data_to_nnunet import load_and_crop
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.predict_from_raw_data import convert_predicted_logits_to_segmentation_with_correct_shape


def _log(*args):
    print("[DEBUG]", *args)


def _normalize_id(p: Path) -> str:
    name = p.name if p.is_dir() else p.stem
    return name.replace("_0000", "").replace("_000", "")


def _classify_region(coord, shape):
    if None in coord or coord[0] is None:
        return "UNKNOWN"
    z, y, x = coord
    h, w = shape[1], shape[2]
    return f"{'RIGHT' if x >= w/2 else 'LEFT'}_{'ANT' if y < h/2 else 'POST'}_{'SUP' if z < shape[0]/2 else 'INF'}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=Path, required=True, help="Folder chứa tất cả series (DICOM hoặc NIfTI)")
    parser.add_argument("-o", "--output-path", type=Path, required=True, help="File CSV submission")
    parser.add_argument("-m", "--model_folder", type=Path, required=True)
    parser.add_argument("-c", "--chk", type=str, required=True)
    parser.add_argument("--fold", type=str, default="('all',)")
    parser.add_argument("--step_size", type=float, default=0.5)
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--use_gaussian", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--heatmap-dir", type=Path, required=False)
    parser.add_argument("--ids-mapping", type=Path, required=False, help="ids_mapping.json (nếu có)")
    parser.add_argument("--cases-csv", type=Path, required=True, help="File ground_truth_coordinates.csv - BẮT BUỘC!")
    args = parser.parse_args()

    # Tạo thư mục output nếu chưa có
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # === 1. ĐỌC FILE ground_truth_coordinates.csv → CHỈ INFER NHỮNG CASE NÀY ===
    if not args.cases_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {args.cases_csv}")
    
    df_cases = pd.read_csv(args.cases_csv)
    # Lấy cả SeriesInstanceUID và short_id (iarsna_xxxx)
    series_uids = set(df_cases["SeriesInstanceUID"].astype(str).str.strip())
    short_ids = set(df_cases["short_id"].astype(str).str.strip())
    # Tạo tập hợp tất cả các ID có thể xuất hiện trong folder
    target_ids = series_uids | short_ids
    target_ids = {x.replace("_0000", "").replace("_000", "") for x in target_ids}
    _log(f"Chỉ infer {len(target_ids)} cases từ ground_truth_coordinates.csv")

    # === 2. ID mapping (nếu có) ===
    id_map_rev = {}  # short_id → SeriesInstanceUID
    if args.ids_mapping and args.ids_mapping.exists():
        id_map = json.loads(args.ids_mapping.read_text())
        id_map_rev = {v: k for k, v in id_map.items()}
        _log(f"Loaded {len(id_map_rev)} ID mappings từ ids_mapping.json")

    # === 3. Khởi tạo predictor ===
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=args.use_gaussian,
        use_mirroring=not args.disable_tta,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    import ast
    folds = ast.literal_eval(args.fold)
    predictor.initialize_from_trained_model_folder(args.model_folder, folds, args.chk)

    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)

    # Label names
    labels_dict = predictor.dataset_json["labels"]
    bg_val = next((int(v) for k, v in labels_dict.items() if "background" in k.lower()), 0)
    class_names = [k for k, v in labels_dict.items() if int(v) != bg_val]
    _log(f"Classes: {class_names}")

    header = ["SeriesInstanceUID"] + class_names + ["Aneurysm Present", "peak_z", "peak_y", "peak_x", "peak_prob", "peak_label_idx"]
    results = []
    peak_logs = []

    if args.heatmap_dir:
        args.heatmap_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for series_path in args.input_dir.iterdir():
        sid_raw = _normalize_id(series_path)
        series_id_from_path = id_map_rev.get(sid_raw, sid_raw)  # nếu là iarsna_xxxx → đổi thành SeriesInstanceUID

        # === CHỈ XỬ LÝ NẾU CASE NẰM TRONG ground_truth_coordinates.csv ===
        if sid_raw not in target_ids and series_id_from_path not in target_ids:
            continue

        processed += 1
        series_id = id_map_rev.get(sid_raw, sid_raw)  # ưu tiên SeriesInstanceUID thật

        _log(f"\n{'='*80}")
        _log(f"[{processed}/{len(target_ids)}] Processing: {series_path.name} → ID: {series_id}")

        # Load image
        if series_path.is_file() and series_path.suffix.lower() in {".nii", ".nii.gz"}:
            from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
            img, props = SimpleITKIO().read_images([str(series_path)])
        else:
            img, props = load_and_crop(series_path)

        # Preprocess + Predict
        data, _, data_props = preprocessor.run_case_npy(
            np.array([img]), None, props,
            predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json
        )
        data_t = torch.from_numpy(data).to(device)

        with torch.no_grad():
            logits = predictor.predict_logits_from_preprocessed_data(data_t)
        probs = torch.sigmoid(logits)
        probs_np = probs.squeeze(0).cpu().numpy()

        # Max per class
        spatial_axes = tuple(range(1, probs_np.ndim))
        max_per_class = probs_np.max(axis=spatial_axes)
        fg_max = max_per_class[1:] if bg_val == 0 else max_per_class

        # Aneurysm Present
        fg_probs = probs_np[1:] if bg_val == 0 else probs_np
        prob_any = np.clip(fg_probs.sum(axis=0), 0.0, 1.0)
        aneurysm_present = float(prob_any.max())

        # Peak detection
        peak_coord = peak_prob = peak_label_idx = None
        if args.heatmap_dir:
            if "bbox_used_for_cropping" in data_props:
                _, prob_resampled = convert_predicted_logits_to_segmentation_with_correct_shape(
                    logits[0], predictor.plans_manager, predictor.configuration_manager,
                    predictor.label_manager, data_props, return_probabilities=True
                )
                prob_all = prob_resampled
            else:
                prob_all = probs_np

            fg_all = prob_all[1:] if bg_val == 0 else prob_all
            if fg_all.size and fg_all.max() > 0:
                loc = np.where(fg_all == fg_all.max())
                ch_rel = loc[0][0]
                spatial_loc = tuple(loc[i][0] for i in range(1, len(loc)))
                channel_idx = ch_rel + (1 if bg_val == 0 else 0)
                peak_label_idx = int(channel_idx)
                peak_prob = float(fg_all.max())

                if "bbox_used_for_cropping" in data_props and len(spatial_loc) == 3:
                    bbox = data_props["bbox_used_for_cropping"]
                    peak_coord = (
                        int(spatial_loc[0] + bbox[0][0]),
                        int(spatial_loc[1] + bbox[1][0]),
                        int(spatial_loc[2] + bbox[2][0]),
                    )
                else:
                    peak_coord = spatial_loc if len(spatial_loc) == 3 else (None, *spatial_loc)

                class_name = next((k for k, v in labels_dict.items() if int(v) == channel_idx), f"class_{channel_idx}")
                region = _classify_region(peak_coord, img.shape)
                _log(f"Peak → {class_name} | Prob: {peak_prob:.4f} | Coord: {peak_coord} | Region: {region}")

               # === VẼ HEATMAP – ĐÃ SỬA ĐÚNG 100% (điểm đỏ trúng tuyệt đối) ===
                mip = img.max(axis=0) if img.ndim == 3 else img
                mip = (mip - mip.min()) / (mip.ptp() + 1e-8)

                plt.figure(figsize=(8, 8))
                plt.imshow(mip, cmap="gray", origin="lower")  # ← DÒNG QUAN TRỌNG NHẤT!!!

                if peak_coord[1] is not None and peak_coord[2] is not None:
                    y, x = peak_coord[1], peak_coord[2]  # y = row, x = col → đúng với origin="lower"
                    plt.scatter(x, y, c="red", s=200, edgecolors="white", linewidth=3, zorder=5)

                    # Vòng tròn heatmap
                    yy, xx = np.ogrid[:mip.shape[0], :mip.shape[1]]
                    blob = np.exp(-((xx - x)**2 + (yy - y)**2) / (2*12**2))
                    blob /= blob.max() + 1e-8
                    plt.imshow(blob, cmap="hot", alpha=blob*0.75, origin="lower")  # ← cũng thêm origin="lower"

                plt.title(f"{series_id} | {class_name} | p={peak_prob:.4f} | {region}", 
                        fontsize=14, color="white", pad=20)
                plt.axis("off")
                plt.tight_layout()

                png_path = args.heatmap_dir / f"{series_id}_mip.png"
                plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="#111111")
                plt.close()

                label_dir = args.heatmap_dir / class_name
                label_dir.mkdir(exist_ok=True)
                shutil.move(png_path, label_dir / png_path.name)

                peak_logs.append({**locals(), "class_name": class_name, "region": region})

        # Ghi dòng kết quả
        row = [series_id] + fg_max.tolist() + [
            aneurysm_present,
            peak_coord[0] if peak_coord else None,
            peak_coord[1] if peak_coord and len(peak_coord) > 1 else None,
            peak_coord[2] if peak_coord and len(peak_coord) > 2 else None,
            peak_prob,
            peak_label_idx
        ]
        results.append(row)
        pd.DataFrame(results, columns=header).to_csv(args.output_path, index=False)

    _log(f"HOÀN TẤT! Đã infer {len(results)} cases → lưu tại {args.output_path}")
    if args.heatmap_dir and peak_logs:
        pd.DataFrame(peak_logs).to_csv(args.heatmap_dir / "peaks.csv", index=False)


if __name__ == "__main__":
    main()