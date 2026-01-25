import argparse
import ast
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ---------------------------- helpers ---------------------------- #

NIFTI_EXTS = {".nii", ".nii.gz"}


def window_hu(vol, level=400.0, width=700.0):
    low = level - width / 2.0
    high = level + width / 2.0
    vol = np.clip(vol, low, high)
    vol = (vol - low) / (high - low)
    return vol


def save_heatmap_mip(volume, coords, label_name, prob, output_path):
    try:
        mip = volume.max(axis=0) if volume.ndim == 3 else volume
        mip = (mip - mip.min()) / (mip.ptp() + 1e-8)

        z_peak, y_peak, x_peak = map(int, coords)
        h, w = mip.shape

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
        ax.imshow(mip, cmap="gray", origin="lower", vmin=0, vmax=1)

        yy, xx = np.ogrid[:h, :w]
        dist_sq = (xx - x_peak) ** 2 + (yy - y_peak) ** 2
        blob = np.exp(-dist_sq / (2 * 12 ** 2))
        blob /= blob.max() + 1e-8
        ax.imshow(blob, cmap="hot", alpha=blob * 0.75, origin="lower")

        ax.scatter(x_peak, y_peak, c="red", s=180, edgecolors="white", linewidth=2.5, zorder=5)
        ax.set_title(f"{label_name} | p={prob:.4f} | z={z_peak}", color="white", fontsize=14, pad=10)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#111111")
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Lỗi vẽ heatmap: {e}")


def load_nifti_simple(nifti_path: Path):
    img_itk = sitk.ReadImage(str(nifti_path))
    img_npy = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # (Z, Y, X)
    spacing = np.array(img_itk.GetSpacing())[::-1]  # to (Z, Y, X)
    props = {
        "sitk_stuff": None,
        "spacing": spacing,
        "shape_before_cropping": img_npy.shape,
        "bbox_used_for_cropping": None,
    }
    return img_npy, props


def uid_of(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    # drop nnU-Net channel suffix
    for i in range(10):
        suf = f"_000{i}"
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name


# ---------------------------- main ---------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-dir", type=Path, required=True, help="Input directory containing .nii/.nii.gz files")
    p.add_argument("-o", "--output-path", type=Path, required=True, help="Output CSV file")
    p.add_argument("-m", "--model_folder", type=Path, required=True, help="Model folder")
    p.add_argument("-c", "--chk", type=str, required=True, help="Checkpoint name")
    p.add_argument("--fold", type=ast.literal_eval, help="Fold tuple, e.g. \"('all',)\" or \"(0,1,2,3,4)\"")
    p.add_argument("--step_size", type=float, default=0.5)
    p.add_argument("--disable_tta", action="store_true", default=False)
    p.add_argument("--use_gaussian", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--viz_threshold", type=float, default=0.2, help="Threshold to save MIP")
    p.add_argument("--whitelist-csv", type=Path, required=False)
    p.add_argument("--mapping-json", type=Path, default=Path(r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\ids_mapping.json"),
                   help="JSON map SeriesInstanceUID -> short_id; sẽ đảo chiều để ghi SeriesInstanceUID chuẩn")
    p.add_argument("--num-workers", type=int, default=2, help="Threads for loading NIfTI in parallel (inference remains serial)")
    return p.parse_args()


def main():
    args = parse_args()

    # prepare output dirs
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    mip_dir = args.output_path.parent / (args.output_path.stem + "_heatmap_mips")
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
    labels_dict = predictor.dataset_json["labels"]
    labels = ["SeriesInstanceUID"] + list(labels_dict.keys())[1:] + ["Aneurysm Present"]
    idx_to_label = {v: k for k, v in labels_dict.items() if v != 0}

    # collect NIfTI files
    series_list = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in NIFTI_EXTS
    )
    if not series_list:
        print(f"⚠️ Không tìm thấy file .nii/.nii.gz trong {args.input_dir}")
        return

    # load mapping json (SeriesInstanceUID -> short), then invert
    short_to_full = {}
    if args.mapping_json and args.mapping_json.exists():
        try:
            import json
            with open(args.mapping_json, "r") as f:
                full_to_short = json.load(f)
            for full, short in full_to_short.items():
                short = str(short).strip()
                full = str(full).strip()
                short_to_full[short] = full
                short_to_full[short + "_0000"] = full
            print(f"-> Loaded mapping: {len(short_to_full)} entries")
        except Exception as e:
            print(f"⚠️ Lỗi đọc mapping json: {e}")

    # whitelist
    if args.whitelist_csv:
        target_csv = args.whitelist_csv
    else:
        target_csv = Path(r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\analysis_results\ground_truth_coordinates.csv")

    if target_csv.exists():
        try:
            print(f"-> Đọc whitelist từ: {target_csv}")
            df_sub = pd.read_csv(target_csv)
            allowed_ids = set(df_sub["SeriesInstanceUID"].astype(str).str.strip())
            if "short_id" in df_sub.columns:
                shorts = df_sub["short_id"].astype(str).str.strip()
                allowed_ids.update(shorts)
                for s, full in zip(shorts, df_sub["SeriesInstanceUID"]):
                    short_to_full.setdefault(s, str(full).strip())
                    short_to_full.setdefault(s + "_0000", str(full).strip())
            before = len(series_list)
            series_list = [p for p in series_list if uid_of(p) in allowed_ids]
            print(f"-> Đã lọc: {before} -> {len(series_list)} case cần xử lý.")
        except Exception as e:
            print(f"⚠️ Lỗi đọc whitelist: {e}")
    else:
        print(f"⚠️ Không tìm thấy whitelist: {target_csv}. Chạy toàn bộ thư mục.")

    # resume
    processed_ids = set()
    if args.output_path.exists():
        try:
            processed_ids = set(pd.read_csv(args.output_path)["SeriesInstanceUID"].astype(str))
        except Exception:
            pass
    else:
        pd.DataFrame(columns=labels).to_csv(args.output_path, index=False)

    print(f"🚀 Bắt đầu Inference NIfTI... (Ảnh lưu tại {mip_dir})")

    def load_case(nifti_path: Path):
        short_uid = uid_of(nifti_path)
        series_uid = short_to_full.get(short_uid, short_uid)
        if series_uid in processed_ids:
            return None
        img, props = load_nifti_simple(nifti_path)
        img = np.flip(img, 1)
        return short_uid, series_uid, img, props

    loader_iter = (
        ThreadPoolExecutor(max_workers=args.num_workers).map(load_case, series_list)
        if args.num_workers and args.num_workers > 0
        else (load_case(p) for p in series_list)
    )

    for item in tqdm(loader_iter, total=len(series_list)):
        if item is None:
            continue
        short_uid, series_uid, img, properties = item

        try:
            
            input_data = img[np.newaxis, ...]  # (1, Z, Y, X)

            data, _, _ = preprocessor.run_case_npy(
                input_data, None, properties,
                predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json,
            )

            logits = predictor.predict_logits_from_preprocessed_data(
                torch.from_numpy(data)
            ).cpu()
            probs = torch.sigmoid(logits)

            max_per_c = torch.amax(probs, dim=(1, 2, 3)).to(dtype=torch.float32, device="cpu")
            res_row = [series_uid] + max_per_c.numpy().tolist()
            pd.DataFrame([res_row], columns=labels).to_csv(args.output_path, mode="a", header=False, index=False)

            fg_probs = max_per_c.numpy()[1:]  # skip bg
            if len(fg_probs) == 0:
                continue

            best_prob = np.max(fg_probs)
            if best_prob > args.viz_threshold:
                best_cls_idx = np.argmax(fg_probs) + 1
                label_name = idx_to_label.get(best_cls_idx, "Unknown")
                prob_map = probs[best_cls_idx]
                peak_idx = torch.argmax(prob_map).item()
                z, y, x = np.unravel_index(peak_idx, prob_map.shape)
                safe_name = label_name.replace(" ", "_").replace("/", "-")
                png_name = f"{short_uid}_{safe_name}_p{best_prob:.2f}.png"
                save_heatmap_mip(data[0], (z, y, x), label_name, best_prob, mip_dir / png_name)

        except Exception as e:
            print(f"❌ Error {series_uid}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
