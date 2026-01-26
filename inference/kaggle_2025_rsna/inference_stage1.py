import argparse
import ast
import subprocess
import tempfile
import sys
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

"""
Stage 1: DICOM/.nii -> Segment CoW -> Overlay -> Lưu NIfTI overlay
Usage:
python inference_stage1.py \
  -i /path/to/series_or_nii \
  -o /path/to/overlay_out_dir \
  --seg-model-root /path/to/nnUNet_results_seg \
  --seg-dataset Dataset113_CTMulSegWholeData \
  --seg-chk checkpoint_final.pth \
  --seg-fold 4 \
  --overlay-boost 200 \
  --temp-dir /tmp
"""

# ---------------- Defaults (có thể override qua CLI) ---------------- #
DEFAULT_SEG_MODEL_ROOT = r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\TopCoWSubmissions\nnUNet\model"
DEFAULT_SEG_DATASET = "Dataset113_CTMulSegWholeData"
DEFAULT_SEG_CHK = "checkpoint_final.pth"
DEFAULT_SEG_FOLD = "4"
DEFAULT_OVERLAY_BOOST = 200
DEFAULT_TEMP_DIR = Path(r"E:\temp")

# -------------------------------------------------------------------- #

def convert_dicom_to_nifti(dicom_dir: Path, out_path: Path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise ValueError(f"Không thấy DICOM trong {dicom_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(out_path))
    return image


def run_vessel_segmentation(input_nii: Path, output_dir: Path,
                            seg_model_root: str, seg_dataset: str,
                            seg_chk: str, seg_fold: str, seg_device: str = "cuda"):
    case_id = input_nii.name.replace(".nii.gz", "").replace(".nii", "")
    temp_input_dir = input_nii.parent / "nnunet_seg_input"
    temp_input_dir.mkdir(exist_ok=True)
    target_input_name = f"{case_id}_0000.nii.gz"
    shutil.copy(input_nii, temp_input_dir / target_input_name)

    env = os.environ.copy()
    env["nnUNet_results"] = seg_model_root
    env["nnUNet_raw"] = str(input_nii.parent)
    env["nnUNet_preprocessed"] = str(input_nii.parent)

    cmd = [
        "nnUNetv2_predict",
        "-d", seg_dataset,
        "-i", str(temp_input_dir),
        "-o", str(output_dir),
        "-f", seg_fold,
        "-tr", "nnUNetTrainer",
        "-c", "3d_fullres",
        "-p", "nnUNetPlans",
        "-chk", seg_chk,
        "--disable_tta",
        "-device", seg_device,
    ]
    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # In case of failure, print stderr for debugging then re-raise
        err = e.stderr.decode(errors="ignore") if e.stderr else ""
        print(f"\n❌ nnUNetv2_predict failed for {input_nii.name}:\n{err}")
        raise
    expected_mask = output_dir / f"{case_id}.nii.gz"
    shutil.rmtree(temp_input_dir, ignore_errors=True)
    if not expected_mask.exists():
        raise FileNotFoundError(f"Không thấy mask: {expected_mask}")
    return expected_mask


def apply_overlay(img_nii: Path, mask_nii: Path, boost: int):
    img_itk = sitk.ReadImage(str(img_nii))
    img = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    spacing = np.array(img_itk.GetSpacing())[::-1]

    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_nii)))
    if mask.shape == img.shape and (mask > 0).any():
        img[mask > 0] = np.clip(img[mask > 0] + boost, -1024, 3000)
    return img, spacing, img_itk.GetOrigin(), img_itk.GetDirection()


def save_overlay(arr, spacing, origin, direction, out_path: Path):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(tuple(spacing[::-1]))
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, str(out_path))


def prepare_targets(input_path: Path):
    if input_path.is_file():
        return [input_path]
    if list(input_path.glob("*.dcm")):
        return [input_path]
    return sorted([d for d in input_path.iterdir() if d.is_dir()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output-dir", type=Path, required=True, help="Thư mục lưu overlay NIfTI")
    ap.add_argument("--seg-model-root", type=Path, default=Path(DEFAULT_SEG_MODEL_ROOT))
    ap.add_argument("--seg-dataset", type=str, default=DEFAULT_SEG_DATASET)
    ap.add_argument("--seg-chk", type=str, default=DEFAULT_SEG_CHK)
    ap.add_argument("--seg-fold", type=str, default=DEFAULT_SEG_FOLD)
    ap.add_argument("--seg-device", type=str, default="cuda", help="Thiết bị chạy nnUNetv2_predict (cuda/cpu)")
    ap.add_argument("--overlay-boost", type=int, default=DEFAULT_OVERLAY_BOOST)
    ap.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    args = ap.parse_args()

    targets = prepare_targets(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(targets)} cases. Saving overlay to {args.output_dir}")
    with tempfile.TemporaryDirectory(dir=str(args.temp_dir)) as tmp_root:
        tmp_root = Path(tmp_root)
        for t in tqdm(targets):
            uid = t.name.replace(".nii.gz", "").replace(".nii", "")
            case_tmp = tmp_root / uid
            case_tmp.mkdir(exist_ok=True)
            raw_nii = case_tmp / f"{uid}.nii.gz"
            mask_dir = case_tmp / "masks"; mask_dir.mkdir(exist_ok=True)

            if t.is_dir():
                convert_dicom_to_nifti(t, raw_nii)
            else:
                shutil.copy(t, raw_nii)

            mask_nii = run_vessel_segmentation(raw_nii, mask_dir,
                                               str(args.seg_model_root), args.seg_dataset,
                                               args.seg_chk, args.seg_fold, args.seg_device)
            overlay, spacing, origin, direction = apply_overlay(raw_nii, mask_nii, args.overlay_boost)
            out_path = args.output_dir / f"{uid}.nii.gz"
            save_overlay(overlay, spacing, origin, direction, out_path)
    print("Done stage1.")


if __name__ == "__main__":
    main()
