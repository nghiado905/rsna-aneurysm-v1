import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nnunetv2.dataset_conversion.kaggle_2025_rsna.official_data_to_nnunet_default import (
    load_and_crop,
)
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def extract_aneurysm_coordinates(
    probs: torch.Tensor, label_names, threshold: float = 0.5
):
    recs = []
    n_labels = min(len(label_names), int(probs.shape[0]))
    for c in range(n_labels):
        p = probs[c].detach().cpu().numpy()
        max_prob = float(np.max(p))
        max_idx = np.unravel_index(np.argmax(p), p.shape)
        z_peak, y_peak, x_peak = [int(v) for v in max_idx]

        fg = p >= threshold
        if np.any(fg):
            coords = np.argwhere(fg)
            zmin, ymin, xmin = coords.min(axis=0).tolist()
            zmax, ymax, xmax = coords.max(axis=0).tolist()
            weights = p[fg]
            wz, wy, wx = (coords * weights[:, None]).sum(axis=0) / np.sum(weights)
            cz, cy, cx = int(round(float(wz))), int(round(float(wy))), int(round(float(wx)))
            voxel_count = int(coords.shape[0])
        else:
            zmin = ymin = xmin = zmax = ymax = xmax = -1
            cz, cy, cx = z_peak, y_peak, x_peak
            voxel_count = 0

        recs.append(
            {
                "label": label_names[c],
                "max_prob": max_prob,
                "peak_z": z_peak,
                "peak_y": y_peak,
                "peak_x": x_peak,
                "coord_z": cz,
                "coord_y": cy,
                "coord_x": cx,
                "bbox_zmin": int(zmin),
                "bbox_zmax": int(zmax),
                "bbox_ymin": int(ymin),
                "bbox_ymax": int(ymax),
                "bbox_xmin": int(xmin),
                "bbox_xmax": int(xmax),
                "voxel_count_above_thr": voxel_count,
                "coord_threshold": float(threshold),
            }
        )
    return recs


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _get_source_axis_value(rec: dict, source_prefix: str, axis: str) -> float:
    direct_key = f"{source_prefix}_{axis}"
    coord_key = f"{source_prefix}_coord_{axis}"
    if direct_key in rec:
        return _to_float(rec.get(direct_key, 0))
    return _to_float(rec.get(coord_key, 0))


def extract_aneurysm_coordinates_with_prefix(
    probs: np.ndarray, label_names, threshold: float, prefix: str
):
    recs = []
    n_labels = min(len(label_names), int(probs.shape[0]))
    for c in range(n_labels):
        p = probs[c]
        max_prob = float(np.max(p))
        max_idx = np.unravel_index(np.argmax(p), p.shape)
        z_peak, y_peak, x_peak = [int(v) for v in max_idx]

        fg = p >= threshold
        if np.any(fg):
            coords = np.argwhere(fg)
            zmin, ymin, xmin = coords.min(axis=0).tolist()
            zmax, ymax, xmax = coords.max(axis=0).tolist()
            weights = p[fg]
            wz, wy, wx = (coords * weights[:, None]).sum(axis=0) / np.sum(weights)
            cz, cy, cx = int(round(float(wz))), int(round(float(wy))), int(round(float(wx)))
            voxel_count = int(coords.shape[0])
        else:
            zmin = ymin = xmin = zmax = ymax = xmax = -1
            cz, cy, cx = z_peak, y_peak, x_peak
            voxel_count = 0

        recs.append(
            {
                "label": label_names[c],
                "max_prob": max_prob,
                f"{prefix}_peak_z": z_peak,
                f"{prefix}_peak_y": y_peak,
                f"{prefix}_peak_x": x_peak,
                f"{prefix}_coord_z": cz,
                f"{prefix}_coord_y": cy,
                f"{prefix}_coord_x": cx,
                f"{prefix}_bbox_zmin": int(zmin),
                f"{prefix}_bbox_zmax": int(zmax),
                f"{prefix}_bbox_ymin": int(ymin),
                f"{prefix}_bbox_ymax": int(ymax),
                f"{prefix}_bbox_xmin": int(xmin),
                f"{prefix}_bbox_xmax": int(xmax),
                f"{prefix}_voxel_count_above_thr": voxel_count,
            }
        )
    return recs


def merge_record_dicts(base_recs, extra_recs):
    by_label = {r["label"]: dict(r) for r in base_recs}
    for r in extra_recs:
        lb = r["label"]
        if lb not in by_label:
            by_label[lb] = dict(r)
        else:
            by_label[lb].update(r)
    return [by_label[k] for k in by_label.keys()]


def _write_coord_transform(
    rec: dict,
    source_prefix: str,
    output_prefix: str,
    crop_shape_zyx: tuple,
    crop_bbox_zyx,
    spacing_zyx,
    origin_xyz,
    direction_flat_xyz,
):
    z_cf = int(round(_get_source_axis_value(rec, source_prefix, "z")))
    y_cf = int(round(_get_source_axis_value(rec, source_prefix, "y")))
    x_cf = int(round(_get_source_axis_value(rec, source_prefix, "x")))

    y_crop = int((crop_shape_zyx[1] - 1) - y_cf)
    z_crop = z_cf
    x_crop = x_cf

    z0, _, y0, _, x0, _ = [int(v) for v in crop_bbox_zyx]
    z_orig = int(z_crop + z0)
    y_orig = int(y_crop + y0)
    x_orig = int(x_crop + x0)

    spacing_xyz = np.array([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]], dtype=np.float64)
    idx_xyz = np.array([x_orig, y_orig, z_orig], dtype=np.float64)
    origin_xyz = np.array(origin_xyz, dtype=np.float64)
    direction = np.array(direction_flat_xyz, dtype=np.float64).reshape(3, 3)
    world_xyz = origin_xyz + direction.dot(idx_xyz * spacing_xyz)

    rec[f"{output_prefix}_z_crop"] = z_crop
    rec[f"{output_prefix}_y_crop"] = y_crop
    rec[f"{output_prefix}_x_crop"] = x_crop
    rec[f"{output_prefix}_z_crop_unflip"] = z_crop
    rec[f"{output_prefix}_y_crop_unflip"] = y_crop
    rec[f"{output_prefix}_x_crop_unflip"] = x_crop
    rec[f"{output_prefix}_z_orig"] = z_orig
    rec[f"{output_prefix}_y_orig"] = y_orig
    rec[f"{output_prefix}_x_orig"] = x_orig
    rec[f"{output_prefix}_world_x_mm"] = float(world_xyz[0])
    rec[f"{output_prefix}_world_y_mm"] = float(world_xyz[1])
    rec[f"{output_prefix}_world_z_mm"] = float(world_xyz[2])
    rec["crop_bbox_z0"] = int(z0)
    rec["crop_bbox_z1"] = int(crop_bbox_zyx[1])
    rec["crop_bbox_y0"] = int(y0)
    rec["crop_bbox_y1"] = int(crop_bbox_zyx[3])
    rec["crop_bbox_x0"] = int(x0)
    rec["crop_bbox_x1"] = int(crop_bbox_zyx[5])
    rec["crop_shape_z"] = int(crop_shape_zyx[0])
    rec["crop_shape_y"] = int(crop_shape_zyx[1])
    rec["crop_shape_x"] = int(crop_shape_zyx[2])
    return rec


def undo_to_original_voxel_and_mm(
    rec: dict,
    crop_shape_zyx: tuple,
    crop_bbox_zyx,
    spacing_zyx,
    origin_xyz,
    direction_flat_xyz,
):
    rec = _write_coord_transform(
        rec,
        source_prefix="cropflip",
        output_prefix="coord",
        crop_shape_zyx=crop_shape_zyx,
        crop_bbox_zyx=crop_bbox_zyx,
        spacing_zyx=spacing_zyx,
        origin_xyz=origin_xyz,
        direction_flat_xyz=direction_flat_xyz,
    )
    rec = _write_coord_transform(
        rec,
        source_prefix="cropflip_peak",
        output_prefix="peak",
        crop_shape_zyx=crop_shape_zyx,
        crop_bbox_zyx=crop_bbox_zyx,
        spacing_zyx=spacing_zyx,
        origin_xyz=origin_xyz,
        direction_flat_xyz=direction_flat_xyz,
    )
    return rec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-dir", type=Path, required=True)
    p.add_argument("-o", "--output-path", type=Path, required=True)
    p.add_argument("-m", "--model_folder", type=Path, required=True)
    p.add_argument("-c", "--chk", type=str, required=True)
    p.add_argument("--fold", type=ast.literal_eval)
    p.add_argument("--step_size", type=float, required=False, default=0.5)
    p.add_argument("--disable_tta", action="store_true", required=False, default=False)
    p.add_argument("--use_gaussian", action="store_true", required=False, default=False)
    p.add_argument("--device", type=str, default="cuda", required=False)
    p.add_argument("--coords-output", type=Path, default=None)
    p.add_argument("--coord-threshold", type=float, default=0.5)
    p.add_argument("--ids-mapping-json", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_path.suffix.lower() != ".csv":
        args.output_path.mkdir(parents=True, exist_ok=True)
        args.output_path = args.output_path / "submission.csv"
        print(f"Output path is a directory; writing CSV to: {args.output_path}")
    if args.coords_output is None:
        args.coords_output = args.output_path.with_name(args.output_path.stem + "_coords.csv")

    series_to_case = {}
    if args.ids_mapping_json is not None and args.ids_mapping_json.exists():
        with open(args.ids_mapping_json, "r", encoding="utf-8") as f:
            series_to_case = {str(k): str(v) for k, v in json.load(f).items()}
        print(f"Loaded ids mapping: {args.ids_mapping_json} ({len(series_to_case)} entries)")
    elif args.ids_mapping_json is not None:
        print(f"[WARN] ids mapping not found: {args.ids_mapping_json}")

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
    labels = (
        ["SeriesInstanceUID"]
        + list(predictor.dataset_json["labels"].keys())[1:]
        + ["Aneurysm Present"]
    )
    aneurysm_label_names = list(predictor.dataset_json["labels"].keys())[1:-1]

    existing_ids = set()
    res = []
    coord_rows = []
    if args.output_path.exists():
        try:
            df_prev = pd.read_csv(args.output_path)
            if "SeriesInstanceUID" in df_prev.columns:
                existing_ids = set(df_prev["SeriesInstanceUID"].astype(str))
                res = df_prev.values.tolist()
                print(f"Skipping {len(existing_ids)} already processed cases from {args.output_path}")
        except Exception as e:
            print(f"Could not load existing results ({e}); starting fresh.")
    if args.coords_output.exists():
        try:
            coord_prev = pd.read_csv(args.coords_output)
            if len(coord_prev) > 0 and "SeriesInstanceUID" in coord_prev.columns:
                coord_rows = coord_prev.to_dict("records")
                print(f"Loaded existing coordinate rows: {len(coord_rows)} from {args.coords_output}")
        except Exception as e:
            print(f"Could not load existing coords ({e}); starting fresh.")

    series_dirs = [p for p in args.input_dir.iterdir() if p.is_dir()]
    for series_dir in tqdm(series_dirs):
        if series_dir.name in existing_ids:
            continue

        img, properties = load_and_crop(series_dir)
        # Training data was saved after flipping crop along Y axis.
        img = np.flip(img, 1).astype(np.float32, copy=False)
        data, _, _ = preprocessor.run_case_npy(
            np.array([img], dtype=np.float32),
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
        res.append([series_dir.name] + max_per_c.numpy().tolist())

        coord_recs = extract_aneurysm_coordinates(probs, aneurysm_label_names, threshold=args.coord_threshold)
        _, probs_cropflip = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.label_manager,
            properties,
            return_probabilities=True,
        )
        coord_recs_cropflip = extract_aneurysm_coordinates_with_prefix(
            probs_cropflip,
            aneurysm_label_names,
            threshold=args.coord_threshold,
            prefix="cropflip",
        )
        coord_recs = merge_record_dicts(coord_recs, coord_recs_cropflip)

        crop_bbox = properties.get("crop_bbox_zyx", [0, img.shape[0], 0, img.shape[1], 0, img.shape[2]])
        spacing_zyx = properties.get("spacing", [1.0, 1.0, 1.0])
        origin_xyz = properties.get("origin", [0.0, 0.0, 0.0])
        direction_flat_xyz = properties.get("direction", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        case_id = series_to_case.get(series_dir.name, "")
        for r in coord_recs:
            r = undo_to_original_voxel_and_mm(
                r,
                crop_shape_zyx=tuple(img.shape),
                crop_bbox_zyx=crop_bbox,
                spacing_zyx=spacing_zyx,
                origin_xyz=origin_xyz,
                direction_flat_xyz=direction_flat_xyz,
            )
            r["SeriesInstanceUID"] = series_dir.name
            r["case_id"] = case_id
            coord_rows.append(r)

        pd.DataFrame(res, columns=labels).to_csv(args.output_path, index=False)
        pd.DataFrame(coord_rows).to_csv(args.coords_output, index=False)

    print(f"Results saved to {args.output_path}")
    print(f"Coordinate table saved to {args.coords_output}")


if __name__ == "__main__":
    main()
