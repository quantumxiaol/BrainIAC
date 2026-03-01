import os
import json
import yaml
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import csv
import numpy as np
from scipy import ndimage as ndi

# Import from segmentation model
from dataset_segmentation import get_segmentation_dataloader
from segmentation_model import ViTUNETRSegmentationModel

def load_model(config, checkpoint_path):
    """
    Loads a ViTUNETRSegmentationModel and populates it with weights from a
    PyTorch Lightning checkpoint.
    """
   #load the model
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=config['pretrain']['simclr_checkpoint_path'],
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )

    # 2load the state_dic
    state_dict = torch.load(checkpoint_path)['state_dict']
    
    # remove model. for lightning compatibiltiy 
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[len('model.'):]
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    return model.eval().cuda()

def get_test_dataloader(config, test_csv):
    """
    Creates a DataLoader for the test set.
    """
   
    test_ds = get_segmentation_dataloader(
        csv_file=test_csv,
        img_size=tuple(config['model']['img_size']),
        batch_size=1, 
        num_workers=1,
        is_train=False
    )
    # spinup dataloader
    return DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

def _parse_thresholds(threshold_str: str) -> list[float]:
    values = []
    for item in threshold_str.split(","):
        token = item.strip()
        if not token:
            continue
        t = float(token)
        if not (0.0 < t < 1.0):
            raise ValueError(f"Threshold must be in (0, 1), got: {t}")
        values.append(t)
    if not values:
        raise ValueError("No valid thresholds parsed from --sweep-thresholds.")
    return sorted(set(values))

def _extract_case_id(batch, fallback_index: int) -> str:
    meta_keys = ["image_meta_dict", "image_dwi_meta_dict", "image_adc_meta_dict"]
    for meta_key in meta_keys:
        if meta_key in batch and "filename_or_obj" in batch[meta_key]:
            filename = batch[meta_key]["filename_or_obj"][0]
            return str(filename)
    return f"case_{fallback_index}"

def _postprocess_prediction(prob_map: np.ndarray, threshold: float, min_lesion_voxels: int) -> np.ndarray:
    pred_bin = (prob_map >= threshold).astype(np.uint8)
    if min_lesion_voxels <= 0:
        return pred_bin
    if pred_bin.max() == 0:
        return pred_bin

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    pred_cc, pred_n = ndi.label(pred_bin.astype(bool), structure=structure)
    if pred_n == 0:
        return pred_bin

    sizes = np.bincount(pred_cc.ravel())
    keep_mask = np.isin(pred_cc, np.where(sizes >= min_lesion_voxels)[0])
    keep_mask[pred_cc == 0] = False
    return keep_mask.astype(np.uint8)

def _collect_predictions(model, loader, config, desc="Evaluating"):
    cases = []
    for batch in tqdm(loader, desc=desc):
        image = batch["image"].cuda()
        label = batch["label"].cuda()

        pred_logits = sliding_window_inference(
            inputs=image,
            roi_size=tuple(config["model"]["img_size"]),
            sw_batch_size=config["training"]["sw_batch_size"],
            predictor=model,
            overlap=0.5,
        )
        pred_prob = torch.sigmoid(pred_logits)
        label_bin = (label > 0.5).to(torch.uint8)

        case_id = _extract_case_id(batch, fallback_index=len(cases))
        cases.append(
            {
                "case_id": case_id,
                "pred_prob": pred_prob[0, 0].detach().cpu().numpy().astype(np.float32),
                "label": label_bin[0, 0].detach().cpu().numpy().astype(np.uint8),
            }
        )
    return cases

def _compute_case_lesion_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, iou_threshold: float = 0.1) -> dict:
    pred_bin = pred_mask.astype(bool)
    gt_bin = gt_mask.astype(bool)

    voxel_pred = int(pred_bin.sum())
    voxel_gt = int(gt_bin.sum())
    avd_percent = abs(voxel_pred - voxel_gt) / max(voxel_gt, 1) * 100.0

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    pred_cc, pred_n = ndi.label(pred_bin, structure=structure)
    gt_cc, gt_n = ndi.label(gt_bin, structure=structure)

    if pred_n == 0 and gt_n == 0:
        return {
            "lesion_f1": 1.0,
            "lesion_precision": 1.0,
            "lesion_recall": 1.0,
            "lesion_tp": 0,
            "lesion_fp": 0,
            "lesion_fn": 0,
            "lesion_count_pred": 0,
            "lesion_count_gt": 0,
            "lesion_count_diff": 0,
            "avd_percent": 0.0,
        }

    pred_sizes = np.bincount(pred_cc.ravel())[1:]
    gt_sizes = np.bincount(gt_cc.ravel())[1:]

    candidate_matches = []
    for pred_idx in range(1, pred_n + 1):
        overlap = gt_cc[pred_cc == pred_idx]
        overlap = overlap[overlap > 0]
        if overlap.size == 0:
            continue
        gt_ids, inter_counts = np.unique(overlap, return_counts=True)
        pred_vol = int(pred_sizes[pred_idx - 1])
        for gt_idx, inter in zip(gt_ids.tolist(), inter_counts.tolist()):
            gt_vol = int(gt_sizes[gt_idx - 1])
            union = pred_vol + gt_vol - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou >= iou_threshold:
                candidate_matches.append((iou, pred_idx, gt_idx))

    candidate_matches.sort(key=lambda x: x[0], reverse=True)
    used_pred = set()
    used_gt = set()
    tp = 0
    for _, pred_idx, gt_idx in candidate_matches:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        tp += 1

    fp = pred_n - tp
    fn = gt_n - tp
    lesion_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    lesion_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    lesion_f1 = (
        2.0 * lesion_precision * lesion_recall / (lesion_precision + lesion_recall)
        if (lesion_precision + lesion_recall) > 0
        else 0.0
    )

    return {
        "lesion_f1": float(lesion_f1),
        "lesion_precision": float(lesion_precision),
        "lesion_recall": float(lesion_recall),
        "lesion_tp": int(tp),
        "lesion_fp": int(fp),
        "lesion_fn": int(fn),
        "lesion_count_pred": int(pred_n),
        "lesion_count_gt": int(gt_n),
        "lesion_count_diff": int(abs(pred_n - gt_n)),
        "avd_percent": float(avd_percent),
    }

def _safe_div(num: float, den: float, empty_value: float = 0.0) -> float:
    return float(num / den) if den > 0 else float(empty_value)

def evaluate_cases(cases, threshold: float, min_lesion_voxels: int):
    """
    Computes voxel/lesion metrics from cached prediction probabilities and labels.
    """
    per_case_dice = {}
    per_case_lesion_f1 = {}
    per_case_avd_percent = {}
    per_case_lesion_count_diff = {}
    per_case_lesion_count_pred = {}
    per_case_lesion_count_gt = {}

    voxel_tp = 0
    voxel_fp = 0
    voxel_fn = 0
    lesion_tp = 0
    lesion_fp = 0
    lesion_fn = 0
    avd_values = []
    lesion_count_diffs = []
    per_case_dice_values = []

    for case in cases:
        image_id = case["case_id"]
        label_np = case["label"]
        pred_np = _postprocess_prediction(
            prob_map=case["pred_prob"],
            threshold=threshold,
            min_lesion_voxels=min_lesion_voxels,
        )

        tp = int(np.logical_and(pred_np == 1, label_np == 1).sum())
        fp = int(np.logical_and(pred_np == 1, label_np == 0).sum())
        fn = int(np.logical_and(pred_np == 0, label_np == 1).sum())
        dice_den = 2 * tp + fp + fn
        case_dice = _safe_div(2 * tp, dice_den, empty_value=1.0)
        per_case_dice_values.append(case_dice)
        per_case_dice[image_id] = case_dice

        lesion_case = _compute_case_lesion_metrics(pred_np, label_np, iou_threshold=0.1)
        per_case_lesion_f1[image_id] = lesion_case["lesion_f1"]
        per_case_avd_percent[image_id] = lesion_case["avd_percent"]
        per_case_lesion_count_diff[image_id] = lesion_case["lesion_count_diff"]
        per_case_lesion_count_pred[image_id] = lesion_case["lesion_count_pred"]
        per_case_lesion_count_gt[image_id] = lesion_case["lesion_count_gt"]

        voxel_tp += tp
        voxel_fp += fp
        voxel_fn += fn
        lesion_tp += int(lesion_case["lesion_tp"])
        lesion_fp += int(lesion_case["lesion_fp"])
        lesion_fn += int(lesion_case["lesion_fn"])
        avd_values.append(float(lesion_case["avd_percent"]))
        lesion_count_diffs.append(float(lesion_case["lesion_count_diff"]))

    # Aggregate metrics
    dice = _safe_div(2 * voxel_tp, (2 * voxel_tp + voxel_fp + voxel_fn), empty_value=1.0)
    jaccard = _safe_div(voxel_tp, (voxel_tp + voxel_fp + voxel_fn), empty_value=1.0)
    precision = _safe_div(voxel_tp, (voxel_tp + voxel_fp))
    recall = _safe_div(voxel_tp, (voxel_tp + voxel_fn))

    lesion_precision = _safe_div(lesion_tp, (lesion_tp + lesion_fp))
    lesion_recall = _safe_div(lesion_tp, (lesion_tp + lesion_fn))
    lesion_f1 = (
        2.0 * lesion_precision * lesion_recall / (lesion_precision + lesion_recall)
        if (lesion_precision + lesion_recall) > 0
        else 0.0
    )
    avd_percent = float(np.mean(avd_values)) if avd_values else 0.0
    lesion_count_diff = float(np.mean(lesion_count_diffs)) if lesion_count_diffs else 0.0
    mean_case_dice = float(np.mean(per_case_dice_values)) if per_case_dice_values else 0.0

    metrics = {
        "dice": float(dice),
        "mean_case_dice": float(mean_case_dice),
        "jaccard": float(jaccard),
        "precision": float(precision),
        "recall": float(recall),
        "lesion_f1": float(lesion_f1),
        "lesion_precision": float(lesion_precision),
        "lesion_recall": float(lesion_recall),
        "avd_percent": float(avd_percent),
        "lesion_count_diff": float(lesion_count_diff),
        "per_case_dice": per_case_dice,
        "per_case_lesion_f1": per_case_lesion_f1,
        "per_case_avd_percent": per_case_avd_percent,
        "per_case_lesion_count_diff": per_case_lesion_count_diff,
        "per_case_lesion_count_pred": per_case_lesion_count_pred,
        "per_case_lesion_count_gt": per_case_lesion_count_gt,
    }
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--config', type=str, required=False, default="config_finetune_segmentation.yml", help="Path to config YAML")
    parser.add_argument('--output_json', type=str, required=False, default="./inference/model_outputs/segmentation.json", help="Path to save combined metrics JSON")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to test CSV file")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to checkpoint file")
    parser.add_argument('--experiment_name', type=str, required=False, default="segmentation_task", help="Name for the experiment")
    parser.add_argument('--csv_output_dir', type=str, required=False, default="./inference/per_case_results", help="Directory to save per-case CSV files")
    parser.add_argument('--threshold', type=float, default=0.5, help="Probability threshold for binarizing prediction.")
    parser.add_argument('--min_lesion_voxels', type=int, default=0, help="Remove predicted connected components smaller than this size.")
    parser.add_argument('--sweep_thresholds', type=str, default="", help="Comma-separated thresholds, e.g. '0.3,0.4,0.5,0.6'.")
    parser.add_argument('--sweep_csv', type=str, default="", help="CSV for threshold sweep (recommended: validation CSV).")
    parser.add_argument('--sweep_metric', type=str, default="dice", choices=["dice", "lesion_f1"], help="Metric used to pick best threshold.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if not (0.0 < args.threshold < 1.0):
        raise ValueError(f"--threshold must be in (0,1), got: {args.threshold}")
    if args.min_lesion_voxels < 0:
        raise ValueError(f"--min_lesion_voxels must be >=0, got: {args.min_lesion_voxels}")

    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']['visible_device']

    # --- Checkpoints to evaluate ---
    checkpoints_to_evaluate = {
        args.experiment_name: args.checkpoint_path,
    }
    # -----------------------------------------

    if not checkpoints_to_evaluate or not args.checkpoint_path:
        print("No checkpoint path provided. Please specify --checkpoint_path")
        exit()
        
    
    output_json_dir = os.path.dirname(args.output_json)
    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(args.csv_output_dir, exist_ok=True)

    all_metrics = {}

    # store results 
    mean_dice_scores = {}

    for experiment_name, ckpt_path in checkpoints_to_evaluate.items():
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for '{experiment_name}'")
            continue

        print(f" Evaluating : '{experiment_name}' ")
        model = load_model(config, ckpt_path)

        selected_threshold = float(args.threshold)
        sweep_results = []
        if args.sweep_thresholds.strip():
            sweep_thresholds = _parse_thresholds(args.sweep_thresholds)
            sweep_csv = args.sweep_csv.strip() or config.get("data", {}).get("val_csv", "")
            if not sweep_csv:
                raise ValueError("--sweep_thresholds provided but no sweep CSV found. Set --sweep_csv or config.data.val_csv.")
            print(
                f"Sweeping thresholds on: {sweep_csv} | thresholds={sweep_thresholds} | "
                f"metric={args.sweep_metric} | min_lesion_voxels={args.min_lesion_voxels}"
            )
            sweep_loader = get_test_dataloader(config, sweep_csv)
            sweep_cases = _collect_predictions(model, sweep_loader, config, desc="Collecting sweep predictions")
            for threshold in sweep_thresholds:
                sweep_metrics = evaluate_cases(
                    sweep_cases,
                    threshold=threshold,
                    min_lesion_voxels=args.min_lesion_voxels,
                )
                score = float(sweep_metrics[args.sweep_metric])
                sweep_results.append({"threshold": threshold, "score": score})
                print(f"  threshold={threshold:.3f} {args.sweep_metric}={score:.4f}")
            best_item = max(sweep_results, key=lambda x: x["score"])
            selected_threshold = float(best_item["threshold"])
            print(f"Selected threshold={selected_threshold:.3f} by {args.sweep_metric}={best_item['score']:.4f}")

        test_loader = get_test_dataloader(config, args.test_csv)
        test_cases = _collect_predictions(model, test_loader, config, desc="Collecting test predictions")
        metrics = evaluate_cases(
            test_cases,
            threshold=selected_threshold,
            min_lesion_voxels=args.min_lesion_voxels,
        )
        metrics["postprocess"] = {
            "threshold": selected_threshold,
            "min_lesion_voxels": int(args.min_lesion_voxels),
        }
        if sweep_results:
            metrics["threshold_sweep"] = sweep_results
        
        all_metrics[experiment_name] = metrics
        summary = {k: v for k, v in metrics.items() if not k.startswith("per_case_") and k != "threshold_sweep"}
        print(f"Metrics summary for '{experiment_name}': {summary}")

        # --- Save per-case Dice to CSV ---
        csv_path = os.path.join(args.csv_output_dir, f"{experiment_name}_per_case_dice.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "case_id",
                    "dice",
                    "lesion_f1",
                    "avd_percent",
                    "lesion_count_diff",
                    "lesion_count_pred",
                    "lesion_count_gt",
                    "threshold",
                    "min_lesion_voxels",
                ]
            )
            for case_id, dice in metrics["per_case_dice"].items():
                writer.writerow(
                    [
                        case_id,
                        dice,
                        metrics["per_case_lesion_f1"].get(case_id),
                        metrics["per_case_avd_percent"].get(case_id),
                        metrics["per_case_lesion_count_diff"].get(case_id),
                        metrics["per_case_lesion_count_pred"].get(case_id),
                        metrics["per_case_lesion_count_gt"].get(case_id),
                        metrics["postprocess"]["threshold"],
                        metrics["postprocess"]["min_lesion_voxels"],
                    ]
                )
        
        
        # Print mean Dice 
        mean_dice = metrics["mean_case_dice"]
        mean_dice_scores[experiment_name] = mean_dice
        print(f"Mean Dice for '{experiment_name}': {mean_dice:.4f}")

    # Print summary 
   
    print("SUMMARY :")
    
    for experiment_name, mean_dice in mean_dice_scores.items():
        print(f"{experiment_name}: {mean_dice:.4f}")
    
    #save results 
    with open(args.output_json, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    
