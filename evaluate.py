import json
import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelAveragePrecision,
)


def eval_parser():
    
    """
    Parser for evaluating CVS predictions from a JSON file.
    The JSON should be a list of dicts, each containing at least:
  - "video_name": str
  - "frame_id": int
  - "label": list of int (0/1) for each class
  - "probs": list of float (0..1) for each class
  - Optional: "confidence_aware_label" or "confidence_aware_labels": list of float (0..1) for each class, if you want to use custom CA labels for Brier score instead of GT hard labels.    
    """
    
    parser = argparse.ArgumentParser(description="Parser for evaluating CVS predictions")
    
    parser.add_argument(
        "--pred_path",
        type=str,
        default="preds.json",
        help="Path to prediction json",
        required=True
    )
    
    return parser
    
    
def print_fancy_metrics_df(metrics: Dict):
    rows = []
    for cls in metrics["mAP"]:
        rows.append({
            "Class": cls,
            "mAP": metrics["mAP"][cls],
            "BAcc": metrics.get("bacc", {}).get(cls, float("nan")),
            "Brier": metrics["brier_score"][cls],
        })

    df = pd.DataFrame(rows).set_index("Class")

    print("\n=== CVS Evaluation Summary ===")
    print(f"N samples           : {metrics['N']}")
    print(f"Exact Match Acc     : {metrics['accuracy']:.4f}")
    print(f"Macro F1            : {metrics['f1']:.4f}")
    print(f"Macro mAP           : {metrics['mAP macro']:.4f}")
    print(f"Macro BAcc          : {metrics.get('bacc macro', float('nan')):.4f}")

    print("\nPer-class metrics:")
    print(df.round(4))
    

def _stack_from_json(
    data: List[dict],
    split: Optional[str] = "test",
    classes: int = 3,
    device: str = "cpu",
    ca_label_key: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      overall_raw_labels: (N, C) int tensor {0,1}
      overall_confidence_aware_labels: (N, C) float tensor (can be continuous)
      overall_outputs: (N, C) float tensor in [0,1]
    """
    if split is not None:
        data = [d for d in data if d.get("split") == split]
    if len(data) == 0:
        raise ValueError("No samples found. Check split filter or JSON content.")

    raw_labels = []
    outputs = []
    ca_labels = []

    for d in data:
        lab = d["label"]
        prob = d["probs"]

        if len(lab) != classes or len(prob) != classes:
            raise ValueError(f"Class count mismatch for item video={d.get('video_name')} frame={d.get('frame_id')}")

        raw_labels.append(lab)
        outputs.append(prob)

        # confidence-aware labels: prefer explicit key, else fallback to GT
        if ca_label_key is not None and ca_label_key in d:
            ca = d[ca_label_key]
        elif "confidence_aware_label" in d:
            ca = d["confidence_aware_label"]
        elif "confidence_aware_labels" in d:
            ca = d["confidence_aware_labels"]
        else:
            ca = lab  # fallback: standard Brier score vs GT

        if len(ca) != classes:
            raise ValueError("confidence-aware labels length mismatch.")

        ca_labels.append(ca)

    overall_raw_labels = torch.tensor(raw_labels, dtype=torch.int64, device=device)
    overall_outputs = torch.tensor(outputs, dtype=torch.float32, device=device)
    overall_confidence_aware_labels = torch.tensor(ca_labels, dtype=torch.float32, device=device)

    return overall_raw_labels, overall_confidence_aware_labels, overall_outputs


def compute_overall_metrics_torchmetrics(
    overall_labels: torch.Tensor,              # (N,C) int 0/1
    confidence_aware_labels: torch.Tensor,     # (N,C) float (0..1)
    overall_confidences: torch.Tensor,         # (N,C) float (0..1)
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Mirrors your original outputs: f1, mAP, accuracy, brier_score per class.
    """
    if overall_labels.ndim != 2:
        raise ValueError("overall_labels must be (N,C)")
    N, C = overall_labels.shape

    # Predicted labels from confidences (same as original code)
    overall_predicted_labels = (overall_confidences > threshold).to(torch.int64)

    # Torchmetrics (multi
    exact_match = (overall_predicted_labels == overall_labels).all(dim=1).float().mean().item()

    f1_metric = MultilabelF1Score(num_labels=C, average="macro", threshold=threshold, zero_division=1).to(overall_labels.device)
    map_metric = MultilabelAveragePrecision(num_labels=C, average="none").to(overall_labels.device)

    with torch.no_grad():
        f1 = float(f1_metric(overall_confidences, overall_labels).item())
        mAP_per_class = map_metric(overall_confidences, overall_labels)

    # Brier score, mAP, and BAcc per class
    brier = {}
    map_dict = {}
    bacc_dict = {}

    names = class_names if (class_names is not None and len(class_names) == C) else [f"c{i+1}" for i in range(C)]

    for i, key in enumerate(names):
        conf = overall_confidences[:, i]
        ca = confidence_aware_labels[:, i]
        pred_i = overall_predicted_labels[:, i]
        true_i = overall_labels[:, i]

        brier[key] = float(torch.mean((ca - conf) ** 2).item())
        map_dict[key] = float(mAP_per_class[i].item())

        TP = ((pred_i == 1) & (true_i == 1)).sum().float()
        TN = ((pred_i == 0) & (true_i == 0)).sum().float()
        FP = ((pred_i == 1) & (true_i == 0)).sum().float()
        FN = ((pred_i == 0) & (true_i == 1)).sum().float()
        sensitivity = TP / (TP + FN).clamp(min=1)
        specificity = TN / (TN + FP).clamp(min=1)
        bacc_dict[key] = float(((sensitivity + specificity) / 2).item())

    metrics = {
        "f1": f1,
        "mAP": map_dict,
        "mAP macro": float(mAP_per_class.mean().item()),
        "bacc": bacc_dict,
        "bacc macro": float(sum(bacc_dict.values()) / len(bacc_dict)),
        "accuracy": exact_match,
        "brier_score": brier,
        "N": int(N),
    }
    return metrics


def evaluate_json_file(
    json_path: str,
    split: Optional[str] = "test",
    device: str = "cpu",
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
    ca_label_key: Optional[str] = None,
) -> Dict:
    with open(json_path, "r") as f:
        data = json.load(f)

    overall_raw_labels, overall_confidence_aware_labels, overall_outputs = _stack_from_json(
        data=data,
        split=split,
        device=device,
        ca_label_key=ca_label_key,
    )

    metrics = compute_overall_metrics_torchmetrics(
        overall_labels=overall_raw_labels,
        confidence_aware_labels=overall_confidence_aware_labels,
        overall_confidences=overall_outputs,
        threshold=threshold,
        class_names=class_names,
    )

    return metrics


if __name__ == "__main__":
    
    parser = eval_parser()
    args = parser.parse_args()
    
    json_path = args.pred_path
    metrics = evaluate_json_file(
        json_path=json_path,
        split="test",
        device="cpu",          # "cuda" if you want
        threshold=0.5,
        class_names=["c1", "c2", "c3"],  # replace with real CVS label names if you have them
        ca_label_key=None,     # set if your json has a custom key
    )
    
    print_fancy_metrics_df(metrics=metrics)
