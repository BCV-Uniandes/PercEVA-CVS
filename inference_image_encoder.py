import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors

import timm
from torchmetrics.classification import MultilabelAveragePrecision

from cvs_datasets.CVS_Dataset import CVSData, collate_fn as cvs_collate_fn
from cvs_datasets.Endoscapes_CVS_Dataset import EndoscapesImageDataset
from utils import ensure_dir, save_json, load_yaml, now_experiment_name, infer_split_name

#Model wrapper for timm 
class TimmMultiLabelModel(nn.Module):
    def __init__(self, model_name, num_classes=3, pretrained=False, img_size=None):
        super().__init__()
        kwargs = {"img_size": img_size} if img_size is not None else {}
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values=None, labels=None):
        logits = self.model(pixel_values)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
        return {"loss": loss, "logits": logits}


#model ckp loading utilities
def clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module."):]

        if nk.startswith("model.model."):
            nk = nk[len("model."):]   # model.model.xxx -> model.xxx
        elif nk.startswith("model."):
            nk = nk[len("model."):]

        cleaned[nk] = v
    return cleaned


def find_checkpoint_file(model_path: str):
    if os.path.isfile(model_path):
        return model_path

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"No existe model_path: {model_path}")

    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "checkpoint.pt",
        "model.pt",
        "weights.pt",
        "best_model.pt",
        "model.pth",
        "weights.pth",
    ]

    for name in candidates:
        p = os.path.join(model_path, name)
        if os.path.isfile(p):
            return p

    valid_exts = (".safetensors", ".bin", ".pt", ".pth")
    found = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(valid_exts)]

    if len(found) == 1:
        return found[0]

    if len(found) > 1:
        raise RuntimeError(
            f"Se encontraron múltiples checkpoints en {model_path}: {found}. "
            "Pasa directamente el archivo deseado en --model_path."
        )

    raise FileNotFoundError(f"No encontré checkpoint válido dentro de: {model_path}")


def extract_state_dict(obj):
    if not isinstance(obj, dict):
        raise RuntimeError(f"Formato de checkpoint no soportado: {type(obj)}")

    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]

    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]

    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]

    return obj  # posiblemente ya es el state_dict directo


def load_model_weights(model: nn.Module, model_path: str, device: torch.device):
    ckpt_file = find_checkpoint_file(model_path)
    print(f"[INFO] Cargando checkpoint desde: {ckpt_file}")

    if ckpt_file.endswith(".safetensors"):
        state_dict = load_safetensors(ckpt_file, device="cpu")
    else:
        obj = torch.load(ckpt_file, map_location="cpu")
        state_dict = extract_state_dict(obj)

    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] Missing keys: {len(missing)}")
    if missing:
        print("       ", missing[:20])

    print(f"[INFO] Unexpected keys: {len(unexpected)}")
    if unexpected:
        print("       ", unexpected[:20])

    model.to(device)
    model.eval()

    return ckpt_file, missing, unexpected


# =========================
# Inference + metrics
# =========================
@torch.no_grad()
def evaluate_and_save_predictions(
    model: nn.Module,
    dataset: Dataset,
    collator: cvs_collate_fn,
    batch_size: int,
    num_workers: int,
    threshold: float,
    device: torch.device,
    split_name: str,
    num_classes: int,
    out_json_path: str,
)-> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    metric = MultilabelAveragePrecision(num_labels=num_classes, average=None).to(device)
    results = []

    model.eval()

    for batch in tqdm(loader, desc=f"Inference [{split_name}]", leave=True):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = model(pixel_values=pixel_values)
        logits = out["logits"]

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int()
        labels_bin = (labels > 0.5).int()

        metric.update(probs, labels_bin)

        probs_np = probs.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        for i in range(len(batch["frame_id"])):
            labels_hard = (labels_np[i] > 0.5).astype(np.int32)
            results.append({
                "split": split_name,
                "video_name": str(batch["video_name"][i]),
                "frame_id": int(batch["frame_id"][i]),
                "probs": [float(x) for x in probs_np[i]],
                "pred": [int(x) for x in preds_np[i]],
                "label": [int(x) for x in labels_hard],
                "confidence_aware_label": [float(x) for x in labels_np[i]],
            })

    ap_per_class = metric.compute().detach().cpu().numpy()
    mAP = float(np.mean(ap_per_class))

    metrics = {
        "split": split_name,
        "mAP": mAP,
        "threshold": float(threshold),
        "num_samples": len(results),
    }

    for i, ap in enumerate(ap_per_class):
        metrics[f"AP_c{i+1}"] = float(ap)

    save_json(results, out_json_path)
    return metrics


# =========================
# Args
# =========================
def parse_args():
    p = argparse.ArgumentParser("CVS Inference (YAML + model_path + split_path)")
    p.add_argument("--config", type=str, required=True, help="Path al YAML.")
    p.add_argument("--model_path", type=str, required=True, help="Path al checkpoint o carpeta de checkpoint.")
    p.add_argument("--split_path", type=str, required=True, help="Ruta directa al split (ej: .../test o .../val).")
    p.add_argument("--out_dir", type=str, default=None, help="Directorio de salida opcional.")
    p.add_argument("--dataset", type=str, default="sages", choices=["sages", "endoscapes"],
                   help="Dataset format: 'sages' (frames/+labels/ subdirs) or "
                        "'endoscapes' (flat split folder with COCO JSON).")
    return p.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    # siempre intentar CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # config desde yaml
    model_name = cfg["model_name"]
    img_size = int(cfg.get("img_size", 448))
    num_classes = int(cfg.get("num_classes", 3))
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 8))
    threshold = float(cfg.get("test_threshold", 0.5))
    exp_tag = cfg.get("exp_tag", "inference")
    out_root = cfg.get("out_root", "outputs/inference")

    split_name = infer_split_name(args.split_path)

    if args.out_dir is not None:
        out_dir = ensure_dir(args.out_dir)
    else:
        run_name = f"{now_experiment_name()}__infer__{exp_tag}__{split_name}"
        out_dir = ensure_dir(os.path.join(out_root, run_name))

    preds_json_path = os.path.join(out_dir, f"{split_name}_predictions.json")
    metrics_json_path = os.path.join(out_dir, f"{split_name}_metrics.json")
    info_json_path = os.path.join(out_dir, "run_info.json")

    print(f"[INFO] model_path: {args.model_path}")
    print(f"[INFO] split_path: {args.split_path}")
    print(f"[INFO] model_name: {model_name}")
    print(f"[INFO] img_size: {img_size}")
    print(f"[INFO] num_classes: {num_classes}")
    print(f"[INFO] batch_size: {batch_size}")
    print(f"[INFO] num_workers: {num_workers}")
    print(f"[INFO] threshold: {threshold}")
    print(f"[INFO] out_dir: {out_dir}")

    tfm = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.dataset == "endoscapes":
        ds = EndoscapesImageDataset(args.split_path, label_type="hard", transform=tfm)
    else:
        frames_path = os.path.join(args.split_path, "frames")
        labels_path = os.path.join(args.split_path, "labels")
        ds = CVSData(frames_path, labels_path, transform=tfm, label_type="hard")
    collator = cvs_collate_fn

    print(f"[INFO] dataset={args.dataset} | dataset size ({split_name}): {len(ds)}")

    model = TimmMultiLabelModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        img_size=img_size,
    )

    ckpt_file, missing, unexpected = load_model_weights(model, args.model_path, device)

    metrics = evaluate_and_save_predictions(
        model=model,
        dataset=ds,
        collator=collator,
        batch_size=batch_size,
        num_workers=num_workers,
        threshold=threshold,
        device=device,
        split_name=split_name,
        num_classes=num_classes,
        out_json_path=preds_json_path,
    )

    metrics["checkpoint_file"] = ckpt_file
    metrics["model_name"] = model_name
    metrics["img_size"] = img_size
    metrics["num_classes"] = num_classes

    save_json(data_dict=metrics, save_path=metrics_json_path)
    save_json(data_dict={
        "config_path": args.config,
        "model_path": args.model_path,
        "resolved_checkpoint_file": ckpt_file,
        "split_path": args.split_path,
        "split_name": split_name,
        "device": str(device),
        "model_name": model_name,
        "img_size": img_size,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "threshold": threshold,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        }, save_path=info_json_path)

    print(f"[OK] Predicciones guardadas en: {preds_json_path}")
    print(f"[OK] Métricas guardadas en: {metrics_json_path}")
    print("\n========== RESULTADOS ==========")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()