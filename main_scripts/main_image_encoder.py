import os
import sys
from os.path import join as path_join

import json
import logging
import argparse

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import timm
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
from torchmetrics.classification import MultilabelAveragePrecision

# main_scripts/ lives one level below the repo root — put the root on
# sys.path so the cvs_datasets/models/utils imports below resolve
# regardless of cwd (same convention as feature_extractor/extract_ft.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvs_datasets.CVS_Dataset import CVSData, collate_fn as cvs_collate_fn
from cvs_datasets.Endoscapes_CVS_Dataset import EndoscapesImageDataset

from utils import now_experiment_name, ensure_dir, load_yaml, save_json, create_directory_if_not_exists


# ============================================================
# Metrics
# ============================================================
def make_compute_metrics(num_labels: int):
    metric = MultilabelAveragePrecision(num_labels=num_labels, average=None)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits_t = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else torch.tensor(logits)
        labels_t = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else torch.tensor(labels)

        labels_bin = (labels_t > 0.5).to(torch.int32)
        probs = torch.sigmoid(logits_t).float()

        metric.reset()
        metric.update(probs, labels_bin)

        ap_per_class = metric.compute().detach().cpu().numpy()
        mAP = float(np.mean(ap_per_class))

        out = {"mAP": mAP}
        for i, ap in enumerate(ap_per_class):
            out[f"AP_c{i+1}"] = float(ap)
        return out

    return compute_metrics


# ============================================================
# Trainer wrapper: StepLR + non-model-key stripping
# ============================================================
class StepLRTrainer(Trainer):
    def __init__(self, *args, step_size=3, gamma=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_size_epochs = step_size
        self._gamma = gamma

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer

        num_epochs = max(1, int(round(float(self.args.num_train_epochs))))
        steps_per_epoch = max(1, num_training_steps // num_epochs)
        step_size_steps = max(1, int(self._step_size_epochs * steps_per_epoch))

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size_steps,
            gamma=self._gamma,
        )
        return self.lr_scheduler

    def _strip_non_model_keys(self, inputs: dict) -> dict:
        inputs = dict(inputs)
        inputs.pop("video_name", None)
        inputs.pop("frame_id", None)
        inputs.pop("metadata", None)
        inputs.pop("img_path", None)
        return inputs

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._strip_non_model_keys(inputs)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._strip_non_model_keys(inputs)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


# ============================================================
# Model wrapper
# ============================================================
class TimmMultiLabelModel(nn.Module):
    def __init__(self, model_name, num_classes=3, img_size=None, pos_weight=None):
        super().__init__()
        kwargs = {"img_size": img_size} if img_size is not None else {}
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, **kwargs)
        if pos_weight is not None:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pixel_values=None, labels=None):
        logits = self.model(pixel_values)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
        return {"loss": loss, "logits": logits}


# ============================================================
# Test predictions JSON
# ============================================================
@torch.no_grad()
def save_test_predictions_json(trainer: Trainer, test_ds: Dataset, collator,
                               threshold: float, out_json_path: str):
    device = next(trainer.model.parameters()).device
    trainer.model.eval()

    loader = DataLoader(
        test_ds,
        batch_size=trainer.args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=trainer.args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    results = []
    for batch in tqdm(loader, desc="Test inference", leave=True):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = trainer.model(pixel_values=pixel_values)
        logits = out["logits"]

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int()

        probs_np = probs.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        for i in range(len(batch["frame_id"])):
            labels_hard = (labels_np[i] > 0.5).astype(np.int32)
            results.append({
                "split": "test",
                "video_name": str(batch["video_name"][i]),
                "frame_id": int(batch["frame_id"][i]),
                "probs": [float(x) for x in probs_np[i]],
                "pred":  [int(x) for x in preds_np[i]],
                "label": [int(x) for x in labels_hard],
                "confidence_aware_label": [float(x) for x in labels_np[i]],
            })

    ensure_dir(os.path.dirname(out_json_path))
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[OK] Saved test predictions JSON: {out_json_path} (N={len(results)})")


# ============================================================
# Args
# ============================================================
def _str2bool(v):
    """argparse type for booleans passed as 'True'/'False' (e.g. from a sweep agent)."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args():
    p = argparse.ArgumentParser("CVS image-encoder trainer (timm + HF Trainer)")
    p.add_argument("--dataset", type=str, default="sages", choices=["sages", "endoscapes"],
                   help="Dataset format: 'sages' (frames/+labels/ subdirs, CVSData) or "
                        "'endoscapes' (flat split folder + COCO json, EndoscapesImageDataset). "
                        "CLI-only, not read from YAML.")
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config providing defaults. CLI flags override YAML.")

    # Paths / experiment
    p.add_argument("--data_path", type=str, default=None,
                   help="Root with train/, val/, test/ subdirs. Defaults to data/SAGES_2024 "
                        "or the Endoscapes dataset root, depending on --dataset.")
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--exp_tag", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Model / data
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--label_type", type=str, default=None, choices=["ca", "hard"],
                   help="Applied uniformly to train/val/test, matching both original scripts.")

    # Optim
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--step_size", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--no_amp", type=_str2bool, nargs="?", const=True, default=None,
                   help="Disable mixed-precision training. Accepts bare flag, True/False.")
    p.add_argument("--early_stopping_patience", type=int, default=None,
                   help="0 disables. Defaults to 5 for --dataset endoscapes, 0 for --dataset "
                        "sages, matching each original script's recipe.")

    # Loss / eval
    p.add_argument("--use_pos_weight", type=_str2bool, nargs="?", const=True, default=None,
                   help="Use BCE pos_weight from the dataset's ds_weights (Endoscapes only; "
                        "no-op for --dataset sages unless --pos_weight is also given). "
                        "Accepts bare flag, True/False.")
    p.add_argument("--pos_weight", type=float, nargs="+", default=None,
                   help="Manual per-class pos_weight, overrides --use_pos_weight. Works for "
                        "either dataset.")
    p.add_argument("--test_threshold", type=float, default=None)

    return p.parse_args()


def _resolve(cfg: dict, args: argparse.Namespace, key: str, default):
    """CLI takes priority when not None, else YAML, else default."""
    cli = getattr(args, key, None)
    if cli is not None:
        return cli
    if cfg is not None and key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


# ============================================================
# Main
# ============================================================
def main():
    logger = logging.getLogger(__name__)
    args = parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    dataset = args.dataset

    # basics
    seed = int(_resolve(cfg, args, "seed", 42))
    set_seed(seed)

    default_data_path = (
        "data/SAGES_2024" if dataset == "sages"
        else "/home/scanar/endovis/Datasets/endoscapes_2023/endoscapes"
    )
    data_path = _resolve(cfg, args, "data_path", default_data_path)
    out_root  = _resolve(cfg, args, "out_root", "outputs/Eva02" if dataset == "sages" else "outputs/Eva02_Endoscapes")
    exp_tag   = _resolve(cfg, args, "exp_tag",  "EVA02" if dataset == "sages" else "EVA02-Endoscapes")

    exp_name = f"{now_experiment_name()}__{exp_tag}"
    exp_dir = ensure_dir(path_join(out_root, exp_name))
    create_directory_if_not_exists(exp_dir)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO,
        filename=path_join(exp_dir, "training.log"),
    )

    best_dir = path_join(exp_dir, _resolve(cfg, args, "ckpt_name", "best"))
    best_valid_metrics_path = path_join(exp_dir, _resolve(cfg, args, "best_valid_metrics_name", "best_valid_metrics.json"))
    best_test_metrics_path  = path_join(exp_dir, _resolve(cfg, args, "best_test_metrics_name",  "best_test_metrics.json"))
    test_preds_path         = path_join(exp_dir, _resolve(cfg, args, "test_json_name",          "test_predictions.json"))

    # train config
    model_name  = _resolve(cfg, args, "model_name", "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
    img_size    = int(_resolve(cfg, args, "img_size", 448))
    num_classes = int(_resolve(cfg, args, "num_classes", 3))
    label_type  = _resolve(cfg, args, "label_type", "ca")

    epochs       = int(_resolve(cfg, args, "epochs", 8))
    batch_size   = int(_resolve(cfg, args, "batch_size", 8))
    num_workers  = int(_resolve(cfg, args, "num_workers", 8))

    lr           = float(_resolve(cfg, args, "lr", 1e-4))
    weight_decay = float(_resolve(cfg, args, "weight_decay", 0.05))
    no_amp       = bool(_resolve(cfg, args, "no_amp", False))

    step_size    = int(_resolve(cfg, args, "step_size", 3))
    gamma        = float(_resolve(cfg, args, "gamma", 0.1))

    test_threshold = float(_resolve(cfg, args, "test_threshold", 0.5))
    use_pos_weight = bool(_resolve(cfg, args, "use_pos_weight", False))
    manual_pos_weight = _resolve(cfg, args, "pos_weight", None)

    default_es_patience = 5 if dataset == "endoscapes" else 0
    early_stopping_patience = int(_resolve(cfg, args, "early_stopping_patience", default_es_patience))

    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Image size: {img_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"StepLR step size (epochs): {step_size}")
    logger.info(f"StepLR gamma: {gamma}")
    logger.info(f"Label type: {label_type}")
    logger.info(f"Early stopping patience: {early_stopping_patience}")

    # transform + datasets
    tfm = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if dataset == "endoscapes":
        train_ds = EndoscapesImageDataset(path_join(data_path, "train"), label_type=label_type, transform=tfm)
        val_ds   = EndoscapesImageDataset(path_join(data_path, "val"),   label_type=label_type, transform=tfm)
        test_ds  = EndoscapesImageDataset(path_join(data_path, "test"),  label_type=label_type, transform=tfm)
    else:
        train_path, val_path, test_path = (path_join(data_path, s) for s in ("train", "val", "test"))
        train_ds = CVSData(path_join(train_path, "frames"), path_join(train_path, "labels"), label_type=label_type, transform=tfm)
        val_ds   = CVSData(path_join(val_path,   "frames"), path_join(val_path,   "labels"), label_type=label_type, transform=tfm)
        test_ds  = CVSData(path_join(test_path,  "frames"), path_join(test_path,  "labels"), label_type=label_type, transform=tfm)

    logger.info(f"Sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    collator = cvs_collate_fn

    if manual_pos_weight is not None:
        pos_weight = list(manual_pos_weight)
    elif use_pos_weight:
        pos_weight = getattr(train_ds, "ds_weights", None)
        if pos_weight is None:
            logger.warning(
                f"--use_pos_weight set but dataset={dataset} exposes no ds_weights "
                "(only EndoscapesImageDataset does); training without pos_weight."
            )
    else:
        pos_weight = None
    logger.info(f"pos_weight: {pos_weight}")

    model = TimmMultiLabelModel(model_name=model_name, num_classes=num_classes, img_size=img_size, pos_weight=pos_weight)

    training_args = TrainingArguments(
        output_dir=best_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        learning_rate=lr,
        weight_decay=weight_decay,
        fp16=(not no_amp),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="mAP",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
    )

    compute_metrics = make_compute_metrics(num_classes)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping_patience > 0 else []

    trainer = StepLRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        step_size=step_size,
        gamma=gamma,
    )

    # ---- train
    trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint

    # ---- val metrics json (best-model selection happens here; load_best_model_at_end already restored it)
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    final_dict_val = {
        "split": "val",
        "exp_name": exp_name,
        "best_model_checkpoint": best_ckpt,
        "mAP":   float(val_metrics.get("eval_mAP",   float("nan"))),
        "AP_c1": float(val_metrics.get("eval_AP_c1", float("nan"))),
        "AP_c2": float(val_metrics.get("eval_AP_c2", float("nan"))),
        "AP_c3": float(val_metrics.get("eval_AP_c3", float("nan"))),
    }
    save_json(final_dict_val, best_valid_metrics_path)
    logger.info(f"Saved best valid metrics JSON: {best_valid_metrics_path}")

    # ---- test + preds json
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    final_dict_test = {
        "split": "test",
        "exp_name": exp_name,
        "best_model_checkpoint": best_ckpt,
        "mAP":   float(test_metrics.get("test_mAP",   float("nan"))),
        "AP_c1": float(test_metrics.get("test_AP_c1", float("nan"))),
        "AP_c2": float(test_metrics.get("test_AP_c2", float("nan"))),
        "AP_c3": float(test_metrics.get("test_AP_c3", float("nan"))),
    }
    save_json(final_dict_test, best_test_metrics_path)
    logger.info(f"Saved best test metrics JSON: {best_test_metrics_path}")

    save_test_predictions_json(
        trainer=trainer,
        test_ds=test_ds,
        collator=collator,
        threshold=test_threshold,
        out_json_path=test_preds_path,
    )


if __name__ == "__main__":
    main()
