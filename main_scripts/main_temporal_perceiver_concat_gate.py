import os
import sys
from os.path import join as path_join

import logging
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer, TrainingArguments, set_seed, EarlyStoppingCallback
from torchmetrics.classification import MultilabelAveragePrecision

# main_scripts/ lives one level below the repo root — put the root on
# sys.path so the cvs_datasets/models/evaluate/utils imports below resolve
# regardless of cwd (same convention as feature_extractor/extract_ft.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvs_datasets.CVS_Temporal_Dataset import TemporalWindowCVSDataset, TemporalWindowAugCVSDataset, collate_temporal
from cvs_datasets.Endoscapes_CVS_Temporal_Dataset import EndoscapesTemporalWindowDataset

from models.TemporalPerceiver_concat_gate import PerceiverLiteTemporalGated

from evaluate import evaluate_json_file, print_fancy_metrics_df
from utils import now_experiment_name, ensure_dir, load_yaml, save_json, create_directory_if_not_exists


# Original main_temporal_perceiver_concat_gate.py hardcoded this unconditionally
# (no CLI toggle existed) — kept as the sages default here for the same reason.
SAGES_FIXED_POS_WEIGHT = [6.80, 3.04, 5.44]


# ============================================================
# Trainer wrapper
# ============================================================
class SafeTrainer(Trainer):
    """Ensures checkpoint subdirectories exist before saving optimizer/scheduler."""

    def _save_optimizer_and_scheduler(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        super()._save_optimizer_and_scheduler(output_dir)


class TemporalPerceiverForTraining(nn.Module):
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__()
        self.model = PerceiverLiteTemporalGated(**kwargs)
        if pos_weight is None:
            pw = torch.ones(kwargs.get("n_classes", 3), dtype=torch.float32)
        else:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32)
        self.register_buffer("pos_weight", pw)

    def forward(self, x=None, key_index=None, labels=None, y=None, positions=None, **kwargs):
        if labels is None and y is not None:
            labels = y

        logits = self.model(x, key_index, positions)
        loss = None
        if labels is not None:
            labels = labels.to(dtype=logits.dtype)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=self.pos_weight.to(device=logits.device, dtype=logits.dtype),
                reduction="none",
            ).mean()

        return {"loss": loss, "logits": logits}


# ============================================================
# Collator
# ============================================================
def make_temporal_collator(pe_max_len: int):
    """Wraps the shared ``collate_temporal`` (which already prefers
    video-relative ``meta['position_ids']`` when the dataset provides them,
    e.g. Endoscapes — see cvs_datasets/CVS_Temporal_Dataset.py and CLAUDE.md
    gotcha #4) with a defense-in-depth hard-clamp of positions to
    ``[0, pe_max_len - 1]``.
    """
    pos_cap = max(0, int(pe_max_len) - 1)

    def temporal_collator(batch):
        out = collate_temporal(batch)
        out.pop("meta", None)
        out["labels"] = out.pop("y")
        out["positions"] = out["positions"].clamp(min=0, max=pos_cap)
        return out

    return temporal_collator


# ============================================================
# Predictions JSON
# ============================================================
@torch.no_grad()
def make_predictions_json(trainer, dataset, out_path, split_name, threshold=0.5):
    pred = trainer.predict(test_dataset=dataset)
    logits = pred.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    labels = pred.label_ids

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    labels = np.asarray(labels)
    labels_bin = (labels >= 0.5).astype(int)

    records = []
    for i in range(len(dataset)):
        sample = dataset[i]
        records.append(
            {
                "split": split_name,
                "video_name": sample.meta["video_name"],
                "key_frame_id": sample.meta["key_frame_id"],
                "file_name": sample.meta.get("file_name"),
                "image_id": sample.meta.get("image_id"),
                "probs": probs[i].tolist(),
                "pred": preds[i].tolist(),
                "label": labels_bin[i].tolist(),
                "confidence_aware_label": labels[i].tolist(),
            }
        )

    save_json(records, out_path)
    print(f"Saved json to: {out_path}")


# ============================================================
# Quick metrics (used during training)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    probs = torch.sigmoid(logits_t)

    labels_t = torch.as_tensor(labels, dtype=torch.float32)
    labels_bin = (labels_t >= 0.5).to(torch.int32)

    metric = MultilabelAveragePrecision(num_labels=probs.shape[1], average="macro")
    mAP = float(metric(probs.cpu(), labels_bin.cpu()).item())
    return {"mAP": mAP}


# ============================================================
# Full metrics (evaluate.py) → flat dict
# ============================================================
def _evaluate_and_flatten(json_path: str, split: str, threshold: float):
    metrics = evaluate_json_file(
        json_path=json_path,
        split=split,
        device="cpu",
        threshold=threshold,
        class_names=["c1", "c2", "c3"],
    )
    flat = {
        f"{split}/mAP_macro": metrics["mAP macro"],
        f"{split}/f1": metrics["f1"],
        f"{split}/accuracy": metrics["accuracy"],
    }
    for cls, v in metrics["mAP"].items():
        flat[f"{split}/mAP_{cls}"] = v
    for cls, v in metrics["brier_score"].items():
        flat[f"{split}/brier_{cls}"] = v
    return flat, metrics


# ============================================================
# CLI / config plumbing
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
    p = argparse.ArgumentParser("CVS Temporal Perceiver (concat-gate) trainer")
    p.add_argument("--dataset", type=str, default="sages", choices=["sages", "endoscapes"],
                   help="Dataset format. CLI-only, not read from YAML.")
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config providing defaults. CLI flags override YAML.")

    # Paths / experiment
    p.add_argument("--data_path", type=str, default=None,
                   help="sages: root with {split}/features subdirs. endoscapes: root with "
                        "{split}/annotation_ds_coco.json.")
    p.add_argument("--features_root", type=str, default=None,
                   help="endoscapes only: root with {split}/features subdirs. Ignored for "
                        "--dataset sages (features live under data_path/{split}/features there).")
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--exp_tag", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval_only", action="store_true")

    # Dataset / window
    p.add_argument("--temporal_mode", type=str, default=None, choices=["offline", "online"])
    p.add_argument("--window_size", type=int, default=None)
    p.add_argument("--fps", type=float, default=None, help="Target fps subsampling.")
    p.add_argument("--frame_stride", type=int, default=None,
                   help="endoscapes only: explicit stride in frame-id units, overrides --fps. "
                        "Ignored for --dataset sages.")
    p.add_argument("--label_type", type=str, default=None, choices=["ca", "hard"],
                   help="Training split only — val/test always use 'hard' (both original "
                        "scripts did this; see CLAUDE.md gotcha #5).")
    p.add_argument("--ft_type", type=str, default=None, choices=["image_embeddings", "patch_embeddings"],
                   help="endoscapes only. sages always uses its own 'prelogits' features "
                        "(same underlying tensor, different name), matching the original script.")
    p.add_argument("--pred_threshold", type=float, default=None)

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--no_amp", type=_str2bool, nargs="?", const=True, default=None)
    p.add_argument("--early_stopping_patience", type=int, default=None,
                   help="0 disables. Defaults to 5, matching both original scripts.")

    # Loss
    p.add_argument("--use_pos_weight", type=_str2bool, nargs="?", const=True, default=None,
                   help="endoscapes only: use BCE pos_weight from annotation_ds_coco.json's "
                        f"ds_weights. sages always uses its fixed pos_weight "
                        f"{SAGES_FIXED_POS_WEIGHT} regardless of this flag (matching the "
                        "original script, which had no toggle) unless --pos_weight overrides it.")
    p.add_argument("--pos_weight", type=float, nargs="+", default=None,
                   help="Manual per-class pos_weight. Overrides --use_pos_weight and sages' "
                        "fixed default. Works for either dataset.")

    # Model architecture (fixed defaults; overridable via YAML/CLI)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--d_in", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None, help="Number of Perceiver blocks.")
    p.add_argument("--dim_ff", type=int, default=None)
    p.add_argument("--K", type=int, default=None, help="Number of learnable latent tokens.")
    p.add_argument("--pe_max_len", type=int, default=None,
                   help="Max_len for the PositionalEncoding buffer (must cover max window position).")
    p.add_argument("--mlp_hidden", type=int, default=None, help="0=head linear, >0=head MLP.")
    p.add_argument("--pe_type", type=str, default=None, choices=["sinusoidal", "learned", "none"])
    p.add_argument("--mult", type=_str2bool, nargs="?", const=True, default=None,
                   help="One head per class (multilabel style).")
    p.add_argument("--no_gate", action="store_true", default=False,
                   help="Disable highway gating (plain residuals).")
    p.add_argument("--readout", type=str, default=None, choices=["keyframe", "mean", "cls"],
                   help="Latent readout strategy: keyframe token | mean of K latents | CLS token.")

    return p.parse_args()


def _resolve(cfg, args, key, default):
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

    seed = int(_resolve(cfg, args, "seed", 42))
    set_seed(seed)

    # Paths
    default_data_path = (
        "data/SAGES_2024" if dataset == "sages"
        else "/home/scanar/endovis/Datasets/endoscapes_2023/endoscapes"
    )
    data_path = _resolve(cfg, args, "data_path", default_data_path)
    features_root = _resolve(
        cfg, args, "features_root",
        "/home/scanar/endovis/Datasets/endoscapes_2023/endoscapes_features",
    )

    out_root = _resolve(cfg, args, "out_root", "outputs/CVSPerceiver" if dataset == "sages" else "outputs/CVSPerceiver_Endoscapes")
    exp_tag = _resolve(cfg, args, "exp_tag", "PerceiverConcatGate" if dataset == "sages" else "PerceiverConcatGate-Endoscapes")

    exp_name = f"{now_experiment_name()}__{exp_tag}"
    exp_dir = ensure_dir(path_join(out_root, exp_name))
    create_directory_if_not_exists(exp_dir)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO,
        filename=path_join(exp_dir, "training.log"),
    )

    best_dir = path_join(exp_dir, _resolve(cfg, args, "ckpt_name", "best"))
    pred_dir = ensure_dir(path_join(exp_dir, "preds"))
    val_pred_path = path_join(pred_dir, "val_predictions.json")
    test_pred_path = path_join(pred_dir, "test_predictions.json")
    best_valid_metrics_path = path_join(exp_dir, "best_valid_metrics.json")
    best_test_metrics_path = path_join(exp_dir, "best_test_metrics.json")

    # Dataset / window config
    temporal_mode = _resolve(cfg, args, "temporal_mode", "online")
    window_size = int(_resolve(cfg, args, "window_size", 5))
    fps = _resolve(cfg, args, "fps", 1.0)
    fps = float(fps) if fps is not None else None
    frame_stride = _resolve(cfg, args, "frame_stride", None)
    frame_stride = int(frame_stride) if frame_stride is not None else None
    label_type = _resolve(cfg, args, "label_type", "ca")
    ft_type = _resolve(cfg, args, "ft_type", "image_embeddings")
    pred_threshold = float(_resolve(cfg, args, "pred_threshold", 0.5))

    # Training hyperparameters
    epochs = int(_resolve(cfg, args, "epochs", 15))
    batch_size = int(_resolve(cfg, args, "batch_size", 64))
    num_workers = int(_resolve(cfg, args, "num_workers", 4))
    lr = float(_resolve(cfg, args, "lr", 1e-4))
    weight_decay = float(_resolve(cfg, args, "weight_decay", 5e-4))
    no_amp = bool(_resolve(cfg, args, "no_amp", False))
    early_stopping_patience = int(_resolve(cfg, args, "early_stopping_patience", 5))

    # Loss
    use_pos_weight = bool(_resolve(cfg, args, "use_pos_weight", False))
    manual_pos_weight = _resolve(cfg, args, "pos_weight", None)

    # Model architecture
    n_classes = int(_resolve(cfg, args, "num_classes", 3))
    d_in = int(_resolve(cfg, args, "d_in", 1024))
    d_model = int(_resolve(cfg, args, "d_model", 256))
    nhead = int(_resolve(cfg, args, "nhead", 4))
    num_layers = int(_resolve(cfg, args, "num_layers", 2))
    dim_ff = int(_resolve(cfg, args, "dim_ff", 1024))
    dropout = float(_resolve(cfg, args, "dropout", 0.2))
    K = int(_resolve(cfg, args, "K", 8))
    pe_max_len = int(_resolve(cfg, args, "pe_max_len", 2048))
    mlp_hidden = int(_resolve(cfg, args, "mlp_hidden", 256))
    pe_type = _resolve(cfg, args, "pe_type", "sinusoidal")
    mult = bool(_resolve(cfg, args, "mult", False))
    use_gate = not args.no_gate
    readout = _resolve(cfg, args, "readout", "keyframe")

    do_eval = not args.eval_only

    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Data path: {data_path}")
    if dataset == "endoscapes":
        logger.info(f"Features root: {features_root}")
    logger.info(f"Window: mode={temporal_mode}, size={window_size}, fps={fps}, "
                f"frame_stride={frame_stride}, label_type={label_type}, ft_type={ft_type}")
    logger.info(f"Model: d_model={d_model}, K={K}, nhead={nhead}, "
                f"num_layers={num_layers}, dropout={dropout}, pe_type={pe_type}, use_gate={use_gate}")
    logger.info(f"Optim: lr={lr}, weight_decay={weight_decay}, epochs={epochs}, batch_size={batch_size}")

    # ----------------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------------
    if dataset == "endoscapes":
        def _make_ds(split: str, lt: str):
            return EndoscapesTemporalWindowDataset(
                features_root=path_join(features_root, split, "features"),
                annotation_json_path=path_join(data_path, split, "annotation_ds_coco.json"),
                mode=temporal_mode,
                window_size=window_size,
                target_fps=fps,
                frame_stride=frame_stride,
                label_type=lt,
                ft_type=ft_type,
            )

        train_ds = _make_ds("train", label_type)
        val_ds = _make_ds("val", "hard")
        test_ds = _make_ds("test", "hard")
    else:
        sages_fps = int(fps) if fps is not None else 1
        train_ds = TemporalWindowAugCVSDataset(
            data_root=path_join(data_path, "train", "features"),
            mode=temporal_mode, window_size=window_size, target_fps=sages_fps,
            ft_type="prelogits", label_type=label_type, seed=seed,
        )
        val_ds = TemporalWindowCVSDataset(
            data_root=path_join(data_path, "val", "features"),
            mode=temporal_mode, window_size=window_size, target_fps=sages_fps,
            ft_type="prelogits", label_type="hard",
        )
        test_ds = TemporalWindowCVSDataset(
            data_root=path_join(data_path, "test", "features"),
            mode=temporal_mode, window_size=window_size, target_fps=sages_fps,
            ft_type="prelogits", label_type="hard",
        )

    logger.info(f"Sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ----------------------------------------------------------------
    # Pos-weight resolution
    # ----------------------------------------------------------------
    if manual_pos_weight is not None:
        pos_weight = list(manual_pos_weight)
    elif dataset == "sages":
        pos_weight = list(SAGES_FIXED_POS_WEIGHT)
    elif use_pos_weight and getattr(train_ds, "ds_weights", None) is not None:
        pos_weight = list(train_ds.ds_weights)
    else:
        pos_weight = None
    logger.info(f"BCE pos_weight: {pos_weight}")

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model = TemporalPerceiverForTraining(
        pos_weight=pos_weight,
        d_in=d_in,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        n_classes=n_classes,
        K=K,
        pe_max_len=pe_max_len,
        mlp_hidden=mlp_hidden,
        pe_type=pe_type,
        mult=mult,
        use_gate=use_gate,
        readout=readout,
    )

    # Persist resolved config for reproducibility
    save_json(
        {
            "exp_name": exp_name, "dataset": dataset,
            "data_path": data_path, "features_root": features_root if dataset == "endoscapes" else None,
            "temporal_mode": temporal_mode, "window_size": window_size,
            "fps": fps, "frame_stride": frame_stride,
            "label_type": label_type, "ft_type": ft_type,
            "epochs": epochs, "batch_size": batch_size, "num_workers": num_workers,
            "lr": lr, "weight_decay": weight_decay, "no_amp": no_amp,
            "early_stopping_patience": early_stopping_patience,
            "pos_weight": pos_weight,
            "n_classes": n_classes, "d_in": d_in, "d_model": d_model,
            "nhead": nhead, "num_layers": num_layers, "dim_ff": dim_ff,
            "dropout": dropout, "K": K, "pe_max_len": pe_max_len,
            "mlp_hidden": mlp_hidden, "pe_type": pe_type, "mult": mult,
            "use_gate": use_gate, "readout": readout,
            "seed": seed,
        },
        path_join(exp_dir, "config.json"),
    )

    # ----------------------------------------------------------------
    # Trainer
    # ----------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=best_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        learning_rate=lr,
        weight_decay=weight_decay,
        fp16=(not no_amp) and torch.cuda.is_available(),
        eval_strategy="epoch" if do_eval else "no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=do_eval,
        metric_for_best_model="mAP",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=seed,
    )

    collator = make_temporal_collator(pe_max_len=pe_max_len)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping_patience > 0 else []

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if not args.eval_only else None,
        eval_dataset=val_ds if do_eval else None,
        data_collator=collator,
        compute_metrics=compute_metrics if do_eval else None,
        callbacks=callbacks,
    )

    if not args.eval_only:
        trainer.train()
    best_ckpt = trainer.state.best_model_checkpoint
    logger.info(f"Best checkpoint: {best_ckpt}")

    # ----------------------------------------------------------------
    # Predictions + full metrics
    # ----------------------------------------------------------------
    make_predictions_json(trainer, val_ds, val_pred_path, "val", pred_threshold)
    make_predictions_json(trainer, test_ds, test_pred_path, "test", pred_threshold)

    val_flat, val_metrics = _evaluate_and_flatten(val_pred_path, "val", pred_threshold)
    test_flat, test_metrics = _evaluate_and_flatten(test_pred_path, "test", pred_threshold)

    print("\n--- Val ---")
    print_fancy_metrics_df(val_metrics)
    print("\n--- Test ---")
    print_fancy_metrics_df(test_metrics)

    save_json(
        {"split": "val", "exp_name": exp_name,
         "best_model_checkpoint": best_ckpt, **val_flat},
        best_valid_metrics_path,
    )
    save_json(
        {"split": "test", "exp_name": exp_name,
         "best_model_checkpoint": best_ckpt, **test_flat},
        best_test_metrics_path,
    )
    logger.info(f"Saved best valid metrics: {best_valid_metrics_path}")
    logger.info(f"Saved best test metrics:  {best_test_metrics_path}")


if __name__ == "__main__":
    main()
