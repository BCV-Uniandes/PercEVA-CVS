import os
import sys

# ---- NCCL hardening for multi-GPU RTX-8000-class PCIe (no NVLink) ----
# Set BEFORE importing torch so the c10d init sees them. Harmless/unused
# outside DDP.
_NCCL_DEFAULTS = {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "lo",
    "NCCL_BLOCKING_WAIT": "1",
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "OMP_NUM_THREADS": "4",
}
for _k, _v in _NCCL_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

from os.path import join as path_join

import json
import logging
import argparse
import time

import numpy as np

import torch
import torch.distributed as dist

from torchvision import transforms
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
    EarlyStoppingCallback,
)
from torchmetrics.classification import MultilabelAveragePrecision

# main_scripts/ lives one level below the repo root — put the root on
# sys.path so the cvs_datasets/models/evaluate/utils imports below resolve
# regardless of cwd (same convention as feature_extractor/extract_ft.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvs_datasets.CVS_TemporalImage_Dataset import TemporalImageWindowDataset
from cvs_datasets.Endoscapes_CVS_Dataset import EndoscapesTemporalImageWindowDataset
from models.E2EPerceiverGated import E2EPerceiverGated, _is_main_process

from evaluate import evaluate_json_file, print_fancy_metrics_df
from utils import now_experiment_name, ensure_dir, load_yaml, save_json, create_directory_if_not_exists


# Original main_e2e_sages.py's fixed BCE pos_weight, applied when
# --use_pos_weight is set for --dataset sages (endoscapes instead reads
# ds_weights from annotation_ds_coco.json — see pos-weight resolution below).
SAGES_POS_WEIGHT_DEFAULT = [6.80, 3.04, 5.44]


def _barrier(tag: str = ""):
    """Best-effort cross-rank barrier — no-op outside DDP."""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
            if tag and _is_main_process():
                logging.getLogger(__name__).info(f"DDP barrier passed: {tag}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"DDP barrier failed ({tag}): {e}")


# ============================================================
# Trainer wrapper / progress logging
# ============================================================
class SafeTrainer(Trainer):
    """Ensures checkpoint subdirectories exist before saving optimizer/scheduler."""
    def _save_optimizer_and_scheduler(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        super()._save_optimizer_and_scheduler(output_dir)


class LineProgressCallback(TrainerCallback):
    """Discrete progress logging — one line every `log_every` global steps.
    Works under torchrun where tqdm's in-place \\r refresh gets stripped."""

    def __init__(self, log_every: int = 25):
        self.log_every = max(1, int(log_every))
        self._t0 = None
        self._last_step = 0
        self._last_t = None

    def on_train_begin(self, args, state, control, **kwargs):
        if int(os.environ.get("RANK", "0")) != 0:
            return
        self._t0 = time.time()
        self._last_t = self._t0
        total = state.max_steps or "?"
        logging.getLogger(__name__).info(
            f"Training started — total steps: {total}, "
            f"per-device batch: {args.per_device_train_batch_size}, "
            f"grad_accum: {args.gradient_accumulation_steps}, world_size: {args.world_size}"
        )

    def on_step_end(self, args, state, control, **kwargs):
        if int(os.environ.get("RANK", "0")) != 0:
            return
        step = state.global_step
        if step == 0 or step % self.log_every != 0:
            return
        now = time.time()
        dt = max(now - self._last_t, 1e-6)
        sps = (step - self._last_step) / dt
        elapsed = now - self._t0
        remaining = (state.max_steps - step) / max(sps, 1e-6) if state.max_steps else 0
        loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    loss = entry["loss"]
                    break
        msg = (f"step {step}/{state.max_steps} ({100.0 * step / max(state.max_steps, 1):.1f}%) — "
               f"{sps:.2f} steps/s — elapsed {elapsed/60:.1f}min — ETA {remaining/60:.1f}min")
        if loss is not None:
            msg += f" — loss {loss:.4f}"
        logging.getLogger(__name__).info(msg)
        self._last_step = step
        self._last_t = now

    def on_epoch_end(self, args, state, control, **kwargs):
        if int(os.environ.get("RANK", "0")) != 0:
            return
        logging.getLogger(__name__).info(f"Epoch {state.epoch:.2f} complete — step {state.global_step}")


# ============================================================
# Collator (raw-image temporal sample), with position-overflow clamp
# ============================================================
def make_e2e_collator(pe_max_len: int):
    pos_cap = max(0, int(pe_max_len) - 1)

    def collator(batch):
        pixel_values = torch.stack([b.pixel_values for b in batch], dim=0)
        y = torch.stack([b.y for b in batch], dim=0)
        key_index = torch.tensor([b.key_index for b in batch])
        valid_mask = torch.stack([b.valid_mask for b in batch], dim=0)

        if all("position_ids" in b.meta for b in batch):
            positions = torch.tensor([b.meta["position_ids"] for b in batch], dtype=torch.long)
        else:
            stride = batch[0].meta["stride"]
            frame_ids = torch.tensor([b.meta["frame_ids"] for b in batch], dtype=torch.long)
            positions = (frame_ids // stride).clamp(min=0)
        # Defense-in-depth: hard-clamp even correctly-computed positions to
        # the PE table's range, in case of an edge case neither branch above
        # accounts for (see cvs_datasets/CVS_Temporal_Dataset.py::collate_temporal
        # for the bug this pattern guards against).
        positions = positions.clamp(min=0, max=pos_cap)

        return {
            "pixel_values": pixel_values,
            "labels": y,
            "key_index": key_index,
            "valid_mask": valid_mask,
            "positions": positions,
        }

    return collator


# ============================================================
# Metrics / predictions
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32))
    labels_bin = (torch.as_tensor(labels, dtype=torch.float32) >= 0.5).to(torch.int32)
    metric = MultilabelAveragePrecision(num_labels=probs.shape[1], average="macro")
    return {"mAP": float(metric(probs.cpu(), labels_bin.cpu()).item())}


def _load_ca_labels_for_dataset(dataset):
    """sages only: CA (soft) labels for every sample, aligned by index — reads
    label CSVs only, doesn't touch images, so this is fast."""
    n = len(dataset)
    ca = np.zeros((n, 3), dtype=np.float32)
    prev_lt = getattr(dataset, "label_type", None)
    try:
        dataset.label_type = "ca"
        for i in range(n):
            s = dataset.samples[i]
            vname, fid = (s["video_name"], s["key_frame_id"]) if isinstance(s, dict) else s
            ca[i] = dataset._load_label(vname, fid).numpy()
    finally:
        if prev_lt is not None:
            dataset.label_type = prev_lt
    return ca


def _write_predictions(pred, dataset, out_path, split_name, dataset_kind, threshold=0.5):
    logits = pred.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    labels = np.asarray(pred.label_ids)

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    labels_bin = (labels >= 0.5).astype(int)

    if dataset_kind == "sages":
        try:
            ca_labels = _load_ca_labels_for_dataset(dataset)
        except Exception as e:
            logging.warning(f"Could not load CA labels for {split_name}: {e}; using pred.label_ids fallback")
            ca_labels = labels.astype(np.float32)
    else:
        # EndoscapesTemporalImageWindowDataset doesn't expose per-sample CA
        # label loading — matches the original endoscapes script exactly.
        ca_labels = labels.astype(np.float32)

    records = []
    for i in range(len(dataset)):
        # dataset.samples[i] avoids loading images; metas are already there.
        # sages: tuple (video_name, frame_id). endoscapes: dict with extra
        # file_name/image_id fields.
        sample = dataset.samples[i]
        if isinstance(sample, dict):
            vname, fid = sample["video_name"], sample["key_frame_id"]
            extra = {"file_name": sample.get("file_name"), "image_id": sample.get("image_id")}
        else:
            vname, fid = sample
            extra = {}

        records.append({
            "split": split_name,
            "video_name": vname,
            "key_frame_id": int(fid),
            **extra,
            "probs": probs[i].tolist(),
            "pred": preds[i].tolist(),
            "label": labels_bin[i].tolist(),
            "confidence_aware_label": ca_labels[i].tolist(),
        })

    save_json(records, out_path)
    print(f"Saved predictions: {out_path}")


def _evaluate_and_flatten(json_path: str, split: str, threshold: float):
    """Uses evaluate.py's own bACC (via evaluate_json_file), consistent with
    every other training/inference script in this repo — the upstream SAGES
    script instead independently recomputed bACC via sklearn; dropped here
    as redundant."""
    metrics = evaluate_json_file(
        json_path=json_path, split=split, device="cpu", threshold=threshold,
        class_names=["c1", "c2", "c3"],
    )
    flat = {
        f"{split}/mAP_macro": metrics["mAP macro"],
        f"{split}/bACC_macro": metrics.get("bacc macro"),
        f"{split}/f1": metrics["f1"],
        f"{split}/accuracy": metrics["accuracy"],
    }
    for cls, v in metrics["mAP"].items():
        flat[f"{split}/mAP_{cls}"] = v
    for cls, v in metrics["brier_score"].items():
        flat[f"{split}/brier_{cls}"] = v
    for cls, v in metrics.get("bacc", {}).items():
        flat[f"{split}/bACC_{cls}"] = v
    return flat, metrics


# ============================================================
# CLI plumbing
# ============================================================
def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args():
    p = argparse.ArgumentParser("CVS End-to-End (EVA-02 + Perceiver concat-gate)")
    p.add_argument("--dataset", type=str, default="sages", choices=["sages", "endoscapes"],
                   help="Dataset format. CLI-only, not read from YAML.")
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--data_path", type=str, default=None,
                   help="sages: root with {split}/frames + {split}/labels subdirs. "
                        "endoscapes: root with {split}/annotation_ds_coco.json + flat *.jpg frames.")
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--exp_tag", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--encoder_pretrained", type=_str2bool, nargs="?", const=True, default=None)
    p.add_argument("--freeze_encoder", type=_str2bool, nargs="?", const=True, default=None)
    p.add_argument("--unfreeze_blocks", type=int, default=None,
                   help="Unfreeze only the last N transformer blocks of EVA-02 (+ final norm). "
                        "0 = full freeze; omit = full unfreeze unless --freeze_encoder is set. "
                        "Paper recipe: 4.")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--encoder_ckpt", type=str, default=None,
                   help="Optional path to a fine-tuned encoder checkpoint "
                        "(safetensors / .bin / .pt) — loaded into self.encoder.")

    p.add_argument("--temporal_mode", type=str, default=None, choices=["offline", "online"])
    p.add_argument("--window_size", type=int, default=None)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--frame_stride", type=int, default=None,
                   help="endoscapes: explicit stride in frame-id units, overrides --fps. "
                        "sages: parsed but not wired into TemporalImageWindowDataset, "
                        "matching the original sages-only script (no-op there).")
    p.add_argument("--pred_threshold", type=float, default=None)
    p.add_argument("--label_type", type=str, default=None, choices=["hard", "ca"],
                   help="sages: training split only, val/test always 'hard'. endoscapes: "
                        "applied to all three splits (intentional upstream divergence).")
    p.add_argument("--border_padding", type=str, default=None, choices=["keyframe", "zero"],
                   help="sages only: how invalid window slots are filled ('keyframe' copies "
                        "the keyframe image, default). Ignored for --dataset endoscapes.")

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="Learning rate for the temporal head.")
    p.add_argument("--encoder_lr", type=float, default=None,
                   help="Learning rate for the EVA-02 encoder. If unset, uses --lr.")
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--no_amp", type=_str2bool, nargs="?", const=True, default=None)
    p.add_argument("--early_stopping_patience", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)

    p.add_argument("--use_pos_weight", type=_str2bool, nargs="?", const=True, default=None,
                   help="sages: use the fixed pos_weight [6.80, 3.04, 5.44]. endoscapes: use "
                        "BCE pos_weight from annotation_ds_coco.json's ds_weights. Both default "
                        "to no pos_weight unless this is set.")
    p.add_argument("--pos_weight", type=float, nargs="+", default=None,
                   help="Manual per-class pos_weight (overrides --use_pos_weight). Works for "
                        "either dataset.")

    p.add_argument("--K", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--dim_ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--mlp_hidden", type=int, default=None)
    p.add_argument("--pe_type", type=str, default=None, choices=["sinusoidal", "learned", "none"])
    p.add_argument("--pe_max_len", type=int, default=None)
    p.add_argument("--mult", type=_str2bool, nargs="?", const=True, default=None)

    # DDP knobs — inert on a single GPU
    p.add_argument("--ddp_timeout", type=int, default=3600)
    p.add_argument("--ddp_bucket_cap_mb", type=int, default=25)
    p.add_argument("--ddp_find_unused_parameters", type=_str2bool, nargs="?", const=True, default=None,
                   help="If unset, auto-True when the encoder is partially frozen.")
    p.add_argument("--log_every", type=int, default=25)

    return p.parse_args()


def _resolve(cfg, args, key, default):
    cli = getattr(args, key, None)
    if cli is not None:
        return cli
    if cfg is not None and key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


def _load_state_dict(path: str):
    if os.path.isdir(path):
        sf = path_join(path, "model.safetensors")
        pt = path_join(path, "pytorch_model.bin")
        if os.path.isfile(sf):
            from safetensors.torch import load_file
            return load_file(sf)
        if os.path.isfile(pt):
            return torch.load(pt, map_location="cpu", weights_only=False)
        raise FileNotFoundError(f"No checkpoint found under {path}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _strip_prefix(state_dict, prefix):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


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

    default_data_path = (
        "data/SAGES_2024" if dataset == "sages"
        else "/home/scanar/endovis/Datasets/endoscapes_2023/endoscapes"
    )
    data_path = _resolve(cfg, args, "data_path", default_data_path)
    out_root = _resolve(cfg, args, "out_root", "outputs/End2End_SAGES" if dataset == "sages" else "outputs/End2End_Endoscapes")
    exp_tag = _resolve(cfg, args, "exp_tag", "End2End-SAGES" if dataset == "sages" else "End2End-Endoscapes")

    exp_name = f"{now_experiment_name()}__{exp_tag}"
    exp_dir = ensure_dir(path_join(out_root, exp_name))
    create_directory_if_not_exists(exp_dir)
    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    if not _root.handlers:
        _fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        _fh = logging.FileHandler(path_join(exp_dir, "training.log"))
        _fh.setFormatter(_fmt)
        _root.addHandler(_fh)
        if _is_main_process():
            _sh = logging.StreamHandler()
            _sh.setFormatter(_fmt)
            _root.addHandler(_sh)

    best_dir = path_join(exp_dir, _resolve(cfg, args, "ckpt_name", "best"))
    pred_dir = ensure_dir(path_join(exp_dir, "preds"))
    val_pred_path = path_join(pred_dir, "val_predictions.json")
    test_pred_path = path_join(pred_dir, "test_predictions.json")
    best_valid_metrics_path = path_join(exp_dir, "best_valid_metrics.json")
    best_test_metrics_path = path_join(exp_dir, "best_test_metrics.json")

    model_name = _resolve(cfg, args, "model_name", "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
    encoder_pretrained = bool(_resolve(cfg, args, "encoder_pretrained", True))
    freeze_encoder = bool(_resolve(cfg, args, "freeze_encoder", False))
    unfreeze_blocks_val = _resolve(cfg, args, "unfreeze_blocks", None)
    unfreeze_blocks = int(unfreeze_blocks_val) if unfreeze_blocks_val is not None else None
    img_size = int(_resolve(cfg, args, "img_size", 448))
    encoder_ckpt = _resolve(cfg, args, "encoder_ckpt", None)

    temporal_mode = _resolve(cfg, args, "temporal_mode", "online")
    window_size = int(_resolve(cfg, args, "window_size", 5))
    fps = _resolve(cfg, args, "fps", 1.0)
    fps = float(fps) if fps is not None else None
    frame_stride = _resolve(cfg, args, "frame_stride", None)
    frame_stride = int(frame_stride) if frame_stride is not None else None
    pred_threshold = float(_resolve(cfg, args, "pred_threshold", 0.5))
    label_type = str(_resolve(cfg, args, "label_type", "hard"))
    border_padding = str(_resolve(cfg, args, "border_padding", "keyframe"))

    epochs = int(_resolve(cfg, args, "epochs", 15))
    batch_size = int(_resolve(cfg, args, "batch_size", 4))
    num_workers = int(_resolve(cfg, args, "num_workers", 4))
    lr = float(_resolve(cfg, args, "lr", 1e-5))
    encoder_lr_val = _resolve(cfg, args, "encoder_lr", None)
    encoder_lr = float(encoder_lr_val) if encoder_lr_val is not None else lr
    weight_decay = float(_resolve(cfg, args, "weight_decay", 5e-4))
    no_amp = bool(_resolve(cfg, args, "no_amp", False))
    early_stopping_patience = int(_resolve(cfg, args, "early_stopping_patience", 5))
    grad_accum = int(_resolve(cfg, args, "gradient_accumulation_steps", 1))

    ddp_timeout = int(_resolve(cfg, args, "ddp_timeout", 3600))
    ddp_bucket_cap_mb = int(_resolve(cfg, args, "ddp_bucket_cap_mb", 25))
    ddp_find_unused_cli = _resolve(cfg, args, "ddp_find_unused_parameters", None)

    use_pos_weight = bool(_resolve(cfg, args, "use_pos_weight", False))
    manual_pos_weight = _resolve(cfg, args, "pos_weight", None)

    n_classes = int(_resolve(cfg, args, "num_classes", 3))
    d_model = int(_resolve(cfg, args, "d_model", 256))
    nhead = int(_resolve(cfg, args, "nhead", 4))
    num_layers = int(_resolve(cfg, args, "num_layers", 2))
    dim_ff = int(_resolve(cfg, args, "dim_ff", 1024))
    dropout = float(_resolve(cfg, args, "dropout", 0.2))
    K = int(_resolve(cfg, args, "K", 8))
    mlp_hidden = int(_resolve(cfg, args, "mlp_hidden", 256))
    pe_type = _resolve(cfg, args, "pe_type", "learned")
    pe_max_len = int(_resolve(cfg, args, "pe_max_len", 2048))
    mult = bool(_resolve(cfg, args, "mult", False))

    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Dataset: {dataset}")

    tfm = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if dataset == "endoscapes":
        def _make_ds(split: str):
            return EndoscapesTemporalImageWindowDataset(
                split_path=path_join(data_path, split),
                mode=temporal_mode, window_size=window_size, frame_stride=frame_stride,
                target_fps=fps, transform=tfm, label_type=label_type,
            )

        train_ds = _make_ds("train")
        val_ds = _make_ds("val")
        test_ds = _make_ds("test")
    else:
        def _make_ds(split: str, ltype: str):
            return TemporalImageWindowDataset(
                frames_path=path_join(data_path, split, "frames"),
                labels_path=path_join(data_path, split, "labels"),
                mode=temporal_mode, window_size=window_size, target_fps=fps,
                label_type=ltype, transform=tfm, border_padding=border_padding,
            )

        train_ds = _make_ds("train", label_type)
        val_ds = _make_ds("val", "hard")
        test_ds = _make_ds("test", "hard")

    logger.info(f"Sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ----------------------------------------------------------------
    # Pos-weight resolution
    # ----------------------------------------------------------------
    if manual_pos_weight is not None:
        pos_weight = list(manual_pos_weight)
    elif use_pos_weight and dataset == "sages":
        pos_weight = list(SAGES_POS_WEIGHT_DEFAULT)
    elif use_pos_weight:
        ds_weights = None
        try:
            with open(path_join(data_path, "train", "annotation_ds_coco.json")) as f:
                ds_weights = json.load(f).get("ds_weights", None)
        except FileNotFoundError:
            pass
        pos_weight = list(ds_weights) if ds_weights is not None else None
    else:
        pos_weight = None
    logger.info(f"BCE pos_weight: {pos_weight}")

    model = E2EPerceiverGated(
        model_name=model_name, encoder_pretrained=encoder_pretrained, freeze_encoder=freeze_encoder,
        d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, dropout=dropout,
        n_classes=n_classes, K=K, mlp_hidden=mlp_hidden, mult=mult,
        pe_type=pe_type, pe_max_len=pe_max_len, pos_weight=pos_weight,
    )

    if encoder_ckpt:
        logger.info(f"Loading encoder checkpoint: {encoder_ckpt}")
        sd = _load_state_dict(encoder_ckpt)
        sd = _strip_prefix(_strip_prefix(sd, "model."), "backbone.")
        missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
        logger.info(f"Encoder load — missing: {len(missing)}, unexpected: {len(unexpected)}")

    if unfreeze_blocks is not None:
        model.unfreeze_encoder_last_n_blocks(unfreeze_blocks)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Params — trainable: {trainable:,} / total: {total:,} "
                f"({100.0 * trainable / max(total, 1):.1f}%)")

    encoder_total = sum(1 for _ in model.encoder.parameters())
    encoder_trainable = sum(1 for p in model.encoder.parameters() if p.requires_grad)
    partially_frozen = 0 < encoder_trainable < encoder_total
    ddp_find_unused = partially_frozen if ddp_find_unused_cli is None else bool(ddp_find_unused_cli)
    logger.info(f"DDP — find_unused_parameters={ddp_find_unused} "
                f"(encoder trainable {encoder_trainable}/{encoder_total})")

    if _is_main_process():
        save_json(
            {
                "exp_name": exp_name, "dataset": dataset, "data_path": data_path,
                "temporal_mode": temporal_mode, "window_size": window_size,
                "fps": fps, "frame_stride": frame_stride, "img_size": img_size,
                "model_name": model_name, "encoder_pretrained": encoder_pretrained,
                "freeze_encoder": freeze_encoder, "encoder_ckpt": encoder_ckpt,
                "unfreeze_blocks": unfreeze_blocks, "label_type": label_type,
                "border_padding": border_padding if dataset == "sages" else None,
                "epochs": epochs, "batch_size": batch_size, "num_workers": num_workers,
                "lr": lr, "encoder_lr": encoder_lr, "weight_decay": weight_decay,
                "no_amp": no_amp, "early_stopping_patience": early_stopping_patience,
                "gradient_accumulation_steps": grad_accum, "pos_weight": pos_weight,
                "n_classes": n_classes, "d_model": d_model, "nhead": nhead,
                "num_layers": num_layers, "dim_ff": dim_ff, "dropout": dropout,
                "K": K, "pe_max_len": pe_max_len, "mlp_hidden": mlp_hidden,
                "pe_type": pe_type, "mult": mult, "seed": seed,
                "ddp_world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "ddp_timeout": ddp_timeout, "ddp_bucket_cap_mb": ddp_bucket_cap_mb,
                "ddp_find_unused_parameters": ddp_find_unused,
            },
            path_join(exp_dir, "config.json"),
        )

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    training_args = TrainingArguments(
        output_dir=best_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        learning_rate=lr,
        weight_decay=weight_decay,
        gradient_accumulation_steps=grad_accum,
        fp16=(not no_amp) and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="mAP",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=seed,
        ddp_find_unused_parameters=ddp_find_unused,
        ddp_bucket_cap_mb=ddp_bucket_cap_mb,
        ddp_broadcast_buffers=False,
        ddp_timeout=ddp_timeout,
        disable_tqdm=False,
    )

    collator = make_e2e_collator(pe_max_len=pe_max_len)

    logger.info("Building Trainer (DDP init + model wrap if under torchrun)...")
    trainer = SafeTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            LineProgressCallback(log_every=int(_resolve(cfg, args, "log_every", 25))),
        ],
        optimizers=(optimizer, None),
    )

    _barrier("pre-train")
    trainer.train()
    best_ckpt = trainer.state.best_model_checkpoint
    logger.info(f"Best checkpoint: {best_ckpt}")

    _barrier("pre-predict-val")
    val_pred = trainer.predict(test_dataset=val_ds)
    _barrier("post-predict-val")
    test_pred = trainer.predict(test_dataset=test_ds)
    _barrier("post-predict-test")

    if _is_main_process():
        _write_predictions(val_pred, val_ds, val_pred_path, "val", dataset, pred_threshold)
        _write_predictions(test_pred, test_ds, test_pred_path, "test", dataset, pred_threshold)

        val_flat, val_metrics = _evaluate_and_flatten(val_pred_path, "val", pred_threshold)
        test_flat, test_metrics = _evaluate_and_flatten(test_pred_path, "test", pred_threshold)

        print("\n--- Val ---")
        print_fancy_metrics_df(val_metrics)
        print("\n--- Test ---")
        print_fancy_metrics_df(test_metrics)

        save_json({"split": "val", "exp_name": exp_name, "best_model_checkpoint": best_ckpt, **val_flat},
                   best_valid_metrics_path)
        save_json({"split": "test", "exp_name": exp_name, "best_model_checkpoint": best_ckpt, **test_flat},
                   best_test_metrics_path)

    _barrier("final")


if __name__ == "__main__":
    main()
