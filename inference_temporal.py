import os
import time
from os.path import join as path_join
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from transformers import Trainer, TrainingArguments, set_seed
from torchmetrics.classification import MultilabelAveragePrecision

from cvs_datasets.CVS_Temporal_Dataset import TemporalWindowCVSDataset, collate_temporal
from cvs_datasets.Endoscapes_CVS_Temporal_Dataset import EndoscapesTemporalWindowDataset
from evaluate import evaluate_json_file, print_fancy_metrics_df
from utils import save_json
# NOTE: weights/best_perceiver was trained with the gated variant
# (main_temporal_perceiver_concat_gate.py -> PerceiverLiteTemporalGated),
# not the plain PerceiverLiteTemporal these MaskCVS scripts default to.
from models.TemporalPerceiver_concat_gate import PerceiverLiteTemporalGated as PerceiverLiteTemporal


POS_WEIGHT = torch.tensor([6.80, 3.04, 5.44], dtype=torch.float32)


# ============================================================
# Wrapper HF Trainer
# ============================================================
class TemporalPerceiverForTraining(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = PerceiverLiteTemporal(**kwargs)
        self.register_buffer("pos_weight", POS_WEIGHT)

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
def temporal_collator(batch):
    out = collate_temporal(batch)
    out.pop("meta", None)
    out["labels"] = out["y"]
    return out


# ============================================================
# Fast metric
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
# Predictions JSON
# ============================================================
@torch.no_grad()
def make_predictions_json(trainer, dataset, out_path, split_name="test", threshold=0.5):
    pred = trainer.predict(test_dataset=dataset)
    logits = pred.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    labels = pred.label_ids

    probs = 1.0 / (1.0 + np.exp(-logits))
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
                "probs": probs[i].tolist(),
                "pred": preds[i].tolist(),
                "label": labels_bin[i].tolist(),
                "confidence_aware_label": labels[i].tolist(),
            }
        )

    save_json(records, out_path)
    print(f"[OK] Saved json to: {out_path}")


# ============================================================
# Full metrics
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
# Path helpers
# ============================================================
def is_checkpoint_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith((".safetensors", ".bin", ".pt", ".pth"))


def resolve_run_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    if os.path.isfile(model_path):
        parent = os.path.dirname(model_path)
        if os.path.basename(parent).startswith("checkpoint-"):
            return os.path.dirname(parent)
        return parent
    raise FileNotFoundError(f"No existe model_path: {model_path}")


def find_config_json(run_dir: str) -> str:
    direct = path_join(run_dir, "config.json")
    if os.path.isfile(direct):
        return direct

    found = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f == "config.json":
                found.append(path_join(root, f))

    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(f"Se encontraron múltiples config.json en {run_dir}: {found}")

    raise FileNotFoundError(f"No encontré config.json dentro de: {run_dir}")


def find_checkpoint_file(model_path: str) -> str:
    if is_checkpoint_file(model_path):
        return model_path

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"No existe model_path: {model_path}")

    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.pt",
        "checkpoint.pt",
        "model.pt",
        "best_model.pt",
        "model.pth",
        "weights.pt",
        "weights.pth",
    ]

    for name in candidates:
        p = path_join(model_path, name)
        if os.path.isfile(p):
            return p

    ckpt_dirs = []
    for name in os.listdir(model_path):
        full = path_join(model_path, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            ckpt_dirs.append(full)

    if ckpt_dirs:
        def _ckpt_num(p):
            try:
                return int(os.path.basename(p).split("-")[-1])
            except Exception:
                return -1

        ckpt_dirs = sorted(ckpt_dirs, key=_ckpt_num)
        best_ckpt_dir = ckpt_dirs[-1]

        for name in candidates:
            p = path_join(best_ckpt_dir, name)
            if os.path.isfile(p):
                return p

        found = [
            path_join(best_ckpt_dir, f)
            for f in os.listdir(best_ckpt_dir)
            if f.endswith((".safetensors", ".bin", ".pt", ".pth"))
        ]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise RuntimeError(f"Múltiples checkpoints en {best_ckpt_dir}: {found}")

    found = []
    for root, _, files in os.walk(model_path):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt", ".pth")):
                low = f.lower()
                # NOTE: "training_args" added to the exclusion list below — the
                # original upstream list missed it, so training_args.bin (a
                # non-weights HF TrainingArguments pickle) got mistaken for a
                # second checkpoint candidate whenever the real checkpoint sat
                # one level deeper than the run dir (e.g. run_dir/best/checkpoint-N/,
                # as in weights/perceiver_endoscapes, vs. weights/best_perceiver's
                # flatter run_dir/checkpoint-N/), causing a false "multiple
                # checkpoints" error.
                if any(tok in low for tok in ["optimizer", "scheduler", "scaler", "trainer_state", "rng", "training_args"]):
                    continue
                found.append(path_join(root, f))

    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(f"Se encontraron múltiples checkpoints candidatos en {model_path}: {found}")

    raise FileNotFoundError(f"No encontré checkpoint válido dentro de: {model_path}")


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# Checkpoint helpers
# ============================================================
def clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module."):]

        # wrapper completo
        if nk.startswith("model.model."):
            nk = nk[len("model."):]
        elif nk.startswith("model."):
            nk = nk[len("model."):]

        cleaned[nk] = v
    return cleaned


def extract_state_dict(obj):
    if not isinstance(obj, dict):
        raise RuntimeError(f"Formato de checkpoint no soportado: {type(obj)}")

    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]
    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]

    return obj


def load_checkpoint_state_dict(ckpt_file: str):
    if ckpt_file.endswith(".safetensors"):
        state_dict = load_safetensors(ckpt_file, device="cpu")
    else:
        obj = torch.load(ckpt_file, map_location="cpu")
        state_dict = extract_state_dict(obj)

    return clean_state_dict_keys(state_dict)


def inspect_state_dict_shapes(state_dict: dict):
    info = {
        "pe_len": None,
        "d_model_from_pe": None,
        "num_classes": None,
        "K_from_latents": None,
    }

    if "pos_enc.pe" in state_dict:
        pe = state_dict["pos_enc.pe"]
        if pe.ndim == 3:
            info["pe_len"] = int(pe.shape[1])
            info["d_model_from_pe"] = int(pe.shape[2])

    # PerceiverLite suele tener latents [K, d_model] o [1, K, d_model]
    for key in ["latents", "latent", "latent_tokens"]:
        if key in state_dict:
            lat = state_dict[key]
            if lat.ndim == 2:
                info["K_from_latents"] = int(lat.shape[0])
            elif lat.ndim == 3:
                info["K_from_latents"] = int(lat.shape[1])
            break

    for key in ["head.weight", "classifier.weight", "fc.weight"]:
        if key in state_dict and state_dict[key].ndim == 2:
            info["num_classes"] = int(state_dict[key].shape[0])
            break

    return info


def load_model_weights(wrapper_model: nn.Module, state_dict: dict, device: torch.device):
    print("[INFO] Loading state_dict into model...")
    missing, unexpected = wrapper_model.model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] Missing keys: {len(missing)}")
    if missing:
        print("       ", missing[:20])

    print(f"[INFO] Unexpected keys: {len(unexpected)}")
    if unexpected:
        print("       ", unexpected[:20])

    wrapper_model.to(device)
    wrapper_model.eval()
    return list(missing), list(unexpected)


# ============================================================
# Latency benchmark (same methodology as benchmark_latency.py, inlined here
# so it runs as part of normal inference and gets logged with the metrics)
# ============================================================
@torch.no_grad()
def bench_latency(model: nn.Module, x, key_index, positions, warmup: int, iters: int,
                   device: torch.device) -> dict:
    """Batch=1 latency of one forward pass, on the real window already
    loaded for this run. Weight values don't affect GPU kernel latency
    (only op shapes/dtypes do), so this reuses whatever's already loaded
    rather than needing a separate benchmarking pass. One sync boundary per
    iteration (isolated per-call latency, not pipelined/amortized across
    iterations). Times with torch.cuda.Event on CUDA (lower overhead / less
    measurement noise than time.perf_counter() + synchronize(), same
    semantics) and falls back to perf_counter on CPU.

    Runs under fp16 autocast when on CUDA, matching this script's own
    Trainer (built with fp16=torch.cuda.is_available() — always on with
    CUDA, no opt-out) — a plain, unwrapped call here would silently
    benchmark fp32 while the actual evaluate()/predict() calls above run
    fp16, understating real deployed latency."""
    def step():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                model(x, key_index, positions)
        else:
            model(x, key_index, positions)

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    if device.type == "cuda":
        for _ in range(iters):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            step()
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            step()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times_ms)
    return {
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "fps": float(1000.0 / t.mean()),
        "params_M": float(sum(p.numel() for p in model.parameters()) / 1e6),
        "window_size": int(x.shape[1]),
        "warmup": warmup,
        "iters": iters,
    }


# ============================================================
# Config helpers
# ============================================================
def get_cfg(cfg: dict, key: str, default=None):
    return cfg[key] if key in cfg else default


def build_model_from_config(cfg: dict, state_info: dict):
    pe_max_len = int(get_cfg(cfg, "pe_max_len", 512))
    if state_info["pe_len"] is not None:
        pe_max_len = int(state_info["pe_len"])

    d_model = int(get_cfg(cfg, "d_model", 256))
    if state_info["d_model_from_pe"] is not None:
        d_model = int(state_info["d_model_from_pe"])

    n_classes = 3
    if state_info["num_classes"] is not None:
        n_classes = int(state_info["num_classes"])

    K = int(get_cfg(cfg, "K", 8))
    if state_info["K_from_latents"] is not None:
        K = int(state_info["K_from_latents"])

    model_kwargs = {
        "d_in": int(get_cfg(cfg, "d_in", 1024)),
        "d_model": d_model,
        "nhead": int(get_cfg(cfg, "nhead", 4)),
        "num_layers": int(get_cfg(cfg, "num_layers", 2)),
        "dim_ff": int(get_cfg(cfg, "dim_ff", 1024)),
        "dropout": float(get_cfg(cfg, "dropout", 0.2)),
        "n_classes": n_classes,
        "K": K,
        "pe_type": get_cfg(cfg, "pe_type", "sinusoidal"),
        "pe_max_len": pe_max_len,
        "mlp_hidden": int(get_cfg(cfg, "mlp_hidden", 256)),
        "use_gate": bool(get_cfg(cfg, "use_gate", True)),
        "readout": get_cfg(cfg, "readout", "keyframe"),
        "mult": bool(get_cfg(cfg, "mult", False)),
    }

    print("[INFO] Model kwargs resolved:")
    print(json.dumps(model_kwargs, indent=4))

    return TemporalPerceiverForTraining(**model_kwargs)


def build_dataset_from_config(split_path: str, cfg: dict, dataset: str = "sages"):
    """Build the eval dataset for --split_path, dispatching on --dataset.

    NOTE: this used to always build a SAGES TemporalWindowCVSDataset
    regardless of which dataset the checkpoint was actually trained on.
    Endoscapes windows have different keyframe/stride geometry (annotated
    frames vs. SAGES' fixed 150-frame-id-modulo rule) — feeding Endoscapes
    features through the SAGES windower produces bogus temporal `positions`
    that overflow the model's positional-embedding table and crash with a
    CUDA "gather kernel index out of bounds" assertion deep in the forward
    pass. Route Endoscapes checkpoints through EndoscapesTemporalWindowDataset
    instead (same TemporalSample/collate_temporal contract, different
    windowing rule) via --dataset endoscapes.
    """
    features_dir = path_join(split_path, "features")
    if not os.path.isdir(features_dir):
        raise FileNotFoundError(f"No existe carpeta features: {features_dir}")

    # Both training scripts (main_temporal_perceiver_concat_gate*.py) build
    # their own val/test datasets with label_type="hard", reserving cfg's
    # (training-split) label_type — typically "ca" — for the train split
    # only. Match that here rather than reusing cfg's label_type for eval,
    # so the "label"/Brier numbers this script reports line up with what
    # the training run's own held-out evaluation actually used.
    if dataset == "endoscapes":
        split_name = os.path.basename(os.path.normpath(split_path))
        data_path = get_cfg(
            cfg, "data_path", "/home/scanar/endovis/Datasets/endoscapes_2023/endoscapes"
        )
        return EndoscapesTemporalWindowDataset(
            features_root=features_dir,
            annotation_json_path=path_join(data_path, split_name, "annotation_ds_coco.json"),
            mode=get_cfg(cfg, "temporal_mode", "online"),
            window_size=int(get_cfg(cfg, "window_size", 5)),
            target_fps=get_cfg(cfg, "fps", 1.0),
            frame_stride=get_cfg(cfg, "frame_stride", None),
            label_type="hard",
            ft_type=get_cfg(cfg, "ft_type", "image_embeddings"),
        )

    return TemporalWindowCVSDataset(
        data_root=features_dir,
        mode=get_cfg(cfg, "temporal_mode", "online"),
        window_size=int(get_cfg(cfg, "window_size", 5)),
        target_fps=int(get_cfg(cfg, "fps", 1)),
        ft_type="prelogits",
        label_type="hard",
    )


# ============================================================
# Parser
# ============================================================
def build_parser():
    p = argparse.ArgumentParser("Temporal Perceiver inference auto")
    p.add_argument("--model_path", type=str, required=True, help="Carpeta del run o checkpoint directo")
    p.add_argument("--split_path", type=str, required=True, help="Ruta directa al split, ej: .../test")
    p.add_argument("--out_dir", type=str, default=None, help="Directorio de salida opcional")
    p.add_argument("--dataset", type=str, default="sages", choices=["sages", "endoscapes"],
                   help="Dataset format: 'sages' (fixed-stride keyframes) or "
                        "'endoscapes' (annotated-frame keyframes, COCO JSON labels).")
    p.add_argument("--no_bench_latency", action="store_true",
                   help="Skip the batch=1 latency micro-benchmark logged alongside the metrics.")
    p.add_argument("--latency_warmup", type=int, default=20)
    p.add_argument("--latency_iters", type=int, default=200)
    return p


# ============================================================
# Main
# ============================================================
def run_inference(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.model_path)
    config_path = find_config_json(run_dir)
    ckpt_file = find_checkpoint_file(args.model_path)

    cfg = load_json(config_path)
    seed = int(get_cfg(cfg, "seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] config_path: {config_path}")
    print(f"[INFO] checkpoint: {ckpt_file}")

    state_dict = load_checkpoint_state_dict(ckpt_file)
    state_info = inspect_state_dict_shapes(state_dict)
    print(f"[INFO] state_info: {state_info}")

    split_name = os.path.basename(os.path.normpath(args.split_path))
    run_name = get_cfg(cfg, "wandb_run_name", get_cfg(cfg, "run_name", os.path.basename(os.path.normpath(run_dir))))

    if args.out_dir is not None:
        output_dir = args.out_dir
    else:
        output_dir = path_join(run_dir, f"infer__{split_name}")

    pred_dir = path_join(output_dir, "preds")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    dataset = build_dataset_from_config(args.split_path, cfg, dataset=args.dataset)

    effective_pe_len = int(state_info["pe_len"]) if state_info["pe_len"] is not None else int(get_cfg(cfg, "pe_max_len", 512))
    window_size = int(get_cfg(cfg, "window_size", 5))
    if effective_pe_len < window_size:
        raise ValueError(f"pe_max_len efectivo ({effective_pe_len}) must be >= window_size ({window_size})")

    model = build_model_from_config(cfg, state_info)
    missing, unexpected = load_model_weights(model, state_dict, device)

    print(f"[INFO] split_path: {args.split_path}")
    print(f"[INFO] dataset size: {len(dataset)}")
    print(f"[INFO] output_dir: {output_dir}")

    hf_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_eval_batch_size=int(get_cfg(cfg, "batch_size", 64)),
        dataloader_num_workers=int(get_cfg(cfg, "num_workers", 16)),
        report_to=[],
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=hf_args,
        eval_dataset=dataset,
        data_collator=temporal_collator,
        compute_metrics=compute_metrics,
    )

    fast_metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=split_name)
    save_json(fast_metrics, path_join(output_dir, f"{split_name}_fast_metrics.json"))

    threshold = float(get_cfg(cfg, "pred_threshold", 0.5))
    pred_path = path_join(pred_dir, f"{split_name}_predictions.json")
    make_predictions_json(trainer, dataset, pred_path, split_name=split_name, threshold=threshold)

    flat, full_metrics = _evaluate_and_flatten(pred_path, split_name, threshold)

    print(f"\n--- {split_name.upper()} ---")
    print_fancy_metrics_df(full_metrics)

    latency = None
    if not args.no_bench_latency:
        lat_batch = temporal_collator([dataset[0]])
        lat_x = lat_batch["x"].to(device)
        lat_key_index = lat_batch["key_index"].to(device)
        lat_positions = lat_batch["positions"].to(device)
        latency = bench_latency(model.model, lat_x, lat_key_index, lat_positions,
                                 args.latency_warmup, args.latency_iters, device)
        print(f"\n--- LATENCY (batch=1, {device}) ---")
        print(f"{latency['mean_ms']:.3f} +/- {latency['std_ms']:.3f} ms "
              f"(median {latency['median_ms']:.3f}, p95 {latency['p95_ms']:.3f}) "
              f"-> {latency['fps']:.1f} FPS | params={latency['params_M']:.2f}M "
              f"| window_size={latency['window_size']}")
        full_metrics["latency"] = latency

    save_json(flat, path_join(output_dir, f"{split_name}_flat_metrics.json"))
    save_json(full_metrics, path_join(output_dir, f"{split_name}_full_metrics.json"))
    save_json(
        {
            "run_dir": run_dir,
            "config_path": config_path,
            "model_path_arg": args.model_path,
            "resolved_checkpoint_file": ckpt_file,
            "split_path": args.split_path,
            "split_name": split_name,
            "device": str(device),
            "threshold": threshold,
            "state_info": state_info,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "latency": latency,
        },
        path_join(output_dir, "run_info.json"),
    )

    print(f"\n[OK] Predictions: {pred_path}")
    print(f"[OK] Fast metrics: {path_join(output_dir, f'{split_name}_fast_metrics.json')}")
    print(f"[OK] Full metrics: {path_join(output_dir, f'{split_name}_full_metrics.json')}")


if __name__ == "__main__":
    parser = build_parser()
    run_inference(parser)