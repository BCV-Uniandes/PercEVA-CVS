import argparse
import json
import time
from os.path import join as path_join

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Lambda, Resize, ToTensor, Normalize

from feature_extractor.extract_ft import load_timm_model, extract_pre_logits
from cvs_datasets.CVS_Temporal_Dataset import TemporalWindowCVSDataset, collate_temporal
from cvs_datasets.Endoscapes_CVS_Temporal_Dataset import EndoscapesTemporalWindowDataset
from cvs_datasets.CVS_TemporalImage_Dataset import TemporalImageWindowDataset
from cvs_datasets.Endoscapes_CVS_Dataset import EndoscapesTemporalImageWindowDataset
from utils import load_yaml
import inference_temporal as it
from main_scripts import main_e2e as me

MODEL_NAME = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"


def count_params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def bench(fn, warmup: int, iters: int, device: torch.device) -> dict:
    """Batch=1 per-call latency: one sync boundary per iteration (so calls
    stay isolated, not pipelined/amortized across iterations — see the
    README's Latency benchmark section for why that distinction matters for
    this repo's online, one-window-at-a-time deployment model). Times with
    torch.cuda.Event on CUDA (lower overhead / less measurement noise than
    time.perf_counter() + synchronize(), same semantics) and falls back to
    perf_counter on CPU, where cuda.Event isn't available.
    """
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    if device.type == "cuda":
        for _ in range(iters):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times_ms)
    return {
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "fps": float(1000.0 / t.mean()),
    }


def bench_encoder(args, device) -> dict:
    tfm = Compose([
        Lambda(lambda x: x.convert("RGB")),
        Resize((args.img_size, args.img_size)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = tfm(Image.open(args.image_path)).unsqueeze(0).to(device)

    model = load_timm_model(MODEL_NAME, 3, args.encoder_weights, device)
    model.eval()
    params_m = count_params_m(model)

    out = {}
    for precision, use_amp in (("fp16 (AMP)", True), ("fp32", False)):
        if args.no_amp and use_amp:
            continue

        def step():
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    extract_pre_logits(model, img)
            else:
                extract_pre_logits(model, img)

        result = bench(step, args.warmup, args.iters, device)
        result["params_M"] = params_m
        out[precision] = result

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def bench_perceiver(args, device) -> dict:
    """NOTE: inference_temporal.py's real Trainer runs with
    fp16=torch.cuda.is_available() — always on when CUDA is available, no
    --no_amp-style opt-out for the perceiver at all — so 'fp16 (AMP)' below
    is the number that actually matches deployment. 'fp32' is included only
    for a complete comparison, mirroring bench_encoder/bench_e2e; it isn't
    reachable through inference_temporal.py itself on a CUDA machine.
    """
    run_dir = it.resolve_run_dir(args.perceiver_path)
    config_path = it.find_config_json(run_dir)
    ckpt_file = it.find_checkpoint_file(args.perceiver_path)
    cfg = it.load_json(config_path)

    state_dict = it.load_checkpoint_state_dict(ckpt_file)
    state_info = it.inspect_state_dict_shapes(state_dict)
    wrapper = it.build_model_from_config(cfg, state_info)
    it.load_model_weights(wrapper, state_dict, device)
    wrapper.eval()
    params_m = count_params_m(wrapper.model)

    # Real window, same code path as inference_temporal.py's build_dataset_from_config.
    dataset = it.build_dataset_from_config(args.split_path, cfg, dataset=args.dataset)
    batch = collate_temporal([dataset[0]])
    x = batch["x"].to(device)
    key_index = batch["key_index"].to(device)
    positions = batch["positions"].to(device)

    out = {}
    for precision, use_amp in (("fp16 (AMP)", True), ("fp32", False)):
        if args.no_amp and use_amp:
            continue
        if use_amp and device.type != "cuda":
            continue

        def step():
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    wrapper.model(x, key_index, positions)
            else:
                wrapper.model(x, key_index, positions)

        result = bench(step, args.warmup, args.iters, device)
        result["params_M"] = params_m
        result["window_size"] = int(x.shape[1])
        out[precision] = result

    return out


def bench_e2e(args, device) -> dict:
    """Real jointly-trained E2E model: one fused forward pass over a raw
    (unfeaturized) image window, using the exact architecture/warm-start
    config main_e2e.py trains with. No trained E2E checkpoint needs to exist
    yet — latency depends on the compute graph (architecture + which
    encoder blocks are unfrozen), not on the specific trained weight
    values, so warm-start-only weights measure the same latency a fully
    trained checkpoint would.
    """
    cfg = load_yaml(args.e2e_config)
    img_size = int(cfg.get("img_size", 448))

    tfm = Compose([
        Lambda(lambda img: img.convert("RGB")),
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data_path = args.e2e_data_path or cfg.get("data_path")
    if args.dataset == "endoscapes":
        ds = EndoscapesTemporalImageWindowDataset(
            split_path=path_join(data_path, "test"),
            mode=cfg.get("temporal_mode", "online"),
            window_size=int(cfg.get("window_size", 15)),
            frame_stride=cfg.get("frame_stride"),
            target_fps=float(cfg.get("fps", 1.0)),
            transform=tfm,
            label_type=cfg.get("label_type", "hard"),
        )
    else:
        ds = TemporalImageWindowDataset(
            frames_path=path_join(data_path, "test", "frames"),
            labels_path=path_join(data_path, "test", "labels"),
            mode=cfg.get("temporal_mode", "online"),
            window_size=int(cfg.get("window_size", 15)),
            target_fps=float(cfg.get("fps", 1.0)),
            label_type="hard",
            transform=tfm,
            border_padding=cfg.get("border_padding", "keyframe"),
        )

    collator = me.make_e2e_collator(pe_max_len=int(cfg.get("pe_max_len", 2048)))
    batch = collator([ds[0]])
    pixel_values = batch["pixel_values"].to(device)
    key_index = batch["key_index"].to(device)
    positions = batch["positions"].to(device)

    model = me.E2EPerceiverGated(
        model_name=cfg.get("model_name", MODEL_NAME),
        encoder_pretrained=bool(cfg.get("encoder_pretrained", True)),
        d_model=int(cfg.get("d_model", 1024)),
        nhead=int(cfg.get("nhead", 16)),
        num_layers=int(cfg.get("num_layers", 2)),
        dim_ff=int(cfg.get("dim_ff", 512)),
        dropout=float(cfg.get("dropout", 0.0)),
        n_classes=int(cfg.get("num_classes", 3)),
        K=int(cfg.get("K", 64)),
        mlp_hidden=int(cfg.get("mlp_hidden", 128)),
        mult=bool(cfg.get("mult", False)),
        pe_type=cfg.get("pe_type", "learned"),
        pe_max_len=int(cfg.get("pe_max_len", 2048)),
    ).to(device)

    ckpt = args.e2e_ckpt or cfg.get("encoder_ckpt")
    if ckpt:
        sd = me._load_state_dict(ckpt)
        sd = me._strip_prefix(me._strip_prefix(sd, "model."), "backbone.")
        model.encoder.load_state_dict(sd, strict=False)

    unfreeze_blocks = cfg.get("unfreeze_blocks")
    if unfreeze_blocks is not None:
        model.unfreeze_encoder_last_n_blocks(int(unfreeze_blocks))

    model.eval()
    params_m = count_params_m(model)

    out = {}
    for precision, use_amp in (("fp16 (AMP)", True), ("fp32", False)):
        if args.no_amp and use_amp:
            continue
        if use_amp and device.type != "cuda":
            continue

        def step():
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    model(pixel_values=pixel_values, key_index=key_index, positions=positions)
            else:
                model(pixel_values=pixel_values, key_index=key_index, positions=positions)

        result = bench(step, args.warmup, args.iters, device)
        result["params_M"] = params_m
        result["window_size"] = int(pixel_values.shape[1])
        out[precision] = result

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def print_latex_table(encoder: dict, perceiver: dict, two_stage_total: dict, e2e: dict, gpu_name: str):
    enc = encoder.get("fp16 (AMP)", next(iter(encoder.values())))
    print("\n% --- paste into paper.tex ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(rf"\caption{{Measured wall-clock latency (batch=1, {gpu_name}). "
          r"Stage~1 uses fp16 autocast, matching the deployed feature-extraction pipeline.}")
    print(r"\label{tab:latency}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Method & Params (M) & Latency (ms) & FPS \\")
    print(r"\midrule")
    print(rf"EVA-02 encoder only (Stage 1) & {enc['params_M']:.1f} & "
          rf"{enc['mean_ms']:.2f}~$\pm$~{enc['std_ms']:.2f} & {enc['fps']:.1f} \\")
    perc = perceiver.get("fp16 (AMP)", next(iter(perceiver.values())))
    print(rf"CVSPerceiver only (Stage 2) & {perc['params_M']:.2f} & "
          rf"{perc['mean_ms']:.2f}~$\pm$~{perc['std_ms']:.2f} & {perc['fps']:.1f} \\")
    print(r"\midrule")
    print(rf"Two-stage total (sequential) & -- & "
          rf"{two_stage_total['mean_ms']:.2f} & {two_stage_total['fps']:.1f} \\")
    if e2e is not None:
        e2e_row = e2e.get("fp16 (AMP)", next(iter(e2e.values())))
        print(rf"End-to-end (jointly trained) & {e2e_row['params_M']:.1f} & "
              rf"{e2e_row['mean_ms']:.2f}~$\pm$~{e2e_row['std_ms']:.2f} & {e2e_row['fps']:.1f} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def parse_args():
    p = argparse.ArgumentParser("PercEVA-CVS latency benchmark")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--no_amp", action="store_true", help="Skip the fp16/AMP encoder timing, fp32 only.")

    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--image_path", type=str, default=None,
                    help="Any single extracted frame, e.g. "
                         "data/SAGES_2024/test/frames/<video-uuid>/frame_0000.jpg. Required.")
    p.add_argument("--encoder_weights", type=str, default="weights/best_eva02_enc_cvs.pt")

    p.add_argument("--perceiver_path", type=str, default="weights/best_perceiver")
    p.add_argument("--split_path", type=str, default="data/SAGES_2024/test")
    p.add_argument("--dataset", type=str, default="sages", choices=["sages", "endoscapes"])

    p.add_argument("--no_e2e", action="store_true",
                    help="Skip the jointly-trained E2E model benchmark (builds EVA-02 a "
                         "2nd time from scratch, so it roughly doubles startup cost).")
    p.add_argument("--e2e_config", type=str, default=None,
                    help="Defaults to configs/end2end_sages.yaml or "
                         "configs/end2end_endoscapes.yaml, matching --dataset.")
    p.add_argument("--e2e_ckpt", type=str, default=None,
                    help="Encoder warm-start checkpoint. Defaults to the e2e_config's own "
                         "encoder_ckpt. No trained end-to-end checkpoint is required — "
                         "latency depends on the compute graph, not the trained weight values.")
    p.add_argument("--e2e_data_path", type=str, default=None,
                    help="Defaults to the e2e_config's own data_path.")

    p.add_argument("--out_json", type=str, default="benchmark_latency_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.image_path is None:
        raise SystemExit("--image_path is required — pass any single extracted frame, "
                          "e.g. data/SAGES_2024/test/frames/<video-uuid>/frame_0000.jpg")
    if args.e2e_config is None:
        args.e2e_config = (
            "configs/end2end_sages.yaml" if args.dataset == "sages"
            else "configs/end2end_endoscapes.yaml"
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    print(f"[INFO] device: {device} ({gpu_name})")
    print(f"[INFO] warmup={args.warmup} iters={args.iters}")

    print("\n[INFO] Benchmarking encoder only (Stage 1)...")
    encoder_results = bench_encoder(args, device)
    for precision, r in encoder_results.items():
        print(f"  [{precision}] {r['mean_ms']:.3f} +/- {r['std_ms']:.3f} ms "
              f"(median {r['median_ms']:.3f}, p95 {r['p95_ms']:.3f}) -> {r['fps']:.1f} FPS "
              f"| params={r['params_M']:.1f}M")

    print("\n[INFO] Benchmarking perceiver only (Stage 2, from cached features)...")
    perceiver_result = bench_perceiver(args, device)
    for precision, r in perceiver_result.items():
        print(f"  [{precision}] {r['mean_ms']:.3f} +/- {r['std_ms']:.3f} ms "
              f"(median {r['median_ms']:.3f}, p95 {r['p95_ms']:.3f}) -> {r['fps']:.1f} FPS "
              f"| params={r['params_M']:.2f}M | window_size={r['window_size']}")

    # Both halves of the sum at the precision inference_temporal.py actually
    # deploys with (fp16, always-on when CUDA is available).
    enc_for_sum = encoder_results.get("fp16 (AMP)", next(iter(encoder_results.values())))
    perc_for_sum = perceiver_result.get("fp16 (AMP)", next(iter(perceiver_result.values())))
    two_stage_mean = enc_for_sum["mean_ms"] + perc_for_sum["mean_ms"]
    two_stage_total = {"mean_ms": two_stage_mean, "fps": 1000.0 / two_stage_mean}
    print(f"\n[INFO] Two-stage total (encoder + perceiver, summed): "
          f"{two_stage_total['mean_ms']:.3f} ms -> {two_stage_total['fps']:.1f} FPS")

    e2e_result = None
    if not args.no_e2e:
        print(f"\n[INFO] Benchmarking e2e (jointly-trained model, config={args.e2e_config})...")
        e2e_result = bench_e2e(args, device)
        for precision, r in e2e_result.items():
            print(f"  [{precision}] {r['mean_ms']:.3f} +/- {r['std_ms']:.3f} ms "
                  f"(median {r['median_ms']:.3f}, p95 {r['p95_ms']:.3f}) -> {r['fps']:.1f} FPS "
                  f"| params={r['params_M']:.1f}M | window_size={r['window_size']}")

    results = {
        "gpu": gpu_name,
        "warmup": args.warmup,
        "iters": args.iters,
        "encoder": encoder_results,
        "perceiver": perceiver_result,
        "two_stage_total": two_stage_total,
        "e2e": e2e_result,
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved: {args.out_json}")

    print_latex_table(encoder_results, perceiver_result, two_stage_total, e2e_result, gpu_name)


if __name__ == "__main__":
    main()
