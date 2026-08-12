import os
import sys
import logging
import argparse
from os.path import join as path_join
from concurrent.futures import ThreadPoolExecutor

from torchvision.transforms import Compose, Lambda, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

import torch
import timm
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvs_datasets.CVS_Dataset import CVSAllFrames, collate_fn_all
from cvs_datasets.Endoscapes_CVS_Dataset import EndoscapesAllFrames


# ─── Model ────────────────────────────────────────────────────────────────────
def load_timm_model(model_name: str, num_classes: int, pt_path: str, device: torch.device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    if pt_path.endswith(".safetensors"):
        state = load_safetensors(pt_path, device="cpu")
        sd = state
    else:
        state = torch.load(pt_path, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    logging.info(f"Weights loaded | missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


@torch.inference_mode()
def extract_pre_logits(model, x):
    feats = model.forward_features(x)
    pre = model.forward_head(feats, pre_logits=True)
    if pre.ndim > 2:
        pre = pre.view(pre.shape[0], -1)
    return pre, feats   # [B, D], [B, N+1, D]


# ─── Extraction + save ────────────────────────────────────────────────────────
@torch.inference_mode()
def process_split(split_root, output_dir, model, transform, device,
                  no_amp, batch_size, num_workers, dataset="sages"):
    """
    Saves one .pth per frame:
      image_embeddings : Tensor[1024]       float16  — ALL frames
      patch_embeddings : Tensor[N+1, 1024]  float16  — keyframes only, else None
      labels_hard      : Tensor[3]          float32  — annotated frames, else None
      labels_ca        : Tensor[3]          float32  — annotated frames, else None

    SAGES:       patch_embeddings saved for frame_id % 30 == 0
    Endoscapes:  patch_embeddings saved for annotated frames (those with CVS labels)
    """
    split_name = os.path.basename(split_root)

    if dataset == "endoscapes":
        ds = EndoscapesAllFrames(split_root, transform=transform)
        # output dirs named by video_name parsed from flat filenames
        video_names = {path_join(split_root, p.split("/")[-1]).split("_")[0]
                       for p in ds.image_paths}
        # Use the parsed video names from image paths directly
        from cvs_datasets.Endoscapes_CVS_Dataset import parse_endoscapes_filename
        seen_videos = set()
        for p in ds.image_paths:
            vname, _ = parse_endoscapes_filename(p)
            if vname not in seen_videos:
                os.makedirs(os.path.join(output_dir, vname), exist_ok=True)
                seen_videos.add(vname)
    else:
        frames_root = os.path.join(split_root, "frames")
        labels_root = os.path.join(split_root, "labels")
        ds = CVSAllFrames(frames_root, labels_root, transform=transform, frame_stride=30)
        for p in ds.image_paths:
            os.makedirs(os.path.join(output_dir, p.split("/")[-2]), exist_ok=True)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_all,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    logging.info(f"[{split_name}] {len(ds)} frames | dataset={dataset}")

    amp_enabled = (device.type == "cuda") and (not no_amp)
    total_saved = 0
    total_skipped = 0

    def _save(payload, path):
        torch.save(payload, path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for batch in tqdm(loader, desc=f"[{split_name}]"):
            paths_in_batch = [
                os.path.join(output_dir, vname, f"frame_{int(fid):06d}.pth")
                for vname, fid in zip(batch["video_name"], batch["frame_id"])
            ]
            if all(os.path.exists(p) for p in paths_in_batch):
                total_skipped += len(paths_in_batch)
                continue

            x = batch["pixel_values"].to(device, non_blocking=True)

            if amp_enabled:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pre, feats = extract_pre_logits(model, x)
            else:
                pre, feats = extract_pre_logits(model, x)

            pre_h   = pre.half().cpu()    # [B, 1024]
            feats_h = feats.half().cpu()  # [B, N+1, 1024]

            is_annotated = batch["is_annotated"]  # BoolTensor[B]

            for i, (vname, fid) in enumerate(zip(batch["video_name"], batch["frame_id"])):
                fid = int(fid)
                annot = bool(is_annotated[i])
                path = paths_in_batch[i]

                if os.path.exists(path):
                    total_skipped += 1
                    continue

                if dataset == "endoscapes":
                    save_patches = annot
                else:
                    save_patches = (fid % 900 == 0)

                payload = {
                    "video_name":        vname,
                    "frame_id":          fid,
                    "image_embeddings":  pre_h[i],
                    "patch_embeddings":  feats_h[i] if save_patches else None,
                    "labels_hard":       batch["labels_hard"][i],
                    "labels_ca":         batch["labels_ca"][i],
                }

                futures.append(pool.submit(_save, payload, path))
                total_saved += 1

            futures = [f for f in futures if not f.done()]

    logging.info(f"[OK] Split '{split_name}': {total_saved} saved, {total_skipped} skipped -> {output_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str,
                   default="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--weights_path", type=str, required=True)
    p.add_argument("--img_size",     type=int, default=448)
    p.add_argument("--no_amp",       action="store_true")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--compile",      action="store_true")
    p.add_argument("--dataset",      type=str, default="sages",
                   choices=["sages", "endoscapes"],
                   help="Dataset format: 'sages' (frames/+labels/ subdirs) or "
                        "'endoscapes' (flat split folder with COCO JSON).")
    p.add_argument("--data_path",    type=str, default=None,
                   help="Root data directory containing train/, val/, test/ splits. "
                        "Required for endoscapes; defaults to built-in SAGES path.")
    p.add_argument("--output_root",  type=str, default=None,
                   help="Root output directory. Defaults to <data_path>/<split>/features.")
    p.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                   help="Which splits to process (default: train val test).")
    return p.parse_args()


def main(data_dir, split, output_dir, model, tfm, device, args):
    split_dir = path_join(data_dir, split)
    assert os.path.exists(split_dir), f"Split dir not found: {split_dir}"
    logging.info(f"Processing split: {split} | {split_dir} -> {output_dir}")

    process_split(split_dir, output_dir, model, tfm, device,
                  args.no_amp, args.batch_size, args.num_workers,
                  dataset=args.dataset)


if __name__ == "__main__":
    this_dir  = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(path_join(parent_dir, "get_ft.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    args = parse_args()

    if args.data_path is None:
        if args.dataset == "endoscapes":
            raise ValueError("--data_path is required for --dataset endoscapes")
        data_dir = path_join(parent_dir, "data", "SAGES_2024")
    else:
        data_dir = args.data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    tfm = Compose([
        Lambda(lambda x: x.convert("RGB")),
        Resize((args.img_size, args.img_size)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    model = load_timm_model(args.model_name, args.num_classes, args.weights_path, device)
    if args.compile and device.type == "cuda":
        logging.info("Compiling model with torch.compile ...")
        model = torch.compile(model)

    for split in args.splits:
        if args.output_root is not None:
            output_dir = path_join(args.output_root, split, "features")
        else:
            output_dir = path_join(data_dir, split, "features")
        os.makedirs(output_dir, exist_ok=True)
        main(data_dir, split, output_dir, model, tfm, device, args)

    print("Done!")
