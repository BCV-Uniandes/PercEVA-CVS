import os
import glob
import logging
from os.path import join as path_join
from dataclasses import dataclass
from statistics import mode
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

KEYFRAME_INTERVAL = 150  # 5 s @ 30 fps
ORIG_FPS = 30


@dataclass
class TemporalImageSample:
    pixel_values: torch.Tensor   # [T, C, H, W]
    y: torch.Tensor              # [3]
    key_index: int
    valid_mask: torch.Tensor     # [T] bool
    meta: Dict[str, Any]


def _build_image_index(frames_path: str):
    """{ video_name: { "frame_ids": sorted list, "id2path": {fid: path} } }"""
    videos: Dict[str, Dict] = {}
    pattern = path_join(frames_path, "**", "*.jpg")
    for img_path in glob.glob(pattern, recursive=True):
        video = img_path.split("/")[-2]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        try:
            frame_id = int(fname.split("_")[-1])
        except (ValueError, IndexError):
            continue
        if video not in videos:
            videos[video] = {"frame_ids": [], "id2path": {}}
        videos[video]["frame_ids"].append(frame_id)
        videos[video]["id2path"][frame_id] = img_path
    for v in videos:
        videos[v]["frame_ids"] = sorted(set(videos[v]["frame_ids"]))
    return videos


class TemporalImageWindowDataset(Dataset):
    """Temporal window dataset over raw image frames (end-to-end training).

    Mirrors the geometry of TemporalWindowCVSDataset but loads .jpg images
    instead of pre-extracted .pth feature files.

    Parameters
    ----------
    frames_path : path to ``{video_name}/*.jpg`` frames.
    labels_path : path to ``{video_name}/frame.csv`` + ``video.csv`` labels.
    mode        : "offline" (centred) or "online" (causal).
    window_size : number of frames in the temporal window.
    target_fps  : effective fps of the window (stride = ORIG_FPS / target_fps).
    label_type  : "hard" (majority vote) or "ca" (confidence-aware).
    transform   : torchvision transform applied to each PIL image.
    border_padding : "keyframe" (default, fills invalid slots with a copy of
                  the keyframe image) or "zero" (fills with an all-zero tensor
                  of the same shape as a transformed frame).
    """

    def __init__(
        self,
        frames_path: str,
        labels_path: str,
        mode: str = "offline",
        window_size: int = 5,
        target_fps: int = 1,
        label_type: str = "ca",
        transform=None,
        border_padding: str = "keyframe",
    ):
        assert mode in ("offline", "online"), f"Unknown mode: {mode}"
        assert label_type in ("hard", "ca"), f"Unknown label_type: {label_type}"
        assert border_padding in ("keyframe", "zero"), (
            f"Unknown border_padding: {border_padding!r} (expected 'keyframe' or 'zero')"
        )

        self.frames_path = frames_path
        self.labels_path = labels_path
        self.mode = mode
        self.window_size = window_size
        self.target_fps = target_fps
        self.label_type = label_type
        self.transform = transform
        self.border_padding = border_padding

        fps_conv = target_fps / ORIG_FPS
        self.stride = max(1, int(round(1.0 / fps_conv)))

        self.videos = _build_image_index(frames_path)
        if not self.videos:
            raise FileNotFoundError(f"No .jpg files found under {frames_path}")

        # Keyframe samples: (video_name, key_frame_id)
        self.samples: List[tuple] = []
        for vname in sorted(self.videos):
            for fid in self.videos[vname]["frame_ids"]:
                if fid % KEYFRAME_INTERVAL == 0:
                    self.samples.append((vname, fid))

        logging.info(
            "TemporalImageWindowDataset: %d samples across %d videos",
            len(self.samples), len(self.videos),
        )
        if not self.samples:
            raise RuntimeError(f"No keyframes (frame_id % {KEYFRAME_INTERVAL} == 0) found")

        # Pre-load label CSVs
        self._label_cache: Dict[str, tuple] = {}
        for vname in self.videos:
            frame_csv = path_join(labels_path, vname, "frame.csv")
            video_csv = path_join(labels_path, vname, "video.csv")
            if os.path.isfile(frame_csv) and os.path.isfile(video_csv):
                fdf = pd.read_csv(frame_csv).set_index("frame_id")
                vdf = pd.read_csv(video_csv)
                confs = [float(vdf[f"confidence_rater{i+1}"].iloc[0]) for i in range(3)]
            else:
                fdf = pd.DataFrame()
                confs = []
            self._label_cache[vname] = (fdf, confs)

    # ------------------------------------------------------------------
    def _build_window(self, video_name, key_frame_id):
        if self.mode == "offline":
            half = (self.window_size - 1) // 2
            start = key_frame_id - half * self.stride
            end = key_frame_id + (self.window_size - 1 - half) * self.stride
        else:
            start = key_frame_id - (self.window_size - 1) * self.stride
            end = key_frame_id

        requested = list(range(start, end + 1, self.stride))
        id2path = self.videos[video_name]["id2path"]

        chosen_ids, chosen_paths, valid_flags = [], [], []
        key_index = None
        for req_id in requested:
            chosen_ids.append(req_id)
            if req_id in id2path:
                chosen_paths.append(id2path[req_id])
                valid_flags.append(True)
            else:
                chosen_paths.append(None)
                valid_flags.append(False)
            if req_id == key_frame_id:
                key_index = len(chosen_ids) - 1

        if key_index is None:
            key_index = len(chosen_ids) - 1 if self.mode == "online" else len(chosen_ids) // 2

        return chosen_ids, chosen_paths, valid_flags, key_index

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _load_label(self, video_name, key_frame_id):
        fdf, confs = self._label_cache[video_name]
        if len(fdf) == 0 or key_frame_id not in fdf.index:
            return torch.zeros(3, dtype=torch.float32)
        row = fdf.loc[key_frame_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        if self.label_type == "hard":
            label = torch.tensor([
                mode([int(row[f"c{c}_rater{r}"]) for r in range(1, 4)])
                for c in range(1, 4)
            ], dtype=torch.float32)
        else:
            conf = np.array(confs, dtype=np.float32)
            ca_vals = []
            for c in range(1, 4):
                rls = np.array([float(row[f"c{c}_rater{r}"]) for r in range(1, 4)], dtype=np.float32)
                ca_vals.append(float(0.5 + (1.0 / 3.0) * np.dot(rls - 0.5, conf)))
            label = torch.tensor(ca_vals, dtype=torch.float32)
        return label

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_name, key_frame_id = self.samples[idx]
        chosen_ids, chosen_paths, valid_flags, key_index = self._build_window(video_name, key_frame_id)

        # Load keyframe image (always needed: returned even if all other slots
        # are valid, and used as a shape reference for zero-padding).
        kf_img = self._load_img(self.videos[video_name]["id2path"][key_frame_id])
        if self.border_padding == "zero":
            pad_img = torch.zeros_like(kf_img)
        else:
            pad_img = kf_img

        frames = []
        for path, is_valid in zip(chosen_paths, valid_flags):
            frames.append(self._load_img(path) if is_valid else pad_img)

        pixel_values = torch.stack(frames, dim=0)  # [T, C, H, W]
        y = self._load_label(video_name, key_frame_id)
        valid_mask = torch.tensor(valid_flags, dtype=torch.bool)

        meta = {
            "video_name": video_name,
            "key_frame_id": key_frame_id,
            "frame_ids": chosen_ids,
            "window_size": self.window_size,
            "stride": self.stride,
            "target_fps": self.target_fps,
            "mode": self.mode,
        }
        return TemporalImageSample(
            pixel_values=pixel_values, y=y,
            key_index=key_index, valid_mask=valid_mask, meta=meta,
        )


def collate_temporal_images(batch: List[TemporalImageSample]) -> Dict[str, Any]:
    pixel_values = torch.stack([b.pixel_values for b in batch], dim=0)  # [B, T, C, H, W]
    y = torch.stack([b.y for b in batch], dim=0)                        # [B, 3]
    key_index = torch.tensor([b.key_index for b in batch])              # [B]
    valid_mask = torch.stack([b.valid_mask for b in batch], dim=0)      # [B, T]

    frame_ids = torch.tensor([b.meta["frame_ids"] for b in batch], dtype=torch.long)
    meta = [b.meta for b in batch]
    if all("position_ids" in b.meta for b in batch):
        positions = torch.tensor([b.meta["position_ids"] for b in batch], dtype=torch.long)
    else:
        stride = batch[0].meta["stride"]
        positions = (frame_ids // stride).clamp(min=0)                  # [B, T]

    return {
        "pixel_values": pixel_values,
        "y": y,
        "key_index": key_index,
        "valid_mask": valid_mask,
        "positions": positions,
        "meta": meta,
    }
