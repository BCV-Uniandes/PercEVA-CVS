import glob
import json
import logging
import os
import re
from os.path import join as path_join
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

# populated lazily so callers that don't use numpy don't pay the import cost
import numpy as np

try:
    from cvs_datasets.CVS_TemporalImage_Dataset import TemporalImageSample
except ImportError:
    try:
        from CVS_TemporalImage_Dataset import TemporalImageSample
    except ImportError:
        TemporalImageSample = None  # only needed by EndoscapesTemporalImageWindowDataset


_ENDOSCAPES_NAME_RE = re.compile(r"^(?P<video>.+)_(?P<frame>\d+)$")


def parse_endoscapes_filename(file_name: str) -> Tuple[str, int]:
    """Parse Endoscapes flat names like ``100_27200.jpg``."""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    match = _ENDOSCAPES_NAME_RE.match(stem)
    if match is None:
        raise ValueError(f"Cannot parse Endoscapes frame name: {file_name}")
    return match.group("video"), int(match.group("frame"))


def _sort_video_key(video_name: str):
    return int(video_name) if str(video_name).isdigit() else str(video_name)


def _build_flat_image_index(frames_path: str) -> Dict[str, Dict[str, Any]]:
    """Build an index for flat Endoscapes split folders.

    Returns:
        {video_name: {"frame_ids": [...], "id2path": {frame_id: jpg_path}}}
    """
    videos: Dict[str, Dict[str, Any]] = {}
    for img_path in glob.glob(path_join(frames_path, "*.jpg")):
        try:
            video_name, frame_id = parse_endoscapes_filename(img_path)
        except ValueError:
            continue
        if video_name not in videos:
            videos[video_name] = {"frame_ids": [], "id2path": {}}
        videos[video_name]["frame_ids"].append(frame_id)
        videos[video_name]["id2path"][frame_id] = img_path

    for video_name in videos:
        videos[video_name]["frame_ids"] = sorted(set(videos[video_name]["frame_ids"]))
    return videos


def _infer_frame_stride(videos: Dict[str, Dict[str, Any]]) -> int:
    diffs: List[int] = []
    for video_info in videos.values():
        frame_ids = video_info["frame_ids"]
        diffs.extend(
            b - a
            for a, b in zip(frame_ids, frame_ids[1:])
            if b > a
        )
    if not diffs:
        return 1
    return max(1, int(round(median(diffs))))


def _default_annotation_json(split_path: str) -> str:
    """Prefer the CVS-focused Endoscapes JSON when present."""
    for name in ("annotation_ds_coco.json", "annotation_coco.json", "annotation_coco_vid.json"):
        candidate = path_join(split_path, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No Endoscapes annotation JSON found. Expected one of "
        "annotation_ds_coco.json, annotation_coco.json, annotation_coco_vid.json "
        f"under {split_path}"
    )


class EndoscapesTemporalImageWindowDataset(Dataset):
    """Temporal E2E dataset for Endoscapes flat split folders.

    Endoscapes stores frames as ``<video_id>_<frame_id>.jpg`` and stores CVS
    labels in the COCO image entries under ``ds``. This dataset ignores the
    anatomy/tool object annotations and converts ``ds`` into the 3 CVS labels
    expected by the MaskCVS E2E model.

    The returned sample type matches ``TemporalImageWindowDataset`` so the
    existing E2E collator and inference loop can be reused.
    """

    def __init__(
        self,
        split_path: Optional[str] = None,
        frames_path: Optional[str] = None,
        coco_json_path: Optional[str] = None,
        mode: str = "offline",
        window_size: int = 5,
        frame_stride: Optional[int] = None,
        target_fps: Optional[int] = None,
        label_threshold: float = 0.5,
        cvs_key: str = "ds",
        require_cvs_keyframe: bool = True,
        transform=None,
        strict: bool = False,
        label_type: str = "hard",
        border_padding: str = "keyframe",
    ):
        assert mode in ("offline", "online"), f"Unknown mode: {mode}"
        if label_type not in ("hard", "ca"):
            raise ValueError(f"Invalid label_type: {label_type}. Must be 'hard' or 'ca'.")
        if border_padding not in ("keyframe", "zero"):
            raise ValueError(
                f"Invalid border_padding: {border_padding!r} (expected 'keyframe' or 'zero')"
            )

        if split_path is None and frames_path is None:
            raise ValueError("Pass either split_path or frames_path")

        self.split_path = split_path or frames_path
        self.frames_path = frames_path or split_path
        self.coco_json_path = coco_json_path or _default_annotation_json(self.split_path)
        self.mode = mode
        self.window_size = int(window_size)
        self.target_fps = target_fps
        self.label_threshold = float(label_threshold)
        self.cvs_key = cvs_key
        self.require_cvs_keyframe = require_cvs_keyframe
        self.transform = transform
        self.strict = strict
        self.label_type = label_type
        self.border_padding = border_padding

        self.videos = _build_flat_image_index(self.frames_path)
        if not self.videos:
            raise FileNotFoundError(f"No flat Endoscapes .jpg files found under {self.frames_path}")

        self.stride = int(frame_stride) if frame_stride is not None else _infer_frame_stride(self.videos)
        if self.stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {self.stride}")

        self.samples: List[Dict[str, Any]] = []
        self._load_cvs_samples()

        logging.info(
            "EndoscapesTemporalImageWindowDataset: %d CVS samples across %d videos "
            "(json=%s, stride=%d, threshold=%.3f)",
            len(self.samples),
            len(self.videos),
            self.coco_json_path,
            self.stride,
            self.label_threshold,
        )
        if not self.samples:
            raise RuntimeError(f"No CVS-labeled Endoscapes samples found in {self.coco_json_path}")

    def _load_cvs_samples(self):
        with open(self.coco_json_path, "r") as f:
            coco = json.load(f)

        missing_files = 0
        skipped_without_cvs = 0

        for image in coco.get("images", []):
            if self.require_cvs_keyframe and image.get("is_ds_keyframe") is False:
                continue

            if self.cvs_key not in image or image[self.cvs_key] is None:
                skipped_without_cvs += 1
                continue

            ds = image[self.cvs_key]
            if not isinstance(ds, (list, tuple)) or len(ds) < 3:
                skipped_without_cvs += 1
                continue

            file_name = image.get("file_name")
            if not file_name:
                skipped_without_cvs += 1
                continue

            video_name, frame_id = parse_endoscapes_filename(file_name)
            id2path = self.videos.get(video_name, {}).get("id2path", {})
            if frame_id not in id2path:
                missing_files += 1
                if self.strict:
                    raise FileNotFoundError(
                        f"JSON image is missing on disk: {file_name} under {self.frames_path}"
                    )
                continue

            soft = [float(v) for v in ds[:3]]
            hard = [1.0 if v >= self.label_threshold else 0.0 for v in soft]

            self.samples.append({
                "video_name": video_name,
                "key_frame_id": frame_id,
                "file_name": file_name,
                "image_id": image.get("id"),
                "soft_label": soft,
                "hard_label": hard,
                "image": image,
            })

        self.samples.sort(key=lambda s: (_sort_video_key(s["video_name"]), s["key_frame_id"]))

        if missing_files:
            logging.warning("Skipped %d JSON images that were not found on disk", missing_files)
        if skipped_without_cvs:
            logging.warning("Skipped %d JSON images without usable CVS labels", skipped_without_cvs)

    def _build_window(self, video_name: str, key_frame_id: int):
        if self.mode == "offline":
            half = (self.window_size - 1) // 2
            start = key_frame_id - half * self.stride
            end = key_frame_id + (self.window_size - 1 - half) * self.stride
        else:
            start = key_frame_id - (self.window_size - 1) * self.stride
            end = key_frame_id

        requested = list(range(start, end + 1, self.stride))
        id2path = self.videos[video_name]["id2path"]
        video_min_frame = self.videos[video_name]["frame_ids"][0]

        chosen_ids, chosen_paths, valid_flags, position_ids = [], [], [], []
        key_index = None
        for req_id in requested:
            chosen_ids.append(req_id)
            position_ids.append(max(0, (req_id - video_min_frame) // self.stride))
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

        return chosen_ids, chosen_paths, valid_flags, position_ids, key_index

    def _load_img(self, path: str):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_name = sample["video_name"]
        key_frame_id = sample["key_frame_id"]

        chosen_ids, chosen_paths, valid_flags, position_ids, key_index = self._build_window(
            video_name,
            key_frame_id,
        )

        key_path = self.videos[video_name]["id2path"][key_frame_id]
        key_img = self._load_img(key_path)
        if self.border_padding == "zero":
            pad_img = torch.zeros_like(key_img)
        else:
            pad_img = key_img

        frames = []
        for path, is_valid in zip(chosen_paths, valid_flags):
            frames.append(self._load_img(path) if is_valid else pad_img)

        meta = {
            "dataset": "endoscapes",
            "video_name": video_name,
            "key_frame_id": key_frame_id,
            "file_name": sample["file_name"],
            "image_id": sample["image_id"],
            "frame_ids": chosen_ids,
            "position_ids": position_ids,
            "window_size": self.window_size,
            "stride": self.stride,
            "target_fps": self.target_fps,
            "mode": self.mode,
            "cvs_soft_label": sample["soft_label"],
            "cvs_label_threshold": self.label_threshold,
        }

        label = sample["soft_label"] if self.label_type == "ca" else sample["hard_label"]
        return TemporalImageSample(
            pixel_values=torch.stack(frames, dim=0),
            y=torch.tensor(label, dtype=torch.float32),
            key_index=key_index,
            valid_mask=torch.tensor(valid_flags, dtype=torch.bool),
            meta=meta,
        )


class EndoscapesImageDataset(Dataset):
    """Single-frame Endoscapes-CVS201 dataset for image-encoder training.

    Reads ``annotation_ds_coco.json`` from a split folder (train/val/test) and
    yields one annotated CVS keyframe per item. Returns the same 5-tuple as
    ``cvs_datasets.CVS_Dataset.CVSData`` so it can be used with the existing
    ``collate_fn`` from that module.
    """

    def __init__(
        self,
        split_path: str,
        label_type: str = "ca",
        transform=None,
        label_threshold: float = 0.5,
        annotation_name: str = "annotation_ds_coco.json",
        cvs_key: str = "ds",
        require_cvs_keyframe: bool = True,
    ):
        if label_type not in ("ca", "hard"):
            raise ValueError(f"Invalid label_type: {label_type}. Must be 'hard' or 'ca'.")

        self.split_path = split_path
        self.label_type = label_type
        self.label_threshold = float(label_threshold)
        self.transform = transform
        self.cvs_key = cvs_key
        self.require_cvs_keyframe = require_cvs_keyframe

        json_path = path_join(split_path, annotation_name)
        if not os.path.isfile(json_path):
            json_path = _default_annotation_json(split_path)
        self.coco_json_path = json_path

        with open(self.coco_json_path, "r") as f:
            coco = json.load(f)

        self.ds_weights = coco.get("ds_weights", None)

        self.samples: List[Dict[str, Any]] = []
        missing_files = 0
        skipped_without_cvs = 0

        for image in coco.get("images", []):
            if self.require_cvs_keyframe and image.get("is_ds_keyframe") is False:
                continue

            ds = image.get(self.cvs_key)
            if not isinstance(ds, (list, tuple)) or len(ds) < 3:
                skipped_without_cvs += 1
                continue

            file_name = image.get("file_name")
            if not file_name:
                skipped_without_cvs += 1
                continue

            img_path = path_join(split_path, file_name)
            if not os.path.isfile(img_path):
                missing_files += 1
                continue

            try:
                video_name, frame_id = parse_endoscapes_filename(file_name)
            except ValueError:
                skipped_without_cvs += 1
                continue

            soft = [float(v) for v in ds[:3]]
            self.samples.append({
                "video_name": video_name,
                "frame_id": frame_id,
                "file_name": file_name,
                "image_id": image.get("id"),
                "img_path": img_path,
                "soft_label": soft,
            })

        self.samples.sort(key=lambda s: (_sort_video_key(s["video_name"]), s["frame_id"]))

        if missing_files:
            logging.warning(
                "EndoscapesImageDataset: skipped %d JSON images not found on disk",
                missing_files,
            )
        if skipped_without_cvs:
            logging.warning(
                "EndoscapesImageDataset: skipped %d JSON images without usable CVS labels",
                skipped_without_cvs,
            )
        if not self.samples:
            raise RuntimeError(f"No CVS-labeled Endoscapes samples found in {self.coco_json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        soft = sample["soft_label"]
        hard = [1.0 if v >= self.label_threshold else 0.0 for v in soft]

        if self.label_type == "hard":
            label = torch.as_tensor(hard, dtype=torch.float32)
        else:
            label = torch.as_tensor(soft, dtype=torch.float32)

        metadata = {
            "raw_labels": {k: hard[i] for i, k in enumerate(("c1", "c2", "c3"))},
            "confidence_aware_labels": {k: soft[i] for i, k in enumerate(("c1", "c2", "c3"))},
            "image_id": sample["image_id"],
            "file_name": sample["file_name"],
        }

        return img, label, sample["video_name"], sample["frame_id"], metadata


class EndoscapesAllFrames(Dataset):
    """Like CVSAllFrames but for Endoscapes flat split folders.

    Indexes ALL .jpg files in a flat split directory (``<video>_<frame>.jpg``).
    Labels come from ``annotation_ds_coco.json``; frames not in the JSON are
    unannotated and return ``labels_ca = labels_hard = None``.

    Returns the same 7-tuple as ``CVSAllFrames`` so it is compatible with
    ``collate_fn_all`` and ``extract_ft.process_split``.
    """

    def __init__(
        self,
        split_path: str,
        transform=None,
        label_threshold: float = 0.5,
        cvs_key: str = "ds",
    ):
        self.split_path = split_path
        self.transform = transform
        self.label_threshold = float(label_threshold)

        self.image_paths = sorted(glob.glob(path_join(split_path, "*.jpg")))
        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg files found in {split_path}")

        json_path = _default_annotation_json(split_path)
        with open(json_path, "r") as f:
            coco = json.load(f)

        # index: (video_name, frame_id) -> (soft [3], hard [3])
        self._label_index: Dict[Tuple[str, int], Tuple[List[float], List[float]]] = {}
        for image in coco.get("images", []):
            ds = image.get(cvs_key)
            if not isinstance(ds, (list, tuple)) or len(ds) < 3:
                continue
            file_name = image.get("file_name", "")
            try:
                video_name, frame_id = parse_endoscapes_filename(file_name)
            except ValueError:
                continue
            soft = [float(v) for v in ds[:3]]
            hard = [1.0 if v >= self.label_threshold else 0.0 for v in soft]
            self._label_index[(video_name, frame_id)] = (soft, hard)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            video_name, frame_id = parse_endoscapes_filename(img_path)
        except ValueError:
            raise RuntimeError(f"Cannot parse Endoscapes filename: {img_path}")

        img = Image.open(img_path).convert("RGB")
        original_size = img.size  # (w, h)
        if self.transform:
            img = self.transform(img)

        entry = self._label_index.get((video_name, frame_id))
        if entry is not None:
            soft, hard = entry
            labels_ca   = torch.tensor(soft, dtype=torch.float32)
            labels_hard = torch.tensor(hard, dtype=torch.float32)
            is_annotated = True
        else:
            labels_ca    = None
            labels_hard  = None
            is_annotated = False

        return img, labels_ca, labels_hard, video_name, frame_id, is_annotated, original_size
