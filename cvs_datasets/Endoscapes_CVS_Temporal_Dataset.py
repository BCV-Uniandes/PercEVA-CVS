import os
import re
import json
import logging
from os.path import join as path_join
from glob import glob
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    from cvs_datasets.CVS_Temporal_Dataset import TemporalSample, collate_temporal
except ImportError:
    from CVS_Temporal_Dataset import TemporalSample, collate_temporal


ORIG_FPS = 30  # Endoscapes videos are 30 fps

_FRAME_FILE_RE = re.compile(r"^frame_(?P<frame>\d+)$")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _sort_video_key(video_name: str):
    return int(video_name) if str(video_name).isdigit() else str(video_name)


def _build_video_index(root: str, ext: str = "pth") -> Dict[str, Dict[str, Any]]:
    """
    Scan ``root/{video_name}/frame_*.{ext}`` and return:
        { video_name: { "frame_ids": sorted list, "id2path": {fid: path} } }
    """
    videos: Dict[str, Dict[str, Any]] = {}
    pattern = path_join(root, "*", f"frame_*.{ext}")

    for ft_path in glob(pattern):
        video = os.path.basename(os.path.dirname(ft_path))
        base = os.path.splitext(os.path.basename(ft_path))[0]
        m = _FRAME_FILE_RE.match(base)
        if m is None:
            continue
        frame_id = int(m.group("frame"))

        if video not in videos:
            videos[video] = {"frame_ids": [], "id2path": {}}
        videos[video]["frame_ids"].append(frame_id)
        videos[video]["id2path"][frame_id] = ft_path

    for v in videos:
        videos[v]["frame_ids"] = sorted(set(videos[v]["frame_ids"]))
    return videos


def _infer_frame_stride(videos: Dict[str, Dict[str, Any]]) -> int:
    diffs: List[int] = []
    for video_info in videos.values():
        frame_ids = video_info["frame_ids"]
        diffs.extend(b - a for a, b in zip(frame_ids, frame_ids[1:]) if b > a)
    if not diffs:
        return 1
    return max(1, int(round(median(diffs))))


def _default_annotation_json(split_path: str) -> str:
    for name in ("annotation_ds_coco.json", "annotation_coco.json", "annotation_coco_vid.json"):
        candidate = path_join(split_path, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No Endoscapes annotation JSON found. Expected one of "
        "annotation_ds_coco.json, annotation_coco.json, annotation_coco_vid.json "
        f"under {split_path}"
    )


def _parse_endoscapes_filename(file_name: str) -> Tuple[str, int]:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse Endoscapes frame name: {file_name}")
    return parts[0], int(parts[1])


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class EndoscapesTemporalWindowDataset(Dataset):
    """
    Temporal-window CVS dataset for Endoscapes-CVS201 pre-extracted features.

    Mirrors ``TemporalWindowCVSDataset`` (SAGES) in API and output shape but:

    - Discovers keyframes from the COCO-style ``annotation_ds_coco.json``
      (entries flagged with ``is_ds_keyframe`` and a valid ``ds`` triplet)
      rather than from a modulo-of-frame-id rule.
    - Reads labels from that same JSON instead of from each ``.pth`` payload.
      (Most Endoscapes ``.pth`` files store only ``image_embeddings``;
      labels and patch tokens are only populated at keyframes.)
    - Uses the median spacing between available feature files as the
      nominal frame stride (typically 25 @ 30 fps for Endoscapes).

    Output (per ``__getitem__``) is the same ``TemporalSample`` produced by
    ``cvs_datasets.CVS_Temporal_Dataset``, so the existing
    ``collate_temporal`` collator and downstream models work unchanged.

    Window layout:
      - ``mode="offline"``: centred — key at index ``window_size // 2``.
      - ``mode="online"``: causal  — key at index ``window_size - 1``.

    Out-of-bounds slots are zero-padded and marked invalid in ``valid_mask``.

    Parameters
    ----------
    features_root : str
        Directory containing ``{video_name}/frame_*.pth``, e.g.
        ``/.../endoscapes_features/train/features``.
    annotation_json_path : str, optional
        Path to ``annotation_ds_coco.json`` (or another COCO JSON with a
        ``ds`` field). If ``None``, ``split_path`` must be given.
    split_path : str, optional
        Directory whose annotation JSON should be auto-discovered when
        ``annotation_json_path`` is not provided.
    mode : {"offline", "online"}
        Window layout.
    window_size : int
        Number of frames in the temporal window.
    target_fps : float, optional
        Desired effective FPS. Converted to a stride of
        ``round(ORIG_FPS / target_fps)`` and snapped to the nearest multiple
        of the underlying feature stride so the window lands on real files.
        Ignored if ``frame_stride`` is given.
    frame_stride : int, optional
        Explicit stride in frame-id units between window slots. Overrides
        ``target_fps``. Defaults to the median feature spacing.
    label_type : {"hard", "ca"}
        ``"hard"`` thresholds ``ds`` at ``label_threshold``; ``"ca"`` keeps
        the raw confidence-aware floats.
    label_threshold : float
        Threshold for ``"hard"`` labels.
    ft_type : {"image_embeddings", "patch_embeddings"}
        Which tensor to read from each ``.pth``. Note that for Endoscapes
        only keyframes carry ``patch_embeddings``; non-keyframe slots will
        zero-pad when this is selected.
    cvs_key : str
        JSON image-entry key holding the CVS triplet (default ``"ds"``).
    require_cvs_keyframe : bool
        If ``True``, skip JSON images with ``is_ds_keyframe == False``.
    strict : bool
        Raise instead of skipping when a JSON image has no matching feature
        file on disk.
    """

    def __init__(
        self,
        features_root: str,
        annotation_json_path: Optional[str] = None,
        split_path: Optional[str] = None,
        mode: str = "offline",
        window_size: int = 5,
        target_fps: Optional[float] = None,
        frame_stride: Optional[int] = None,
        label_type: str = "hard",
        label_threshold: float = 0.5,
        ft_type: str = "image_embeddings",
        cvs_key: str = "ds",
        require_cvs_keyframe: bool = True,
        strict: bool = False,
    ):
        assert mode in ("offline", "online"), f"Unknown mode: {mode}"
        assert label_type in ("hard", "ca"), f"Unknown label_type: {label_type}"
        assert window_size >= 1, f"window_size must be >= 1, got {window_size}"
        assert ft_type in ("image_embeddings", "patch_embeddings"), \
            f"Unknown ft_type: {ft_type}"

        if annotation_json_path is None:
            if split_path is None:
                raise ValueError("Pass annotation_json_path or split_path")
            annotation_json_path = _default_annotation_json(split_path)

        self.features_root = features_root
        self.annotation_json_path = annotation_json_path
        self.mode = mode
        self.window_size = int(window_size)
        self.target_fps = target_fps
        self.label_type = label_type
        self.label_threshold = float(label_threshold)
        self.ft_type = ft_type
        self.cvs_key = cvs_key
        self.require_cvs_keyframe = require_cvs_keyframe
        self.strict = strict

        # Per-video feature index
        self.videos = _build_video_index(features_root, "pth")
        if not self.videos:
            raise FileNotFoundError(
                f"No frame_*.pth files found under {features_root}"
            )

        # Frame stride between window slots
        self.feature_stride = _infer_frame_stride(self.videos)
        self.stride = self._resolve_stride(frame_stride, target_fps)

        # Keyframe samples from JSON
        self.samples: List[Dict[str, Any]] = []
        self._load_cvs_samples()

        logging.info(
            "EndoscapesTemporalWindowDataset: %d keyframes across %d videos "
            "(stride=%d, feature_stride=%d, mode=%s, window_size=%d, ft=%s, label=%s)",
            len(self.samples), len(self.videos), self.stride,
            self.feature_stride, mode, window_size, ft_type, label_type,
        )
        if not self.samples:
            raise RuntimeError(
                f"No CVS-labeled Endoscapes samples found in {annotation_json_path}"
            )

    # ----------------------------------------------------------------
    # Setup helpers
    # ----------------------------------------------------------------
    def _resolve_stride(self, frame_stride: Optional[int],
                        target_fps: Optional[float]) -> int:
        if frame_stride is not None:
            s = int(frame_stride)
            if s <= 0:
                raise ValueError(f"frame_stride must be positive, got {s}")
            return s
        if target_fps is not None and target_fps > 0:
            nominal = max(1, int(round(ORIG_FPS / float(target_fps))))
            # Snap to the closest non-zero multiple of the feature stride
            mult = max(1, int(round(nominal / self.feature_stride)))
            return mult * self.feature_stride
        return self.feature_stride

    def _load_cvs_samples(self):
        with open(self.annotation_json_path, "r") as f:
            coco = json.load(f)

        self.ds_weights = coco.get("ds_weights", None)

        missing_features = 0
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

            try:
                video_name, frame_id = _parse_endoscapes_filename(file_name)
            except ValueError:
                skipped_without_cvs += 1
                continue

            id2path = self.videos.get(video_name, {}).get("id2path", {})
            if frame_id not in id2path:
                missing_features += 1
                if self.strict:
                    raise FileNotFoundError(
                        f"Feature missing on disk for keyframe {file_name} "
                        f"under {self.features_root}"
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
            })

        self.samples.sort(
            key=lambda s: (_sort_video_key(s["video_name"]), s["key_frame_id"])
        )

        if missing_features:
            logging.warning(
                "EndoscapesTemporalWindowDataset: %d JSON keyframes had no "
                "feature file on disk and were skipped", missing_features,
            )
        if skipped_without_cvs:
            logging.warning(
                "EndoscapesTemporalWindowDataset: %d JSON images had no usable "
                "CVS labels and were skipped", skipped_without_cvs,
            )

    # ----------------------------------------------------------------
    # Window building
    # ----------------------------------------------------------------
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
            # Video-relative position so the learned PE embedding stays in range
            # regardless of how late in the procedure the keyframe is.
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

    def _load_window(self, chosen_ids, chosen_paths, valid_flags):
        feats: List[Optional[torch.Tensor]] = []
        feat_shape = None

        for fpath, is_valid in zip(chosen_paths, valid_flags):
            if not is_valid:
                feats.append(None)
                continue

            payload = torch.load(fpath, map_location="cpu", weights_only=False)
            feat = payload.get(self.ft_type)
            if feat is None:
                feats.append(None)
                continue

            feat = feat.float()
            if self.ft_type == "patch_embeddings":
                # Drop the CLS token for parity with PatchTokensCVSDataset
                feat = feat[1:]
            if feat_shape is None:
                feat_shape = feat.shape
            feats.append(feat)

        if feat_shape is None:
            raise RuntimeError(
                f"No valid '{self.ft_type}' tensor found in window "
                f"frames={chosen_ids}"
            )

        feats = [f if f is not None else torch.zeros(feat_shape) for f in feats]
        return torch.stack(feats, dim=0)  # [T, D] or [T, N, D]

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> TemporalSample:
        sample = self.samples[idx]
        video_name = sample["video_name"]
        key_frame_id = sample["key_frame_id"]

        chosen_ids, chosen_paths, valid_flags, position_ids, key_index = self._build_window(
            video_name, key_frame_id,
        )

        x = self._load_window(chosen_ids, chosen_paths, valid_flags)

        label = sample["hard_label"] if self.label_type == "hard" else sample["soft_label"]
        y = torch.tensor(label, dtype=torch.float32)
        valid_mask = torch.tensor(valid_flags, dtype=torch.bool)

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
        }

        return TemporalSample(
            x=x, y=y, key_index=key_index, valid_mask=valid_mask, meta=meta,
        )


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    features_root = "/home/scanar/endovis/Datasets/endoscapes_2023/endoscapes_features/train/features"
    annotation = "/home/scanar/endovis/Datasets/endoscapes_2023/endoscapes/train/annotation_ds_coco.json"

    ds_off = EndoscapesTemporalWindowDataset(
        features_root=features_root,
        annotation_json_path=annotation,
        mode="offline",
        window_size=5,
        label_type="ca",
    )
    print(f"[offline] len={len(ds_off)}")
    s = ds_off[0]
    print(f"  x: {tuple(s.x.shape)}, y: {s.y.tolist()}, key_index: {s.key_index}")
    print(f"  frame_ids: {s.meta['frame_ids']}, valid: {s.valid_mask.tolist()}")

    ds_on = EndoscapesTemporalWindowDataset(
        features_root=features_root,
        annotation_json_path=annotation,
        mode="online",
        window_size=5,
        label_type="hard",
    )
    print(f"\n[online] len={len(ds_on)}")
    s = ds_on[0]
    print(f"  x: {tuple(s.x.shape)}, y: {s.y.tolist()}, key_index: {s.key_index}")
    print(f"  frame_ids: {s.meta['frame_ids']}, valid: {s.valid_mask.tolist()}")
