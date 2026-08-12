import os
import logging
from os.path import join as path_join

from glob import glob
from dataclasses import dataclass
from typing import Dict, Any, List, Set, Optional

import pandas as pd

import torch
from torch.utils.data import Dataset


KEYFRAME_INTERVAL = 150  # 5 seconds @ 30 fps
ORIG_FPS = 30            # SAGES frames are saved at 30 fps


@dataclass
class TemporalSample:
    x: torch.Tensor                         # [T, D]
    y: torch.Tensor                         # [3] (c1, c2, c3)
    key_index: int                          # index in [0..T-1] where the keyframe sits
    valid_mask: torch.Tensor                # [T] bool — True = real frame, False = padding
    meta: Dict[str, Any]
    obj_features: Optional[torch.Tensor] = None   # [N_obj, d_obj] top-8 MaskDINO instances


# ------------------------------------------------------------------
# Video index builder
# ------------------------------------------------------------------
def _build_video_index(root: str, ext: str):
    """
    Scan `root/{video_name}/frame_*.{ext}` and return:
      { video_name: { "frame_ids": sorted list, "id2path": {fid: path} } }
    """
    videos: Dict[str, Dict] = {}
    pattern = path_join(root, "*", f"frame_*.{ext}")
    
    for ft_path in glob(pattern):
        video = os.path.basename(os.path.dirname(ft_path))
        base = os.path.splitext(os.path.basename(ft_path))[0]  # frame_XXXX
        try:
            frame_id = int(base.split("_")[-1])
        except (ValueError, IndexError):
            print(f"Warning: skipping file with unexpected name format: {ft_path}")

        if video not in videos:
            videos[video] = {"frame_ids": [], "id2path": {}}
        videos[video]["frame_ids"].append(frame_id)
        videos[video]["id2path"][frame_id] = ft_path

    # Sort frame_ids for each video
    for v in videos:
        videos[v]["frame_ids"] = sorted(set(videos[v]["frame_ids"]))
    return videos


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class TemporalWindowCVSDataset(Dataset):
    """
    Temporal window dataset built around keyframes (every 5 s @ 30 fps).

    Window layout depends on ``mode``:

    - **offline** (training): centred window.
      ``window_size=5`` → ``[t-2, t-1, t, t+1, t+2]``,
      keyframe at index ``window_size // 2``.
      
    - **online** (inference): causal window — no future frames.
      ``window_size=5`` → ``[t-4, t-3, t-2, t-1, t]``,
      keyframe at index ``window_size - 1``.

    The stride between consecutive slots is derived from
    ``target_fps``: stride = ``ORIG_FPS / target_fps``.

    - ``target_fps=30`` → stride 1  (every frame).
    - ``target_fps=1``  → stride 30 (1 frame per second).

    Slots that fall outside the available frames (start / end of
    video) are filled by repeating the keyframe so the output is
    always exactly ``window_size`` entries.

    Labels are always read from the ``.pt`` payloads (both modes).

    Parameters
    ----------
    data_root : str
        Root directory containing ``{video_name}/frame_*.pt``.
    mode : str
        ``"offline"`` (centred) or ``"online"`` (causal).
    window_size : int
        Number of frames in the temporal window.
    target_fps : int
        Effective FPS of the window.  ``1`` → stride 30,
        ``30`` → stride 1.
    label_type : str
        ``"hard"`` for majority-vote 0/1 labels or ``"ca"`` for
        confidence-aware float labels.
    ft_type : str
        Which features to load from the .pt files: ``"patch_embeddings"``
        or ``"prelogits"``.
    """

    def __init__(
        self,
        data_root: str,
        mode: str = "offline",
        window_size: int = 5,
        target_fps: int = 1,
        label_type: str = "hard",
        ft_type: str = "patch_embeddings",
        oversample_factor: int = 0,
        obj_root: Optional[str] = None,
    ):
        assert mode in ("offline", "online"), f"Unknown mode: {mode}"
        assert label_type in ("hard", "ca"), f"Unknown label_type: {label_type}"
        assert window_size >= 1, f"window_size must be >= 1, got {window_size}"
        assert ft_type in ("patch_embeddings", "prelogits"), f"Unknown ft_type: {ft_type}"
        assert oversample_factor >= 0, "oversample_factor must be >= 0"

        self.data_root = data_root
        self.mode = mode
        self.window_size = window_size
        self.target_fps = target_fps
        self.label_type = label_type
        self.ft_type = ft_type
        self.obj_root = obj_root

        fps_conv = target_fps / ORIG_FPS  # e.g. 1/30 or 30/30
        self.stride = max(1, int(1.0 / fps_conv))

        # Creat video idx dict for easier data lookup
        self.videos = _build_video_index(data_root, "pth")

        if len(self.videos) == 0:
            raise FileNotFoundError(
                f"No .pt files found under {data_root}"
            )

        # Collect (video_name, key_frame_id) pairs — one per sample
        self.samples: List[tuple] = []
        for vname in sorted(self.videos):
            for fid in self.videos[vname]["frame_ids"]:
                if fid % KEYFRAME_INTERVAL == 0:
                    self.samples.append((vname, fid))

        logging.info(f"Found {len(self.samples)} keyframes across {len(self.videos)} videos")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No keyframes (frame_id % {KEYFRAME_INTERVAL} == 0) found"
            )

        # Oversample unanimous-positive keyframes
        if oversample_factor > 0:
            self._oversample_unanimous(data_root, oversample_factor)

    # ----------------------------------------------------------------
    # Oversampling
    # ----------------------------------------------------------------
    def _oversample_unanimous(self, data_root: str, oversample_factor: int):
        """
        Extend self.samples by repeating keyframes where all 3 raters agree
        on at least one positive criterion (c1, c2, or c3).

        Labels are read from the per-video frame.csv files in the sibling
        `labels/` directory (derived from data_root by replacing `features`).
        """
        labels_root = data_root.replace("features", "labels")
        if not os.path.isdir(labels_root):
            logging.warning(
                f"oversample_unanimous: labels dir not found at {labels_root}, skipping."
            )
            return

        # Build set of (video_name, frame_id) that are unanimous-positive
        unanimous: Set[tuple] = set()
        for vname in self.videos:
            csv_path = path_join(labels_root, vname, "frame.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                fid = int(row["frame_id"])
                for c in ["c1", "c2", "c3"]:
                    raters = [row[f"{c}_rater1"], row[f"{c}_rater2"], row[f"{c}_rater3"]]
                    if all(int(r) == 1 for r in raters):
                        unanimous.add((vname, fid))
                        break  # one positive criterion is enough

        # Only oversample keyframes that are in self.samples
        samples_set = set(self.samples)
        to_repeat = [(v, f) for (v, f) in unanimous if (v, f) in samples_set]

        n_before = len(self.samples)
        self.samples = self.samples + to_repeat * oversample_factor
        n_after = len(self.samples)

        logging.info(
            f"Oversampling: {len(to_repeat)} unanimous-positive keyframes repeated x{oversample_factor} "
            f"→ dataset size {n_before} → {n_after}"
        )
        print(
            f"[Oversample] {len(to_repeat)} unanimous-pos keyframes x{oversample_factor} "
            f"| {n_before} → {n_after} samples"
        )

    # ----------------------------------------------------------------
    # Object feature loading
    # ----------------------------------------------------------------
    def _load_obj_features(self, video_name: str, key_frame_id: int):
        """
        Load keyframe object features from ``obj_root``.

        Returns
        -------
        obj_features : Tensor [N_obj, d_obj] or None
            Top-8 MaskDINO instance features. ``None`` if ``obj_root`` is
            not set or the file does not exist for this keyframe.
        """
        if self.obj_root is None:
            return None

        obj_path = path_join(
            self.obj_root, video_name, f"frame_{key_frame_id:04d}.pt"
        )
        if not os.path.exists(obj_path):
            logging.warning(f"Object features not found: {obj_path}")
            return None

        payload = torch.load(obj_path, map_location="cpu", weights_only=False)
        return payload["object_features"].float()   # [N_obj, d_obj]

    # ----------------------------------------------------------------
    # Window building
    # ----------------------------------------------------------------
    def _build_window(self, video_name: str, key_frame_id: int):
        """
        Return (chosen_ids, chosen_paths, key_index) for a fixed-size
        window of ``window_size`` frames.

        - offline (centred): key at ``window_size // 2``
        - online  (causal) : key at ``window_size - 1`` (last slot)
        """
                
        if self.mode == "offline":
            # Centred: half before, keyframe, half after
            half = (self.window_size - 1) // 2
            start = key_frame_id - half * self.stride
            end = key_frame_id + (self.window_size - 1 - half) * self.stride
        else:
            # Causal: all context before + keyframe at end
            start = key_frame_id - (self.window_size - 1) * self.stride
            end = key_frame_id

        requested = list(range(start, end + 1, self.stride))

        id2path = self.videos[video_name]["id2path"]
        key_path = id2path[key_frame_id]  # must exist

        chosen_ids = []
        chosen_paths = []
        valid_flags = []   # True = real frame, False = padding
        key_index = None

        for req_id in requested:
            if req_id in id2path:
                chosen_ids.append(req_id)
                chosen_paths.append(id2path[req_id])
                valid_flags.append(True)
            else:
                # Out-of-bounds slot: zero-pad, mark as invalid
                chosen_ids.append(req_id)
                chosen_paths.append(None)
                valid_flags.append(False)

            if req_id == key_frame_id:
                key_index = len(chosen_ids) - 1

        # Safety fallback
        if key_index is None:
            key_index = len(chosen_ids) - 1 if self.mode == "online" else len(chosen_ids) // 2

        return chosen_ids, chosen_paths, valid_flags, key_index

    @staticmethod
    def _parse_annots(annots):
        """Convert label to float32 tensor[3]. Accepts Tensor[3], list, or None."""
        if annots is None:
            return torch.zeros(3, dtype=torch.float32)
        if isinstance(annots, torch.Tensor):
            return annots.float()
        return torch.tensor([float(v) if v is not None else 0.0 for v in annots],
                            dtype=torch.float32)

    def _load_window(self, key_frame_id, chosen_ids, chosen_paths, valid_flags):
        feat_key  = "patch_embeddings" if self.ft_type == "patch_embeddings" else "image_embeddings"
        annot_key = "labels_hard" if self.label_type == "hard" else "labels_ca"

        feats = []
        annots_key = None
        feat_shape = None

        for fid, fpath, is_valid in zip(chosen_ids, chosen_paths, valid_flags):
            if is_valid:
                payload = torch.load(fpath, map_location="cpu", weights_only=False)
                feat = payload[feat_key]
                if feat is not None:
                    feat = feat.float()
                    if feat_shape is None:
                        feat_shape = feat.shape
                    feats.append(feat)
                else:
                    feats.append(None)  # preheads not saved for this frame
                if fid == key_frame_id:
                    annots_key = payload[annot_key]
            else:
                feats.append(None)

        assert feat_shape is not None, "No valid frames with features found in window"
        feats = [f if f is not None else torch.zeros(feat_shape) for f in feats]

        x = torch.stack(feats, dim=0)  # [T, D] or [T, P, D]
        y = self._parse_annots(annots_key)
        return x, y

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> TemporalSample:
        
        #Load the sample given the idx and build the window around the keyframe
        video_name, key_frame_id = self.samples[idx]
        
        chosen_ids, chosen_paths, valid_flags, key_index = self._build_window(
            video_name, key_frame_id
        )

        if len(chosen_ids) == 0:
            raise RuntimeError(
                f"Empty window for {video_name} key={key_frame_id}"
            )

        x, y = self._load_window(
            key_frame_id, chosen_ids, chosen_paths, valid_flags
        )

        valid_mask = torch.tensor(valid_flags, dtype=torch.bool)  # [T]

        meta = {
            "video_name": video_name,
            "key_frame_id": key_frame_id,
            "frame_ids": chosen_ids,
            "window_size": self.window_size,
            "stride": self.stride,
            "target_fps": self.target_fps,
            "mode": self.mode,
        }

        obj_features = self._load_obj_features(video_name, key_frame_id)

        return TemporalSample(
            x=x, y=y, key_index=key_index, valid_mask=valid_mask, meta=meta,
            obj_features=obj_features,
        )


# ------------------------------------------------------------------
# Augmented Dataset — random jitter within keyframe interval
# ------------------------------------------------------------------
class TemporalWindowAugCVSDataset(TemporalWindowCVSDataset):
    """
    Data-augmented variant of TemporalWindowCVSDataset.

    Each window slot independently picks a random frame within its fps
    interval ``[slot, slot + stride - 1]``, so the temporal spacing
    between consecutive frames varies slightly each epoch:

        nominal slots (1fps, stride=30): …, -30,   0,  30,  60, …
        jittered slots (example):        …, -23,   7,  31,  58, …

    This gives richer augmentation than a global shift — the model sees
    different intra-second frames for every slot on every pass.

    Labels are **always** read from the original ``key_frame_id`` payload.

    All constructor parameters are identical to TemporalWindowCVSDataset.
    Jitter uses ``torch.randint``, which PyTorch seeds independently per
    DataLoader worker, ensuring uncorrelated augmentation across the batch.
    """

    def __init__(self, *args, seed: int = None, frame_drop_p: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_drop_p = frame_drop_p
        if seed is not None:
            logging.warning(
                "TemporalWindowAugCVSDataset: `seed` is ignored — jitter now uses "
                "torch.randint, which is seeded independently per DataLoader worker."
            )
        logging.info(
            "TemporalWindowAugCVSDataset: main-process torch seed = %d. "
            "Each DataLoader worker will receive a unique seed automatically.",
            torch.initial_seed(),
        )

    def __getitem__(self, idx: int) -> TemporalSample:
        video_name, key_frame_id = self.samples[idx]
        id2path = self.videos[video_name]["id2path"]

        # Compute nominal slot positions (same geometry as parent)
        if self.mode == "offline":
            half = (self.window_size - 1) // 2
            start = key_frame_id - half * self.stride
            end = key_frame_id + (self.window_size - 1 - half) * self.stride
        else:
            start = key_frame_id - (self.window_size - 1) * self.stride
            end = key_frame_id

        nominal_ids = list(range(start, end + 1, self.stride))

        chosen_ids = []
        chosen_paths = []
        valid_flags = []
        jitters = []
        key_index = None

        for i, nom_id in enumerate(nominal_ids):
            # Keyframe slot is always exact; context slots jitter within their interval
            j = 0 if nom_id == key_frame_id else int(torch.randint(0, self.stride, (1,)).item())
            actual_id = nom_id + j
            jitters.append(j)

            if actual_id in id2path:
                chosen_ids.append(actual_id)
                chosen_paths.append(id2path[actual_id])
                valid_flags.append(True)
            else:
                chosen_ids.append(actual_id)
                chosen_paths.append(None)
                valid_flags.append(False)

            if nom_id == key_frame_id:
                key_index = i

        if key_index is None:
            key_index = len(chosen_ids) - 1 if self.mode == "online" else len(chosen_ids) // 2

        # Load features only (pass sentinel -1 so _load_window skips label extraction)
        x, _ = self._load_window(-1, chosen_ids, chosen_paths, valid_flags)

        # Frame dropout: randomly zero out context frames (never the keyframe)
        # This forces the model to rely on generalizable patterns, not specific sequences
        if self.frame_drop_p > 0.0:
            drop_mask = torch.rand(len(chosen_ids)) < self.frame_drop_p
            drop_mask[key_index] = False   # never drop the keyframe slot
            x[drop_mask] = 0.0
            for i in range(len(valid_flags)):
                if drop_mask[i]:
                    valid_flags[i] = False

        # Load labels from the original keyframe
        annot_key = "labels_hard" if self.label_type == "hard" else "labels_ca"
        kf_payload = torch.load(id2path[key_frame_id], map_location="cpu", weights_only=False)
        annots_key = kf_payload[annot_key]
        y = self._parse_annots(annots_key)

        valid_mask = torch.tensor(valid_flags, dtype=torch.bool)

        meta = {
            "video_name": video_name,
            "key_frame_id": key_frame_id,
            "frame_ids": chosen_ids,
            "jitters": jitters,
            "window_size": self.window_size,
            "stride": self.stride,
            "target_fps": self.target_fps,
            "mode": self.mode,
        }

        obj_features = self._load_obj_features(video_name, key_frame_id)

        return TemporalSample(
            x=x, y=y, key_index=key_index, valid_mask=valid_mask, meta=meta,
            obj_features=obj_features,
        )


# ------------------------------------------------------------------
# Collation
# ------------------------------------------------------------------
def collate_temporal(batch: List[TemporalSample]) -> Dict[str, Any]:
    x = torch.stack([b.x for b in batch], dim=0)                    # [B, T, D]
    y = torch.stack([b.y for b in batch], dim=0)                     # [B, 3]
    key_index = torch.tensor([b.key_index for b in batch])           # [B]
    valid_mask = torch.stack([b.valid_mask for b in batch], dim=0)   # [B, T]
    if batch[0].meta.get("position_ids") is not None:
        # EndoscapesTemporalWindowDataset already computes video-relative
        # positions (frame_id - video_start) // stride and stores them in
        # meta — use those. Recomputing from meta["frame_ids"] (raw, absolute
        # frame ids) // stride, as below, overflows the learned PE table
        # (pe_max_len) for any video past frame ~stride * pe_max_len — e.g.
        # frame 51300 // stride=25 = 2052 > pe_max_len=2048 — which crashes
        # the embedding lookup with a CUDA "gather kernel index out of
        # bounds" assertion for later frames in longer Endoscapes videos.
        positions = torch.tensor([b.meta["position_ids"] for b in batch], dtype=torch.long)  # [B, T]
    else:
        stride = batch[0].meta["stride"]
        frame_ids = torch.tensor([b.meta["frame_ids"] for b in batch], dtype=torch.long)  # [B, T]
        positions = (frame_ids // stride).clamp(min=0)               # [B, T]
    meta = [b.meta for b in batch]

    out = {
        "x": x, "y": y, "key_index": key_index,
        "valid_mask": valid_mask, "positions": positions, "meta": meta,
    }

    # Object features — only stacked when the dataset was built with obj_root
    if batch[0].obj_features is not None:
        out["obj_features"] = torch.stack([b.obj_features for b in batch])  # [B, N_obj, d_obj]

    return out
