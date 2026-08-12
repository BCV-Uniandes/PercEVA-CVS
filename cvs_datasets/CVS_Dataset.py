import os
from os.path import join as path_join

import glob
from statistics import mode

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def collate_fn(batch: list) -> dict:
    images, labels, video_names, frame_ids, metadata = zip(*batch)
    return {
        "pixel_values": torch.stack(images),
        "labels": torch.stack(labels),
        "video_name": video_names,
        "frame_id": frame_ids,
        "metadata": metadata
    }

class CVSData(Dataset):
    """Dataloader that returns annotated frames and their
    corresponding labels (majority), video names, and frame ids.
      Args:
        frames_path (str):   Path to folder containing the extracted frames
        labels_path (str):   Path to folder containing a label csv file for each video
        label_type (str):    Type of labels to return ("ca" for confidence-aware, "hard" for hard labels)
        transform (callable, optional): Optional transform to be applied on a sample.

      Returns:
        img, label, video_name, frame_id, metadata
    """

    def __init__(self, frames_path, labels_path, label_type="ca", transform=None):
        
        #Attribute initialization
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.label_type = label_type
        self.transform = transform
        search_pattern = os.path.join(frames_path, "**", "*.jpg")
        
        frames = sorted(glob.glob(search_pattern, recursive=True))
        
        self.image_paths = [file for file in frames if self.is_annotated(file)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        video_name = img_path.split("/")[-2]
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        frame_name = frame_name.split('_')[-1]
        frame_id = int(frame_name)
        # frame_id = int(img_path.split("/")[-1].replace(".jpg", "")) - 1
        label_path = os.path.join(self.labels_path, f"{video_name}/frame.csv")
        video_label_path = os.path.join(self.labels_path, f"{video_name}/video.csv")

        # Load image
        img = Image.open(img_path)

        # Load c1,c2,c3 label
        label_df = pd.read_csv(label_path)
        video_label_df = pd.read_csv(video_label_path)
        confidences = [float(video_label_df[f'confidence_rater{i+1}'].iloc[0]) for i in range(3)]
        label_df = label_df[label_df["frame_id"] == frame_id]
        
        #Hard labels
        
        if self.label_type == "hard":
            c1, c2, c3 = (
                self.majority_vote(label_df, "c1"),
                self.majority_vote(label_df, "c2"),
                self.majority_vote(label_df, "c3"),
            )
            label = torch.as_tensor([c1, c2, c3], dtype=torch.float32)
        elif self.label_type == "ca":            
            #Soft labels using confidence multiplexed majority vote from dataset
            ca_c1 = self.confidence_multiplexed_majority_vote(label_df, "c1",confidences)
            ca_c2 = self.confidence_multiplexed_majority_vote(label_df, "c2",confidences)
            ca_c3 = self.confidence_multiplexed_majority_vote(label_df, "c3",confidences)
            label = torch.as_tensor([ca_c1, ca_c2, ca_c3], dtype=torch.float32)
        else: 
            raise ValueError(f"Invalid label_type: {self.label_type}. Must be 'hard' or 'ca'.")    
            
        
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)
            
        metadata = {}
        
        #Labels list
        
        metadata["raw_labels"] = {}
        
        for i,key in enumerate(["c1", "c2", "c3"]):
            metadata["raw_labels"][key] = self.raw_labels(label_df, key)[i]
        
        if not self.label_type == "hard":
            ca_labels = [ca_c1,ca_c2,ca_c3]
            metadata["confidence_aware_labels"] = {}
            for i,key in enumerate(["c1", "c2", "c3"]):
                metadata["confidence_aware_labels"][key] = ca_labels[i]
            
        return img, label, video_name, frame_id, metadata

    def is_annotated(self, fname):
        frame_name = os.path.splitext(os.path.basename(fname))[0]
        frame_name = frame_name.split('_')[-1]
        frame_id = int(frame_name)
        # frame_id = int(os.path.splitext(os.path.basename(fname))[0]) - 1
        return frame_id % 150 == 0

    def raw_labels(self, row, category):
        labels = [
            row[f"{category}_rater1"].iloc[0],
            row[f"{category}_rater2"].iloc[0],
            row[f"{category}_rater3"].iloc[0],
        ]

        return labels

    def majority_vote(self, row, category):
        labels = [
            row[f"{category}_rater1"].iloc[0],
            row[f"{category}_rater2"].iloc[0],
            row[f"{category}_rater3"].iloc[0],
        ]
        return mode(labels)

    def confidence_multiplexed_majority_vote(self, row, category,confidences):
        labels = np.array([
            row[f"{category}_rater1"].iloc[0],
            row[f"{category}_rater2"].iloc[0],
            row[f"{category}_rater3"].iloc[0],
        ])
        labeler_confidences = np.array(confidences)
        confidence_aware_label = 0.5+1/3.0*np.dot(labels-0.5,labeler_confidences)
        return confidence_aware_label


def collate_fn_all(batch: list) -> dict:
    """Collate for CVSAllFrames. labels_ca / labels_hard are tuples that may contain None
    for unannotated frames — they are NOT stacked so callers can handle None explicitly."""
    images, labels_ca, labels_hard, video_names, frame_ids, is_annotated, original_sizes = zip(*batch)
    return {
        "pixel_values":   torch.stack(images),
        "labels_ca":      labels_ca,     # tuple[Tensor[3] | None]
        "labels_hard":    labels_hard,   # tuple[Tensor[3] | None]
        "video_name":     video_names,
        "frame_id":       frame_ids,
        "is_annotated":   torch.tensor(is_annotated, dtype=torch.bool),
        "original_sizes": original_sizes,  # tuple[(w, h)]
    }


class CVSAllFrames(Dataset):
    """Like CVSData but indexes ALL frames in the video folders, not just annotated keyframes.
    Labels for unannotated frames (frame_id not in frame.csv) are returned as None.
    All label CSVs for the split are pre-loaded at __init__ time.

      Args:
        frames_path (str):  Path to folder containing the extracted frames
        labels_path (str):  Path to folder containing a label csv file for each video
        transform (callable, optional): Transform applied to the PIL image.

      Returns:
        img, labels_ca, labels_hard, video_name, frame_id, is_annotated
        labels_ca / labels_hard are Tensor[3] for annotated frames, None otherwise.
    """

    def __init__(self, frames_path, labels_path, transform=None, frame_stride=1):
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.transform = transform

        search_pattern = os.path.join(frames_path, "**", "*.jpg")
        all_paths = sorted(glob.glob(search_pattern, recursive=True))
        if frame_stride > 1:
            self.image_paths = [
                p for p in all_paths
                if int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]) % frame_stride == 0
            ]
        else:
            self.image_paths = all_paths

        # Pre-load ALL label CSVs for the split at init time.
        # Keyed by video_name -> (frame_df indexed by frame_id, confidences list).
        # Videos with missing CSVs map to (empty DataFrame, []).
        self._label_cache = {}
        video_names = {p.split("/")[-2] for p in self.image_paths}
        for vname in video_names:
            frame_csv = os.path.join(labels_path, vname, "frame.csv")
            video_csv = os.path.join(labels_path, vname, "video.csv")
            if os.path.isfile(frame_csv) and os.path.isfile(video_csv):
                fdf = pd.read_csv(frame_csv).set_index("frame_id")
                vdf = pd.read_csv(video_csv)
                confidences = [float(vdf[f"confidence_rater{i+1}"].iloc[0]) for i in range(3)]
            else:
                fdf = pd.DataFrame()
                confidences = []
            self._label_cache[vname] = (fdf, confidences)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        video_name = img_path.split("/")[-2]
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])

        img = Image.open(img_path).convert("RGB")
        original_size = img.size  # (w, h) before any transform
        if self.transform:
            img = self.transform(img)

        fdf, confidences = self._label_cache[video_name]

        if len(fdf) > 0 and frame_id in fdf.index:
            row = fdf.loc[frame_id]
            is_annotated = True

            labels_hard = torch.tensor([
                mode([int(row[f"c{c}_rater{r}"]) for r in range(1, 4)])
                for c in range(1, 4)
            ], dtype=torch.float32)

            conf = np.array(confidences, dtype=np.float32)
            ca_vals = []
            for c in range(1, 4):
                rater_labels = np.array([float(row[f"c{c}_rater{r}"]) for r in range(1, 4)], dtype=np.float32)
                ca_vals.append(float(0.5 + (1.0 / 3.0) * np.dot(rater_labels - 0.5, conf)))
            labels_ca = torch.tensor(ca_vals, dtype=torch.float32)
        else:
            is_annotated = False
            labels_ca   = None
            labels_hard = None

        return img, labels_ca, labels_hard, video_name, frame_id, is_annotated, original_size


if __name__ == "__main__":
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    data_dir = path_join(parent_dir, "data")
    sages_2024_dir = path_join(data_dir, "SAGES_2024")
    frames_path = path_join(sages_2024_dir, "train", "frames")
    labels_path = path_join(sages_2024_dir, "train", "labels")
    dataset = CVSData(frames_path, labels_path, label_type="ca")
    print(f"Dataset length: {len(dataset)}")
    img, label, video_name, frame_id, metadata = dataset[0]
    print(f"Image shape: {img.size}")
    print(f"Label: {label}")
    print(f"Video name: {video_name}")
    print(f"Frame ID: {frame_id}")
    print(f"Metadata: {metadata}")
    