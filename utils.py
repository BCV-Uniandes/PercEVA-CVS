import os
import cv2
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from os.path import join as path_join
from datetime import datetime

def infer_split_name(split_path: str) -> str:
    return os.path.basename(os.path.normpath(split_path))


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def now_experiment_name():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def cvs_data_collator(batch, feature_extractor):
    images = [img for img, label, video_name, frame_id, metadata in batch]
    labels = torch.stack([label for img, label, video_name, frame_id, metadata in batch], dim=0)   # [B, 3]

    enc = feature_extractor(images, return_tensors="pt")
    pixel_values = enc["pixel_values"]  # [B, 3, 224, 224]

    # Return only what the model forward expects:
    return {"pixel_values": pixel_values, "labels": labels}

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(coco_json_path: str)->dict:
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data_dict: dict, save_path: str)->None:
    with open(save_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
        
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)
