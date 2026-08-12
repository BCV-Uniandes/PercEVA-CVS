import argparse
import json
import os
import shutil


def organize(data_path, splits_json_path):
    with open(splits_json_path) as f:
        splits = json.load(f)

    moved = 0
    for target_split, mapping in splits.items():
        # The official dataset only extracts a "train" pool (which this repo's
        # val is carved out of) and a "test" pool — val's videos physically
        # live under train/ until this script moves them.
        source_split = "train" if target_split in ("train", "val") else target_split

        dst_frames = os.path.join(data_path, target_split, "frames")
        dst_labels = os.path.join(data_path, target_split, "labels")
        os.makedirs(dst_frames, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)

        for video_name, uuid in sorted(mapping.items()):
            any_moved = False
            for src_root, dst_root in (
                (os.path.join(data_path, source_split, "frames"), dst_frames),
                (os.path.join(data_path, source_split, "labels"), dst_labels),
            ):
                src = os.path.join(src_root, uuid)
                dst = os.path.join(dst_root, video_name)
                if os.path.exists(dst):
                    continue
                if not os.path.exists(src):
                    print(f"[{target_split}] SKIP {uuid} -> {video_name} — not found under {src_root}")
                    continue
                shutil.move(src, dst)
                any_moved = True
            if any_moved:
                moved += 1

    total = sum(len(m) for m in splits.values())
    print(f"Organized {moved}/{total} videos into train/val/test, renamed to video_XXXX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finish organizing a freshly-extracted SAGES 2024 download: carve the "
                     "paper's 200-video val split out of the official train pool, and rename "
                     "every video (train/val/test) from its UUID to the video_XXXX scheme, "
                     "using the mapping in data/sages_splits.json. Run this after extracting "
                     "frames with tools/preprocess_videos.py for both the train and test "
                     "splits (val does not exist yet at that point — it's produced by this "
                     "script)."
    )
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--data_path", type=str, default=os.path.join(here, "SAGES_2024"))
    parser.add_argument("--splits_json", type=str, default=os.path.join(here, "sages_splits.json"))
    args = parser.parse_args()

    organize(args.data_path, args.splits_json)
