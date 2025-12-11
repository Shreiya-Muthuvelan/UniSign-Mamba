import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import gc

def patchify_frame(frame, patch_size=16):
    H, W, C = frame.shape

    new_H = H - (H % patch_size)
    new_W = W - (W % patch_size)
    frame = cv2.resize(frame, (new_W, new_H))

    patches = []
    for i in range(0, new_H, patch_size):
        for j in range(0, new_W, patch_size):
            patch = frame[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)

    patches = np.stack(patches)                   
    return torch.from_numpy(patches).float()


def patchify_video(frames, patch_size=16):
    all_patches = []

    for frame in frames:
        all_patches.append(patchify_frame(frame, patch_size))

    return torch.stack(all_patches)  


def load_video(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f" Cannot open: {path}")
        return None

    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None:
            continue

        # Convert BGR → RGB
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            continue

        frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()

    if len(frames) == 0:
        print(f"No readable frames in: {path}")
        return None

    return frames


def process_dataset(video_dir, save_dir,split,patch_size=16, out_size=224):
    os.makedirs(save_dir, exist_ok=True)

    video_files = sorted([
        f for f in os.listdir(video_dir)
        if f.endswith(".mp4")
    ])
    skipped=[]
    print(f"Found {len(video_files)} videos in {video_dir}")

    for video_name in tqdm(video_files):
        video_path = os.path.join(video_dir, video_name)

        try:
            frames = load_video(video_path)
            if frames is None:
                print(f"Skipping: {video_name}")
                skipped.append(video_name)
                continue

            patches = patchify_video(frames, patch_size)

            base_name = os.path.splitext(video_name)[0]
            video_save_dir = os.path.join(save_dir, base_name)
            os.makedirs(video_save_dir, exist_ok=True)

            torch.save(patches, os.path.join(video_save_dir, "patches.pt"))

        except Exception as e:
            print(f" Failed on: {video_path}")
            skipped.append(video_name)
            continue
    
        # memory cleanup
        del frames
        del patches
        gc.collect()

    # writting skipped video to text file
    file_name=f"skipped_videos_{split}.txt"
    skip_file=os.path.join(save_dir, file_name)
    with open(skip_file, "w") as f:
        for item in skipped:
            f.write(item+"\n")
    print(f"Done, Skipped {len(skipped)} videos. Skipped videos are listed in {skip_file}")


if __name__ == "__main__":
    base_dir=os.path.dirname(os.path.abspath(__file__))
    phoenix_root=os.path.abspath(os.path.join(base_dir, "..", "..", "data", "videos_phoenix"))

    splits = ["train", "dev", "test"]

    patch_save_root = os.path.join(base_dir,"patches")

    for split in splits:
        video_dir = os.path.join(phoenix_root, split)
        save_dir = os.path.join(patch_save_root, split)

        process_dataset(
            video_dir=video_dir,
            save_dir=save_dir,
            patch_size=16,
            split=split,
            out_size=224
        )
