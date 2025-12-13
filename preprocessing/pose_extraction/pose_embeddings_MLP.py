import os
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# pose embedding MLP for 258 to 512 dim
class PoseEmbeddingMLP(nn.Module):
    def __init__(self, input_dim=258, embed_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, pose):
        assert pose.dim() == 2
        return self.mlp(pose)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_FEATURES_ROOT = os.path.join(BASE_DIR, "pose_features") 
POSE_EMBEDDINGS_BASE_ROOT = os.path.join(BASE_DIR, "pose_embeddings")
SUBSETS = ['train', 'dev', 'test']
os.makedirs(POSE_EMBEDDINGS_BASE_ROOT, exist_ok=True)

INPUT_DIM = 258
EMBED_DIM = 512
pose_model = PoseEmbeddingMLP(input_dim=INPUT_DIM, embed_dim=EMBED_DIM)

for subset in SUBSETS:
    print(F"Starting embedding for subeset:{subset}")
    POSE_INPUT_ROOT=os.path.join(POSE_FEATURES_ROOT, subset)
    POSE_EMBED_SAVE_ROOT=os.path.join(POSE_EMBEDDINGS_BASE_ROOT, subset)
    os.makedirs(POSE_EMBED_SAVE_ROOT, exist_ok=True)



    pose_files = glob.glob(os.path.join(POSE_INPUT_ROOT, "*.npy"))
    print(f"Found {len(pose_files)} pose files.")
    device="cpu"
    for pose_path in tqdm(pose_files):
        try:
            pose_np = np.load(pose_path)  # shape: (T, 258)
            pose_tensor = torch.from_numpy(pose_np).float().to(device)

            with torch.no_grad():
                embeddings = pose_model(pose_tensor)  # (T, 512)

            base_name = os.path.splitext(os.path.basename(pose_path))[0]
            save_path = os.path.join(POSE_EMBED_SAVE_ROOT, base_name + ".pt")
            torch.save(embeddings.cpu(), save_path)
        except Exception as e:
            print(f"Failed on {pose_path}: {e}")

    print(f" Pose embeddings saved in: {POSE_EMBED_SAVE_ROOT}")
