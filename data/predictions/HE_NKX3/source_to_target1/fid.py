from glob import glob
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
fid = FrechetInceptionDistance(feature=2048).to(device)
img_names = os.listdir('data/predictions/HE_NKX3/source_to_target/epoch_0')
gt_paths   = sorted([os.path.join('data/images/NKX3/', name) for name in img_names])
os.makedirs('data/predictions/HE_NKX3/source_to_target/gt', exist_ok=True)
for epoch in range(0, 20, 5):
    pred_paths = sorted(glob(f'data/predictions/HE_NKX3/source_to_target/epoch_{epoch}/*.png'))

    # Reset FID object for each epoch
    fid.reset()

    for pred_path, gt_path in zip(pred_paths, gt_paths):
        # Load and convert to RGB
        pred_img = Image.open(pred_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        print(os.path.basename(pred_path) , os.path.basename(gt_path))
        #save gt_img
        # gt_img.save(f'data/predictions/HE_NKX3/source_to_target/gt/{os.path.basename(gt_path)}')
        #show images

        # Convert to tensor [C, H, W] in range [0, 255]
        pred_tensor = torch.tensor(np.array(pred_img)).permute(2, 0, 1).unsqueeze(0)
        gt_tensor   = torch.tensor(np.array(gt_img)).permute(2, 0, 1).unsqueeze(0)

        # Add to FID calculator
        fid.update(pred_tensor.to(device), real=False)  # fake images
        fid.update(gt_tensor.to(device), real=True)     # real images

    # Compute FID
    score = fid.compute()
    print(f"Epoch {epoch} - FID: {score.item():.4f}")
