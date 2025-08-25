import os
import numpy as np
import json


patch_json = 'patch_coordinates_HE.json'
bbox_dir = '/home/comppec/Downloads/VirtualMultiplexer-master/data/bbox_info/HE_NKX3/HE'
patch_bbox_dir = 'data/bbox_info/HE_NKX3/HE_patch'
patch_size = 512 

os.makedirs(patch_bbox_dir, exist_ok=True)

#load patch coordinates
with open(patch_json, 'r') as f:
    patch_coords = json.load(f)

for img_name, patches in patch_coords.items():
    bbox_path = os.path.join(bbox_dir, img_name + '.npz')
    if not os.path.exists(bbox_path):
        print(f"Warning: bbox file not found for {bbox_path}")
        continue

    bboxes = np.load(bbox_path)['bbox']  
    for patch in patches:
        y, x = patch['y'], patch['x']
        y0_patch, x0_patch = y, x
        y1_patch, x1_patch = y + patch_size, x + patch_size

        
        if bboxes.size == 0:
            patch_bboxes = np.empty((0, 5), dtype=int)
        else:
            mask = (
                (bboxes[:, 0] >= y0_patch) & (bboxes[:, 1] >= x0_patch) &
                (bboxes[:, 2] <= y1_patch) & (bboxes[:, 3] <= x1_patch)
            )
            patch_bboxes = bboxes[mask]
            
            patch_bboxes = patch_bboxes - np.array([y,x,y,x,0])
       
        patch_name = f"{img_name}_patch_y_{y}_x_{x}.npz"
        patch_path = os.path.join(patch_bbox_dir, patch_name)
        np.savez_compressed(patch_path, bbox=patch_bboxes)
        
print("Done! Patch bbox files created in", patch_bbox_dir)