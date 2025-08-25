import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from preprocess.tissue_mask import GaussianTissueMask
import pandas as  pd
src_marker = 'HE'
dst_marker = 'NKX3'
images_dir = 'data/images'  
patches_dir = 'data/i2i_patches'  
mask_dir = 'data/tissue_masks'
patch_size = 512
patch_tissue_threshold = 0.7 




def read_image(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path))

def read_mask(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path).convert('L'))   
def extract_and_save_patches(image: np.ndarray, tissue_mask: np.ndarray,
                             save_dir: str, image_id: str,
                             patch_size: int, patch_tissue_threshold: float):
    os.makedirs(save_dir, exist_ok=True)
    h, w, _ = image.shape
    pad_h = patch_size - h % patch_size
    pad_w = patch_size - w % patch_size

    
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
    tissue_mask = np.pad(tissue_mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    #make tissue mask binary
    tissue_mask = (tissue_mask > 0).astype(np.uint8) #fixed the problem with tissue mask being not binary
    tissue_thresh = int(patch_size * patch_size * patch_tissue_threshold)
    
    for y in range(0, image.shape[0], patch_size):
        for x in range(0, image.shape[1], patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = tissue_mask[y:y+patch_size, x:x+patch_size]
            if np.sum(mask_patch) >= tissue_thresh:
                patch_fname = f"{image_id}_patch_y_{y}_x_{x}.png"
                patch_path = os.path.join(save_dir, patch_fname)
                Image.fromarray(patch).save(patch_path)

def process_images_in_folder(marker: str):
    image_folder = os.path.join(images_dir, marker)
    mask_folder = os.path.join(mask_dir, marker)
    save_folder = os.path.join(patches_dir, marker)
    all_splits = pd.read_csv(f'/home/comppec/Downloads/VirtualMultiplexer-master/data/data_splits/i2i/{marker}_splits.csv')
    train_files = all_splits['train_cores']
    test_files = all_splits['test_cores']

    # Combine into one giant list
    image_files = list(train_files) + list(test_files)
    for img_file in tqdm(image_files):



        
        img_id = img_file
        img_path = os.path.join(image_folder, f"{img_file}.png")
        mask_path = os.path.join(mask_folder, f"{img_file}.png")
        image = read_image(img_path)
        tissue_mask = read_mask(mask_path)
        extract_and_save_patches(image, tissue_mask, save_folder, img_id,
                                 patch_size, patch_tissue_threshold)
       
if __name__ == '__main__':
    process_images_in_folder(dst_marker)
 