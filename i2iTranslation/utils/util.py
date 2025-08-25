"""This module contains simple helper functions """
from __future__ import print_function
from typing import Dict, List
import os
import time
import collections.abc
import numpy as np
import pandas as pd
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def get_split(splits_path: str, split_key: str):
    all_splits = pd.read_csv(splits_path)
    split = all_splits[split_key]
    split = split.dropna().reset_index(drop=True)
    return split

def set_seed(device, seed=0):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def random_seed():
    return np.random.seed(int(1000 * time.time()) % 2 ** 32)

def rescale_tensor(img, scale_factor):
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img = img.unsqueeze(0)

    return F.interpolate(img, scale_factor=(scale_factor, scale_factor), mode='bilinear', align_corners=True).squeeze()

def downsample_image(img, downsample):
    new_size = (
        int(img.shape[1] // downsample),
        int(img.shape[0] // downsample),
    )
    img = cv2.resize(
        img,
        new_size,
        interpolation=cv2.INTER_LINEAR,
    )
    return img

def upsample_image(img, upsample):
    new_size = (
        int(img.shape[0] * upsample),
        int(img.shape[1] * upsample),
    )
    img = cv2.resize(
        img,
        new_size,
        interpolation=cv2.INTER_LINEAR,
    )
    return img

def delete_tensor_gpu(tensor_dict: Dict):
    for k, v in tensor_dict.items():
        del v
        
'''def delete_tensor_gpu(tensor_dict: Dict):
    keys = list(tensor_dict.keys())
    for k in keys:
        tensor = tensor_dict[k]
        if torch.is_tensor(tensor) and tensor.is_cuda:
            del tensor
        del tensor_dict[k]
    gc.collect()
    torch.cuda.empty_cache()
'''  
def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):  # ✅ fixed here
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_image(path: str):
    return np.asarray(Image.open(path))

def unnormalize(patches: torch.Tensor, norm_mean: float, norm_std: float):
    patches = [x * norm_std + norm_mean for x in patches]
    return patches

def clip_img(patches: torch.Tensor):
    patches = [torch.clip(x, 0.0, 1.0) for x in patches]
    return patches

def tensor2img(patches: torch.Tensor) -> List[np.ndarray]:
    transform = transforms.Compose([transforms.ToPILImage()])
    patches = [np.array(transform(x.squeeze())) for x in patches]
    return patches