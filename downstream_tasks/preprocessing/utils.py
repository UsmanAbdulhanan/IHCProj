from typing import Any
import numpy as np
import cv2
import importlib
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


def dynamic_import_from(source_file: str, class_name: str) -> Any:
    """Do a from source_file import class_name dynamically

    Args:
        source_file (str): Where to import from
        class_name (str): What to import

    Returns:
        Any: The class to be imported
    """
    module = importlib.import_module(source_file)
    return getattr(module, class_name)

def get_torchvision_feature_extractor(feature_extractor: str) -> nn.Module:
    """
    Returns a torchvision model from a given architecture string.

    Args:
        architecture (str): Torchvision model description.

    Returns:
        nn.Module: A pretrained pytorch model.
    """
    model_class = dynamic_import_from("torchvision.models", feature_extractor)
    model = model_class(pretrained=True)

    # remove classification_gt head
    if hasattr(model, "model"):
        model = model.model
    if isinstance(model, torchvision.models.resnet.ResNet):
        model.fc = nn.Sequential()
    else:
        model.classifier[-1] = nn.Sequential()
    return model

def get_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]
    )

def extract_patch_features(
        model,
        transforms,
        device,
        image: np.ndarray,
        tissue_mask: np.ndarray,
        patch_size: int,
        stride: int,
        patch_threshold: float,
        batch_size: int,
        **kwargs
):
    # extract patches
    h, w, _ = image.shape
    image = np.pad(image, ((0, patch_size), (0, patch_size), (0, 0)), mode='constant', constant_values=255)
    patches = []
    coords = []
    patch_threshold = int(patch_size * patch_size * patch_threshold)

    y = 0
    while y <= h:
        x = 0
        while x <= w:
            mask = tissue_mask[y:y + patch_size, x:x + patch_size]
            if mask.sum() >= patch_threshold:
                patch = image[y:y + patch_size, x:x + patch_size, :]
                patch = cv2.resize(
                    patch,
                    (224, 224),
                    interpolation=cv2.INTER_NEAREST,
                )
                patches.append(patch)
                coords.append([x, y])
            x += stride
        y += stride

    # normalize patches
    patches = torch.stack([transforms(x) for x in patches])
    coords = np.asarray(coords)

    # extract patch features
    features = []
    with torch.no_grad():
        for j in range(0, len(patches), batch_size):
            batch = patches[j: j + batch_size]
            batch = batch.to(device)
            features_ = model(batch)
            features_ = features_.cpu()
            features.extend(features_)

    features = torch.stack(features, dim=0)
    return features, coords

def extract_adj_matrix(
        coords: np.ndarray,
        patch_size: int,
        **kwargs
):
    N = coords.shape[0]
    adj_s = np.zeros(shape=(N, N))

    # convert co-ordinates to positional indices
    posit = (coords/patch_size).astype(int)

    # sptial adjacency
    for i in range(N-1):
        x_i, y_i = posit[i]

        for j in range(i+1, N):
            x_j, y_j = posit[j]
            if abs(int(x_i)-int(x_j)) <=1 and abs(int(y_i)-int(y_j)) <= 1:
                adj_s[i][j] = 1
                adj_s[j][i] = 1

    return torch.from_numpy(adj_s)