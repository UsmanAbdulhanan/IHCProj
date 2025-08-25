import numpy as np
from PIL import Image

def read_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img)


def _extract_patches(path: str):
        # useful during testing
        image = read_image(path)
        h, w, _ = image.shape
        image = np.pad(image, ((0, 512), (0, 512), (0, 0)), mode='constant', constant_values=255)
        patches = list()
        patch_coords = list()

        # loop over the padded image to extract patches
        y = 0
        while y <= h:
            x = 0
            while x <= w:
                patch = image[y:y + 512, x:x + 512, :]
                
                patches.append(patch)
                patch_coords.append([y, x])
                x += 512
            y += 512
        del image

        return patches, np.array(patch_coords)
