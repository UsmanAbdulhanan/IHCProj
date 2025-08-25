import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import gaussian, threshold_otsu

Image.MAX_IMAGE_PIXELS = 100000000000


def get_tissue_mask(
    image: np.ndarray,
    n_thresholding_steps: int = 1,
    sigma: float = 0.0,
    min_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get binary tissue mask

    Args:
        image (np.ndarray):
            (m, n, 3) nd array of thumbnail RGB image
            or (m, n) nd array of thumbnail grayscale image
        n_thresholding_steps (int, optional): number of gaussian smoothign steps. Defaults to 1.
        sigma (float, optional): sigma of gaussian filter. Defaults to 0.0.
        min_size (int, optional): minimum size (in pixels) of contiguous tissue regions to keep. Defaults to 500.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            np int32 array
                each unique value represents a unique tissue region
            np bool array
                largest contiguous tissue region.
    """
    if len(image.shape) == 3:
        # grayscale thumbnail (inverted)
        thumbnail = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    else:
        thumbnail = image

    if len(np.unique(thumbnail)) == 1:
        return None, None

    for _ in range(n_thresholding_steps):

        # gaussian smoothing of grayscale thumbnail
        if sigma > 0.0:
            thumbnail = gaussian(
                thumbnail,
                sigma=sigma,
                output=None,
                mode="nearest",
                preserve_range=True)

        # get threshold to keep analysis region
        try:
            thresh = threshold_otsu(thumbnail[thumbnail > 0])
        except ValueError:  # all values are zero
            thresh = 0

        # replace pixels outside analysis region with upper quantile pixels
        thumbnail[thumbnail < thresh] = 0

    # convert to binary
    mask = 0 + (thumbnail > 0)

    # find connected components
    labeled, _ = ndimage.label(mask)

    # only keep
    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
    if len(unique) != 0:
        discard = np.in1d(labeled, unique[counts < min_size])
        discard = discard.reshape(labeled.shape)
        labeled[discard] = 0
        # largest tissue region
        mask = labeled == unique[np.argmax(counts)]
        return labeled, mask
    else:
        return None, None


class GaussianTissueMask:
    """Helper class to extract tissue mask from images"""

    def __init__(
        self,
        n_thresholding_steps: int = 1,
        sigma: int = 20,
        min_size: int = 10,
        kernel_size: int = 20,
        dilation_steps: int = 1,
        background_gray_value: int = 228,
        downsampling_factor: int = 4,
        **kwargs,
    ) -> None:
        """
        Args:
            n_thresholding_steps (int, optional): Number of gaussian smoothing steps. Defaults to 1.
            sigma (int, optional): Sigma of gaussian filter. Defaults to 20.
            min_size (int, optional): Minimum size (in pixels) of contiguous tissue regions to keep. Defaults to 10.
            kernel_size (int, optional): Dilation kernel size. Defaults to 20.
            dilation_steps (int, optional): Number of dilation steps. Defaults to 1.
            background_gray_value (int, optional): Gray value of background pixels (usually high). Defaults to 228.
            downsampling_factor (int, optional): Downsampling factor from the input image
                                                 resolution. Defaults to 4.
        """
        self.n_thresholding_steps = n_thresholding_steps
        self.sigma = sigma
        self.min_size = min_size
        self.kernel_size = kernel_size
        self.dilation_steps = dilation_steps
        self.background_gray_value = background_gray_value
        self.downsampling_factor = downsampling_factor
        super().__init__(**kwargs)
        self.kernel = np.ones((self.kernel_size, self.kernel_size), "uint8")

    @staticmethod
    def _downsample(image: np.ndarray, downsampling_factor: int) -> np.ndarray:
        """Downsample an input image with a given downsampling factor
        Args:
            image (np.array): Input tensor
            downsampling_factor (int): Factor to downsample
        Returns:
            np.array: Output tensor
        """
        height, width = image.shape[0], image.shape[1]
        new_height = math.floor(height / downsampling_factor)
        new_width = math.floor(width / downsampling_factor)
        downsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return downsampled_image

    @staticmethod
    def _upsample(
            image: np.ndarray,
            new_height: int,
            new_width: int) -> np.ndarray:
        """Upsample an input image to a speficied new height and width
        Args:
            image (np.array): Input tensor
            new_height (int): Target height
            new_width (int): Target width
        Returns:
            np.array: Output tensor
        """
        upsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return upsampled_image

    # type: ignore[override]
    def process(self, image: np.ndarray) -> np.ndarray:
        """Return the superpixels of a given input image
        Args:
            image (np.array): Input image
        Returns:
            np.array: Extracted tissue mask
        """
        # Downsample image
        original_height, original_width = image.shape[0], image.shape[1]
        if self.downsampling_factor != 1:
            image = self._downsample(image, self.downsampling_factor)

        tissue_mask = np.zeros(shape=(image.shape[0], image.shape[1]))
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect tissue region
        while True:
            _, mask_ = get_tissue_mask(
                image,
                n_thresholding_steps=self.n_thresholding_steps,
                sigma=self.sigma,
                min_size=self.min_size,
            )
            if mask_ is None:
                break
            mask_ = cv2.dilate(
                mask_.astype(
                    np.uint8),
                self.kernel,
                iterations=self.dilation_steps)
            image_masked = mask_ * image_gray

            if image_masked[image_masked > 0].mean(
            ) < self.background_gray_value:
                tissue_mask[mask_ != 0] = 1
                image[mask_ != 0] = (255, 255, 255)
            else:
                break
        tissue_mask = tissue_mask.astype(np.uint8)

        tissue_mask = self._upsample(
            tissue_mask, original_height, original_width)
        return tissue_mask
