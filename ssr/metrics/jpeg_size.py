from typing import Union

from basicsr.utils.registry import METRIC_REGISTRY

import torch
from torch import Tensor
from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import to_pil_image
import numpy as np


def size(image: Union[Tensor, Image.Image], **kwargs) -> float:
    r"""
    Adapted from: https://github.com/sprout-ai/torchjpeg/blob/main/src/torchjpeg/metrics/_size.py
    Computes the size in bytes of a JPEG

    Args:
        image (Tensor or PIL Image): The image to compress
        kwargs: Arguments to pass to the PIL JPEG compressor (like quality or quantization matrices)

    Returns
    -------
        Tensor
            A single element tensor containing the size in bytes of the image after JPEG compression

    Warning:
        The output of this function is **not** differentiable. It compresses the image to memory and reads the size of
        the resulting buffer.
    """
    if isinstance(image, Tensor) or isinstance(image, np.ndarray):
        pimage: Image.Image = to_pil_image(image)
    else:
        pimage = image

    with BytesIO() as f:
        pimage.save(f, "png", optimize=False, compress_level=0)
        f.seek(0)
        size_before = f.getbuffer().nbytes

    with BytesIO() as f:
        pimage.save(f, "jpeg", **kwargs)
        f.seek(0)
        size_after = f.getbuffer().nbytes

    return size_after / size_before

@METRIC_REGISTRY.register()
def calculate_jpeg_size(img, img2, quality):
    return size(img, quality=quality)