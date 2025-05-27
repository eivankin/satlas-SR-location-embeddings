from compressai.zoo import mbt2018
import math
import torch

from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import to_pil_image

from basicsr.utils.registry import METRIC_REGISTRY

compression_model = mbt2018(8, pretrained=True)

@METRIC_REGISTRY.register()
def calculate_bpp(img, img2, **kwargs):
    tensor = torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0).float()/255
    out_net = compression_model.forward(tensor)
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()

@METRIC_REGISTRY.register()
def calculate_compressed_size(img, img2, **kwargs):
    tensor = torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0).float()/255
    out_net = compression_model.forward(tensor)
    x_hat = out_net['x_hat']

    image_before = to_pil_image(tensor[0])
    image_after = to_pil_image(x_hat[0])

    with BytesIO() as f:
        image_before.save(f, "webp", lossless=True, quality=100)
        f.seek(0)
        size_before = f.getbuffer().nbytes

    with BytesIO() as f:
        image_after.save(f, "webp", lossless=True, quality=100)
        f.seek(0)
        size_after = f.getbuffer().nbytes

    return size_after / size_before