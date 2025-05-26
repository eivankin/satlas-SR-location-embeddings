from compressai.zoo import mbt2018
import math
import torch

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
