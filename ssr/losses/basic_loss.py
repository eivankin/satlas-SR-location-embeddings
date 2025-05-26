import math

import clip
import torch
import kornia
import open_clip
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Normalize

from compressai.zoo import mbt2018

from basicsr.losses.loss_util import weighted_loss
from basicsr.utils.registry import LOSS_REGISTRY

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@LOSS_REGISTRY.register()
class BPPLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(BPPLoss, self).__init__()
        self.loss_weight = loss_weight

        self.compression_model = mbt2018(8, pretrained=True)

    def forward(self, x, gt):
        """Bpp calculation from https://github.com/InterDigitalInc/CompressAI/blob/master/examples/CompressAI%20Inference%20Demo.ipynb"""
        out_net = self.compression_model.forward(x)
        size = out_net['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                   for likelihoods in out_net['likelihoods'].values())


@LOSS_REGISTRY.register()
class CLIPLoss(nn.Module):
    
    def __init__(self, clip_loss_model, loss_weight=1.0):
        super(CLIPLoss, self).__init__()
        self.loss_weight = loss_weight

        if clip_loss_model == 'EVA02-E-14-plus':
            self.img_size = (224,224)
            self.sim_model, _, _ = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained='laion2b_s9b_b144k')
        elif clip_loss_model == 'ViT-B-16-SigLIP-256':
            self.img_size = (256,256)
            self.sim_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP-256', pretrained='webli')
        elif clip_loss_model == 'RN50':
            self.img_size = (224,224)
            self.sim_model, _ = clip.load("RN50")

        self.normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        self.cos_emb_loss = nn.CosineEmbeddingLoss()

    def forward(self, x, gt):
        x = F.interpolate(x, self.img_size)
        gt = F.interpolate(gt, self.img_size)

        x = self.normalize(x)
        gt = self.normalize(gt)

        x_feats = self.sim_model.encode_image(x)
        gt_feats = self.sim_model.encode_image(gt)
        # l1 = l1_loss(x_feats, gt_feats)
        # return l1 * self.loss_weight
        cos_dissim = self.cos_emb_loss(x_feats, gt_feats, torch.ones(x_feats.shape[0]).to(x_feats.device))
        return cos_dissim * self.loss_weight

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, gt):
        ssim_loss = kornia.losses.ssim_loss(x, gt, window_size=5, reduction="none")
        ssim_loss = torch.mean(ssim_loss.mean(dim=(-1,-2,-3)))
        return ssim_loss * self.loss_weight
