import torch
import torch.nn as nn

from ssr.archs.osm_obj_discriminator_arch import OSMObjDiscriminator
from basicsr.utils.registry import ARCH_REGISTRY


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return self.lrelu(out + x)

class LocationProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

class LocationMatchingDiscriminator(nn.Module):
    def __init__(self, img_in_ch=3, base_feat=64, n_blocks=4, satclip_emb_dim=256, proj_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(img_in_ch, base_feat, 4, 2, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.blocks = nn.Sequential(
            *[ResidualBlock(base_feat) for _ in range(n_blocks)]
        )
        self.conv2 = nn.Conv2d(base_feat, proj_dim, 4, 2, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_uncond = nn.Linear(proj_dim, 1)
        self.loc_proj = LocationProjection(satclip_emb_dim, proj_dim)

    def forward(self, img, loc_emb):
        # img: (B, C, H, W), loc_emb: (B, satclip_emb_dim)
        feat = self.lrelu(self.conv1(img))
        feat = self.blocks(feat)
        feat = self.lrelu(self.conv2(feat))  # (B, proj_dim, H', W')
        pooled = self.pool(feat).view(feat.size(0), -1)  # (B, proj_dim)
        uncond_score = self.fc_uncond(pooled).squeeze(1)  # (B,)
        cond_proj = self.loc_proj(loc_emb)  # (B, proj_dim)
        cond_score = torch.sum(pooled * cond_proj, dim=1)  # (B,)
        return uncond_score + cond_score  # (B,)

@ARCH_REGISTRY.register()
class DoubleDiscriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, n_blocks=4, satclip_emb_dim=256, proj_dim=512, skip_connection=True):
        super(DoubleDiscriminator, self).__init__()
        self.loc_matching_discriminator = LocationMatchingDiscriminator(
            num_in_ch, num_feat, n_blocks, satclip_emb_dim, proj_dim
        )
        self.osm_obj_discriminator = OSMObjDiscriminator(
            num_in_ch=num_in_ch, num_feat=num_feat, skip_connection=skip_connection)

    def loc_matching(self, img, loc_emb):
        return self.loc_matching_discriminator(img, loc_emb)

    def osm_obj(self, x, osm_objs):
        return self.osm_obj_discriminator(x, osm_objs)