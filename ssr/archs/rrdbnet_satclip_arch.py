from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Self-Attention Block ---
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=2)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

# --- Cross-Attention Block ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, img_channels, emb_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(img_channels, img_channels, 1)
        self.key_fc = nn.Linear(emb_dim, img_channels)
        self.value_fc = nn.Linear(emb_dim, img_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, loc_emb):
        B, C, H, W = x.size()
        query = self.query_conv(x).view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        key = self.key_fc(loc_emb).unsqueeze(1)  # (B, 1, C)
        value = self.value_fc(loc_emb).unsqueeze(1)  # (B, 1, C)
        attn = torch.softmax(torch.bmm(query, key.transpose(1, 2)) / (C ** 0.5), dim=2)  # (B, HW, 1)
        out = torch.bmm(attn, value)  # (B, HW, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return self.gamma * out + x

# --- MLP Projection ---
class MLPProjection(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# --- RRDB Block (import or define as in rrdbnet_arch.py) ---
from .rrdbnet_arch import RRDB

# --- Main Generator ---
@ARCH_REGISTRY.register()
class SSR_RRDBNet_LocAttn(nn.Module):
    def __init__(
        self,
        num_in_ch,
        num_out_ch,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        satclip_emb_dim=256,
        loc_emb_dim=64,
    ):
        super().__init__()
        self.scale = scale
        self.satclip_emb_dim = satclip_emb_dim
        self.loc_emb_dim = loc_emb_dim

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.loc_proj = MLPProjection(satclip_emb_dim, loc_emb_dim)

        self.upsample_blocks = nn.ModuleList()
        self.self_attn_blocks = nn.ModuleList()
        self.cross_attn_blocks = nn.ModuleList()
        n_upsample = int(torch.log2(torch.tensor(scale)).item())
        for _ in range(n_upsample):
            self.upsample_blocks.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.self_attn_blocks.append(SelfAttentionBlock(num_feat))
            self.cross_attn_blocks.append(CrossAttentionBlock(num_feat, loc_emb_dim))

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, coords=None, satclip_model=None):
        # x: (B, C, H, W), coords: (B, 2)
        with torch.no_grad():
            loc_emb = satclip_model(coords.double()).float()  # (B, satclip_emb_dim)
        loc_emb = self.loc_proj(loc_emb)  # (B, loc_emb_dim)
        # else:
        #     # fallback: zeros if no location info
        #     B = x.shape[0]
        #     device = x.device
        #     loc_emb = torch.zeros(B, self.loc_emb_dim, device=device)

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        for up, sa, ca in zip(self.upsample_blocks, self.self_attn_blocks, self.cross_attn_blocks):
            feat = self.lrelu(up(F.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = sa(feat)
            feat = ca(feat, loc_emb)

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out