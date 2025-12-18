# models/unet3d.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1):
        super().__init__()
        p = k//2
        g = min(16, max(1, c_out//2))
        self.block = nn.Sequential(
            nn.Conv3d(c_in, c_out, k, s, p, bias=False),
            nn.GroupNorm(g, c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class UNet3DInstBEV(nn.Module):
    def __init__(self, in_ch, n_classes, n_profiles, base=32, enable_bev_head=True, bev_band_only=True):
        super().__init__()
        self.enable_bev_head = enable_bev_head
        self.bev_band_only = bev_band_only

        self.enc1 = nn.Sequential(ConvGNReLU(in_ch, base), ConvGNReLU(base, base))
        self.down1 = nn.Conv3d(base, base*2, 2, 2)
        self.enc2 = nn.Sequential(ConvGNReLU(base*2, base*2), ConvGNReLU(base*2, base*2))
        self.down2 = nn.Conv3d(base*2, base*4, 2, 2)
        self.enc3 = nn.Sequential(ConvGNReLU(base*4, base*4), ConvGNReLU(base*4, base*4))
        self.down3 = nn.Conv3d(base*4, base*8, 2, 2)
        self.bott = nn.Sequential(ConvGNReLU(base*8, base*8), ConvGNReLU(base*8, base*8))

        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, 2)
        self.dec3 = nn.Sequential(ConvGNReLU(base*8, base*4), ConvGNReLU(base*4, base*4))
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.dec2 = nn.Sequential(ConvGNReLU(base*4, base*2), ConvGNReLU(base*2, base*2))
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, 2)
        self.dec1 = nn.Sequential(ConvGNReLU(base*2, base), ConvGNReLU(base, base))

        self.head_sem  = nn.Conv3d(base, n_classes, 1)
        self.head_prof = nn.Conv3d(base, n_profiles, 1) if n_profiles>0 else None
        self.head_ctr  = nn.Conv3d(base, 1, 1)
        self.head_off  = nn.Conv3d(base, 3, 1)

        if self.enable_bev_head:
            self.bev_reduce = nn.Conv3d(base, base, 1)
            self.bev_head = nn.Sequential(
                nn.Conv2d(base, base, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base, 1, 1)
            )

    def forward(self, x, band=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b  = self.bott(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logits_c  = self.head_sem(d1)
        logits_p  = self.head_prof(d1) if self.head_prof is not None else None
        logits_ctr= self.head_ctr(d1)
        logits_off= self.head_off(d1)

        logits_bev = None
        if self.enable_bev_head:
            F3 = self.bev_reduce(d1)  # (B,C,D,H,W)
            if (band is not None) and self.bev_band_only:
                F3 = F3 * band
            F_bev = F3.max(dim=4).values.transpose(2,3)  # (B,C,H,D)
            logits_bev = self.bev_head(F_bev)            # (B,1,H,D)

        return logits_c, logits_p, logits_ctr, logits_off, logits_bev
