import torch
import torch.nn as nn
import torch.nn.functional as F

def GN(c: int) -> nn.GroupNorm:
    """
    Pick the largest group size ≤ 8 that divides c.
    Falls back to LayerNorm-like behavior with 1 group if needed.
    """
    for g in (8, 6, 4, 3, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)  # safety fallback

class SeperableConv(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        pad = dilation
        self.dw = nn.Conv2d(cin, cin, 3, padding=pad, dilation=dilation, groups=cin, bias=False)
        self.bn1 = GN(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = GN(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class DoubleSeperableConv(nn.Module):
    def __init__(self, cin, cout, dropout_p=0.0, dilation=1):
        super().__init__()
        self.c1 = SeperableConv(cin, cout, dilation=dilation)
        self.c2 = SeperableConv(cout, cout, dilation=1)
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
    def forward(self, x):
        return self.drop(self.c2(self.c1(x)))

class Up(nn.Module):
    """Bilinear upsample -> 1x1 reduce; project skip -> concat"""
    def __init__(self, cin, cout, skip_in, skip_proj_to):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn_up = GN(cout)
        self.skip_proj = nn.Conv2d(skip_in, skip_proj_to, 1, bias=False)
        self.bn_skip = GN(skip_proj_to)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, skip):
        x = self.act(self.bn_up(self.reduce(self.up(x))))
        skip = self.act(self.bn_skip(self.skip_proj(skip)))
        return torch.cat([x, skip], dim=1)

class LACM(nn.Module):
    """ASPP-lite with reduced channels per branch + shared fuse."""
    def __init__(self, channels, dilations=(1, 3, 5), reduction=2):
        super().__init__()
        br_c = channels // reduction
        self.branches = nn.ModuleList([
            SeperableConv(channels, br_c, dilation=d) for d in dilations
        ])
        fused_in = br_c * len(dilations)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, channels, 1, bias=False),
            GN(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.fuse(torch.cat(feats, dim=1))

class BoundaryRefinementHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            GN(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, 1)
        )
    def forward(self, x):
        bl = self.conv(x)
        bp = torch.sigmoid(bl)
        return bp, bl

class UNetDW512_BR_LACM(nn.Module):
    def __init__(self, out_channels=1, bottleneck_drop=0.1, return_boundary=False):
        super().__init__()
        self.return_boundary = return_boundary

        # ↓ slightly trimmed channels to drop MACs without hurting params
        self.enc1 = DoubleSeperableConv(3,   32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleSeperableConv(32,  64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleSeperableConv(64,  112)   # 128→112
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleSeperableConv(112, 160)   # 192→160

        self.lacm   = LACM(160, dilations=(1,3,5), reduction=2)
        self.bottle = DoubleSeperableConv(160, 224, dropout_p=bottleneck_drop)  # 256→224

        self.up1 = Up(cin=224, cout=112, skip_in=112, skip_proj_to=56)
        self.dec1 = DoubleSeperableConv(168, 112)   # 112+56
        self.up2 = Up(cin=112, cout=56,  skip_in=64,  skip_proj_to=32)
        self.dec2 = DoubleSeperableConv(88,  56)
        self.up3 = Up(cin=56,  cout=28,  skip_in=32,  skip_proj_to=16)
        self.dec3 = DoubleSeperableConv(44,  32)    # end at 32 for head

        self.brh  = BoundaryRefinementHead(32)
        self.head = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        c  = self.lacm(e4)
        b  = self.bottle(c)

        x  = self.dec1(self.up1(b, e3))
        x  = self.dec2(self.up2(x, e2))
        x  = self.dec3(self.up3(x, e1))

        boundary_prob, boundary_logits = self.brh(x)
        x = x * (1.0 + boundary_prob)   # residual edge gating
        mask_logits = self.head(x)

        if self.return_boundary:
            return {"logits": mask_logits, "boundary_logits": boundary_logits}
        return mask_logits

