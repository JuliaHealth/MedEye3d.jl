#!/usr/bin/env python3
"""
Self-contained TotalSegEdgeNet v1 (High LR) model and diffusivity computation
for MedEye3d.jl CT edge detection.

Checkpoint: assets/models/totalseg_edge_v1_high_lr/best_model.pth
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_V1_CKPT = os.path.join(
    REPO_ROOT, "assets", "models", "totalseg_edge_v1_high_lr", "best_model.pth"
)


class ConvBlock(nn.Module):
    """Single conv block: Conv3d -> InstanceNorm -> LeakyReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=True)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class StackedConvBlock(nn.Module):
    """Two stacked conv blocks (matching nnUNet's StackedConvBlocks)."""
    def __init__(self, in_ch, out_ch, first_stride=1):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvBlock(in_ch, out_ch, stride=first_stride),
            ConvBlock(out_ch, out_ch, stride=1),
        ])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class TotalSegEncoder(nn.Module):
    """5-stage encoder matching TotalSegmentator's PlainConvUNet encoder."""
    def __init__(self, in_channels=2):
        super().__init__()
        channels = [32, 64, 128, 256, 320]
        self.stages = nn.ModuleList()
        self.stages.append(nn.ModuleList([StackedConvBlock(in_channels, channels[0], first_stride=1)]))
        for i in range(1, 5):
            self.stages.append(nn.ModuleList([
                StackedConvBlock(channels[i - 1], channels[i], first_stride=2)
            ]))

    def forward(self, x):
        skips = []
        for stage in self.stages:
            for block in stage:
                x = block(x)
            skips.append(x)
        return skips


class EdgeDecoder(nn.Module):
    """Edge detection decoder with HED-style multi-scale side outputs."""
    def __init__(self, encoder_channels=(32, 64, 128, 256, 320)):
        super().__init__()
        dec_channels = [256, 128, 64, 32]
        self.transpconvs = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(4):
            in_ch_transp = encoder_channels[4 - i]
            out_ch_transp = dec_channels[i]
            skip_ch = encoder_channels[3 - i]
            self.transpconvs.append(
                nn.ConvTranspose3d(in_ch_transp, out_ch_transp, kernel_size=2, stride=2)
            )
            self.stages.append(StackedConvBlock(out_ch_transp + skip_ch, dec_channels[i]))

        self.side_outputs = nn.ModuleList([
            nn.Conv3d(320, 1, 1),
            nn.Conv3d(256, 1, 1),
            nn.Conv3d(128, 1, 1),
            nn.Conv3d(64, 1, 1),
            nn.Conv3d(32, 1, 1),
        ])
        self.fusion = nn.Conv3d(5, 1, 1)

    def forward(self, skips):
        target_shape = skips[0].shape[2:]
        sides = [
            F.interpolate(self.side_outputs[0](skips[4]), size=target_shape, mode="trilinear", align_corners=False)
        ]
        x = skips[4]
        for i in range(4):
            x = self.transpconvs[i](x)
            skip = skips[3 - i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.stages[i](x)
            side = self.side_outputs[i + 1](x)
            if side.shape[2:] != target_shape:
                side = F.interpolate(side, size=target_shape, mode="trilinear", align_corners=False)
            sides.append(side)

        fused = torch.cat(sides, dim=1)
        edge_logit = self.fusion(fused)
        edge_prob = torch.clamp(torch.sigmoid(edge_logit), 1e-6, 1.0 - 1e-6)
        side_preds = [torch.clamp(torch.sigmoid(s), 1e-6, 1.0 - 1e-6) for s in sides]
        return edge_prob, side_preds, edge_logit, sides


class TotalSegEdgeNet(nn.Module):
    """Full edge detection model: TotalSeg encoder + edge decoder."""
    def __init__(self, in_channels=2):
        super().__init__()
        self.encoder = TotalSegEncoder(in_channels=in_channels)
        self.decoder = EdgeDecoder()

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)


def percentile_contrast_stretch(edge_map: torch.Tensor, p_low: float = 0.05, p_high: float = 0.95) -> torch.Tensor:
    flat = edge_map.reshape(-1)
    if flat.numel() > 16_000_000:
        # Subsample deterministically when tensor exceeds PyTorch quantile element limit
        step = max(1, flat.numel() // 8_000_000)
        sample = flat[::step]
        p_lo = torch.quantile(sample, p_low)
        p_hi = torch.quantile(sample, p_high)
    else:
        p_lo = torch.quantile(flat, p_low)
        p_hi = torch.quantile(flat, p_high)
    denom = (p_hi.detach() - p_lo.detach()).clamp(min=1e-6)
    return torch.clamp((edge_map - p_lo.detach()) / denom, 0.0, 1.0)


def compute_diffusivity(edge_map: torch.Tensor, eps_D: float = 0.001, gamma: float = 11.0,
                        contrast_stretch: bool = True, p_low: float = 0.05, p_high: float = 0.95) -> torch.Tensor:
    if contrast_stretch:
        edge_map = percentile_contrast_stretch(edge_map, p_low=p_low, p_high=p_high)
    D = eps_D + (1.0 - eps_D) * torch.exp(-gamma * edge_map)
    return torch.clamp(D, min=eps_D, max=1.0)


def load_v1_model(device: torch.device, ckpt_path: str = DEFAULT_V1_CKPT) -> TotalSegEdgeNet:
    model = TotalSegEdgeNet(in_channels=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def infer_ct_edge_and_diffusivity_v1(
    ct_arr: np.ndarray,
    device: torch.device,
    model: TotalSegEdgeNet = None,
    ckpt_path: str = DEFAULT_V1_CKPT,
    exclusion_mask: np.ndarray = None,
    patch_size: int = 64,
    step: int = 48,
    batch_size: int = 8,
) -> tuple:
    """Run sliding-window TotalSegEdgeNet v1 on CT (ch1=zeros) and return (edge_ct, diff_ct) float32."""
    owns_model = model is None
    if owns_model:
        model = load_v1_model(device, ckpt_path)

    ct_norm = ((np.clip(ct_arr.astype(np.float32), -1024.0, 1500.0) - (-200.0)) / 400.0).astype(np.float32)
    D, H, W = ct_norm.shape
    out_edge = torch.zeros((D, H, W), dtype=torch.float32, device=device)
    weight_map = torch.zeros((D, H, W), dtype=torch.float32, device=device)

    z = np.sin(np.linspace(0, np.pi, patch_size))
    y = np.sin(np.linspace(0, np.pi, patch_size))
    x = np.sin(np.linspace(0, np.pi, patch_size))
    w3d = torch.from_numpy((z[:, None, None] * y[None, :, None] * x[None, None, :]).astype(np.float32)).to(device)
    w3d = torch.clamp(w3d, min=1e-4)

    d_steps = list(range(0, max(1, D - patch_size + 1), step))
    if not d_steps or d_steps[-1] != max(0, D - patch_size):
        d_steps.append(max(0, D - patch_size))
    h_steps = list(range(0, max(1, H - patch_size + 1), step))
    if not h_steps or h_steps[-1] != max(0, H - patch_size):
        h_steps.append(max(0, H - patch_size))
    w_steps = list(range(0, max(1, W - patch_size + 1), step))
    if not w_steps or w_steps[-1] != max(0, W - patch_size):
        w_steps.append(max(0, W - patch_size))

    patches = [(d, h, w) for d in d_steps for h in h_steps for w in w_steps]
    ct_t = torch.from_numpy(ct_norm).to(device)

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            coords = patches[i:i + batch_size]
            b_inp = torch.zeros((len(coords), 2, patch_size, patch_size, patch_size), dtype=torch.float32, device=device)
            for j, (d, h, w) in enumerate(coords):
                b_inp[j, 0] = ct_t[d:d + patch_size, h:h + patch_size, w:w + patch_size]
            e_prob, _, _, _ = model(b_inp)
            e_prob = e_prob[:, 0]
            for j, (d, h, w) in enumerate(coords):
                out_edge[d:d + patch_size, h:h + patch_size, w:w + patch_size] += e_prob[j] * w3d
                weight_map[d:d + patch_size, h:h + patch_size, w:w + patch_size] += w3d

        out_edge /= torch.clamp(weight_map, min=1e-6)
        out_edge = torch.clamp(out_edge, 0.0, 1.0)
        diff_t = compute_diffusivity(out_edge, eps_D=0.001, gamma=11.0, contrast_stretch=True, p_low=0.05, p_high=0.95)
        edge_ct = out_edge.cpu().numpy().astype(np.float32)
        diff_ct = diff_t.cpu().numpy().astype(np.float32)

    if exclusion_mask is not None:
        diff_ct[exclusion_mask > 0] = 0.0

    del ct_t, out_edge, weight_map, w3d, diff_t
    if owns_model:
        del model
    torch.cuda.empty_cache()
    return edge_ct, diff_ct
