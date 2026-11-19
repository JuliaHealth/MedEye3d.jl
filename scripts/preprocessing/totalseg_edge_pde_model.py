#!/usr/bin/env python3
"""
TotalSeg-Edge-PDE: End-to-End Lesion Segmentation with TotalSegmentator Backbone.

Architecture:
  InputFusion(2ch CT+PET → 1ch) → TotalSeg_UNet(1ch → 118 classes) → Adapter(118 → 1ch edge) → PDE → Mask

The model fine-tunes TotalSegmentator's PlainConvUNet encoder-decoder to produce
edge probability maps, then applies differentiable heat diffusion PDE for seed-based
lesion segmentation.

PDE parameters (γ, ε_D, θ) are learnable, initialized from the hyperparameter sweep.

Training data: Combined PET/CT + CT-only lesion datasets (7,129 patches).
CT-only patches have PET=0; the input fusion block learns to handle both modalities.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Building blocks matching nnUNet's PlainConvUNet
# ═══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv3d → InstanceNorm3d → LeakyReLU (matches nnUNet)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=True)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        # Create all_modules as a list for weight loading compatibility
        self.all_modules = nn.ModuleList([self.conv, self.norm])

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


# ═══════════════════════════════════════════════════════════════════
# Input Fusion: PET + CT → 1 channel (TotalSeg-compatible)
# ═══════════════════════════════════════════════════════════════════

class InputFusion(nn.Module):
    """Fuse 2-channel (CT + PET) input into 1-channel for TotalSeg encoder.
    
    For CT-only patches (PET=0), learns to pass CT through.
    For PET/CT patches, learns complementary fusion.
    
    Architecture: 2ch → 16ch → 16ch → 1ch with residual skip.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(2, 16, 3, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm3d(16, affine=True)
        self.conv2 = nn.Conv3d(16, 16, 3, padding=1, bias=True)
        self.norm2 = nn.InstanceNorm3d(16, affine=True)
        self.conv3 = nn.Conv3d(16, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        
        # Initialize so CT channel passes through by default
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        # Make first filter respond to CT channel
        nn.init.kaiming_normal_(self.conv1.weight)
        
    def forward(self, x):
        """x: (B, 2, D, H, W) — ch0=CT, ch1=PET."""
        ct = x[:, 0:1]  # (B, 1, D, H, W) for residual
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        h = self.conv3(h)
        return ct + h  # Residual: start from CT, learn correction from PET


# ═══════════════════════════════════════════════════════════════════
# TotalSeg Encoder (PlainConvUNet encoder, 5 stages)
# ═══════════════════════════════════════════════════════════════════

class TotalSegEncoder(nn.Module):
    """5-stage encoder matching TotalSegmentator's PlainConvUNet.
    
    Stages: [1→32, 32→64, 64→128, 128→256, 256→320]
    Downsampling via stride-2 in first conv of stages 1-4.
    """
    CHANNELS = [32, 64, 128, 256, 320]
    
    def __init__(self, in_channels=1):
        super().__init__()
        ch = self.CHANNELS
        self.stages = nn.ModuleList()
        
        # Stage 0: in → 32 (no downsampling)
        self.stages.append(nn.ModuleList([StackedConvBlock(in_channels, ch[0], first_stride=1)]))
        
        # Stages 1-4: stride-2 downsample
        for i in range(1, 5):
            self.stages.append(nn.ModuleList([
                StackedConvBlock(ch[i-1], ch[i], first_stride=2)
            ]))
    
    def forward(self, x):
        skips = []
        for stage in self.stages:
            for block in stage:
                x = block(x)
            skips.append(x)
        return skips


# ═══════════════════════════════════════════════════════════════════
# TotalSeg Decoder (PlainConvUNet decoder)
# ═══════════════════════════════════════════════════════════════════

class TotalSegDecoder(nn.Module):
    """4-stage decoder matching TotalSegmentator's UNet decoder.
    
    transpconv upsampling + skip concatenation + conv blocks.
    Multi-scale segmentation heads (seg_layers) at each scale.
    """
    def __init__(self, num_classes=118):
        super().__init__()
        enc_ch = [32, 64, 128, 256, 320]
        dec_ch = [256, 128, 64, 32]
        
        self.transpconvs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        
        for i in range(4):
            # Transposed conv: upsample from deeper to current scale
            self.transpconvs.append(
                nn.ConvTranspose3d(enc_ch[4-i] if i == 0 else dec_ch[i-1],
                                   dec_ch[i], kernel_size=2, stride=2)
            )
            # After concat with skip: dec_ch[i] + enc_ch[3-i] → dec_ch[i]
            self.stages.append(StackedConvBlock(dec_ch[i] + enc_ch[3-i], dec_ch[i]))
            # Segmentation head at this scale
            self.seg_layers.append(nn.Conv3d(dec_ch[i], num_classes, 1))
        
    def forward(self, skips):
        """skips: [s0(32), s1(64), s2(128), s3(256), s4(320)]"""
        x = skips[4]  # Bottleneck
        decoder_outputs = []
        
        for i in range(4):
            x = self.transpconvs[i](x)
            skip = skips[3 - i]
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.stages[i](x)
            decoder_outputs.append(x)
        
        return decoder_outputs  # [dec0(256), dec1(128), dec2(64), dec3(32)]


# ═══════════════════════════════════════════════════════════════════
# Output Adapter: 118-class seg → 1-channel edge probability
# ═══════════════════════════════════════════════════════════════════

class EdgeAdapter(nn.Module):
    """Collapse TotalSeg's multi-class outputs to a single edge probability.
    
    Uses multi-scale features from decoder + seg_layers.
    Final output: edge probability map ∈ [0, 1].
    """
    def __init__(self, num_classes=118, dec_channels=[256, 128, 64, 32]):
        super().__init__()
        # Process full-res decoder output (32ch) with seg logits context
        self.pre_conv = nn.Sequential(
            nn.Conv3d(32 + num_classes, 64, 3, padding=1, bias=True),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(64, 32, 3, padding=1, bias=True),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.edge_head = nn.Conv3d(32, 1, 1, bias=True)
        
    def forward(self, decoder_outputs, seg_logits_fullres):
        """
        decoder_outputs: list from decoder, last one is full-res 32ch
        seg_logits_fullres: (B, 118, D, H, W) upsampled seg logits
        """
        x = decoder_outputs[-1]  # (B, 32, D, H, W) full-res features
        # Concatenate full-res features with seg logits for rich context
        x = torch.cat([x, seg_logits_fullres], dim=1)  # (B, 32+118, D, H, W)
        x = self.pre_conv(x)
        edge_logit = self.edge_head(x)  # (B, 1, D, H, W)
        edge_prob = torch.sigmoid(edge_logit)
        return edge_prob.squeeze(1), edge_logit.squeeze(1)  # (B, D, H, W)


# ═══════════════════════════════════════════════════════════════════
# Full End-to-End Model
# ═══════════════════════════════════════════════════════════════════

class TotalSegEdgePDE(nn.Module):
    """
    End-to-End TotalSeg-based edge detection + heat PDE segmentation.
    
    Pipeline:
      InputFusion(CT+PET → 1ch) → TotalSeg encoder-decoder(1ch → 118 seg logits)
      → EdgeAdapter(features + seg → 1ch edge) → Diffusivity → Heat PDE → Mask
    
    Learnable PDE parameters:
      - log_gamma: log(γ), initialized from sweep champion
      - log_eps_D: log(ε_D), minimum diffusivity floor  
      - logit_theta: logit(θ), mask extraction threshold
    
    All PDE ops use native PyTorch → differentiable + Julia-portable.
    """
    
    # PDE parameter bounds (enforced via reparameterization)
    GAMMA_RANGE = (3.0, 25.0)    # Edge sensitivity
    EPS_D_RANGE = (1e-5, 0.01)   # Min diffusivity
    THETA_RANGE = (1e-5, 0.01)   # Mask threshold
    
    def __init__(self, 
                 # PDE init from sweep champion
                 gamma_init=11.0,
                 eps_D_init=0.001,
                 theta_init=0.0001,
                 K=200,           # Fixed number of PDE steps
                 dt=0.16,
                 num_seg_classes=118,
                 ):
        super().__init__()
        
        # Components
        self.input_fusion = InputFusion()
        self.encoder = TotalSegEncoder(in_channels=1)
        self.decoder = TotalSegDecoder(num_classes=num_seg_classes)
        self.edge_adapter = EdgeAdapter(num_classes=num_seg_classes)
        
        # PDE hyperparams (learnable, reparameterized)
        self.log_gamma = nn.Parameter(torch.tensor(float(gamma_init)).log())
        self.log_eps_D = nn.Parameter(torch.tensor(float(eps_D_init)).log())
        self.logit_theta = nn.Parameter(torch.tensor(float(theta_init)).log() - torch.tensor(1.0 - float(theta_init)).log())
        
        # Fixed PDE params
        self.K = K
        self.dt = dt
        self.tau = 0.0001  # Sigmoid sharpness for mask extraction
        
        self.num_seg_classes = num_seg_classes
    
    @property
    def gamma(self):
        return self.log_gamma.exp().clamp(*self.GAMMA_RANGE)
    
    @property
    def eps_D(self):
        return self.log_eps_D.exp().clamp(*self.EPS_D_RANGE)
    
    @property
    def theta(self):
        return torch.sigmoid(self.logit_theta).clamp(*self.THETA_RANGE)
    
    def get_pde_params(self):
        """Return current PDE parameter values (detached, moved to CPU for logging)."""
        with torch.no_grad():
            return {
                'gamma': self.gamma.cpu().item(),
                'eps_D': self.eps_D.cpu().item(),
                'theta': self.theta.cpu().item(),
                'K': self.K,
                'dt': self.dt,
            }
    
    def forward_edge(self, ct_pet):
        """
        Forward pass through edge network only (no PDE).
        
        Args:
            ct_pet: (B, 2, D, H, W) — ch0=CT z-score, ch1=PET normalized
            
        Returns:
            edge_prob: (B, D, H, W) edge probability ∈ [0, 1]
            edge_logit: (B, D, H, W) pre-sigmoid logit
        """
        # Input fusion: 2ch → 1ch
        x = self.input_fusion(ct_pet)  # (B, 1, D, H, W)
        
        # TotalSeg encoder
        skips = self.encoder(x)
        
        # TotalSeg decoder
        dec_outputs = self.decoder(skips)
        
        # Get full-res seg logits from last seg_layer
        seg_logits = self.decoder.seg_layers[-1](dec_outputs[-1])  # (B, 118, D, H, W)
        
        # Edge adapter
        edge_prob, edge_logit = self.edge_adapter(dec_outputs, seg_logits)
        
        return edge_prob, edge_logit, seg_logits
    
    def forward(self, ct_pet, seed_mask, return_edge=False):
        """
        Full end-to-end forward: image → edge → PDE → mask.
        
        Args:
            ct_pet: (B, 2, D, H, W) — ch0=CT z-score, ch1=PET normalized
            seed_mask: (B, D, H, W) — seed point mask (1 at seed, 0 elsewhere)
            return_edge: bool, also return edge map for auxiliary losses
            
        Returns:
            mask: (B, D, H, W) soft segmentation mask ∈ [0, 1]
            edge_prob: (B, D, H, W) edge probability (if return_edge=True)
        """
        from heat_pde_torch import compute_diffusivity, heat_diffusion, extract_mask, diffuse_step
        
        # Edge detection
        edge_prob, edge_logit, seg_logits = self.forward_edge(ct_pet)
        
        # Diffusivity from edges (NO .item() calls — driver segfaults on cuStreamSynchronize)
        gamma = self.gamma
        eps_D = self.eps_D
        D = eps_D + (1.0 - eps_D) * torch.exp(-gamma * edge_prob)
        # Use torch.clamp with tensor min to avoid .item() host-device sync
        D = torch.clamp(D, max=1.0)
        D = torch.max(D, eps_D.detach().expand_as(D))
        
        # Heat PDE diffusion
        # diffuse_step expects 4D tensors (B, X, Y, Z), but u and D are 5D (B, C, X, Y, Z)
        u = seed_mask.squeeze(1).clone()
        D_4d = D.squeeze(1)
        for k in range(self.K):
            u = diffuse_step(u, D_4d, self.dt, stencil='arithmetic', u_seed=seed_mask.squeeze(1))
        u = u.unsqueeze(1)
        
        # Extract mask
        theta = self.theta
        mask = torch.sigmoid((u - theta) / self.tau)
        
        if return_edge:
            return mask, edge_prob, edge_logit, seg_logits
        return mask
    
    def load_totalseg_weights(self, checkpoint_path):
        """Load pretrained TotalSegmentator encoder+decoder weights.
        
        Maps nnUNet key format to our key format.
        First conv is kept at 1ch (fusion block handles 2ch→1ch).
        """
        print(f"Loading TotalSegmentator weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        weights = ckpt['network_weights']
        
        loaded_enc, loaded_dec, skipped = 0, 0, 0
        
        # Load encoder weights
        for our_key, param in self.encoder.state_dict().items():
            nnunet_key = 'encoder.' + our_key
            if nnunet_key in weights:
                src = weights[nnunet_key]
                if src.shape == param.shape:
                    param.data.copy_(src)
                    loaded_enc += 1
                else:
                    print(f"  Shape mismatch: {our_key}: ckpt={src.shape} model={param.shape}")
                    skipped += 1
            else:
                skipped += 1
        
        # Load decoder weights  
        for our_key, param in self.decoder.state_dict().items():
            nnunet_key = 'decoder.' + our_key
            if nnunet_key in weights:
                src = weights[nnunet_key]
                if src.shape == param.shape:
                    param.data.copy_(src)
                    loaded_dec += 1
                else:
                    skipped += 1
            else:
                skipped += 1
        
        print(f"  Encoder: {loaded_enc} loaded")
        print(f"  Decoder: {loaded_dec} loaded")
        print(f"  Skipped: {skipped}")
        
        return loaded_enc, loaded_dec
    
    def count_params(self):
        """Count parameters by component."""
        counts = {}
        for name, module in [
            ('input_fusion', self.input_fusion),
            ('encoder', self.encoder),
            ('decoder', self.decoder),
            ('edge_adapter', self.edge_adapter),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['pde_params'] = 3  # gamma, eps_D, theta
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


def test_model():
    """Quick smoke test."""
    print("Testing TotalSegEdgePDE model...")
    model = TotalSegEdgePDE(gamma_init=11.0, eps_D_init=0.001, theta_init=0.0001, K=5)
    
    counts = model.count_params()
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    
    print(f"\nPDE params: {model.get_pde_params()}")
    
    # Forward pass
    x = torch.randn(1, 2, 32, 32, 32)
    seed = torch.zeros(1, 32, 32, 32)
    seed[0, 16, 16, 16] = 1.0
    
    with torch.no_grad():
        mask, edge, edge_logit, seg = model(x, seed, return_edge=True)
    
    print(f"\nShapes:")
    print(f"  mask:       {mask.shape}")
    print(f"  edge:       {edge.shape}")
    print(f"  edge_logit: {edge_logit.shape}")
    print(f"  seg_logits: {seg.shape}")
    print(f"  mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"  edge range: [{edge.min():.4f}, {edge.max():.4f}]")
    
    # Test gradient flow
    model.train()
    x.requires_grad_(True)
    mask2 = model(x, seed)
    loss = mask2.mean()
    loss.backward()
    
    print(f"\nGradient check:")
    print(f"  log_gamma grad: {model.log_gamma.grad}")
    print(f"  log_eps_D grad: {model.log_eps_D.grad}")
    print(f"  logit_theta grad: {model.logit_theta.grad}")
    print(f"  input_fusion conv1 grad norm: {model.input_fusion.conv1.weight.grad.norm():.6f}")
    
    print("\n✅ Model test passed!")


if __name__ == "__main__":
    test_model()
