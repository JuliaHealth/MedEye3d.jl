#!/usr/bin/env python3
"""
TotalSeg-Edge-PDE PET-Only Variant: Edge Detection from PET Images Only.

Architecture identical to TotalSegEdgePDE but with a PET-specific InputFusion:
  PETFusion(1ch PET → 1ch) → TotalSeg_UNet(1ch → 118 classes) → Adapter(118 → 1ch edge) → PDE → Mask

Fine-tuned from the best CT+PET checkpoint. Encoder/decoder weights are shared;
only the fusion block is re-initialized for single-channel PET input.

Rationale: PET images have strong metabolic contrast at lesion boundaries.
A PET-only model can exploit this without interference from CT normalization
artifacts that confuse the edge network on heterogeneous CT backgrounds.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared components from the base model
from totalseg_edge_pde_model import (
    ConvBlock, StackedConvBlock,
    TotalSegEncoder, TotalSegDecoder, EdgeAdapter,
)


# ═══════════════════════════════════════════════════════════════════
# PET-Only Input Fusion: 1ch PET → 1ch (TotalSeg-compatible)
# ═══════════════════════════════════════════════════════════════════

class PETFusion(nn.Module):
    """Adapt single-channel PET input for TotalSeg encoder.
    
    PET volumes have different contrast characteristics than CT:
    - High uptake in metabolically active regions (lesions)
    - Low/zero background outside the body
    - SUV values typically 0-20+
    
    Architecture: 1ch → 16ch → 16ch → 1ch with residual skip.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm3d(16, affine=True)
        self.conv2 = nn.Conv3d(16, 16, 3, padding=1, bias=True)
        self.norm2 = nn.InstanceNorm3d(16, affine=True)
        self.conv3 = nn.Conv3d(16, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        
        # Initialize so PET passes through with learned correction
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        nn.init.kaiming_normal_(self.conv1.weight)
        
    def forward(self, x):
        """x: (B, 1, D, H, W) — PET only."""
        pet = x  # (B, 1, D, H, W) for residual
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        h = self.conv3(h)
        return pet + h  # Residual: start from PET, learn correction


# ═══════════════════════════════════════════════════════════════════
# Full PET-Only End-to-End Model
# ═══════════════════════════════════════════════════════════════════

class TotalSegEdgePDE_PET(nn.Module):
    """
    PET-only variant of TotalSeg-Edge-PDE.
    
    Pipeline:
      PETFusion(PET → 1ch) → TotalSeg encoder-decoder(1ch → 118 seg logits)
      → EdgeAdapter(features + seg → 1ch edge) → Diffusivity → Heat PDE → Mask
    
    Identical encoder/decoder/adapter as the CT+PET model.
    Only InputFusion is replaced with PETFusion (1ch input).
    """
    
    # PDE parameter bounds (same as base model)
    GAMMA_RANGE = (3.0, 25.0)
    EPS_D_RANGE = (1e-5, 0.01)
    THETA_RANGE = (1e-5, 0.01)
    
    def __init__(self, 
                 gamma_init=11.0,
                 eps_D_init=0.001,
                 theta_init=0.0001,
                 K=200,
                 dt=0.16,
                 num_seg_classes=118,
                 ):
        super().__init__()
        
        # PET-specific fusion (1ch instead of 2ch)
        self.input_fusion = PETFusion()
        self.encoder = TotalSegEncoder(in_channels=1)
        self.decoder = TotalSegDecoder(num_classes=num_seg_classes)
        self.edge_adapter = EdgeAdapter(num_classes=num_seg_classes)
        
        # PDE hyperparams (learnable)
        self.log_gamma = nn.Parameter(torch.tensor(float(gamma_init)).log())
        self.log_eps_D = nn.Parameter(torch.tensor(float(eps_D_init)).log())
        self.logit_theta = nn.Parameter(torch.tensor(float(theta_init)).log() - torch.tensor(1.0 - float(theta_init)).log())
        
        self.K = K
        self.dt = dt
        self.tau = 0.0001
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
        with torch.no_grad():
            return {
                'gamma': self.gamma.cpu().item(),
                'eps_D': self.eps_D.cpu().item(),
                'theta': self.theta.cpu().item(),
                'K': self.K,
                'dt': self.dt,
            }
    
    def forward_edge(self, pet):
        """
        Forward pass through edge network only (no PDE).
        
        Args:
            pet: (B, 1, D, H, W) — PET only, normalized
            
        Returns:
            edge_prob: (B, D, H, W) edge probability ∈ [0, 1]
            edge_logit: (B, D, H, W) pre-sigmoid logit
            seg_logits: (B, 118, D, H, W)
        """
        x = self.input_fusion(pet)  # (B, 1, D, H, W)
        skips = self.encoder(x)
        dec_outputs = self.decoder(skips)
        seg_logits = self.decoder.seg_layers[-1](dec_outputs[-1])
        edge_prob, edge_logit = self.edge_adapter(dec_outputs, seg_logits)
        return edge_prob, edge_logit, seg_logits
    
    def forward(self, pet, seed_mask, return_edge=False):
        """
        Full end-to-end forward: PET → edge → PDE → mask.
        
        Args:
            pet: (B, 1, D, H, W) — PET only
            seed_mask: (B, D, H, W) — seed point mask
            return_edge: bool
        """
        from heat_pde_torch import diffuse_step
        
        edge_prob, edge_logit, seg_logits = self.forward_edge(pet)
        
        gamma = self.gamma
        eps_D = self.eps_D
        D = eps_D + (1.0 - eps_D) * torch.exp(-gamma * edge_prob)
        D = torch.clamp(D, max=1.0)
        D = torch.max(D, eps_D.detach().expand_as(D))
        
        u = seed_mask.clone()
        for k in range(self.K):
            u = diffuse_step(u, D, self.dt, stencil='arithmetic')
        
        theta = self.theta
        mask = torch.sigmoid((u - theta) / self.tau)
        
        if return_edge:
            return mask, edge_prob, edge_logit, seg_logits
        return mask
    
    def load_from_ctpet_checkpoint(self, checkpoint_path):
        """Load weights from the trained CT+PET TotalSegEdgePDE model.
        
        Transfers: encoder, decoder, edge_adapter, PDE params.
        Does NOT transfer: input_fusion (different architecture: 2ch→1ch vs 1ch→1ch).
        """
        print(f"Loading CT+PET checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        src_state = ckpt['model_state_dict']
        
        loaded, skipped = 0, 0
        our_state = self.state_dict()
        
        for key in our_state:
            if key.startswith('input_fusion.'):
                # Skip fusion layer — architecture mismatch (2ch vs 1ch)
                skipped += 1
                continue
            if key in src_state and src_state[key].shape == our_state[key].shape:
                our_state[key].copy_(src_state[key])
                loaded += 1
            else:
                skipped += 1
        
        self.load_state_dict(our_state)
        
        pde = ckpt.get('pde_params', {})
        print(f"  Loaded: {loaded} params, Skipped: {skipped} (fusion layer)")
        print(f"  Source PDE params: γ={pde.get('gamma','?')}, εD={pde.get('eps_D','?')}, θ={pde.get('theta','?')}")
        
        return loaded, skipped
    
    def count_params(self):
        counts = {}
        for name, module in [
            ('input_fusion', self.input_fusion),
            ('encoder', self.encoder),
            ('decoder', self.decoder),
            ('edge_adapter', self.edge_adapter),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['pde_params'] = 3
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


def test_model():
    """Quick smoke test."""
    print("Testing TotalSegEdgePDE_PET model...")
    model = TotalSegEdgePDE_PET(gamma_init=11.0, eps_D_init=0.001, theta_init=0.0001, K=5)
    
    counts = model.count_params()
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    
    print(f"\nPDE params: {model.get_pde_params()}")
    
    # Forward pass with PET-only input
    pet = torch.randn(1, 1, 32, 32, 32)  # Single channel PET
    seed = torch.zeros(1, 32, 32, 32)
    seed[0, 16, 16, 16] = 1.0
    
    with torch.no_grad():
        mask, edge, edge_logit, seg = model(pet, seed, return_edge=True)
    
    print(f"\nShapes:")
    print(f"  mask:       {mask.shape}")
    print(f"  edge:       {edge.shape}")
    print(f"  edge_logit: {edge_logit.shape}")
    print(f"  seg_logits: {seg.shape}")
    print(f"  mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"  edge range: [{edge.min():.4f}, {edge.max():.4f}]")
    
    # Test gradient flow
    model.train()
    pet.requires_grad_(True)
    mask2 = model(pet, seed)
    loss = mask2.mean()
    loss.backward()
    
    print(f"\nGradient check:")
    print(f"  log_gamma grad: {model.log_gamma.grad}")
    print(f"  log_eps_D grad: {model.log_eps_D.grad}")
    print(f"  logit_theta grad: {model.logit_theta.grad}")
    print(f"  input_fusion conv1 grad norm: {model.input_fusion.conv1.weight.grad.norm():.6f}")
    
    print("\n✅ PET-only model test passed!")


if __name__ == "__main__":
    test_model()
