"""
HMambaEncoderWrapper: Integrates H-Net Dynamic Chunking with ConMamba encoder layers.

This wrapper takes a ConmambaEncoder and splits its layers into:
- Stage 0 (layers 0-5): Frame-level processing at full resolution
- DC Layer: Dynamic Chunking (learned compression via boundary detection)
- Stage 1 (layers 6-11): Chunk-level processing on compressed sequence
- DeChunk: Restore original temporal resolution via EMA interpolation

Architecture:
    Input (B, L, D)
        ↓
    Stage 0: layers[0:split_idx] (frame-level)
        ↓
    DC: RoutingModule → ChunkLayer (compress L → M)
        ↓
    Stage 1: layers[split_idx:] (chunk-level, on M frames)
        ↓
    DeChunk: Expand M → L via EMA
        ↓
    Residual + Norm
        ↓
    Output (B, L, D)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from modules.HMambaEncoder import (
    RoutingModule,
    RoutingModuleOutput,
    ChunkLayer,
    DeChunkLayer,
    load_balancing_loss,
    DCStats,
)
from modules.Conmamba import ConmambaEncoder


class HMambaEncoderWrapper(nn.Module):
    """
    Wraps a ConmambaEncoder with H-Net Dynamic Chunking.
    
    Args:
        conmamba_encoder: ConmambaEncoder instance with layers to split
        d_model: Model dimension (default 144 for ConMamba-Small)
        split_idx: Index to split layers (default 6: layers 0-5 for stage0, 6-11 for stage1)
        target_compression_N: Target downsampling factor (N=2 means r≈0.5, keep 50% of frames)
        headdim: Head dimension for EMA in DeChunkLayer (default 36, gives 4 heads for d=144)
    """
    
    def __init__(
        self,
        conmamba_encoder: ConmambaEncoder,
        d_model: int = 144,
        split_idx: int = 6,
        target_compression_N: float = 2.0,
        headdim: int = 36,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.split_idx = split_idx
        self.target_compression_N = target_compression_N
        self.headdim = headdim
        
        # Extract and split layers from ConmambaEncoder
        all_layers = list(conmamba_encoder.layers)
        num_layers = len(all_layers)
        
        assert split_idx < num_layers, f"split_idx {split_idx} must be < num_layers {num_layers}"
        assert d_model % headdim == 0, f"d_model {d_model} must be divisible by headdim {headdim}"
        
        # Stage 0: First half of layers (frame-level processing)
        self.stage0_layers = nn.ModuleList(all_layers[:split_idx])
        
        # Stage 1: Second half of layers (chunk-level processing)
        self.stage1_layers = nn.ModuleList(all_layers[split_idx:])
        
        # Copy the final norm from original encoder
        self.norm = conmamba_encoder.norm
        
        # Dynamic Chunking components
        self.routing_module = RoutingModule(d_model)
        self.chunk_layer = ChunkLayer()
        self.dechunk_layer = DeChunkLayer(d_model=d_model, headdim=headdim)
        
        # Residual projection (zero-initialized for stable training start)
        self.residual_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.residual_proj.weight)
        self.residual_proj.weight._no_reinit = True
        
        print(f"[HMambaEncoderWrapper] Created with {len(self.stage0_layers)} stage0 layers, "
              f"{len(self.stage1_layers)} stage1 layers")
        print(f"[HMambaEncoderWrapper] Target compression N={target_compression_N} (r≈{1/target_compression_N:.2f})")
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[Any] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass with Dynamic Chunking.
        
        Args:
            src: Input tensor (B, L, D)
            src_mask: Source mask (unused, for API compatibility)
            src_key_padding_mask: Padding mask (B, L) where True = padded position
            pos_embs: Positional embeddings (unused by Mamba)
            dynchunktrain_config: Dynamic chunk training config (unused, for API compatibility)
            return_stats: If True, return compression statistics dict
            
        Returns:
            output: (B, L, D) encoded features at original resolution
            stats: dict with compression stats if return_stats=True, else None
        """
        B, L, D = src.shape
        device = src.device
        
        # Create validity mask from padding mask
        # src_key_padding_mask: True = padded (invalid), we need True = valid
        if src_key_padding_mask is not None:
            validity_mask = ~src_key_padding_mask  # (B, L), True = valid
        else:
            validity_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        
        # ==================== Stage 0: Frame-level processing ====================
        x = src
        for layer in self.stage0_layers:
            x = layer(
                x,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
                dynchunktrain_config=dynchunktrain_config,
            )
        
        stage0_out = x  # (B, L, D)
        
        # Save residual for skip connection (zero-initialized projection)
        residual = self.residual_proj(stage0_out.float()).to(stage0_out.dtype)
        
        # ==================== Dynamic Chunking ====================
        # Get boundary predictions from routing module
        router_output: RoutingModuleOutput = self.routing_module(
            stage0_out, 
            mask=validity_mask,
        )
        # router_output.boundary_prob: (B, L, 2) - [P(not-boundary), P(boundary)]
        # router_output.boundary_mask: (B, L) - True = boundary position
        # router_output.selected_probs: (B, L, 1) - probability of selected action
        
        # Compress sequence by selecting boundary frames
        chunked_states, chunk_mask, max_chunks = self.chunk_layer(
            stage0_out, 
            router_output.boundary_mask, 
            mask=validity_mask,
        )
        # chunked_states: (B, M, D) where M = max boundaries in batch
        # chunk_mask: (B, M) validity mask for chunked sequence
        
        # ==================== Stage 1: Chunk-level processing ====================
        x = chunked_states
        for layer in self.stage1_layers:
            # Note: Stage 1 operates on compressed sequence without original masks
            x = layer(
                x,
                src_mask=None,
                src_key_padding_mask=None,
                pos_embs=None,
                dynchunktrain_config=None,
            )
        
        stage1_out = x  # (B, M, D)
        
        # ==================== DeChunk: Expand back to original length ====================
        expanded_states = self.dechunk_layer(
            stage1_out,
            router_output.boundary_mask,
            router_output.boundary_prob,
            mask=validity_mask,
        )  # (B, L, D)
        
        # ==================== Residual connection + Normalization ====================
        # Combine stage0 residual with expanded stage1 output
        output = stage0_out + residual + expanded_states
        
        # Apply final layer norm
        output = self.norm(output)
        
        # ==================== Compute stats if requested ====================
        if return_stats:
            num_boundaries = router_output.boundary_mask.float().sum().item()
            total_valid = validity_mask.float().sum().item()
            compression_ratio = num_boundaries / total_valid if total_valid > 0 else 0
            
            dc_loss = load_balancing_loss(router_output, N=self.target_compression_N)
            
            stats = {
                "compression_ratio": compression_ratio,
                "num_chunks": max_chunks,
                "avg_chunk_size": total_valid / num_boundaries if num_boundaries > 0 else 0,
                "dc_loss": dc_loss,
                "boundary_prob_mean": router_output.boundary_prob[..., 1].mean().item(),
            }
            return output, stats
        
        return output, None
    
    def compute_dc_loss(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute DC load balancing loss by running stage0 and routing only.
        
        This is useful during training when you need the DC loss term separately.
        Note: For efficiency, prefer using return_stats=True in forward() instead.
        
        Args:
            src: Input tensor (B, L, D)
            src_key_padding_mask: Padding mask (B, L) where True = padded
            
        Returns:
            dc_loss: Scalar loss tensor for load balancing
        """
        B, L, D = src.shape
        device = src.device
        
        # Create validity mask
        if src_key_padding_mask is not None:
            validity_mask = ~src_key_padding_mask
        else:
            validity_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        
        # Run through stage0
        x = src
        for layer in self.stage0_layers:
            x = layer(x)
        
        # Get boundary predictions
        router_output = self.routing_module(x, mask=validity_mask)
        
        # Compute and return load balancing loss
        dc_loss = load_balancing_loss(router_output, N=self.target_compression_N)
        
        return dc_loss
    
    def get_compression_stats(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> DCStats:
        """
        Get compression statistics without full forward pass.
        
        Args:
            src: Input tensor (B, L, D)
            src_key_padding_mask: Padding mask
            
        Returns:
            DCStats with compression_ratio, num_chunks, avg_chunk_size
        """
        B, L, D = src.shape
        device = src.device
        
        if src_key_padding_mask is not None:
            validity_mask = ~src_key_padding_mask
        else:
            validity_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        
        # Run stage0
        x = src
        with torch.no_grad():
            for layer in self.stage0_layers:
                x = layer(x)
            
            router_output = self.routing_module(x, mask=validity_mask)
            _, _, max_chunks = self.chunk_layer(x, router_output.boundary_mask, mask=validity_mask)
        
        num_boundaries = router_output.boundary_mask.sum().item()
        total_valid = validity_mask.sum().item()
        
        return DCStats(
            compression_ratio=num_boundaries / total_valid if total_valid > 0 else 0,
            num_chunks=max_chunks,
            avg_chunk_size=total_valid / num_boundaries if num_boundaries > 0 else 0,
        )


def create_hmamba_from_conmamba(
    conmamba_encoder: ConmambaEncoder,
    d_model: int = 144,
    split_idx: int = 6,
    target_compression_N: float = 2.0,
    headdim: int = 36,
) -> HMambaEncoderWrapper:
    """
    Factory function to create HMambaEncoderWrapper from a ConmambaEncoder.
    
    Args:
        conmamba_encoder: ConmambaEncoder instance (typically with 12 layers)
        d_model: Model dimension (must match encoder)
        split_idx: Where to split layers for DC insertion (default 6 for 6-6 split)
        target_compression_N: Compression target (N=2 → 50%, N=3 → 33%)
        headdim: Head dimension for EMA (d_model must be divisible by this)
        
    Returns:
        HMambaEncoderWrapper instance ready for training
    """
    return HMambaEncoderWrapper(
        conmamba_encoder=conmamba_encoder,
        d_model=d_model,
        split_idx=split_idx,
        target_compression_N=target_compression_N,
        headdim=headdim,
    )


# ==================== Test ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing HMambaEncoderWrapper")
    print("=" * 60)
    
    from modules.Conmamba import ConmambaEncoder
    
    # Configuration
    d_model = 144
    num_layers = 12
    d_ffn = 1024
    split_idx = 6
    
    mamba_config = {
        "d_state": 16,
        "expand": 2,
        "d_conv": 4,
        "bidirectional": True,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create ConmambaEncoder
    print("\nCreating ConmambaEncoder...")
    conmamba = ConmambaEncoder(
        num_layers=num_layers,
        d_model=d_model,
        d_ffn=d_ffn,
        kernel_size=31,
        dropout=0.1,
        causal=False,
        mamba_config=mamba_config,
    )
    print(f"ConmambaEncoder layers: {len(conmamba.layers)}")
    
    # Wrap with HMamba
    print("\nCreating HMambaEncoderWrapper...")
    hmamba = create_hmamba_from_conmamba(
        conmamba,
        d_model=d_model,
        split_idx=split_idx,
        target_compression_N=2.0,
        headdim=36,
    )
    hmamba = hmamba.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in hmamba.parameters())
    dc_params = sum(p.numel() for n, p in hmamba.named_parameters() 
                    if 'routing' in n or 'chunk' in n or 'residual_proj' in n)
    print(f"Total params: {total_params:,}")
    print(f"DC overhead: {dc_params:,} ({100*dc_params/total_params:.2f}%)")
    
    # Test forward pass
    print("\n--- Forward Pass Test ---")
    B, L, D = 2, 100, d_model
    x = torch.randn(B, L, D, device=device)
    
    # Create padding mask (second sequence is shorter)
    src_key_padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    src_key_padding_mask[1, 80:] = True  # Pad last 20 frames of second sequence
    
    print(f"Input shape: {x.shape}")
    print(f"Valid frames: {(~src_key_padding_mask).sum(dim=1).tolist()}")
    
    # Forward with stats
    hmamba.train()
    output, stats = hmamba(x, src_key_padding_mask=src_key_padding_mask, return_stats=True)
    
    print(f"Output shape: {output.shape}")
    print(f"\n--- Compression Statistics ---")
    print(f"Compression ratio: {stats['compression_ratio']:.3f} (target: {1/2.0:.2f})")
    print(f"Num chunks (max): {stats['num_chunks']}")
    print(f"Avg chunk size: {stats['avg_chunk_size']:.2f} frames")
    print(f"DC loss: {stats['dc_loss'].item():.4f}")
    print(f"Boundary prob mean: {stats['boundary_prob_mean']:.3f}")
    
    # Test backward pass
    print("\n--- Backward Pass Test ---")
    loss = output.sum() + stats['dc_loss'] * 0.03  # Weighted DC loss
    loss.backward()
    print("Backward pass successful!")
    
    # Check gradients
    routing_grad = hmamba.routing_module.q_proj_layer.weight.grad
    if routing_grad is not None:
        print(f"Routing grad norm: {routing_grad.norm().item():.6f}")
    else:
        print("Warning: No gradient for routing module!")
    
    # Test API compatibility (same interface as ConmambaEncoder)
    print("\n--- API Compatibility Test ---")
    hmamba.zero_grad()
    output2, _ = hmamba(x)  # Without stats
    print(f"Output without stats: {output2.shape}")
    
    # Test compute_dc_loss separately
    print("\n--- DC Loss Computation Test ---")
    dc_loss_separate = hmamba.compute_dc_loss(x, src_key_padding_mask)
    print(f"Separate DC loss: {dc_loss_separate.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)