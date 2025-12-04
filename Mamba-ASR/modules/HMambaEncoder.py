"""
H-Mamba Encoder: ConMamba with Dynamic Chunking (DC) for efficient ASR.

This module integrates H-Net's Dynamic Chunking mechanism into the ConMamba encoder,
splitting 12 bi-Mamba layers into two stages with learned compression in between.

Architecture:
    Stage-0: Layers 1-6 (frame-level processing)
    DC Layer: Learns ~r compression ratio (e.g., r=0.5 keeps 50% of frames)
    Stage-1: Layers 7-12 (chunk-level processing)
    Upsample: Restore original temporal resolution via EMA

Usage:
    python HMambaEncoder.py  # Run standalone test with dummy tensors
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

# Try to import mamba kernel, fall back to pure PyTorch if not available
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    MAMBA_KERNEL_AVAILABLE = True
except ImportError:
    MAMBA_KERNEL_AVAILABLE = False
    print("Warning: mamba_ssm kernel not available, using pure PyTorch EMA fallback")


# =============================================================================
# Data Classes for DC Module State
# =============================================================================

@dataclass
class RoutingModuleOutput:
    """Output from the routing module."""
    boundary_prob: torch.Tensor      # (B, L, 2) - probability of [not-boundary, boundary]
    boundary_mask: torch.Tensor      # (B, L) - boolean mask where True = boundary
    selected_probs: torch.Tensor     # (B, L, 1) - probability of selected action


@dataclass
class DCStats:
    """Statistics from Dynamic Chunking for monitoring."""
    compression_ratio: float         # Actual r = num_boundaries / total_frames
    num_chunks: int                  # Number of chunks created
    avg_chunk_size: float            # Average frames per chunk


# =============================================================================
# Routing Module: Learns where to place boundaries
# =============================================================================

class RoutingModule(nn.Module):
    """
    Learns boundary positions using cosine similarity between adjacent frames.
    
    High dissimilarity between frame[t] and frame[t+1] → boundary at t+1
    This naturally places boundaries at acoustic transitions (e.g., phoneme boundaries).
    
    Key improvements:
    - Learnable temperature to control sharpness of decisions
    - Learnable bias to control overall boundary rate
    - Gumbel-Softmax for differentiable sampling during training
    """

    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Learnable projections for computing similarity
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        
        # Initialize with small random weights (not identity!) so frames differ
        nn.init.xavier_uniform_(self.q_proj_layer.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj_layer.weight, gain=0.1)
        
        # Learnable temperature (controls sharpness of boundary decisions)
        # Initialize to make boundaries more likely initially
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # temp = 1.0 initially
        
        # Learnable bias toward boundaries (positive = more boundaries)
        # Initialize to encourage ~30-50% boundary rate
        self.boundary_bias = nn.Parameter(torch.tensor(0.5))  # Start with bias toward boundaries
        
        # Gumbel temperature for training (annealed over time)
        self.gumbel_temperature = 1.0

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> RoutingModuleOutput:
        """
        Compute boundary probabilities based on cosine similarity.
        
        Args:
            hidden_states: (B, L, D) input features
            mask: (B, L) boolean mask where True = valid position
            
        Returns:
            RoutingModuleOutput with boundary predictions
        """
        B, L, D = hidden_states.shape
        
        # Project frames
        q = self.q_proj_layer(hidden_states[:, :-1])  # (B, L-1, D)
        k = self.k_proj_layer(hidden_states[:, 1:])   # (B, L-1, D)
        
        # Compute cosine similarity between adjacent frames
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        cos_sim = torch.einsum("bld,bld->bl", q_norm, k_norm)  # (B, L-1)
        
        # Convert similarity to logits (low similarity → high boundary logit)
        # Use learnable temperature and bias
        temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        dissimilarity = 1 - cos_sim  # Range [0, 2], higher = more different
        
        # Logits for boundary decision: higher dissimilarity + bias → more likely boundary
        boundary_logits = (dissimilarity * temperature + self.boundary_bias)  # (B, L-1)
        
        # First frame is always a boundary (start of sequence) - use high logit
        boundary_logits = F.pad(boundary_logits, (1, 0), "constant", 10.0)  # (B, L)
        
        # Create logits for [not-boundary, boundary]
        logits = torch.stack([-boundary_logits, boundary_logits], dim=-1)  # (B, L, 2)
        
        # Compute probabilities
        boundary_prob = F.softmax(logits, dim=-1)  # (B, L, 2)
        
        # During training: use Gumbel-Softmax for differentiable sampling
        # During eval: use hard argmax
        if self.training:
            # Gumbel-Softmax: differentiable approximation to categorical sampling
            gumbel_out = F.gumbel_softmax(logits, tau=self.gumbel_temperature, hard=True)
            boundary_mask = gumbel_out[..., 1] > 0.5  # (B, L)
        else:
            # Hard decision during evaluation
            selected_idx = torch.argmax(boundary_prob, dim=-1)  # (B, L)
            boundary_mask = selected_idx == 1  # (B, L)
        
        # Mask out invalid positions
        if mask is not None:
            boundary_mask = boundary_mask & mask
        
        # Get probability of selected action (for monitoring)
        selected_idx = boundary_mask.long()
        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))  # (B, L, 1)
        
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )


# =============================================================================
# Chunk Layer: Compress sequence by selecting boundary frames
# =============================================================================

class ChunkLayer(nn.Module):
    """
    Compresses the sequence by keeping only boundary frames.
    
    This is the "downsampling" step - if r=0.5, roughly half the frames are kept.
    """

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        boundary_mask: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Select boundary frames to create compressed sequence.
        
        Args:
            hidden_states: (B, L, D) input features
            boundary_mask: (B, L) boolean mask where True = boundary
            mask: (B, L) validity mask
            
        Returns:
            chunked_states: (B, M, D) compressed features where M = max boundaries
            chunk_mask: (B, M) validity mask for compressed sequence
            max_chunks: maximum number of chunks in batch
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        # Count boundaries per sequence
        num_boundaries = boundary_mask.sum(dim=-1)  # (B,)
        max_chunks = int(num_boundaries.max().item())
        
        if max_chunks == 0:
            # Edge case: no boundaries (shouldn't happen with first frame always being boundary)
            max_chunks = 1
            boundary_mask[:, 0] = True
            num_boundaries = boundary_mask.sum(dim=-1)
        
        # Sort to move boundary frames to the front
        # Non-boundary positions get index L (will be sorted to end)
        token_idx = torch.arange(L, device=device)[None, :].expand(B, -1)  # (B, L)
        token_idx = token_idx + (~boundary_mask).long() * L
        sorted_indices = torch.argsort(token_idx, dim=1)  # (B, L)
        
        # Gather only the first max_chunks frames (the boundary frames)
        gather_indices = sorted_indices[:, :max_chunks, None].expand(-1, -1, D)  # (B, M, D)
        chunked_states = torch.gather(hidden_states, dim=1, index=gather_indices)  # (B, M, D)
        
        # Create validity mask for chunked sequence
        chunk_mask = torch.arange(max_chunks, device=device)[None, :] < num_boundaries[:, None]  # (B, M)
        
        return chunked_states, chunk_mask, max_chunks


# =============================================================================
# DeChunk Layer: Expand compressed sequence back to original length
# =============================================================================

class DeChunkLayer(nn.Module):
    """
    Expands compressed sequence back to original length using EMA interpolation.
    
    Each non-boundary frame is computed as a weighted average of the previous
    output and the next chunk's representation.
    """

    def __init__(
        self,
        d_model: int,
        dtype: torch.dtype = torch.bfloat16,
        block_size: int = 256,
        headdim: int = 36,  # Fixed for ASR: 144 / 36 = 4 heads
    ):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        
        assert d_model % headdim == 0, f"d_model ({d_model}) must be divisible by headdim ({headdim})"
        self.nheads = d_model // headdim

    def forward(
        self,
        chunked_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Expand chunked sequence back to original length.
        
        Args:
            chunked_states: (B, M, D) compressed features
            boundary_mask: (B, L) original boundary positions
            boundary_prob: (B, L, 2) boundary probabilities
            mask: (B, L) original validity mask
            
        Returns:
            expanded_states: (B, L, D) features at original resolution
        """
        B, L = boundary_mask.shape
        D = chunked_states.shape[-1]
        device = chunked_states.device
        
        # Get boundary probability (used for EMA weighting)
        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - 1e-4)  # (B, L)
        
        # Reorder p to match chunked sequence order
        token_idx = torch.arange(L, device=device)[None, :].expand(B, -1)
        token_idx = token_idx + (~boundary_mask).long() * L
        sorted_indices = torch.argsort(token_idx, dim=1)
        
        M = chunked_states.shape[1]
        p_chunked = torch.gather(p, dim=1, index=sorted_indices[:, :M])  # (B, M)
        
        # Use mamba kernel for efficient EMA if available, else pure PyTorch
        if MAMBA_KERNEL_AVAILABLE and chunked_states.is_cuda:
            expanded_chunked = self._ema_mamba_kernel(chunked_states, p_chunked)
        else:
            expanded_chunked = self._ema_pytorch(chunked_states, p_chunked)
        
        # Map back to original positions
        # For each position in original sequence, find which chunk it belongs to
        chunk_indices = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
        chunk_indices = chunk_indices.clamp(min=0, max=M-1)
        
        # Gather from expanded chunked states
        gather_indices = chunk_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, L, D)
        expanded_states = torch.gather(expanded_chunked, dim=1, index=gather_indices)  # (B, L, D)
        
        return expanded_states

    def _ema_mamba_kernel(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """EMA using optimized Mamba2 kernel."""
        original_dtype = x.dtype
        
        # Prepare inputs for mamba kernel
        dt = torch.log(1 / (1 - p)).to(self.dtype)  # (B, M)
        x_scaled = (x / dt.unsqueeze(-1)).to(self.dtype)  # (B, M, D)
        
        A = -torch.ones((self.nheads,), device=x.device, dtype=torch.float32)
        b = p.to(self.dtype)  # (B, M)
        c = torch.ones_like(b)
        
        # Reshape for mamba kernel: (B, M, D) -> (B, M, H, P)
        x_reshaped = rearrange(x_scaled, "b m (h p) -> b m h p", p=self.headdim)
        dt_expanded = repeat(dt, "b m -> b m h", h=self.nheads)
        b_reshaped = rearrange(b, "b m -> b m 1 1")
        c_reshaped = rearrange(c, "b m -> b m 1 1")
        
        out = mamba_chunk_scan_combined(
            x_reshaped,
            dt_expanded,
            A,
            b_reshaped,
            c_reshaped,
            chunk_size=self.block_size,
        )
        
        out = rearrange(out, "b m h p -> b m (h p)")
        return out.to(original_dtype)

    def _ema_pytorch(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch EMA fallback (slower but works everywhere)."""
        B, M, D = x.shape
        
        # Simple iterative EMA: out[t] = p[t] * x[t] + (1-p[t]) * out[t-1]
        # Avoid inplace operations by collecting outputs in a list
        outputs = [x[:, 0]]  # First position is just the input
        
        for t in range(1, M):
            prev = outputs[-1]
            curr = p[:, t:t+1] * x[:, t] + (1 - p[:, t:t+1]) * prev
            outputs.append(curr)
        
        # Stack along sequence dimension
        out = torch.stack(outputs, dim=1)  # (B, M, D)
        return out


# =============================================================================
# Load Balancing Loss: Controls compression ratio
# =============================================================================

def load_balancing_loss(router_output: RoutingModuleOutput, N: float) -> torch.Tensor:
    """
    Compute load balancing loss to control compression ratio.
    
    This loss encourages the boundary ratio to be close to 1/N.
    
    Args:
        router_output: Output from RoutingModule
        N: Target downsampling factor (N=2 → r=0.5, N=3 → r=0.33)
        
    Returns:
        Scalar loss tensor
    """
    boundary_prob = router_output.boundary_prob  # (B, L, 2)
    
    # Target ratio of boundaries
    target_ratio = 1.0 / N
    
    # Average predicted boundary probability across all positions
    avg_boundary_prob = boundary_prob[..., 1].mean()  # P(boundary)
    
    # L2 loss to push average probability toward target
    # This provides gradients that directly push probabilities up or down
    prob_loss = (avg_boundary_prob - target_ratio) ** 2
    
    # Entropy regularization to prevent collapse to all-0 or all-1
    # Encourages exploration of different boundary patterns
    entropy = -(boundary_prob * (boundary_prob + 1e-8).log()).sum(dim=-1).mean()
    entropy_bonus = -0.01 * entropy  # Small bonus for maintaining entropy
    
    # Combined loss
    loss = prob_loss * 10.0 + entropy_bonus  # Scale prob_loss for stronger signal
    
    return loss


# =============================================================================
# STE (Straight-Through Estimator) for gradient flow
# =============================================================================

class STE(torch.autograd.Function):
    """Straight-through estimator: forward returns 1, backward passes gradient."""
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x):
    return STE.apply(x)


# =============================================================================
# H-Mamba Encoder: Main class wrapping ConMamba with DC
# =============================================================================

class HMambaEncoder(nn.Module):
    """
    H-Mamba Encoder: ConMamba with Dynamic Chunking.
    
    Architecture:
        Input: (B, L, D) audio features
        Stage-0: First 6 bi-Mamba layers (frame-level)
        DC: Dynamic Chunking (learned compression)
        Stage-1: Last 6 bi-Mamba layers (chunk-level)
        Upsample: DeChunk back to original resolution
        Output: (B, L, D) features at original resolution
    
    Args:
        d_model: Model dimension (default: 144 for ConMamba-Small)
        stage0_layers: nn.ModuleList of first 6 encoder layers
        stage1_layers: nn.ModuleList of last 6 encoder layers
        headdim: Head dimension for EMA (default: 36, gives 4 heads)
        target_ratio: Target compression ratio N (N=2 → 50% compression)
    """

    def __init__(
        self,
        d_model: int = 144,
        stage0_layers: Optional[nn.ModuleList] = None,
        stage1_layers: Optional[nn.ModuleList] = None,
        headdim: int = 36,
        target_ratio: float = 2.0,  # N=2 means keep 50% of frames
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.d_model = d_model
        self.target_ratio = target_ratio
        self.headdim = headdim
        
        # Store encoder layers (will be set from ConMamba)
        self.stage0_layers = stage0_layers if stage0_layers is not None else nn.ModuleList()
        self.stage1_layers = stage1_layers if stage1_layers is not None else nn.ModuleList()
        
        # Dynamic Chunking components
        self.routing_module = RoutingModule(d_model, **factory_kwargs)
        self.chunk_layer = ChunkLayer()
        self.dechunk_layer = DeChunkLayer(d_model, headdim=headdim)
        
        # Residual projection (initialized to zero for stable training)
        self.residual_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        nn.init.zeros_(self.residual_proj.weight)
        self.residual_proj.weight._no_reinit = True
        
        # Residual function with STE for gradient flow
        self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[RoutingModuleOutput], Optional[DCStats]]:
        """
        Forward pass through H-Mamba encoder.
        
        Args:
            hidden_states: (B, L, D) input features
            mask: (B, L) boolean mask where True = valid position
            return_stats: Whether to return compression statistics
            
        Returns:
            output: (B, L, D) encoded features
            router_output: RoutingModuleOutput for loss computation
            stats: DCStats if return_stats=True, else None
        """
        B, L, D = hidden_states.shape
        
        # Create default mask if not provided
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=hidden_states.device)
        
        # ===== Stage 0: Frame-level processing =====
        for layer in self.stage0_layers:
            hidden_states = layer(hidden_states)
        
        # Save residual for skip connection
        residual = self.residual_proj(hidden_states.float()).to(hidden_states.dtype)
        
        # ===== Dynamic Chunking =====
        # Predict boundaries
        router_output = self.routing_module(hidden_states, mask=mask)
        
        # Compress sequence
        chunked_states, chunk_mask, max_chunks = self.chunk_layer(
            hidden_states, router_output.boundary_mask, mask=mask
        )
        
        # ===== Stage 1: Chunk-level processing =====
        for layer in self.stage1_layers:
            chunked_states = layer(chunked_states)
        
        # ===== Upsample back to original resolution =====
        expanded_states = self.dechunk_layer(
            chunked_states,
            router_output.boundary_mask,
            router_output.boundary_prob,
            mask=mask,
        )
        
        # ===== Residual connection =====
        output = self.residual_func(
            expanded_states.float(),
            residual,
            router_output.selected_probs
        ).to(hidden_states.dtype)
        
        # Compute statistics if requested
        stats = None
        if return_stats:
            num_boundaries = router_output.boundary_mask.sum().item()
            total_frames = mask.sum().item()
            stats = DCStats(
                compression_ratio=num_boundaries / total_frames if total_frames > 0 else 0,
                num_chunks=max_chunks,
                avg_chunk_size=total_frames / num_boundaries if num_boundaries > 0 else 0,
            )
        
        return output, router_output, stats

    def compute_dc_loss(self, router_output: RoutingModuleOutput) -> torch.Tensor:
        """Compute load balancing loss for the given router output."""
        return load_balancing_loss(router_output, self.target_ratio)


# =============================================================================
# Dummy Mamba Layer for Testing
# =============================================================================

class DummyMambaLayer(nn.Module):
    """Simple feedforward layer mimicking Mamba layer interface for testing."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.norm(x))


# =============================================================================
# Test Function
# =============================================================================

def test_hmamba_encoder():
    """Test HMambaEncoder with dummy tensors."""
    print("=" * 60)
    print("Testing HMambaEncoder")
    print("=" * 60)
    
    # Configuration
    B, L, D = 2, 100, 144  # batch=2, length=100 frames, d_model=144
    headdim = 36  # 144 / 36 = 4 heads
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    print(f"\nConfig: B={B}, L={L}, D={D}, headdim={headdim}")
    print(f"Device: {device}")
    print(f"Mamba kernel available: {MAMBA_KERNEL_AVAILABLE}")
    
    # Create dummy encoder layers
    print("\nCreating dummy Mamba layers...")
    stage0_layers = nn.ModuleList([DummyMambaLayer(D) for _ in range(6)])
    stage1_layers = nn.ModuleList([DummyMambaLayer(D) for _ in range(6)])
    
    # Create H-Mamba encoder
    print("Creating HMambaEncoder...")
    encoder = HMambaEncoder(
        d_model=D,
        stage0_layers=stage0_layers,
        stage1_layers=stage1_layers,
        headdim=headdim,
        target_ratio=2.0,  # N=2 → 50% compression target
        device=device,
        dtype=dtype,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    dc_params = sum(p.numel() for n, p in encoder.named_parameters() 
                    if 'routing' in n or 'chunk' in n or 'residual_proj' in n)
    print(f"Total params: {total_params:,}")
    print(f"DC overhead params: {dc_params:,} ({100*dc_params/total_params:.2f}%)")
    
    # Create dummy input
    print("\nCreating dummy input...")
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    # Make second sequence shorter
    mask[1, 80:] = False
    
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Valid frames: {mask.sum(dim=1).tolist()}")
    
    # Forward pass
    print("\nRunning forward pass...")
    encoder.train()
    output, router_output, stats = encoder(x, mask=mask, return_stats=True)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Boundary mask shape: {router_output.boundary_mask.shape}")
    
    # Print statistics
    print(f"\n--- Compression Statistics ---")
    print(f"Compression ratio: {stats.compression_ratio:.3f} (target: 0.5)")
    print(f"Num chunks (max in batch): {stats.num_chunks}")
    print(f"Avg chunk size: {stats.avg_chunk_size:.2f} frames")
    
    # Compute losses
    print(f"\n--- Loss Computation ---")
    dc_loss = encoder.compute_dc_loss(router_output)
    print(f"DC load balancing loss: {dc_loss.item():.4f}")
    
    # Test backward pass
    print("\n--- Backward Pass ---")
    dummy_loss = output.sum() + dc_loss
    dummy_loss.backward()
    print("Backward pass successful!")
    
    # Check gradients
    routing_grad = encoder.routing_module.q_proj_layer.weight.grad
    if routing_grad is not None:
        print(f"Routing module gradient norm: {routing_grad.norm().item():.6f}")
    else:
        print("Warning: No gradient for routing module!")
    
    # Test different compression ratios
    print("\n--- Testing Different Compression Ratios ---")
    for N in [1.0, 2.0, 3.0]:
        encoder.target_ratio = N
        encoder.zero_grad()
        
        with torch.no_grad():
            _, router_output, stats = encoder(x, mask=mask, return_stats=True)
        
        expected_ratio = 1.0 / N if N > 0 else 1.0
        print(f"N={N}: actual ratio={stats.compression_ratio:.3f}, expected≈{expected_ratio:.3f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    
    return encoder, x, mask


def test_individual_components():
    """Test DC components individually."""
    print("\n" + "=" * 60)
    print("Testing Individual Components")
    print("=" * 60)
    
    B, L, D = 2, 50, 144
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test RoutingModule
    print("\n1. Testing RoutingModule...")
    routing = RoutingModule(D).to(device)
    x = torch.randn(B, L, D, device=device)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    
    output = routing(x, mask=mask)
    print(f"   boundary_prob shape: {output.boundary_prob.shape}")
    print(f"   boundary_mask shape: {output.boundary_mask.shape}")
    print(f"   Boundaries per sequence: {output.boundary_mask.sum(dim=1).tolist()}")
    
    # Test ChunkLayer
    print("\n2. Testing ChunkLayer...")
    chunk = ChunkLayer()
    chunked, chunk_mask, max_chunks = chunk(x, output.boundary_mask, mask=mask)
    print(f"   Original shape: {x.shape}")
    print(f"   Chunked shape: {chunked.shape}")
    print(f"   Compression: {L} -> {max_chunks} ({100*max_chunks/L:.1f}%)")
    
    # Test DeChunkLayer
    print("\n3. Testing DeChunkLayer...")
    dechunk = DeChunkLayer(D, headdim=36).to(device)
    expanded = dechunk(chunked, output.boundary_mask, output.boundary_prob, mask=mask)
    print(f"   Chunked shape: {chunked.shape}")
    print(f"   Expanded shape: {expanded.shape}")
    
    print("\nAll component tests passed!")


if __name__ == "__main__":
    # Run tests
    test_individual_components()
    encoder, x, mask = test_hmamba_encoder()
    
    # Interactive: print architecture
    print("\n" + "=" * 60)
    print("Encoder Architecture")
    print("=" * 60)
    print(encoder)