from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from hnet.modules.utils import get_seq_idx


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    """
    The state of the routing module.

    Contains
        - [has_seen_tokens] (batch_size,) bool tensor. Whether that batch element has processed any tokens yet.
        - [last_hidden_state] (batch_size, d_model) tensor. The last hidden state of the batch element (used for boundary prediction).
    """

    has_seen_tokens: torch.Tensor  # (batch_size,)
    last_hidden_state: torch.Tensor  # (batch_size, d_model)


@dataclass
class DeChunkState:
    """
    The state of the dechunk.

    Contains
        - [last_value] (batch_size, d_model) tensor. The last value of the batch element (used for the EMA).
    """

    last_value: torch.Tensor  # (batch_size, d_model)


class RoutingModule(nn.Module):
    """
    Learns boundary positions using cosine similarity between adjacent frames.
    
    Key features:
    - Learnable Q/K projections with random initialization
    - Learnable temperature for decision sharpness control
    - Learnable bias to center decision boundary (critical for achieving target ratios)
    - Gumbel-softmax for differentiable discrete sampling during training
    """

    def __init__(self, d_model, device=None, dtype=None):
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Q/K projection layers
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        
        # Random initialization for diverse projections
        nn.init.normal_(self.q_proj_layer.weight, std=0.02)
        nn.init.normal_(self.k_proj_layer.weight, std=0.02)
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True
        
        # Learnable temperature for decision sharpness (lower = sharper decisions)
        # Initialize to 0.5 for moderately sharp decisions
        self.temperature = nn.Parameter(torch.tensor(0.5, **factory_kwargs))
        
        # Learnable bias to CENTER the decision boundary
        # Mathematical analysis: bias=1.0 gives ~95% frames kept (for avg cos_sim=0.5)
        # Warm-up starts at target=95%, so bias=1.0 is ideal initial value
        # Model will learn to DECREASE bias toward -0.5 as target_N increases to 2.0
        self.boundary_bias = nn.Parameter(torch.tensor(1.0, **factory_kwargs))
        
        # Gumbel temperature for training (annealed during training)
        self.gumbel_tau = 1.0  # Can be adjusted externally

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def _compute_boundary_logits(self, cos_sim):
        """
        Compute boundary logits from cosine similarity.
        
        Formula: logit = (1 - cos_sim + bias) / temperature
        
        - cos_sim high (similar frames) → low logit → low boundary prob
        - cos_sim low (different frames) → high logit → high boundary prob
        - bias shifts the decision threshold
        - temperature controls sharpness
        """
        # Clamp temperature to prevent division issues
        temp = torch.clamp(self.temperature.abs(), min=0.1, max=2.0)
        
        # Compute logits with learnable bias for centering
        # bias < 0 → fewer boundaries, bias > 0 → more boundaries
        logits = (1 - cos_sim + self.boundary_bias) / temp
        
        return logits

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if inference_params is not None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"
            assert (
                ~inference_params.has_seen_tokens
            ).all(), "Cannot have seen tokens when inference_params is not provided"

        if cu_seqlens is not None:
            # We are in packed mode, so hidden_states is (T, D). Make it (B, T, D)
            hidden_states = hidden_states.unsqueeze(0)

        # Compute cosine similarity between adjacent frames
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        
        # Compute boundary logits with learnable bias and temperature
        boundary_logits = self._compute_boundary_logits(cos_sim)
        
        # Convert to probability
        boundary_prob_single = torch.sigmoid(boundary_logits)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob_single = F.pad(boundary_prob_single, (1, 0), "constant", PAD_PROB)

        if cu_seqlens is not None:
            boundary_prob_single = boundary_prob_single.squeeze(0)
            boundary_prob_single[cu_seqlens[:-1]] = PAD_PROB

        # Stack into [P(no-boundary), P(boundary)] format
        boundary_prob = torch.stack(
            (1 - boundary_prob_single, boundary_prob_single), dim=-1
        )
        
        # Discrete boundary decision
        if self.training:
            # Use Gumbel-Softmax for differentiable sampling during training
            # This allows gradients to flow through discrete decisions
            boundary_hard = F.gumbel_softmax(
                torch.log(boundary_prob + 1e-8), 
                tau=self.gumbel_tau, 
                hard=True
            )
            boundary_mask = boundary_hard[..., 1] > 0.5
        else:
            # Use argmax for inference (deterministic)
            selected_idx = torch.argmax(boundary_prob, dim=-1)
            boundary_mask = selected_idx == 1

        if mask is not None:
            # No invalid tokens can be selected
            boundary_mask = boundary_mask & mask

        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask,
                    hidden_states[
                        torch.arange(
                            hidden_states.shape[0], device=hidden_states.device
                        ),
                        last_mask,
                    ],
                    inference_params.last_hidden_state,
                )
            )

        selected_idx = boundary_mask.long()
        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (shape hidden_states.shape[:-1], 1)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
            boundary_mask=boundary_mask,  # (shape hidden_states.shape[:-1])
            selected_probs=selected_probs,  # (shape hidden_states.shape[:-1], 1)
        )

    def step(self, hidden_states, inference_params):
        # hidden_states is (B, 1, D)
        hidden_states = hidden_states.squeeze(1)
        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states), dim=-1),
        )
        
        # Compute boundary logits with learnable bias and temperature
        boundary_logits = self._compute_boundary_logits(cos_sim)
        boundary_prob_single = torch.sigmoid(boundary_logits)
        
        inference_params.last_hidden_state.copy_(hidden_states)
        
        # Force first token to be boundary
        boundary_prob_single = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob_single,
            torch.ones_like(boundary_prob_single),
        )
        
        boundary_prob = torch.stack(
            (1 - boundary_prob_single, boundary_prob_single), dim=-1
        )

        inference_params.has_seen_tokens.copy_(
            torch.ones_like(inference_params.has_seen_tokens)
        )
        
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (B, 2)
            boundary_mask=boundary_prob[..., 1] > 0.5,  # (B,)
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),  # (B, 1)
        )


class ChunkLayer(nn.Module):

    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(
                boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())

            device = hidden_states.device
            L = hidden_states.shape[1]
            token_idx = (
                torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                    -1, -1, hidden_states.shape[-1]
                ),
            )

            next_mask = (
                torch.arange(next_max_seqlen, device=device)[None, :]
                < num_tokens[:, None]
            )
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(self, hidden_states, boundary_mask):
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):

    def __init__(
        self,
        d_model,
        dtype=torch.bfloat16,
        block_size=256,
        headdim=32,
    ):
        super().__init__()
        self.d_model = d_model

        # Just for Mamba2 kernel.
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkState(
            last_value=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        cu_seqlens=None,
        inference_params=None,
        mask=None,
    ):
        if inference_params is not None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"
            assert boundary_mask[
                :, 0
            ].all(), "First token must be a boundary if running prefill"

        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

        if cu_seqlens is not None:
            p = p[boundary_mask].unsqueeze(0)
            seq_idx = get_seq_idx(cu_seqlens, device=hidden_states.device)
        else:
            B, L = boundary_mask.shape
            seq_idx = None

            token_idx = (
                torch.arange(L, device=hidden_states.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            p = torch.gather(
                p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
            )  # (B, M)

        original_dtype = hidden_states.dtype
        # Reuse Mamba2 kernel for EMA Deaggregator.
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones(
            (self.nheads,), device=hidden_states.device, dtype=torch.float32
        )
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
            seq_idx=seq_idx,
        )
        out = rearrange(out, "b l h p -> b l (h p)")

        if cu_seqlens is not None:
            out = out.squeeze(0)
            plug_back_idx = boundary_mask.cumsum(dim=0) - 1
            out = torch.gather(
                out, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, self.d_model)
            )
        else:
            plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
            out = torch.gather(
                out,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1])

        return out.to(original_dtype)

    def step(self, hidden_states, boundary_mask, boundary_prob, inference_params):
        # hidden_states is (B', 1, D), where B' = boundary_mask.sum()
        # boundary_mask is (B,) and boundary_prob is (B, 2)

        B = boundary_mask.shape[0]
        # B_selected = hidden_states.shape[0]
        D = hidden_states.shape[-1]

        p = torch.zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(
            min=1e-4, max=1 - (1e-4)
        )

        current_hidden_states = torch.zeros(
            B, D, device=hidden_states.device, dtype=hidden_states.dtype
        )
        current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        result = p * current_hidden_states + (1 - p) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)
