import math
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops 
from einops import repeat, rearrange
from einops_exts import rearrange_many

from timm.models.vision_transformer import RmsNorm


class StatelessLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)
    



def precompute_freqs_1d(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute frequencies for 1D rotary embeddings.
    For action sequences, we only need position-based frequencies.
    """
    freqs = torch.arange(0, dim, 2).float()
    freqs = theta ** (-freqs / dim)
    
    t = torch.arange(max_seq_len)
    freqs = (t.view(-1, 1) * freqs.view(1, -1))  # [seq_len, dim/2]
    
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def rotate_half(x):
    """Split the features in half and rotate one half with respect to the other."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply 1D rotary position embeddings to queries and keys.
    Properly handles dimension splitting and broadcasting.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine tensor [1, 1, seq_len, head_dim/2]
        sin: Sine tensor [1, 1, seq_len, head_dim/2]
        position_ids: Optional position indices
    """
    seq_len = q.shape[-2]
    
    # If position_ids not provided, assume sequential
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device)
    
    # Get the cos/sin for the positions we need
    cos = cos[position_ids]  # [seq_len, dim/2]
    sin = sin[position_ids]  # [seq_len, dim/2]
    
    # Reshape for broadcasting while preserving head dimension
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
    
    # Split q and k into two halves along the last dimension
    q1, q2 = q.chunk(2, dim=-1)    # Each half: [16, 8, 10, 64]
    k1, k2 = k.chunk(2, dim=-1)    # Each half: [16, 8, 10, 64]
    
    # Apply rotations separately to each half
    q_embed = torch.cat([
        q1 * cos - q2 * sin,    # Rotate first half
        q2 * cos + q1 * sin     # Rotate second half
    ], dim=-1)
    
    k_embed = torch.cat([
        k1 * cos - k2 * sin,    # Rotate first half
        k2 * cos + k1 * sin     # Rotate second half
    ], dim=-1)
    
    return q_embed, k_embed


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class MLP(nn.Module):
    def __init__(
            self, 
            dim, 
            hidden_dim=None, 
            dropout=0.0,
            output_dim=None
        ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        
        if output_dim is None:
            output_dim = dim
        self.c_fc1 = nn.Linear(dim, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(dim, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        return x
    



class MultiHeadLayerNorm(nn.Module):
    def __init__(self, hidden_size=None, eps=1e-5):
        # Copy pasta from 
        # https://github.com/huggingface/transformers/blob/e5f71ecaae50ea476d1e12351003790273c4b2ed/src/transformers/models/cohere/modeling_cohere.py#L78

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(
            variance + self.variance_epsilon
        )
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)


class SingleAttention(nn.Module):

    def __init__(
            self, 
            dim, 
            n_heads, 
            attn_pdrop=0.1, 
            resid_pdrop=0.1,
            use_rope=False,
            max_seq_len=120,
            rope_theta=32,
        ):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.q_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.k_norm = RmsNorm(self.head_dim, eps=1e-6)

        self.use_rope = use_rope
        
        if use_rope:
            # Pre-compute RoPE frequencies
            self.rope_theta = rope_theta
            cos, sin = precompute_freqs_1d(
                self.head_dim,
                max_seq_len,
                theta=rope_theta
            )

            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
            
        self.max_seq_len = max_seq_len
        
    def forward(self, x, custom_attn_mask=None, is_causal=False):
        B, T, C = x.size()

        # Handle sequences longer than max_seq_len
        if T > self.max_seq_len and self.use_rope:
            cos, sin = precompute_freqs_1d(
                self.head_dim,
                T,
                theta=self.rope_theta
            )
            cos = cos.to(x.device)
            sin = sin.to(x.device)
        elif T <= self.max_seq_len and self.use_rope:
            cos, sin = self.cos, self.sin

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if enabled
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if is_causal and custom_attn_mask is None:
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
        elif custom_attn_mask is not None:
            attn = attn.masked_fill(custom_attn_mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.resid_dropout(self.proj(x))
        return x


class CrossAttention(nn.Module):
    def __init__(
            self, 
            dim, 
            n_heads, 
            attn_pdrop=0.1, 
            resid_pdrop=0.1,
            use_rope=False,
            query_seq_len=64,    # For action sequences
            context_seq_len=384, # Padding a bit beyond 300 for safety
            rope_theta=32,       # For shorter action sequences
            context_rope_theta=1000.0  # Smaller than default since sequences are shorter
        ):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Ensure head_dim is even for RoPE if it's used
        if use_rope:
            assert self.head_dim % 2 == 0, "Head dimension must be even when using RoPE"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.q_norm = RmsNorm(self.head_dim, eps=1e-6)
        self.k_norm = RmsNorm(self.head_dim, eps=1e-6)

        # RoPE settings
        self.use_rope = use_rope
        if use_rope:
            # Pre-compute RoPE frequencies for queries (shorter sequences)
            q_cos, q_sin = precompute_freqs_1d(
                self.head_dim,
                query_seq_len,
                theta=rope_theta
            )
            # Pre-compute RoPE frequencies for keys (context length)
            k_cos, k_sin = precompute_freqs_1d(
                self.head_dim,
                context_seq_len,
                theta=context_rope_theta
            )

            self.register_buffer("q_cos", q_cos)
            self.register_buffer("q_sin", q_sin)
            self.register_buffer("k_cos", k_cos)
            self.register_buffer("k_sin", k_sin)
            
            self.query_seq_len = query_seq_len
            self.context_seq_len = context_seq_len
            self.rope_theta = rope_theta
            self.context_rope_theta = context_rope_theta

    def forward(self, x, context, custom_attn_mask=None):
        B, T, C = x.size()
        _, S, _ = context.size()

        # Handle sequences longer than max lengths when RoPE is enabled
        if self.use_rope:
            if T > self.query_seq_len:
                q_cos, q_sin = precompute_freqs_1d(
                    self.head_dim, T, theta=self.rope_theta
                )
                q_cos, q_sin = q_cos.to(x.device), q_sin.to(x.device)
            else:
                q_cos, q_sin = self.q_cos, self.q_sin

            if S > self.context_seq_len:
                k_cos, k_sin = precompute_freqs_1d(
                    self.head_dim, S, theta=self.context_rope_theta
                )
                k_cos, k_sin = k_cos.to(x.device), k_sin.to(x.device)
            else:
                k_cos, k_sin = self.k_cos, self.k_sin

        # Project and reshape q, k, v
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE only if enabled
        if self.use_rope:
            q, _ = apply_rotary_pos_emb(q, q, q_cos, q_sin)  # Only use q output
            k, _ = apply_rotary_pos_emb(k, k, k_cos, k_sin)  # Only use k output

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if custom_attn_mask is not None:
            attn = attn.masked_fill(custom_attn_mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.resid_dropout(self.proj(x))
        return x
    

class DoubleAttention(nn.Module):

    def __init__(
            self, 
            dim, 
            n_heads, 
            attn_pdrop=0.1, 
            resid_pdrop=0.1, 
            mh_qknorm=False
        ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.w1q = nn.Linear(dim, dim, bias=False)
        self.w1k = nn.Linear(dim, dim, bias=False)
        self.w1v = nn.Linear(dim, dim, bias=False)
        self.w1o = nn.Linear(dim, dim, bias=False)

        self.w2q = nn.Linear(dim, dim, bias=False)
        self.w2k = nn.Linear(dim, dim, bias=False)
        self.w2v = nn.Linear(dim, dim, bias=False)
        self.w2o = nn.Linear(dim, dim, bias=False)

        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else RmsNorm(self.head_dim, eps=1e-6)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else RmsNorm(self.head_dim, eps=1e-6)
        )

        self.q_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else RmsNorm(self.head_dim, eps=1e-6)
        )
        self.k_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim))
            if mh_qknorm
            else RmsNorm(self.head_dim, eps=1e-6)
        )

        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def create_custom_attention_mask(self, seqlen1, seqlen2):
        """
        Create a custom attention mask for `c` and `x` tokens:
        - `c` tokens can attend to all `c` tokens but not `x` tokens
        - `x` tokens can attend to all `c` tokens
        - `x` tokens follow causal masking within their own sequence
        
        Args:
            seqlen1 (int): Length of the `c` sequence
            seqlen2 (int): Length of the `x` sequence
        
        Returns:
            torch.Tensor: Attention mask of shape (seqlen1 + seqlen2, seqlen1 + seqlen2)
        """
        seqlen = seqlen1 + seqlen2
        
        # Initialize mask with zeros, using the device from self
        mask = torch.zeros((seqlen, seqlen))
        
        # Block c -> x attention (top-right block)
        mask[:seqlen1, seqlen1:] = float('-inf')
        
        # Create causal mask for x -> x (bottom-right block)
        x_causal_mask = torch.triu(
            torch.full((seqlen2, seqlen2), float('-inf')), 
            diagonal=1
        )
        mask[seqlen1:, seqlen1:] = x_causal_mask
        
        return mask

    def forward(self, c, x, attention_mask=None, use_causal_x_attention=False):
        bsz, seqlen1, _ = c.shape
        bsz, seqlen2, _ = x.shape
        seqlen = seqlen1 + seqlen2

        # Linear projections and reshaping using einops
        cq, ck, cv = self.w1q(c), self.w1k(c), self.w1v(c)
        cq = rearrange(cq, "b s (h d) -> b h s d", h=self.n_heads)
        ck = rearrange(ck, "b s (h d) -> b h s d", h=self.n_heads)
        cv = rearrange(cv, "b s (h d) -> b h s d", h=self.n_heads)

        cq, ck = self.q_norm1(cq), self.k_norm1(ck)

        xq, xk, xv = self.w2q(x), self.w2k(x), self.w2v(x)
        xq = rearrange(xq, "b s (h d) -> b h s d", h=self.n_heads)
        xk = rearrange(xk, "b s (h d) -> b h s d", h=self.n_heads)
        xv = rearrange(xv, "b s (h d) -> b h s d", h=self.n_heads)

        xq, xk = self.q_norm2(xq), self.k_norm2(xk)

        # Concatenate along the sequence dimension
        q = torch.cat([cq, xq], dim=2)  # Concatenate over sequence length
        k = torch.cat([ck, xk], dim=2)
        v = torch.cat([cv, xv], dim=2)

        attention_mask = self.create_custom_attention_mask(seqlen1, seqlen2).to(q.device)

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask.to(q.device),
            dropout_p=self.attn_dropout.p if self.training else 0,
            is_causal=False
        )

        # Reshape back to the original format
        output = rearrange(output, "b h s d -> b s (h d)")

        # Split outputs and apply residual dropout
        c, x = output.split([seqlen1, seqlen2], dim=1)
        c = self.resid_dropout(self.w1o(c))
        x = self.resid_dropout(self.w2o(x))

        return c, x


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        dim,                    # Model dimension
        heads=8,               # Number of attention heads
        attn_pdrop=0.1,       # Attention dropout probability
        resid_pdrop=0.1,      # Residual dropout probability
        mlp_pdrop=0.1,        # MLP dropout probability
        use_rope=False,       # Whether to use rotary position embeddings
        max_seq_len=384,       # Maximum sequence length for RoPE
        rope_theta=10000.0    # Base frequency for RoPE
    ):
        super().__init__()
        
        # Layer normalization layers - using FP32 for stability
        self.norm1 = RmsNorm(dim, eps=1e-6)
        self.norm2 = RmsNorm(dim, eps=1e-6)
        
        # Attention layer with RoPE configuration
        self.attn = SingleAttention(
            dim=dim,
            n_heads=heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta
        )
        
        # MLP remains unchanged as RoPE only affects attention
        self.mlp = MLP(
            dim,
            dropout=mlp_pdrop
        )
        
        # Dropout layers for residual connections
        self.resid_dropout1 = nn.Dropout(resid_pdrop)
        self.resid_dropout2 = nn.Dropout(resid_pdrop)

    def forward(self, cx, attention_mask=None, is_causal=False, **kwargs):
        """
        Forward pass with optional RoPE positioning.
        
        Args:
            cx: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional attention mask
            is_causal: Whether to use causal attention
            positions: Optional position indices for RoPE
            **kwargs: Additional arguments passed to attention
        """
        # Attention block with residual connection
        residual = cx
        cx = self.norm1(cx)
        # Pass positions to attention if using RoPE
        cx = self.attn(cx, attention_mask, is_causal)
        cx = self.resid_dropout1(cx)
        cx = residual + cx

        # MLP block with residual connection
        residual = cx
        cx = self.norm2(cx)
        cx = self.mlp(cx)
        cx = self.resid_dropout2(cx)
        cx = residual + cx
        
        return cx

    

class DiTBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        heads=8, 
        global_conddim=1024, 
        attn_pdrop=0.1, 
        resid_pdrop=0.1, 
        mlp_pdrop=0.1,
        use_cross_attn=False,
        use_rope=False,        # Use RoPE for rotary position embeddings
        query_seq_len=64,      # For action sequences
        context_seq_len=384,   # For context sequences
        rope_theta=32,         # For action sequences
        context_rope_theta=1000.0  # For context
    ):
        super().__init__()

        self.norm1 = RmsNorm(dim, eps=1e-6)
        self.norm2 = RmsNorm(dim, eps=1e-6)
        self.norm3 = RmsNorm(dim, eps=1e-6) if use_cross_attn else None

        # Number of modulation signals needed
        num_mod_signals = 9 if use_cross_attn else 6
        self.modCX = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, num_mod_signals * dim, bias=False),
        )

        self.use_cross_attn = use_cross_attn
        
        # Initialize self-attention with RoPE parameters
        self.self_attn = SingleAttention(
            dim=dim, 
            n_heads=heads, 
            attn_pdrop=attn_pdrop, 
            resid_pdrop=resid_pdrop,
            use_rope=use_rope,
            max_seq_len=query_seq_len,
            rope_theta=rope_theta
        )
        
        # Initialize cross-attention with RoPE parameters if needed
        if use_cross_attn:
            self.cross_attn = CrossAttention(
                dim=dim, 
                n_heads=heads, 
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                use_rope=use_rope,
                query_seq_len=query_seq_len,
                context_seq_len=context_seq_len,
                rope_theta=rope_theta,
                context_rope_theta=context_rope_theta
            )

        self.mlp = MLP(
            dim,
            dropout=mlp_pdrop
        )

    def forward(self, cx, global_cond, context=None, custom_self_attn_mask=None, custom_cross_attn_mask=None, is_causal=False, **kwargs):
        B, L, D = cx.shape
        cxres = cx
        
        # Generate modulation signals
        mod_signals = self.modCX(global_cond)
        
        if self.use_cross_attn:
            # Split into 9 parts for cross-attention path
            chunks = mod_signals.chunk(9, dim=-1)
            shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
            shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
            shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]
            
            # Self-attention
            cx_norm = self.norm1(cx)
            cx_norm = modulate(cx_norm, shift_msa, scale_msa)
            cx_self = self.self_attn(cx_norm, custom_attn_mask=custom_self_attn_mask, is_causal=is_causal)
            cx = cxres + gate_msa.unsqueeze(1) * cx_self
            
            # Cross-attention
            if context is None:
                raise ValueError("Context must be provided when using cross-attention")
            cx_norm = self.norm2(cx)
            cx_norm = modulate(cx_norm, shift_cross, scale_cross)
            cx_cross = self.cross_attn(cx_norm, context, custom_attn_mask=custom_cross_attn_mask)
            cx = cx + gate_cross.unsqueeze(1) * cx_cross
            
            # MLP
            cx_norm = self.norm3(cx)
            cx_norm = modulate(cx_norm, shift_mlp, scale_mlp)
            mlpout = self.mlp(cx_norm)
            cx = cx + gate_mlp.unsqueeze(1) * mlpout
            
        else:
            # Split into 6 parts for self-attention only path
            chunks = mod_signals.chunk(6, dim=-1)
            shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
            shift_mlp, scale_mlp, gate_mlp = chunks[3], chunks[4], chunks[5]
            
            # Self-attention only
            cx_norm = self.norm1(cx)
            cx_norm = modulate(cx_norm, shift_msa, scale_msa)
            cx_self = self.self_attn(cx_norm, custom_attn_mask=custom_attn_mask, is_causal=is_causal)
            cx = cxres + gate_msa.unsqueeze(1) * cx_self
            
            # MLP
            cx_norm = self.norm2(cx)
            cx_norm = modulate(cx_norm, shift_mlp, scale_mlp)
            mlpout = self.mlp(cx_norm)
            cx = cx + gate_mlp.unsqueeze(1) * mlpout

        return cx



class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    # @torch.compile()
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb
    



class SharedAdaLNController(nn.Module):
    """Shared Adaptive Layer Normalization controller for all DiT blocks"""
    def __init__(self, dim, global_conddim, use_cross_attn=False):
        super().__init__()
        # Number of modulation signals needed
        num_mod_signals = 9 if use_cross_attn else 6
        self.modCX = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_conddim, num_mod_signals * dim, bias=False),
        )
        self.use_cross_attn = use_cross_attn

        # Zero initialize the final linear layer
        nn.init.zeros_(self.modCX[-1].weight)
        self.use_cross_attn = use_cross_attn

    def forward(self, global_cond):
        mod_signals = self.modCX(global_cond)
        if self.use_cross_attn:
            # Split into 9 parts for cross-attention path
            return mod_signals.chunk(9, dim=-1)
        else:
            # Split into 6 parts for self-attention only path
            return mod_signals.chunk(6, dim=-1)



class MemoryEfficientDiTBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        heads=8, 
        attn_pdrop=0.1, 
        resid_pdrop=0.1, 
        mlp_pdrop=0.1,
        use_cross_attn=False,
        use_rope=False,        # 1D RoPE
        query_seq_len=64,      # For action sequences
        context_seq_len=384,   # For context sequences
        rope_theta=32,         # For action sequences
        context_rope_theta=1000.0  # For context
    ):
        super().__init__()
        
        self.norm1 = RmsNorm(dim, eps=1e-6)
        self.norm2 = RmsNorm(dim, eps=1e-6)
        self.norm3 = RmsNorm(dim, eps=1e-6) if use_cross_attn else None
        
        self.use_cross_attn = use_cross_attn
        
        # Initialize self-attention with RoPE parameters
        self.self_attn = SingleAttention(
            dim=dim, 
            n_heads=heads, 
            attn_pdrop=attn_pdrop, 
            resid_pdrop=resid_pdrop,
            use_rope=use_rope,
            max_seq_len=query_seq_len,
            rope_theta=rope_theta
        )
        
        # Initialize cross-attention with separate RoPE parameters if needed
        if use_cross_attn:
            self.cross_attn = CrossAttention(
                dim=dim, 
                n_heads=heads, 
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                use_rope=use_rope,
                query_seq_len=query_seq_len,
                context_seq_len=context_seq_len,
                rope_theta=rope_theta,
                context_rope_theta=context_rope_theta
            )
            
        self.mlp = MLP(
            dim,
            dropout=mlp_pdrop
        )

    def forward(self, cx, mod_signals, context=None, custom_attn_mask=None, is_causal=False, **kwargs):
        B, L, D = cx.shape
        cxres = cx
        
        if self.use_cross_attn:
            shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = mod_signals
            
            # Self-attention
            cx_norm = self.norm1(cx)
            cx_norm = modulate(cx_norm, shift_msa, scale_msa)
            cx_self = self.self_attn(cx_norm, custom_attn_mask=custom_attn_mask, is_causal=is_causal)
            cx = cxres + gate_msa.unsqueeze(1) * cx_self
            
            # Cross-attention
            if context is None:
                raise ValueError("Context must be provided when using cross-attention")
            
            cx_norm = self.norm2(cx)
            cx_norm = modulate(cx_norm, shift_cross, scale_cross)
            cx_cross = self.cross_attn(cx_norm, context, custom_attn_mask=custom_attn_mask)
            cx = cx + gate_cross.unsqueeze(1) * cx_cross
            
            # MLP
            cx_norm = self.norm3(cx)
            cx_norm = modulate(cx_norm, shift_mlp, scale_mlp)
            mlpout = self.mlp(cx_norm)
            cx = cx + gate_mlp.unsqueeze(1) * mlpout
            
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_signals
            
            # Self-attention only
            cx_norm = self.norm1(cx)
            cx_norm = modulate(cx_norm, shift_msa, scale_msa)
            cx_self = self.self_attn(cx_norm, custom_attn_mask=custom_attn_mask, is_causal=is_causal)
            cx = cxres + gate_msa.unsqueeze(1) * cx_self
            
            # MLP
            cx_norm = self.norm2(cx)
            cx_norm = modulate(cx_norm, shift_mlp, scale_mlp)
            mlpout = self.mlp(cx_norm)
            cx = cx + gate_mlp.unsqueeze(1) * mlpout

        return cx


class FreqEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=1000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ActionSpaceEmbedderParameter(nn.Module):
    """
    Embeds discrete action indices using direct learnable parameters.
    """
    def __init__(
        self,
        hidden_size,
        max_actions=11,  # 0-10 inclusive
        embedding_size=256,
    ):
        super().__init__()
        # Direct learnable parameters for each action
        self.action_embeddings = nn.Parameter(
            torch.randn(max_actions, embedding_size) * 0.02  # Small initialization
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.max_actions = max_actions
        
    def forward(self, action_indices):
        """
        Convert action indices to embeddings using parameter lookup.
        
        Args:
            action_indices: tensor of shape (batch_size,) containing integers in [0, max_actions-1]
        """
        # Index into the parameter matrix
        embeddings = self.action_embeddings[action_indices]
        
        # Process through MLP
        embeddings = embeddings
        output = self.mlp(embeddings)
        
        return output

    def get_all_embeddings(self):
        """Returns embeddings for all possible actions."""
        return self.mlp(self.action_embeddings)



# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)



class MixtureMLP(nn.Module):
    def __init__(self, dim: int, num_action_types: int, hidden_dim: Optional[int] = None, dropout: float = 0.0, eps: float = 1e-6, use_shared_encoder: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.use_shared_encoder = use_shared_encoder

        if use_shared_encoder:
            self.shared_mlp = nn.Sequential(
                SwishGLU(dim, hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            )
        
        self.action_mlps = nn.ModuleList([
            nn.Sequential(
                SwishGLU(dim, hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_action_types)
        ])

        self.norm = RmsNorm(dim, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor, action_type: torch.Tensor) -> torch.Tensor:
        if self.use_shared_encoder:
            shared_out = self.shared_mlp(x)
        default_dtype = next(self.parameters()).dtype
        action_out = torch.zeros_like(x, dtype=default_dtype, device=x.device)
        for action in range(len(self.action_mlps)):
            mask = (action_type == action)
            if mask.any():
                out = self.action_mlps[action](x[mask])
                action_out[mask] = out

        # Simple averaging of shared and specialized paths
        if self.use_shared_encoder:
            combined = (shared_out + action_out) / 2
        else:
            combined = action_out
        
        return self.norm(x + self.dropout(combined))


class MemoryEfficientActionMixtureDiTBlock(nn.Module):
    """
    Memory efficient DiT Block with mixture of actions and shared AdaLN controller
    """
    def __init__(
        self, 
        dim, 
        num_action_types,
        heads=8,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_pdrop=0.1,
        use_cross_attn=False,
        use_rope=False,
        query_seq_len=128,
        context_seq_len=384,
        rope_theta=32,
        context_rope_theta=1000.0,
        use_shared_norm: bool = False,
        use_shared_encoder: bool = False,
        use_shared_attention: bool = False,
        use_shared_mlp: bool = False,
    ):
        super().__init__()
        self.shared_attention = use_shared_attention
        self.use_shared_encoder = use_shared_encoder
        self.use_shared_mlp = use_shared_mlp
        self.use_shared_norm = use_shared_norm
        self.use_cross_attn = use_cross_attn
        
        if self.shared_attention:
            self.self_attn = SingleAttention(
                dim=dim,
                n_heads=heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                use_rope=use_rope,
                max_seq_len=query_seq_len,
                rope_theta=rope_theta
            )
        else:
            self.self_attn = nn.ModuleList([
                SingleAttention(
                    dim=dim,
                    n_heads=heads,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    use_rope=use_rope,
                    max_seq_len=query_seq_len,
                    rope_theta=rope_theta
                ) for _ in range(num_action_types)
            ])

        if use_cross_attn:
            self.cross_attn = CrossAttention(
                dim=dim,
                n_heads=heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                use_rope=False, # RoPE is only for self-attention
            )

        if self.use_shared_mlp:
            self.mlp = MLP(
                dim=dim,
                output_dim=dim,
                dropout=mlp_pdrop
            )
        else:
            self.mlp = MixtureMLP(
                dim=dim,
                num_action_types=num_action_types,
                hidden_dim=4 * dim,
                dropout=mlp_pdrop,
                use_shared_encoder=use_shared_encoder
            )

        if self.use_shared_norm:
            self.action_norms1 = RmsNorm(dim, eps=1e-6)
            self.action_norms2 = RmsNorm(dim, eps=1e-6)
            if use_cross_attn:
                self.action_norms3 = RmsNorm(dim, eps=1e-6)
        else:
            self.action_norms1 = nn.ModuleList([
                RmsNorm(dim, eps=1e-6) for _ in range(num_action_types)
            ])
            self.action_norms2 = nn.ModuleList([
                RmsNorm(dim, eps=1e-6) for _ in range(num_action_types)
            ])

            if use_cross_attn:
                self.action_norms3 = nn.ModuleList([
                    RmsNorm(dim, eps=1e-6) for _ in range(num_action_types)
                ])


    def normalize_per_action(self, x, action_type, norm_layers):
        out = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        if self.use_shared_norm:
            out = norm_layers(x)
        else:
            for action in range(len(norm_layers)):
                mask = action_type == action
                if mask.any():
                    out[mask] = norm_layers[action](x[mask]).to(out.dtype)
        return out

    def forward(self, cx, mod_signals_list, action_type, context=None, custom_attn_mask=None, is_causal=False):
        if self.use_cross_attn and context is None:
            raise ValueError("Context is required when use_cross_attn=True")

        action_type = action_type.long()
        if cx.shape[0] == 0:
            print("Empty input")
            return cx

        residual = cx
        
        if self.use_cross_attn:
            shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = mod_signals_list
            
            attn_norm = self.normalize_per_action(residual, action_type, self.action_norms1)
            attn_norm = modulate(attn_norm, shift_msa, scale_msa) 
            attn_out = self.attention_per_action(attn_norm, action_type, None, is_causal)
            cx = residual + gate_msa.unsqueeze(1) * attn_out

            residual = cx
            cross_norm = self.normalize_per_action(residual, action_type, self.action_norms2)
            cross_norm = modulate(cross_norm, shift_cross, scale_cross)
            cx = residual + gate_cross.unsqueeze(1) * self.cross_attn(cross_norm, context, custom_attn_mask)

            residual = cx 
            mlp_norm = self.normalize_per_action(residual, action_type, self.action_norms3)
            mlp_norm = modulate(mlp_norm, shift_mlp, scale_mlp)
            cx = residual + gate_mlp.unsqueeze(1) * self.mlp_per_action(mlp_norm, action_type)

        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_signals_list
            
            attn_norm = self.normalize_per_action(residual, action_type, self.action_norms1)
            attn_norm = modulate(attn_norm, shift_msa, scale_msa)
            attn_out = self.attention_per_action(attn_norm, action_type, None, is_causal)
            cx = residual + gate_msa.unsqueeze(1) * attn_out

            residual = cx
            mlp_norm = self.normalize_per_action(residual, action_type, self.action_norms2)
            mlp_norm = modulate(mlp_norm, shift_mlp, scale_mlp)
            cx = residual + gate_mlp.unsqueeze(1) * self.mlp_per_action(mlp_norm, action_type)

        return cx

    def attention_per_action(self, x: torch.Tensor, action_type: torch.Tensor, custom_attn_mask=None, is_causal=False) -> torch.Tensor:
        if self.shared_attention:
            return self.self_attn(x, custom_attn_mask=custom_attn_mask, is_causal=is_causal)
        
        out = torch.zeros_like(x)
        for action in range(len(self.self_attn)):
            mask = action_type == action
            if mask.any():
                out[mask] = self.self_attn[action](x[mask], custom_attn_mask=custom_attn_mask, is_causal=is_causal)
        return out
    
    def mlp_per_action(self, x: torch.Tensor, action_type: torch.Tensor) -> torch.Tensor:
        if self.use_shared_mlp:
            return self.mlp(x)
        return self.mlp(x, action_type)
# next create mlp decoders and encoders for the different action types to project to the latent dim and back to the output dim 
# just a generic MLP class




class ActionHead(nn.Module):
    """
    The final layer of RDT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        self.ffn_final = MLP(
            hidden_size,
            hidden_size,
            0.0,
            output_dim=output_size,
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class MetaActionEncoder(nn.Module):
    def __init__(self, max_action_dim: int, dit_dim: int, num_action_spaces: int):
        super().__init__()
        # Learned embedding for each action space
        self.action_space_embeddings = nn.Embedding(num_action_spaces, 64)  
        # Hyperparameter 64 is arbitrary; tune as needed
        
        # One MLP that takes in [max_action_dim + embedding_dim] -> dit_dim
        input_dim = max_action_dim + 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, dit_dim)
        )
        
    def forward(self, padded_action: torch.Tensor, action_type: torch.Tensor):
        """
        padded_action: (B, T, max_action_dim)
        action_type: (B,)  # single integer per sample
        """
        # Expand action_type to match sequence length T if needed
        # Option 1: same action_type per entire sequence
        B, T, _ = padded_action.shape
        action_type_expanded = action_type.to(torch.long).view(B, 1).expand(-1, T)
        
        # Get embeddings for each token
        # shape: (B, T, 64)
        space_embeds = self.action_space_embeddings(action_type_expanded)
        
        # Concatenate along the feature dimension
        # result shape: (B, T, max_action_dim + 64)
        combined = torch.cat([padded_action, space_embeds], dim=-1)
        
        # Pass through MLP
        encoded = self.net(combined)  # shape: (B, T, dit_dim)
        return encoded
    


class MetaActionDecoder(nn.Module):
    def __init__(self, max_action_dim: int, dit_dim: int, num_action_spaces: int):
        super().__init__()
        # (Optional) embed action_space
        self.action_space_embeddings = nn.Embedding(num_action_spaces, 64)
        
        # MLP from dit_dim + 64 -> max_action_dim
        self.net = nn.Sequential(
            RmsNorm(dit_dim + 64, eps=1e-6),
            nn.Linear(dit_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, max_action_dim)
        )

    def forward(self, latent: torch.Tensor, action_type: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, T, dit_dim)
        action_type: (B,)
        """
        B, T, _ = latent.shape
        # Expand action type across sequence dimension
        action_type_expanded = action_type.to(torch.long).view(B, 1).expand(-1, T)
        space_embeds = self.action_space_embeddings(action_type_expanded)  # (B, T, 64)
        
        combined = torch.cat([latent, space_embeds], dim=-1)  # (B, T, dit_dim + 64)
        out = self.net(combined)  # (B, T, max_action_dim)
        return out



class ScaledImageProjection(nn.Module):
    def __init__(self, image_dim, projection_dim):
        super().__init__()
        self.linear = nn.Linear(image_dim, projection_dim, bias=True)
        self.norm = nn.LayerNorm(projection_dim)
        self.scale = (projection_dim ** -0.5)  # PaLI-GEMMA style scaling
        
    def forward(self, x):
        x = self.linear(x)
        x = x * self.scale  # Scale before norm
        x = self.norm(x)    # Florence style norm
        return x



class ZeroEncoder(nn.Module):
    def __init__(self, dit_dim, device):
        super(ZeroEncoder, self).__init__()
        self.dit_dim = dit_dim
        self.device = device

    def forward(self, x):
        return torch.zeros((x.shape[0], self.dit_dim), device=self.device)
    





class FlowBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        heads=8,
        attn_pdrop=0.1, 
        resid_pdrop=0.1, 
        mlp_pdrop=0.1,
        use_cross_attn=False,
        use_rope=False,
        query_seq_len=128,
        context_seq_len=384,
        rope_theta=32,
        context_rope_theta=1000.0,
        lora_dim: int = 128,  # New: LoRA dimension for AdaLN
        use_global_adaln=True  # New: Whether to use global AdaLN
    ):
        super().__init__()

        self.norm1 = RmsNorm(dim, eps=1e-6)
        self.norm2 = RmsNorm(dim, eps=1e-6)
        self.norm3 = RmsNorm(dim, eps=1e-6) if use_cross_attn else None

        self.use_cross_attn = use_cross_attn
        self.use_global_adaln = use_global_adaln
        
        # Initialize self-attention with RoPE parameters
        self.self_attn = SingleAttention(
            dim=dim, 
            n_heads=heads, 
            attn_pdrop=attn_pdrop, 
            resid_pdrop=resid_pdrop,
            use_rope=use_rope,
            max_seq_len=query_seq_len,
            rope_theta=rope_theta
        )
        
        # Initialize cross-attention if needed
        if use_cross_attn:
            self.cross_attn = CrossAttention(
                dim=dim, 
                n_heads=heads, 
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                use_rope=use_rope,
                query_seq_len=query_seq_len,
                context_seq_len=context_seq_len,
                rope_theta=rope_theta,
                context_rope_theta=context_rope_theta
            )

        self.mlp = MLP(dim, dropout=mlp_pdrop)

        # Initialize AdaLN-LoRA modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, lora_dim),  # Down project
            nn.Linear(lora_dim, 6 * dim)  # Up project
        )


    def forward(self, cx, c, context=None, custom_attn_mask=None, is_causal=False, global_adaln=None):
        B, L, D = cx.shape
        cxres = cx
        
        # Get modulation signals
        block_signals = self.adaLN_modulation(c)
        if self.use_global_adaln and global_adaln is not None:
            mod_signals = block_signals + global_adaln
        else:
            mod_signals = block_signals

        # Split modulation signals
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_signals.chunk(6, dim=1)
            
        # Self-attention
        cx_norm = self.norm1(cx)
        cx_norm = modulate(cx_norm, shift_msa, scale_msa)
        cx_self = self.self_attn(cx_norm, custom_attn_mask=custom_attn_mask, is_causal=is_causal)
        cx = cxres + gate_msa.unsqueeze(1) * cx_self
        
        # Cross-attention if enabled
        if self.use_cross_attn:
            if context is None:
                raise ValueError("Context must be provided when using cross-attention")
            cx_norm = self.norm2(cx)
            cx_cross = self.cross_attn(cx_norm, context, custom_attn_mask=custom_attn_mask)
            cx = cx + cx_cross
            
        # MLP
        cx_norm = self.norm3 if self.use_cross_attn else self.norm2
        cx_norm = modulate(cx_norm(cx), shift_mlp, scale_mlp)
        mlpout = self.mlp(cx_norm)
        cx = cx + gate_mlp.unsqueeze(1) * mlpout
        
        return cx