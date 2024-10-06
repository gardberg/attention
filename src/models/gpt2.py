from venv import create
from base import Array, BaseModule
from transformer import Embedding
from attention import Linear, LayerNorm, create_causal_mask
from states import GPT2DenseState, GPT2AttentionState
from act import gelu_new, dropout, softmax

import jax.numpy as jnp

class GPT2(BaseModule):
    def __init__(self, vocab_size: int=50257, emb_size: int=768):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.context_len = 1024

        self.wte = Embedding(vocab_size, emb_size)
        self.wpe = Embedding(self.context_len, emb_size)

        
        self.lm_head = Linear(emb_size, vocab_size, bias=False)


# activation: gelu_new
# attention: GPT2Attention
class GPT2Block(BaseModule):
    def __init__(self, emb_size: int=768):
        super().__init__()

        self.emb_size = emb_size

        self.ln_1 = LayerNorm(emb_size)
        self.attn = GPT2Attention(emb_size)
        self.ln_2 = LayerNorm(emb_size)

        self.mlp = GPT2Dense(emb_size * 4, emb_size)

        
# Only self attention
class GPT2Attention(BaseModule):
    def __init__(
        self,
        emb_size: int=768,
        n_heads: int=12,
        context_len: int=1024
    ):
        super().__init__()

        self.emb_size = emb_size
        self.n_heads = n_heads

        assert emb_size % n_heads == 0, f"emb_size must be divisible by n_heads, got {emb_size} % {n_heads}"

        self.head_dim = emb_size // n_heads
        self.context_len = context_len

        # We dont seem to care about layer index, no scaling based on it
        
        self.c_attn = Linear(emb_size, emb_size * 3) # Combined layer for q, k, v
        self.c_proj = Linear(emb_size, emb_size) # output projection

        causal_mask = ~create_causal_mask(context_len)
        self.pos_bias = causal_mask.reshape(1, 1, context_len, context_len)

    # TODO: kv cache
    def forward(
        self,
        states: GPT2AttentionState,
        x: Array["batch_size, seq_len, emb_size"],
        rng: Array,
        training: bool=False
    ) -> Array:

        # q, k, v.shape: (batch_size, seq_len, emb_size)
        query, key, value = jnp.split(self.c_attn(states.c_attn, x), 3, axis=-1)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # attention computation
        attn_logits = jnp.matmul(query, key.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)

        # masking
        q_len, k_len = query.shape[-2], key.shape[-2]
        causal_mask = self.pos_bias[:, :, k_len - q_len:k_len, :k_len]
        mask_value = jnp.finfo(attn_logits.dtype).min
        attn_logits = jnp.where(causal_mask, attn_logits, mask_value)

        attn_weights = softmax(attn_logits, dim=-1)
        attn_weights = dropout(attn_weights, 0.1, rng, training)

        attn_output = jnp.matmul(attn_weights, value)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(states.c_proj, attn_output)
        attn_output = dropout(attn_output, 0.1, rng, training)

        return attn_output


    def _split_heads(
        self,
        x: Array["batch_size, seq_len, emb_size"]
    ) -> Array["batch_size, n_heads, seq_len, head_dim"]:
        return x.reshape(x.shape[:-1] + (self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)

    def _merge_heads(
        self,
        x: Array["batch_size, n_heads, seq_len, head_dim"]
    ) -> Array["batch_size, seq_len, emb_size"]:
        return x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], self.emb_size)
    


class GPT2Dense(BaseModule):
    def __init__(self, intermediate_size: int, emb_size: int=768):
        super().__init__()

        self.emb_size = emb_size # input features
        self.intermediate_size = intermediate_size

        self.c_fc = Linear(emb_size, self.intermediate_size)
        self.c_proj = Linear(self.intermediate_size, emb_size)

    def forward(
        self, 
        states: GPT2DenseState,
        x: Array["..., emb_size"],
        rng: Array,
        training: bool=False
    ) -> Array:

        # out.shape: (..., emb_size)
        x = self.c_fc(states.c_fc, x) 
        x = gelu_new(x)
        x = self.c_proj(states.c_proj, x)
        if training: x = dropout(x, 0.1, rng)
        out = x
        return out

