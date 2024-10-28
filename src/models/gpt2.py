from base import Array, BaseModule
from transformer import Embedding
from attention import Linear, LayerNorm, create_causal_mask
from states import GPT2DenseState, GPT2AttentionState, GPT2BlockState, GPT2BaseModelState, GPT2State
from act import gelu_new, dropout, softmax_stable
from log_utils import logger

import jax.numpy as jnp
import jax


# GPT2Model with LM head
class GPT2(BaseModule):
    def __init__(self, vocab_size: int = 50257, emb_size: int = 768):
        super().__init__()
        
        self.transformer = GPT2BaseModel(vocab_size, emb_size)
        self.lm_head = Linear(emb_size, vocab_size, bias=False)

        self.eos_bos_token_id = 50256

    def forward(
        self,
        state: GPT2State,
        input_ids: Array["batch_size, context_len"],
        rng: Array,
        training: bool = False,
    ) -> Array["batch_size, context_len, vocab_size"]:
        x = self.transformer(state.transformer, input_ids, rng, training)
        return self.lm_head(state.lm_head, x) # use shared weights between lm_head and wte

    def generate(
        self,
        state: GPT2State,
        input_ids: Array["context_len,"],
        rng: Array,
        max_new_tokens: int = 50,
    ) -> Array["context_len + max_new_tokens,"]:
        input_ids = input_ids.reshape(1, -1)
        
        pred_token_ids = jnp.concatenate([jnp.array([[self.eos_bos_token_id]]), input_ids], axis=1)
        nbr_new_tokens = 0

        while (nbr_new_tokens < max_new_tokens) and (pred_token_ids.shape[-1] < self.transformer.context_len):
            logits = self.forward(state, pred_token_ids, rng, training=False)
            
            logits = logits[:, -1, :]
            
            probs = softmax_stable(logits, dim=-1)
            next_token_id = self.predict_next_token(probs)
            
            pred_token_ids = jnp.concatenate([pred_token_ids, next_token_id.reshape(1, 1)], axis=1)
            nbr_new_tokens += 1

            if next_token_id == self.eos_bos_token_id:
                break

        return pred_token_ids
            
    def predict_next_token(self, token_probs: Array["1, vocab_size"]) -> Array["1"]:
        return jnp.argmax(token_probs, axis=-1)

    def init_state(self, rng: Array) -> GPT2State:
        rngs = jax.random.split(rng, 2)
        return GPT2State(
            transformer=self.transformer.init_state(rngs[0]),
            lm_head=self.lm_head.init_state(rngs[1]),
        )


class GPT2BaseModel(BaseModule):
    def __init__(self, vocab_size: int = 50257, emb_size: int = 768, n_layers: int = 12):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.context_len = 1024

        self.wte = Embedding(vocab_size, emb_size)
        self.wpe = Embedding(self.context_len, emb_size)

        self.blocks = [GPT2Block(emb_size) for _ in range(n_layers)]
        self.ln_f = LayerNorm(emb_size)

    # token input ids to logits
    def forward(
        self,   
        state: GPT2BaseModelState,
        input_ids: Array["batch_size, context_len"],
        rng: Array,
        training: bool = False,
    ) -> Array["batch_size, context_len, emb_size"]:
        rng1, rng2 = jax.random.split(rng)
        
        input_embeds = self.wte(state.wte, input_ids)

        position_ids = jnp.arange(input_ids.shape[-1])
        position_embeds = self.wpe(state.wpe, position_ids)

        x = input_embeds + position_embeds
        x = dropout(x, 0.1, rng1, training)

        for i, block in enumerate(self.blocks):
            x = block(state.blocks[i], x, rng2, training)

        x = self.ln_f(state.ln_f, x)

        return x

    def init_state(self, rng: Array) -> GPT2BaseModelState:
        rngs = jax.random.split(rng, 3 + len(self.blocks))
        return GPT2BaseModelState(
            wte=self.wte.init_state(rngs[0]),
            wpe=self.wpe.init_state(rngs[1]),
            blocks=[block.init_state(rngs[i + 2]) for i, block in enumerate(self.blocks)],
            ln_f=self.ln_f.init_state(rngs[-1]),
        )


# activation: gelu_new
# attention: GPT2Attention
class GPT2Block(BaseModule):
    def __init__(self, emb_size: int = 768):
        super().__init__()

        self.emb_size = emb_size

        self.ln_1 = LayerNorm(emb_size)
        self.attn = GPT2Attention(emb_size)
        self.ln_2 = LayerNorm(emb_size)

        self.mlp = GPT2Dense(emb_size * 4, emb_size)

    def forward(
        self,
        states: GPT2BlockState,
        x: Array["batch_size, context_len, emb_size"],
        rng: Array,
        training: bool = False,
    ) -> Array["batch_size, context_len, emb_size"]:
        rng1, rng2 = jax.random.split(rng)

        residual = x

        x = self.ln_1(states.ln_1, x)
        x = self.attn(states.attn, x, rng1, training)

        x += residual

        residual = x

        x = self.ln_2(states.ln_2, x)
        x = self.mlp(states.mlp, x, rng2, training)

        x += residual

        return x

    def init_state(self, rng: Array) -> GPT2BlockState:
        rngs = jax.random.split(rng, 4)
        return GPT2BlockState(
            ln_1=self.ln_1.init_state(rngs[0]),
            attn=self.attn.init_state(rngs[1]),
            ln_2=self.ln_2.init_state(rngs[2]),
            mlp=self.mlp.init_state(rngs[3]),
        )

# Only self attention
class GPT2Attention(BaseModule):
    def __init__(self, emb_size: int = 768, n_heads: int = 12, context_len: int = 1024):
        super().__init__()

        self.emb_size = emb_size
        self.n_heads = n_heads

        assert (
            emb_size % n_heads == 0
        ), f"emb_size must be divisible by n_heads, got {emb_size} % {n_heads}"

        self.head_dim = emb_size // n_heads
        self.context_len = context_len

        # We dont seem to care about layer index, no scaling based on it

        self.c_attn = Linear(emb_size, emb_size * 3)  # Combined layer for q, k, v
        self.c_proj = Linear(emb_size, emb_size)  # output projection

        causal_mask = ~create_causal_mask(context_len)
        self.pos_bias = causal_mask.reshape(1, 1, context_len, context_len)

    # TODO: kv cache
    def forward(
        self,
        state: GPT2AttentionState,
        x: Array["batch_size, context_len, emb_size"],
        rng: Array,
        training: bool = False,
    ) -> Array["batch_size, context_len, emb_size"]:
        rng1, rng2 = jax.random.split(rng)
        # q, k, v.shape: (batch_size, context_len, emb_size)
        query, key, value = jnp.split(self.c_attn(state.c_attn, x), 3, axis=-1)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # attention computation
        attn_logits = jnp.matmul(query, key.transpose(0, 1, 3, 2)) / jnp.sqrt(
            self.head_dim
        )

        # masking
        q_len, k_len = query.shape[-2], key.shape[-2]
        causal_mask = self.pos_bias[:, :, k_len - q_len : k_len, :k_len]
        mask_value = jnp.finfo(attn_logits.dtype).min
        attn_logits = jnp.where(causal_mask, attn_logits, mask_value)

        attn_weights = softmax_stable(attn_logits, dim=-1)
        attn_weights = dropout(attn_weights, 0.1, rng1, training)

        attn_output = jnp.matmul(attn_weights, value)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(state.c_proj, attn_output)
        attn_output = dropout(attn_output, 0.1, rng2, training)

        return attn_output

    def _split_heads(
        self, x: Array["batch_size, context_len, emb_size"]
    ) -> Array["batch_size, n_heads, context_len, head_dim"]:
        return x.reshape(x.shape[:-1] + (self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )

    def _merge_heads(
        self, x: Array["batch_size, n_heads, context_len, head_dim"]
    ) -> Array["batch_size, context_len, emb_size"]:
        return x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], self.emb_size)

    def init_state(self, rng: Array) -> GPT2AttentionState:
        rngs = jax.random.split(rng, 2)
        return GPT2AttentionState(
            c_attn=self.c_attn.init_state(rngs[0]),
            c_proj=self.c_proj.init_state(rngs[1]),
        )


class GPT2Dense(BaseModule):
    def __init__(self, intermediate_size: int, emb_size: int = 768):
        super().__init__()

        self.emb_size = emb_size  # input features
        self.intermediate_size = intermediate_size

        self.c_fc = Linear(emb_size, self.intermediate_size)
        self.c_proj = Linear(self.intermediate_size, emb_size)

    def forward(
        self,
        state: GPT2DenseState,
        x: Array["..., emb_size"],
        rng: Array,
        training: bool = False,
    ) -> Array["..., emb_size"]:
        # out.shape: (..., emb_size)
        x = self.c_fc(state.c_fc, x)
        x = gelu_new(x)
        x = self.c_proj(state.c_proj, x)
        if training:
            x = dropout(x, 0.1, rng)
        out = x
        return out

    def init_state(self, rng: Array) -> GPT2DenseState:
        rngs = jax.random.split(rng, 2)
        return GPT2DenseState(
            c_fc=self.c_fc.init_state(rngs[0]),
            c_proj=self.c_proj.init_state(rngs[1]),
        )
