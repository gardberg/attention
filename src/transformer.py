import jax.numpy as jnp
import jax
from attention import *
from act import relu
from jax import Array


class EncoderLayer:
    """
    Norm-first Transformer encoder layer
    """

    def __init__(
        self, emb_size: int, n_heads: int, d_ff: int = 2048, dropout: float = 0
    ):
        self.layer_norm1 = LayerNorm(emb_size)
        self.self_attn = MultiHeadAttention(emb_size, n_heads)
        self.layer_norm2 = LayerNorm(emb_size)
        self.feed_forward = FeedForward(emb_size, d_ff)
        self.dropout = dropout

    def __call__(
        self, state: EncoderLayerState, x: Array, mask: Array, rng: Array
    ) -> Array:
        """
        x: (context_len, batch_size, emb_size)
        """
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        z = self.layer_norm1(state.layer_norm1_state, x)
        attn = self.self_attn(state.self_attn_state, z, z, z, mask)
        x_drop = dropout(attn, self.dropout, rng1)
        x = x + x_drop

        z = self.layer_norm2(state.layer_norm2_state, x)
        ff = self.feed_forward(state.feed_forward_state, z, rng2)
        x_drop = dropout(ff, self.dropout, rng3)
        x = x + x_drop
        return x

    def init_state(self, rng: Array) -> EncoderLayerState:
        rngs = jax.random.split(rng, 4)
        return EncoderLayerState(
            layer_norm1_state=self.layer_norm1.init_state(rngs[0]),
            self_attn_state=self.self_attn.init_state(rngs[1]),
            layer_norm2_state=self.layer_norm2.init_state(rngs[2]),
            feed_forward_state=self.feed_forward.init_state(rngs[3]),
            training=True,
        )
