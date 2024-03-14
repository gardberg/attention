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
        self, emb_size: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.0
    ):
        self.norm_attn = LayerNorm(emb_size)
        self.self_attn = MultiHeadAttention(emb_size, n_heads)
        self.norm_ff = LayerNorm(emb_size)
        self.feed_forward = FeedForward(emb_size, d_ff)
        self.dropout = dropout

    def __call__(
        self,
        state: EncoderLayerState,
        x: Array,
        rng: Array,
        mask: Array = None,
        training: bool = True,
    ) -> Array:
        """
        state:      NamedTuple of parameters
        x:          Source input sequence (context_len, batch_size, emb_size)
        rng:        Jax random key
        mask:       Mask to apply to the input sequence x (context_len, context_len, batch_size, n_heads) (Optional)
        training:   Whether to apply dropout or not
        """
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        z = self.norm_attn(state.layer_norm1, x)
        attn = self.self_attn(state.self_attn, z, z, z, mask)
        x_drop = dropout(attn, self.dropout, rng1, training)
        x = x + x_drop

        z = self.norm_ff(state.layer_norm2, x)
        ff = self.feed_forward(state.feed_forward, z, rng2)
        x_drop = dropout(ff, self.dropout, rng3, training)
        x = x + x_drop
        return x

    def init_state(self, rng: Array) -> EncoderLayerState:
        rngs = jax.random.split(rng, 4)
        return EncoderLayerState(
            layer_norm1=self.norm_attn.init_state(rngs[0]),
            self_attn=self.self_attn.init_state(rngs[1]),
            layer_norm2=self.norm_ff.init_state(rngs[2]),
            feed_forward=self.feed_forward.init_state(rngs[3]),
        )



class DecoderLayer:
    def __init__(
        self, emb_size: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.0
    ):
        self.norm_attn = LayerNorm(emb_size)
        self.self_attn = MultiHeadAttention(emb_size, n_heads)

        self.src_attn = MultiHeadAttention(emb_size, n_heads)
        self.norm_src_attn = LayerNorm(emb_size)

        self.norm_ff = LayerNorm(emb_size)
        self.feed_forward = FeedForward(emb_size, d_ff)

        self.dropout = dropout

    def __call__(
        self,
        state: DecoderLayerState,
        x: Array,
        src: Array,
        rng: Array,
        mask: Array = None,
        src_mask: Array = None,
        training: bool = True,
    ) -> Array:
        """
        state:      NamedTuple of parameters
        x:          Decoder input sequence (context_len, batch_size, emb_size)
        src:        Encoder input sequence (context_len, batch_size, emb_size)
        rng:        Jax random key
        mask:       Mask for x (context_len, context_len, batch_size, n_heads) (Optional)
        src_mask:   Mask for src (context_len, context_len, batch_size, n_heads) (Optional)
        training:   Whether to apply dropout or not
        """
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        # First masked part
        z = self.norm_attn(state.norm_attn, x)
        attn = self.self_attn(state.self_attn, z, z, z, mask)
        x += dropout(attn, self.dropout, rng1, training)

        # Second part
        z = self.norm_src_attn(state.norm_src_attn, x)
        attn_src = self.src_attn(state.src_attn, z, src, src, src_mask)
        x += dropout(attn_src, self.dropout, rng2, training)

        z = self.norm_ff(state.norm_ff, x)
        ff = self.feed_forward(state.feed_forward, z, rng3, training)
        x += dropout(ff, self.dropout, rng4, training)

        return x

    def init_state(self, rng: Array) -> DecoderLayerState:
        rngs = jax.random.split(rng, 6)
        return DecoderLayerState(
            norm_attn=self.norm_attn.init_state(rngs[0]),
            self_attn=self.self_attn.init_state(rngs[1]),
            src_attn=self.src_attn.init_state(rngs[2]),
            norm_src_attn=self.norm_src_attn.init_state(rngs[3]),
            norm_ff=self.norm_ff.init_state(rngs[4]),
            feed_forward=self.feed_forward.init_state(rngs[5]),
        )
