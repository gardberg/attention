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
        self.emb_size = emb_size
        self.norm_attn = LayerNorm(emb_size)
        self.self_attn = MultiHeadAttention(emb_size, n_heads)
        self.norm_ff = LayerNorm(emb_size)
        self.feed_forward = FeedForward(emb_size, d_ff)
        self.dropout = dropout

    def __call__(
        self,
        state: EncoderLayerState,
        src: Array,
        rng: Array,
        mask: Array = None,
        training: bool = True,
    ) -> Array:
        """
        state:      NamedTuple of parameters
        src:        Transformer source input sequence (src_len, batch_size, emb_size)
        rng:        Jax random key
        mask:       Mask to apply to the input sequence (src_len, src_len) (Optional)
        training:   Whether to apply dropout or not
        """
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        z = self.norm_attn(state.layer_norm1, src)
        attn = self.self_attn(state.self_attn, z, z, z, mask)
        src_drop = dropout(attn, self.dropout, rng1, training)
        src += src_drop

        z = self.norm_ff(state.layer_norm2, src)
        ff = self.feed_forward(state.feed_forward, z, rng2)
        src_drop = dropout(ff, self.dropout, rng3, training)
        src += src_drop
        return src

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
        tgt: Array,
        src: Array,
        rng: Array,
        mask: Array = None,
        src_mask: Array = None,
        training: bool = True,
    ) -> Array:
        """
        state:      NamedTuple of parameters
        tgt:        Decoder target input sequence (tgt_len, batch_size, emb_size)
        src:        Sequence from encoder (src_len, batch_size, emb_size)
        rng:        Jax random key
        mask:       Mask for x (tgt_len, tgt_len) (Optional)
        src_mask:   Mask for src (src_len, src_len) (Optional)
        training:   Whether to apply dropout or not
        """
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        # First masked part
        z = self.norm_attn(state.norm_attn, tgt)
        attn = self.self_attn(state.self_attn, z, z, z, mask)
        tgt += dropout(attn, self.dropout, rng1, training)

        # Second part
        z = self.norm_src_attn(state.norm_src_attn, tgt)
        attn_src = self.src_attn(state.src_attn, z, src, src, src_mask)
        tgt += dropout(attn_src, self.dropout, rng2, training)

        z = self.norm_ff(state.norm_ff, tgt)
        ff = self.feed_forward(state.feed_forward, z, rng3, training)
        tgt += dropout(ff, self.dropout, rng4, training)

        return tgt

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


class Encoder:
    """
    Transformer Encoder
    """

    def __init__(self, encoder_layer: EncoderLayer, n_layers: int, norm: LayerNorm=None):
        """
        encoder_layer:  instance of EncoderLayer
        n_layers:       Number of encoder layers
        """

        self.n_layers = n_layers
        self.layers = [encoder_layer for _ in range(n_layers)]
        self.norm = norm

    def __call__(self, state: EncoderState, src: Array, rng: Array, mask: Array=None, training: bool=True) -> Array:
        """
        state:      NamedTuple of parameters
        src:        Source input sequence (src_len, batch_size, emb_size)
        """
        for layer, layer_state in zip(self.layers, state.layers):
            src = layer(layer_state, src, rng, mask, training)

        return self.norm(state.norm, src) if self.norm is not None else src

    def init_state(self, rng: Array) -> EncoderState:
        rngs = jax.random.split(rng, self.n_layers + 1)
        return EncoderState(
            layers=[layer.init_state(rng) for layer, rng in zip(self.layers, rngs[:-1])],
            norm=self.norm.init_state(rngs[-1]) if self.norm is not None else None,
        )
        