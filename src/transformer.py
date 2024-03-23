import jax
from attention import *
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
        src_mask: Array = None,
        training: bool = True,
    ) -> Array:
        """
        state:      NamedTuple of parameters
        src:        Transformer source input sequence (src_len, batch_size, emb_size)
        rng:        Jax random key
        src_mask:   Mask to apply to the input sequence (src_len, src_len) (Optional)
        training:   Whether to apply dropout or not
        """
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        z = self.norm_attn(state.layer_norm1, src)
        attn = self.self_attn(state.self_attn, z, z, z, src_mask)
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


class Encoder:
    """
    Transformer Encoder
    """

    def __init__(
        self, encoder_layer: EncoderLayer, n_layers: int, norm: LayerNorm = None
    ):
        """
        encoder_layer:  instance of EncoderLayer
        n_layers:       Number of encoder layers
        """

        self.n_layers = n_layers
        self.layers = [encoder_layer for _ in range(n_layers)]
        self.norm = norm

    def __call__(
        self,
        state: EncoderState,
        src: Array,
        rng: Array,
        src_mask: Array = None,
        training: bool = True,
    ) -> Array:
        """
        state:      NamedTuple of parameters
        src:        Source input sequence (src_len, batch_size, emb_size)
        rng:        Jax random key
        src_mask:   Mask for the input sequence (src_len, src_len)
        """
        for layer, layer_state in zip(self.layers, state.layers):
            src = layer(layer_state, src, rng, src_mask, training)

        return self.norm(state.norm, src) if self.norm is not None else src

    def init_state(self, rng: Array) -> EncoderState:
        rngs = jax.random.split(rng, self.n_layers + 1)
        return EncoderState(
            layers=[
                layer.init_state(rng) for layer, rng in zip(self.layers, rngs[:-1])
            ],
            norm=self.norm.init_state(rngs[-1]) if self.norm is not None else None,
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
        memory: Array,
        rng: Array,
        tgt_mask: Array = None,
        memory_mask: Array = None,
        training: bool = True,
    ) -> Array:
        """
        state:          NamedTuple of parameters
        tgt:            Decoder target input sequence (tgt_len, batch_size, emb_size)
        memory:         Sequence from encoder (src_len, batch_size, emb_size)
        rng:            Jax random key
        tgt_mask:       Mask for tgt (tgt_len, tgt_len)
        memory_mask:    Mask for src (tgt_len, src_len)
        training:       Whether to apply dropout or not
        """
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        # First masked part
        z = self.norm_attn(state.norm_attn, tgt)
        attn = self.self_attn(state.self_attn, z, z, z, tgt_mask)
        tgt += dropout(attn, self.dropout, rng1, training)

        # Second part
        z = self.norm_src_attn(state.norm_src_attn, tgt)
        attn_src = self.src_attn(state.src_attn, z, memory, memory, memory_mask)
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


class Decoder:
    def __init__(
        self, decoder_layer: DecoderLayer, n_layers: int, norm: LayerNorm = None
    ):
        self.n_layers = n_layers
        self.layers = [decoder_layer for _ in range(n_layers)]
        self.norm = norm

    def __call__(
        self,
        state: DecoderState,
        tgt: Array,
        memory: Array,  # Memory
        rng: Array,
        tgt_mask: Array = None,
        memory_mask: Array = None,
        training: bool = True,
    ) -> Array:
        """
        tgt:            Decoder target input sequence (tgt_len, batch_size, emb_size)
        memory:         Sequence from encoder (src_len, batch_size, emb_size)
        rng:            Jax random key
        tgt_mask:       Mask for tgt (tgt_len, tgt_len)
        memory_mask:    Mask for src (tgt_len, src_len)
        training:       Whether to apply dropout or not
        """

        for layer, layer_state in zip(self.layers, state.layers):
            tgt = layer(layer_state, tgt, memory, rng, tgt_mask, memory_mask, training)

        return self.norm(state.norm, tgt) if self.norm is not None else tgt

    def init_state(self, rng: Array) -> DecoderState:
        rngs = jax.random.split(rng, self.n_layers + 1)
        return DecoderState(
            layers=[
                layer.init_state(rng) for layer, rng in zip(self.layers, rngs[:-1])
            ],
            norm=self.norm.init_state(rngs[-1]) if self.norm is not None else None,
        )


class Transformer:
    """
    Norm-first Encoder-decoder Transformer
    input: (tgt_len, batch_size, emb_size) -> 
    output: (tgt_len, batch_size, emb_size)

    Assumes inputs are positionally encoded embeddings
    """

    def __init__(
        self,
        emb_size: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,  # Number of encoder and decoder layers
        d_ff: int = 2048,
        dropout: float = 0.0,
    ):
        self.encoder = Encoder(
            EncoderLayer(emb_size, n_heads, d_ff, dropout),
            n_layers,
            LayerNorm(emb_size),
        )
        self.decoder = Decoder(
            DecoderLayer(emb_size, n_heads, d_ff, dropout),
            n_layers,
            LayerNorm(emb_size),
        )

    def __call__(
        self,
        state: TransformerState,
        src: Array,
        tgt: Array,
        rng: Array,
        src_mask: Array = None,
        tgt_mask: Array = None,
        memory_mask: Array = None,
        training: bool = True,
    ) -> Array:
        """
        src:           Encoder source input sequence of token embeddings (src_len, batch_size, emb_size)
        tgt:           Decoder target input sequence of token embeddings (tgt_len, batch_size, emb_size)
        rng:           Jax random key
        src_mask:      Mask for the encoder source input sequence (src_len, src_len)
        tgt_mask:      Mask for the decoder target input sequence (tgt_len, tgt_len)
        memory_mask:   Mask for the encoder output (tgt_len, src_len)

        output.shape:  (tgt_len, batch_size, emb_size)
        """
        rng1, rng2 = jax.random.split(rng, 2)

        memory = self.encoder(state.encoder, src, rng1, src_mask, training)
        return self.decoder(
            state.decoder, tgt, memory, rng2, tgt_mask, memory_mask, training
        )
 
    def init_state(self, rng: Array) -> TransformerState:
        rngs = jax.random.split(rng, 3)
        return TransformerState(
            encoder=self.encoder.init_state(rngs[0]),
            decoder=self.decoder.init_state(rngs[1]),
        )


class Seq2SeqTransformer:
    
    def __init__(self, vocab_size: int, emb_size: int, **kwargs):
        self.emb_size = emb_size
        self.transformer = Transformer(emb_size=emb_size, **kwargs)
        # Assume same vocab for input and output sequence
        self.embed_src = Embedding(vocab_size, emb_size)
        self.embed_tgt = Embedding(vocab_size, emb_size)
        self.project_out = Linear(emb_size, vocab_size)
        self.encode_position = PositionalEncoding(emb_size)

    def __call__(
        self,
        state: Seq2SeqTransformerState,
        src: Array,
        tgt: Array,
        rng: Array,
        src_mask: Array = None,
        tgt_mask: Array = None,
        memory_mask: Array = None,
        training: bool = True,
    ) -> Array:
        # src.shape: (src_len, batch_size)
        # tgt.shape: (tgt_len, batch_size)
        # returns: (seq_len, batch_size, vocab_size) of unnormalized token probabilities

        rngs = jax.random.split(rng, 3)

        src_emb = self.embed_src(state.embed_src, src) * jnp.sqrt(self.emb_size)
        src_emb = self.encode_position(src_emb, rngs[0])

        tgt_emb = self.embed_tgt(state.embed_tgt, tgt) * jnp.sqrt(self.emb_size)
        tgt_emb = self.encode_position(tgt_emb, rngs[1])

        outputs = self.transformer(
            state.transformer, src_emb, tgt_emb, rngs[2], src_mask, tgt_mask, memory_mask, training
        )
        return self.project_out(state.project_out, outputs)

    def init_state(self, rng: Array) -> Seq2SeqTransformerState:
        rngs = jax.random.split(rng, 4)
        return Seq2SeqTransformerState(
            transformer=self.transformer.init_state(rngs[0]),
            embed_src=self.embed_src.init_state(rngs[1]),
            embed_tgt=self.embed_tgt.init_state(rngs[2]),
            project_out=self.project_out.init_state(rngs[3]),
        )

    def predict(self, state: Seq2SeqTransformerState, src: Array, tgt: Array, rng: Array):
        # TODO: full greedy decoding using Tokenizer 
        pass