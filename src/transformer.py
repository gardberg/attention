import jax
from attention import *
from base import Array


class EncoderLayer(BaseModule):
    """
    Norm-first Transformer encoder layer
    """

    def __init__(
        self,
        emb_size: int,
        n_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.0,
        layer_norm_first: bool = True,
        ff_activation: Callable = relu,
        attn_out_bias: bool = False,
        attn_qk_bias: bool = False,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.layer_norm_first = layer_norm_first

        self.norm_attn = LayerNorm(emb_size)
        self.self_attn = MultiHeadAttention(emb_size, n_heads, out_bias=attn_out_bias, qk_bias=attn_qk_bias)
        self.norm_ff = LayerNorm(emb_size)
        self.feed_forward = FeedForward(emb_size, d_ff, ff_activation, dropout)
        self.dropout = dropout

    def forward(
        self,
        state: EncoderLayerState,
        src: Array,
        rng: Array,
        src_mask: Array = None,
        training: bool = False,
    ) -> Array:
        """
        state:      NamedTuple of parameters
        src:        Transformer source input sequence (src_len, batch_size, emb_size)
        rng:        Jax random key
        src_mask:   Mask to apply to the input sequence (src_len, src_len) (Optional)
        training:   Whether to apply dropout or not
        """
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        residual = src

        if self.layer_norm_first:
            src = self.norm_attn(state.layer_norm1, src)

        x = self.self_attn(state.self_attn, src, src, src, src_mask)
        x = dropout(x, self.dropout, rng1, training)
        x += residual

        if self.layer_norm_first:
            x_norm = self.norm_ff(state.layer_norm2, x)
            x = x + self.feed_forward(state.feed_forward, x_norm, rng2, training)
        else:
            x = self.norm_attn(state.layer_norm1, x)
            x = x + self.feed_forward(state.feed_forward, x, rng2, training)
            x = self.norm_ff(state.layer_norm2, x)

        return x

    def init_state(self, rng: Array) -> EncoderLayerState:
        rngs = jax.random.split(rng, 4)
        return EncoderLayerState(
            layer_norm1=self.norm_attn.init_state(rngs[0]),
            self_attn=self.self_attn.init_state(rngs[1]),
            layer_norm2=self.norm_ff.init_state(rngs[2]),
            feed_forward=self.feed_forward.init_state(rngs[3]),
        )


class Encoder(BaseModule):
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
        super().__init__()

        self.n_layers = n_layers
        self.layers = [encoder_layer for _ in range(n_layers)]
        self.norm = norm

    def forward(
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
        output:     (src_len, batch_size, emb_size)
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


class DecoderLayer(BaseModule):
    def __init__(
        self, emb_size: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.0
    ):
        super().__init__()
        self.norm_attn = LayerNorm(emb_size)
        self.self_attn = MultiHeadAttention(emb_size, n_heads)

        self.src_attn = MultiHeadAttention(emb_size, n_heads)
        self.norm_src_attn = LayerNorm(emb_size)

        self.norm_ff = LayerNorm(emb_size)
        self.feed_forward = FeedForward(emb_size, d_ff)

        self.dropout = dropout

    def forward(
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


class Decoder(BaseModule):
    def __init__(
        self, decoder_layer: DecoderLayer, n_layers: int, norm: LayerNorm = None
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = [decoder_layer for _ in range(n_layers)]
        self.norm = norm

    def forward(
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


class Transformer(BaseModule):
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
        super().__init__()
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

    def forward(
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


class Seq2SeqTransformer(BaseModule):
    """
    Sequence to Sequence Transformer model, e.g. for machine translation
    https://pytorch.org/tutorials/beginner/translation_transformer.html
    """

    def __init__(
        self, src_vocab_size: int, emb_size: int, tgt_vocab_size: int = None, **kwargs
    ):
        super().__init__()
        if tgt_vocab_size is None:
            tgt_vocab_size = src_vocab_size

        self.emb_size = emb_size
        self.transformer = Transformer(emb_size=emb_size, **kwargs)

        # Different vocabs for src and tgt, e.g. different languages
        self.src_embedding = Embedding(src_vocab_size, emb_size)
        self.tgt_embedding = Embedding(tgt_vocab_size, emb_size)
        self.project_out = Linear(emb_size, tgt_vocab_size)
        self.encode_position = PositionalEncoding(emb_size)

    def embed_src(self, state: EmbeddingState, src: Array, rng: Array) -> Array:
        src_emb = self.src_embedding(state, src) * jnp.sqrt(self.emb_size)
        return self.encode_position(src_emb, rng)

    def embed_tgt(self, state: EmbeddingState, tgt: Array, rng: Array) -> Array:
        tgt_emb = self.tgt_embedding(state, tgt) * jnp.sqrt(self.emb_size)
        return self.encode_position(tgt_emb, rng)

    def forward(
        self,
        state: Seq2SeqTransformerState,
        src_toks: Array,
        tgt_toks: Array,
        rng: Array,
        src_mask: Array = None,
        tgt_mask: Array = None,
        memory_mask: Array = None,
        training: bool = True,
    ) -> Array:
        # src_toks.shape: (src_len, batch_size)
        # tgt_toks.shape: (tgt_len, batch_size)
        # returns: (seq_len, batch_size, tgt_vocab_size) of unnormalized token probabilities

        rngs = jax.random.split(rng, 4)

        src_emb = self.embed_src(state.src_embedding, src_toks, rngs[0])
        tgt_emb = self.embed_tgt(state.tgt_embedding, tgt_toks, rngs[1])

        memory = self.transformer.encoder(
            state.transformer.encoder, src_emb, rngs[2], src_mask, training
        )

        outputs = self.transformer.decoder(
            state.transformer.decoder,
            tgt_emb,
            memory,
            rngs[3],
            tgt_mask,
            memory_mask,
            training,
        )

        return self.project_out(state.project_out, outputs)

    def init_state(self, rng: Array) -> Seq2SeqTransformerState:
        rngs = jax.random.split(rng, 4)
        return Seq2SeqTransformerState(
            transformer=self.transformer.init_state(rngs[0]),
            src_embedding=self.src_embedding.init_state(rngs[1]),
            tgt_embedding=self.tgt_embedding.init_state(rngs[2]),
            project_out=self.project_out.init_state(rngs[3]),
        )

    def generate(
        self,
        state: Seq2SeqTransformerState,
        src_toks: Array,
        rng: Array,
        src_mask: Array = None,
        max_len: int = 10,
        stop_token: int = 100257,  # cl100k_base default
        start_token: int = 0,
    ):
        # Start generation based on start token and info from encoder

        # src_toks.shape (src_len, 1) since batch_size = 1
        # output: array of predicted tokens of shape (*, 1) (however long the model decides to generate)
        # src_mask.shape (src_len, src_len)
        # memory_mask.shape (tgt_len, src_len)

        # TODO: Split RNGs
        rngs = jax.random.split(rng, 4)

        if src_toks.ndim == 1:
            src_toks = src_toks.reshape(-1, 1)
        assert src_toks.shape[1] == 1, "Only batch_size = 1 supported for generation"

        # src.shape: (src_len, 1, emb_size)
        src = self.embed_src(state.src_embedding, src_toks, rngs[0])

        # memory.shape: (src_len, 1, emb_size)
        memory = self.transformer.encoder(
            state.transformer.encoder, src, rngs[1], src_mask, training=False
        )

        # TODO: Do we need to start with the start token here?
        predicted = jnp.array([start_token]).reshape(1, 1)
        for i in range(max_len):
            # Causal mask on previously generated tokens
            tgt_mask = create_causal_mask(predicted.shape[0])

            # tgt_len increases with 1 each iteration,
            # as the predicted token is fed back into the model
            # (tgt_len, 1, emb_size)
            tgt = self.embed_tgt(state.tgt_embedding, predicted, rngs[2])

            # (tgt_len, 1, emb_size)
            out = self.transformer.decoder(
                state.transformer.decoder,
                tgt,
                memory,
                rngs[3],
                tgt_mask,
                memory_mask=None,
                training=False,
            )

            # (tgt_len, 1, tgt_vocab_size)
            logits = self.project_out(state.project_out, out)

            # Greedy decode
            # (1, 1)
            pred_token = jnp.argmax(logits[-1], axis=-1).reshape(1, 1)

            predicted = jnp.concatenate([predicted, pred_token], axis=0)

            if pred_token == stop_token:
                break

        # (*, 1)
        return predicted
