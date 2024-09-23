
# T5 translation model based on google t5-small for translation
# takes in input ids in the form of tokens, and returns predicted tokens (for translation)
from typing import Callable, Optional, Tuple
from jax import random, jit
import jax.numpy as jnp

from attention import Embedding, RMSNorm, Linear, create_causal_mask
from act import relu, dropout, softmax
from states import T5DenseState, T5FeedForwardState, T5MultiHeadAttentionState, EmbeddingState, LinearState, T5AttentionLayerState, T5EncoderBlockState, T5DecoderBlockState, T5EncoderState, T5DecoderState, T5BaseModelState, T5ModelState
from base import Array, BaseModule

from log_utils import logger

# https://github.com/huggingface/transformers/blob/e4ea19b958c89d61e42461fac6ac8441787121f8/src/transformers/models/t5/modeling_t5.py#L646


# Corresponds to T5ForConditionalGeneration
class T5Model(BaseModule):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.base_model = T5BaseModel(vocab_size, emb_size)
        self.lm_head = Linear(emb_size, vocab_size, bias=False)

        self.decoder_start_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 0

        self.emb_size = emb_size

        # Should decoder_input_ids be static here since its shape changes on every call?
        self._jit_forward = jit(self.forward)

    def forward(
        self,
        state: T5ModelState,
        input_ids: Array["batch_size, context_len"],
        decoder_input_ids: Array["batch_size, tgt_len"],
        rng: Array,
        encoder_output: Optional[Array["batch_size, context_len, emb_size"]] = None,
    ) -> Tuple[Array["batch_size, tgt_len, vocab_size"], Array["batch_size, context_len, emb_size"]]:
        # Returns logits over vocab from lm head, as well as encoder output

        # enligt config.json: decoder_start_token_id = 0
        # This means that for the first generation step, forward should be called with decoder_input_ids something like [0]
        # logger.debug(f"input_ids: {input_ids.shape}, decoder_input_ids: {decoder_input_ids.shape}")

        decoder_output, encoder_output = self.base_model(state.base_model, input_ids, decoder_input_ids, rng, encoder_output=encoder_output)
        rescaled_output = decoder_output * (self.emb_size ** -0.5) # From TF Mesh implementation
        return self.lm_head(state.lm_head, rescaled_output), encoder_output

    def generate(
        self,
        state: T5ModelState,
        input_ids: Array["1, context_len"],
        rng: Array,
        max_length: int=300,
    ) -> Array["1, _"]:
        # Generate translation from input_ids
        # Returns an array of token ids of the predicted text
        # until eather max_length of a stop token is reached

        # https://huggingface.co/blog/encoder-decoder#encoder-decoder

        rngs = random.split(rng, 2)

        pred_token_ids = jnp.array([[self.decoder_start_token_id]])
        
        encoder_output = None
        for _ in range(max_length-1):
            
            logits, encoder_output = self.forward(state, input_ids, pred_token_ids, rngs[0], encoder_output)
            logits = logits[:, -1, :]

            probs = softmax(logits, dim=-1)

            next_token_id = self.predict_next_token(probs)

            pred_token_ids = jnp.concatenate([pred_token_ids, next_token_id.reshape(1, 1)], axis=1)

            if next_token_id == self.eos_token_id:
                break

        return pred_token_ids

    # TODO: Beam search
    def predict_next_token(self, token_probs: Array["1, vocab_size"]) -> Array["1"]:
        return jnp.argmax(token_probs, axis=-1)

    def init_state(self, rng: Array) -> T5ModelState:
        rngs = random.split(rng, 2)
        return T5ModelState(
            base_model=self.base_model.init_state(rngs[0]),
            lm_head=self.lm_head.init_state(rngs[1]),
        )

        

class T5BaseModel(BaseModule):
    def __init__(self, vocab_size: int, emb_size: int, n_layers: int=6):
        super().__init__()
        # Embedding layer weights are shared between encode and decoder
        self.encoder = T5Encoder(emb_size, n_layers, vocab_size)
        self.decoder = T5Decoder(emb_size, n_layers, vocab_size)

    def forward(
        self,
        state: T5BaseModelState,
        input_ids: Array["batch_size, context_len"], # context, e.g. source language to transle
        decoder_input_ids: Array["batch_size, tgt_len"], # start of translation, e.g. "Translate English to French: "
        rng: Array,
        encoder_output: Array["batch_size, context_len, emb_size"] = None,
    ) -> Tuple[Array["batch_size, tgt_len, emb_size"], Array["batch_size, context_len, emb_size"]]:

        rngs = random.split(rng, 2)
        
        if encoder_output is None:
            encoder_output = self.encoder(state.encoder, input_ids, rngs[0])

        decoder_output = self.decoder(state.decoder, decoder_input_ids, encoder_output, rngs[1])

        return decoder_output, encoder_output

    def init_state(self, rng: Array) -> T5BaseModelState:
        rngs = random.split(rng, 2)
        return T5BaseModelState(
            encoder=self.encoder.init_state(rngs[0]),
            decoder=self.decoder.init_state(rngs[1]),
        )


# T5Stack with encoder config
class T5Encoder(BaseModule):
    def __init__(self, emb_size: int, n_layers: int, vocab_size: int, n_heads: int=8, dropout_rate: float=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.n_layers = n_layers

        # Shared between enconder and decoder 
        # -> make sure to use same weights from state!
        self.embedding = Embedding(vocab_size, emb_size)

        # Only first block uses relative attention bias
        self.blocks = [
            T5EncoderBlock(emb_size, n_heads, use_rel_attn_bias=bool(i == 0)) for i in range(n_layers)
        ]

        self.norm = RMSNorm(self.emb_size, eps=1e-6)
        self.dropout_rate = dropout_rate

    def forward(
        self,
        state: T5EncoderState,
        input_ids: Array["batch_size, context_len"],
        rng: Array,
        training: bool=False,
    ) -> Array["batch_size, context_len, emb_size"]:
        rngs = random.split(rng, self.n_layers + 2)

        input_embs: Array["batch_size, context_len, emb_size"] = self.embedding(state.embedding, input_ids)

        hidden_states = dropout(input_embs, self.dropout_rate, rng=rngs[0], training=training)

        pos_bias: Array["1, n_heads, context_len, context_len"] = None
        for block, block_state, rng in zip(self.blocks, state.blocks, rngs[1:self.n_layers + 1]):
            # Array["batch_size, context_len, emb_size"]
            hidden_states, pos_bias = block(
                block_state,
                hidden_states,
                rng,
                training=training,
                pos_bias=pos_bias,
                output_pos_bias=True)

        hidden_states_normed = self.norm(state.norm, hidden_states)

        return dropout(hidden_states_normed, self.dropout_rate, rngs[-1], training=training)

    def init_state(self, rng: Array) -> T5EncoderState:
        rngs = random.split(rng, 2)
        return T5EncoderState(
            embedding=self.embedding.init_state(rngs[0]),
            blocks=[block.init_state(rng) for block in self.blocks],
            norm=self.norm.init_state(rngs[1]),
        )


# T5Block as Encoder
class T5EncoderBlock(BaseModule):
    def __init__(
        self,
        emb_size: int,
        n_heads: int,
        dropout_rate: float=0.1,
        use_rel_attn_bias: bool=False,
    ):
        super().__init__()
        self.self_attn_layer = T5SelfAttention(
            emb_size,
            n_heads,
            dropout=dropout_rate,
            use_rel_attn_bias=use_rel_attn_bias)

        self.feed_forward = T5FeedForward(emb_size, 4 * emb_size) # 2048

    def forward(
        self,
        state: T5EncoderBlockState,
        x: Array["batch_size, context_len, emb_size"],
        rng: Array,
        training = False,
        pos_bias = None,
        output_pos_bias = False,
    ) -> Array["batch_size, context_len, emb_size"]:

        rngs = random.split(rng, 2)
        
        self_attn_out = self.self_attn_layer(
            state.self_attn_layer,
            x,
            rngs[0],
            training=training,
            pos_bias=pos_bias,
            output_pos_bias=output_pos_bias)

        if output_pos_bias:
            self_attn_out, pos_bias = self_attn_out # unpack tuple
            ff_out = self.feed_forward(state.feed_forward, self_attn_out, rngs[1], training=training)
            return ff_out, pos_bias
        else:
            return self.feed_forward(state.feed_forward, self_attn_out, rngs[1], training=training)

    def init_state(self, rng: Array) -> T5EncoderBlockState:
        rngs = random.split(rng, 3)
        return T5EncoderBlockState(
            self_attn_layer=self.self_attn_layer.init_state(rngs[0]),
            feed_forward=self.feed_forward.init_state(rngs[1]),
        )


# TODO: Cleanup debug
# T5Stack with decoder config
class T5Decoder(BaseModule):
    def __init__(self, emb_size: int, n_layers: int, vocab_size: int, n_heads: int=8, dropout_rate: float=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.n_layers = n_layers

        # Shared between enconder and decoder 
        # -> make sure to use same weights from state!
        self.embedding = Embedding(vocab_size, emb_size)

        # Only first block uses relative attention bias
        self.blocks = [
            T5DecoderBlock(emb_size, n_heads, use_rel_attn_bias=bool(i == 0)) for i in range(n_layers)
        ]

        self.norm = RMSNorm(self.emb_size, eps=1e-6)
        self.dropout_rate = dropout_rate

        self.debug_states = dict()

    def forward(
        self,
        state: T5DecoderState,
        input_ids: Array["batch_size, context_len"],
        encoder_hidden_states: Array["batch_size, context_len, emb_size"],
        rng: Array,
        training: bool=False,
    ) -> Array["batch_size, context_len, emb_size"]:
        # Transforms encoder output and input ids into decoder output
        # used for next token prediction

        tgt_len = input_ids.shape[1]

        self.debug_states["input_ids"] = input_ids.copy()
        self.debug_states["encoder_hidden_states"] = encoder_hidden_states.copy()

        rngs = random.split(rng, self.n_layers + 2)
        
        input_embeds: Array["batch_size, context_len, emb_size"] = self.embedding(state.embedding, input_ids)

        self.debug_states["input_embeds"] = input_embeds.copy()

        decoder_hidden_states = dropout(input_embeds, self.dropout_rate, rng=rngs[0], training=training)

        self.debug_states["decoder_hidden_states"] = decoder_hidden_states.copy()

        self_attn_mask = create_causal_mask(tgt_len, tgt_len)
        self_attn_mask = jnp.where(self_attn_mask, float("-inf"), 0.0)

        # init pos bias for both self attention and cross attention in decoder
        # and reuse the bias from prev layer
        self_pos_bias: Optional[Array["1, n_heads, context_len, context_len"]] = None
        cross_pos_bias: Array["1, n_heads, context_len, context_len"] = None
        for i, (block, block_state, rng) in enumerate(zip(self.blocks, state.blocks, rngs[1:self.n_layers + 1])):
            block_output = block(
                block_state,
                decoder_hidden_states,
                encoder_hidden_states,
                rng,
                training=training,
                self_pos_bias=self_pos_bias,
                cross_pos_bias=cross_pos_bias,
                mask=self_attn_mask,
                output_pos_bias=True)

            decoder_hidden_states, self_pos_bias, cross_pos_bias = block_output

            self.debug_states[f"post_block_hs_{i}"] = decoder_hidden_states.copy() 
            self.debug_states[f"post_block_position_bias_{i}"] = self_pos_bias.copy()
            self.debug_states[f"post_block_cross_pos_bias_{i}"] = cross_pos_bias.copy()

        decoder_hidden_states_normed = self.norm(state.norm, decoder_hidden_states)
        self.debug_states["decoder_hidden_states_normed"] = decoder_hidden_states_normed.copy()

        dropped = dropout(decoder_hidden_states_normed, self.dropout_rate, rngs[-1], training=training)

        self.debug_states["dropped"] = dropped.copy()

        return dropped

    def init_state(self, rng: Array) -> T5DecoderState:
        rngs = random.split(rng, 2)
        return T5DecoderState(
            embedding=self.embedding.init_state(rngs[0]),
            blocks=[block.init_state(rng) for block in self.blocks],
            norm=self.norm.init_state(rngs[1]),
        )


class T5DecoderBlock(BaseModule):
    def __init__(self, emb_size: int, n_heads: int, dropout_rate: float=0.1, use_rel_attn_bias: bool=False):
        super().__init__()
        self.self_attn_layer = T5SelfAttention(
            emb_size,
            n_heads,
            dropout=dropout_rate,
            use_rel_attn_bias=use_rel_attn_bias,
            bidirectional=False)

        self.cross_attn_layer = T5CrossAttention(emb_size, n_heads, dropout=dropout_rate)
        self.feed_forward = T5FeedForward(emb_size, 4 * emb_size)

    def forward(
        self,
        state: T5DecoderBlockState,
        xq: Array["batch_size, tgt_len, emb_size"],
        xkv: Array["batch_size, src_len, emb_size"], # encoder hidden states
        rng: Array,
        training = False,
        self_pos_bias: Array["1, n_heads, tgt_len, tgt_len"] = None, # TODO: Double check shapes
        cross_pos_bias: Array["1, n_heads, tgt_len, src_len"] = None,
        mask: Array["tgt_len, tgt_len"] = None,
        output_pos_bias = False,
    ) -> Array["batch_size, tgt_len, emb_size"]:

        rngs = random.split(rng, 3)

        self_attn_out = self.self_attn_layer(
            state.self_attn_layer,
            xq,
            rngs[0], 
            pos_bias=self_pos_bias,
            output_pos_bias=output_pos_bias,
            training=training,
            mask=mask)

        if output_pos_bias:
            self_attn_out, self_pos_bias = self_attn_out

        cross_attn_out = self.cross_attn_layer(
            state.cross_attn_layer,
            self_attn_out,
            xkv,
            rngs[1], 
            pos_bias=cross_pos_bias,
            output_pos_bias=output_pos_bias,
            training=training)

        if output_pos_bias:
            # output both self and cross pos bias
            cross_attn_out, cross_pos_bias = cross_attn_out
            assert isinstance(cross_attn_out, jnp.ndarray), f"Expected jnp.ndarray, got {type(cross_attn_out)}"
            ff_out = self.feed_forward(state.feed_forward, cross_attn_out, rngs[2], training=training)
            return ff_out, self_pos_bias, cross_pos_bias
        else:
            return self.feed_forward(state.feed_forward, cross_attn_out, rngs[2], training=training)

    def init_state(self, rng: Array) -> T5DecoderBlockState:
        rngs = random.split(rng, 4)
        return T5DecoderBlockState(
            self_attn_layer=self.self_attn_layer.init_state(rngs[0]),
            cross_attn_layer=self.cross_attn_layer.init_state(rngs[1]),
            feed_forward=self.feed_forward.init_state(rngs[2]),
        )



class T5CrossAttention(BaseModule):
    def __init__(self, emb_size: int, n_heads: int, dropout: float=0.1, use_rel_attn_bias: bool=False):
        super().__init__()
        self.attention = T5MultiHeadAttention(emb_size, n_heads, use_rel_attn_bias=use_rel_attn_bias, dropout=dropout)
        self.norm = RMSNorm(emb_size, eps=1e-6)
        self.dropout_rate = dropout

    def forward(
        self,
        state: T5AttentionLayerState,
        xq: Array["batch_size, tgt_len, emb_size"],
        xkv: Array["batch_size, src_len, emb_size"],
        rng: Array,
        pos_bias: Array["1, n_heads, tgt_len, src_len"] = None,
        output_pos_bias = False,
        training: bool=False,
    ) -> Array["batch_size, tgt_len, emb_size"]:

        xq_normed = self.norm(state.norm, xq)
        
        rngs = random.split(rng, 2)

        attn_out = self.attention(
            state.attention,
            xq_normed,
            xkv,
            xkv,
            rng=rngs[0],
            training=training,
            pos_bias=pos_bias,
            output_pos_bias=output_pos_bias
        )

        if output_pos_bias:
            attn_out, pos_bias = attn_out
            outputs = xq + dropout(attn_out, self.dropout_rate, rngs[1], training)
            return outputs, pos_bias
        else:
            return xq + dropout(attn_out, self.dropout_rate, rngs[1], training)

    def init_state(self, rng: Array) -> T5AttentionLayerState:
        rngs = random.split(rng, 2)
        return T5AttentionLayerState(
            attention=self.attention.init_state(rngs[0]),
            norm=self.norm.init_state(rngs[1]),
        )
        

class T5SelfAttention(BaseModule):
    def __init__(self, emb_size: int, n_heads: int, dropout: float=0.1, use_rel_attn_bias: bool=False, bidirectional: bool=True):
        super().__init__()
        self.attention = T5MultiHeadAttention(emb_size, n_heads, use_rel_attn_bias, dropout=dropout, bidirectional=bidirectional)
        self.norm = RMSNorm(emb_size, eps=1e-06)
        self.dropout_rate = dropout

    def forward(
        self,
        state: T5AttentionLayerState,
        x: Array["batch_size, context_len, emb_size"],
        rng: Array,
        training: bool=False,
        pos_bias: Array["1, n_heads, context_len, context_len"] = None,
        mask: Array["context_len, context_len"] = None,
        output_pos_bias: bool=False,
    ) -> Array["batch_size, context_len, emb_size"]:
        x_normed = self.norm(state.norm, x)

        rngs = random.split(rng, 2)

        attn_out = self.attention(
            state.attention,
            x_normed,
            x_normed,
            x_normed,
            rng=rngs[0],
            training=training,
            pos_bias=pos_bias,
            mask=mask,
            output_pos_bias=output_pos_bias
        )

        if output_pos_bias:
            attn_out, pos_bias = attn_out # unpack tuple
            outputs = x + dropout(attn_out, self.dropout_rate, rngs[1], training)
            return outputs, pos_bias
        else:
            return x + dropout(attn_out, self.dropout_rate, rngs[1], training)

    def init_state(self, rng: Array) -> T5AttentionLayerState:
        rngs = random.split(rng, 2)
        return T5AttentionLayerState(
            attention=self.attention.init_state(rngs[0]),
            norm=self.norm.init_state(rngs[1]),
        )
        


class T5MultiHeadAttention(BaseModule):
    def __init__(
        self, 
        emb_size: int,
        n_heads: int,
        use_rel_attn_bias: bool=False,
        rel_attn_n_buckets: int=32,
        rel_attn_max_distance: int=128,
        dropout: float=0.1,
        bidirectional: bool=True
    ):
        super().__init__()
        
        # TODO: Add masking

        self.n_heads = n_heads
        self.emb_size = emb_size

        assert emb_size % n_heads == 0, "emb_size must be divisible by n_heads"

        self.d_k = emb_size // n_heads # Hidden inner dim per head
        self.inner_dim = self.n_heads * self.d_k

        self.query_dense = Linear(emb_size, self.inner_dim, bias=False)
        self.key_dense = Linear(emb_size, self.inner_dim, bias=False)
        self.value_dense = Linear(emb_size, self.inner_dim, bias=False)
        self.out_dense = Linear(self.inner_dim, emb_size, bias=False)

        self.use_rel_attn_bias = use_rel_attn_bias
        self.rel_attn_n_buckets = rel_attn_n_buckets
        self.rel_attn_max_distance = rel_attn_max_distance
        self.dropout_rate = dropout

        if self.use_rel_attn_bias:
            self.pos_emb = Embedding(self.rel_attn_n_buckets, self.n_heads)

        self.bidirectional = bidirectional

        self.debug_states = dict()

    def get_kv(
        self,
        state: T5MultiHeadAttentionState,
        xk: Array["batch_size, src_len, emb_size"],
        xv: Array,
        use_cache: bool,
        kv_cache: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        if use_cache and kv_cache is not None:
            assert len(kv_cache) == 2, "kv_cache must be a tuple of length 2"
            cached_keys, cached_values = kv_cache
            
            next_k = xk[-1][None, ...]
            next_v = xv[-1][None, ...]

            key = jnp.concatenate(
                [cached_keys, self.apply_dense(state.key, next_k, self.key_dense)], axis=0
            )

            value = jnp.concatenate(
                [cached_values, self.apply_dense(state.value, next_v, self.value_dense)], axis=0
            )
        else:
            key = self.apply_dense(state.key, xk, self.key_dense)
            value = self.apply_dense(state.value, xv, self.value_dense)
            
        # (batch_size, n_heads, src_len, emb_size)
        return key, value

    def apply_dense(self, state: LinearState, x: Array, dense_fn: Linear) -> Array:
        # xq: (batch_size, context_len, emb_size)
        # dense_fn: Linear callable

        batch_size = x.shape[0]
        query: Array["batch_size, context_len, emb_size"] = dense_fn(state, x)
        query: Array["batch_size, n_heads, context_len, d_k"] = query.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        return query

    def forward(
        self,
        state: T5MultiHeadAttentionState,
        xq: Array["batch_size, tgt_len, emb_size"],
        xk: Array["batch_size, src_len, emb_size"],
        xv: Array["batch_size, src_len, emb_size"],
        rng: Array,
        mask: Array["tgt_len, src_len"] = None,
        use_cache: bool = False,
        kv_cache: tuple[Array, Array] = None,
        training = False,
        pos_bias: Array["1, n_heads, tgt_len, src_len"] = None,
        output_pos_bias = False,
    ) -> tuple[Array["batch_size, tgt_len, emb_size"], tuple[Array, Array]]:

        # ! Uses batch dimension first !

        # mask: -inf at pos where not to attend

        # Returns (attn_output, kv_cache) if use_cache, else attn_output
        # kv_cache: tuple[()]
        
        batch_size, tgt_len = xq.shape[:2]
        src_len = xk.shape[1]

        # TODO: Masking

        query: Array["batch_size, n_heads, tgt_len, d_k"] = self.apply_dense(state.query, xq, self.query_dense)

        assert query.shape == (
            batch_size,
            self.n_heads,
            tgt_len,
            self.d_k,
        ), f"Expected shape {(batch_size, self.n_heads, tgt_len, self.d_k)}, got {query.shape}"

        # (batch_size, n_heads, src_len, d_k)
        key, value = self.get_kv(state, xk, xv, use_cache, kv_cache)

        if use_cache: kv_cache = (key, value)

        self.debug_states["query"] = query.copy()
        self.debug_states["key"] = key.copy()
        self.debug_states["value"] = value.copy()

        scores: Array["batch_size, n_heads, tgt_len, src_len"] = jnp.matmul(query, key.transpose((0, 1, 3, 2)))

        self.debug_states["scores"] = scores.copy()

        assert scores.shape == (
            batch_size,
            self.n_heads,
            tgt_len,
            src_len
        ), f"Expected shape {(batch_size, self.n_heads, tgt_len, src_len)}, got {scores.shape}"

        # TODO: Is scaling done in reference impl?
        # scores = scores * (1 / jnp.sqrt(self.d_k))

        # TODO: Only keep values relevant with kv cache
        if pos_bias is None:
            pos_bias: Array["1, n_heads, tgt_len, src_len"] = self.compute_pos_bias(state.pos_emb, tgt_len, src_len)

        self.debug_states["pos_bias"] = pos_bias.copy()

        assert pos_bias.shape == (
            1,
            self.n_heads,
            tgt_len,
            src_len,
        ), f"Expected shape {(1, self.n_heads, tgt_len, src_len)}, got {pos_bias.shape}"
        
        # broadcast to (batch_size, n_heads, tgt_len, src_len)
        pos_bias_br = jnp.repeat(pos_bias, batch_size, axis=0)

        if mask is not None:
            # Create causal mask to add to pos bias
            assert mask.shape == (tgt_len, src_len), f"Expected attention mask shape {(tgt_len, src_len)}, got {mask.shape}" 

            # extend mask to shape (batch_size, n_heads, tgt_len, src_len) by repeating it
            extended_mask = jnp.tile(mask[None, None, ...], (batch_size, self.n_heads, 1, 1))
            pos_bias_br += extended_mask

        scores += pos_bias_br

        attn_weights = softmax(scores, dim=-1)

        self.debug_states["attn_weights"] = attn_weights.copy()

        attn_weights = dropout(attn_weights, self.dropout_rate, rng, training)

        self.debug_states["attn_weights_dropout"] = attn_weights.copy()

        attn: Array["batch_size, n_heads, tgt_len, d_k"] = jnp.matmul(attn_weights, value)
        attn: Array["batch_size, tgt_len, emb_size"] = attn.transpose((0, 2, 1, 3)).reshape(batch_size, tgt_len, self.inner_dim)

        self.debug_states["attn"] = attn.copy()

        assert attn.shape == (
            batch_size,
            tgt_len,
            self.inner_dim,
        ), f"Expected shape {(batch_size, tgt_len, self.inner_dim)}, got {attn.shape}"

        out: Array["tgt_len, batch_size, emb_size"] = self.out_dense(state.output, attn)

        self.debug_states["out"] = out.copy()

        # if we dont use cache or output pos bias, just output out
        if not use_cache and not output_pos_bias:
            return out
        
        out = (out,)
        if use_cache:
            out += (kv_cache,)
            
        if output_pos_bias:
            out += (pos_bias,)
            
        return out
       

    def init_state(self, rng: Array) -> T5MultiHeadAttentionState:
        rngs = random.split(rng, 5)
        return T5MultiHeadAttentionState(
            query=self.query_dense.init_state(rngs[0]),
            key=self.key_dense.init_state(rngs[1]),
            value=self.value_dense.init_state(rngs[2]),
            output=self.out_dense.init_state(rngs[3]),
            pos_emb=self.pos_emb.init_state(rngs[4]) if self.use_rel_attn_bias else None,
        )

    def compute_pos_bias(
        self, 
        pos_emb_state: EmbeddingState,
        query_len: int,
        key_len: int,
    ) -> Array["1, n_heads, query_len, key_len"]:

        if not self.use_rel_attn_bias:
            # return zero tensor
            return jnp.zeros((1, self.n_heads, query_len, key_len))

        # int64 in original
        context_pos = jnp.arange(query_len, dtype=jnp.int32)[:, None]
        memory_pos = jnp.arange(key_len, dtype=jnp.int32)[None, :]

        relative_pos: Array["q_len, k_len"] = memory_pos - context_pos

        relative_pos_bucket = self._calc_bucket(
            relative_pos,
            n_buckets=self.rel_attn_n_buckets,
            max_distance=self.rel_attn_max_distance,
            bidirectional=self.bidirectional
        )

        values = self.pos_emb(pos_emb_state, relative_pos_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]

        return values

    def _calc_bucket(
        self, 
        relative_pos: Array["q_len, k_len"], 
        n_buckets: int=32, 
        max_distance: int=128,
        bidirectional: bool=True
    ) -> Array["q_len, k_len"]:

        # for each index in relative_pos, return which bucket it corresponds to.
        # in range [0, n_buckets)

        rel_buckets = 0
        # assume bidirectional
        if bidirectional:
            n_buckets //= 2
            rel_buckets += (relative_pos > 0).astype(jnp.int32) * n_buckets
            relative_pos = jnp.abs(relative_pos)
        else:
            relative_pos = -jnp.minimum(relative_pos, jnp.zeros_like(relative_pos))

        max_exact = n_buckets // 2
        is_small = relative_pos < max_exact

        rel_pos_if_large = max_exact + (
            jnp.log(relative_pos.astype(jnp.float32) / max_exact)
            / jnp.log(max_distance / max_exact)
            * (n_buckets - max_exact)
        ).astype(jnp.int32)

        rel_pos_if_large = jnp.minimum(
            rel_pos_if_large, jnp.full_like(rel_pos_if_large, n_buckets - 1)
        )

        rel_buckets += jnp.where(is_small, relative_pos, rel_pos_if_large)
        return rel_buckets


class T5Dense(BaseModule):
    def __init__(self, n_in: int, d_ff: int, dropout: float=0.1, act: Callable=relu):
        super().__init__()
        self.wi = Linear(n_in, d_ff, bias=False)
        self.wo = Linear(d_ff, n_in, bias=False)
        self.dropout = dropout
        self.act = act

    def forward(self, state: T5DenseState, x: Array, rng: Array, training=True) -> Array:
        x = self.wi(state.wi, x)
        x = self.act(x)
        x = dropout(x, self.dropout, rng, training)
        x = self.wo(state.wo, x)
        return x

    def init_state(self, rng: Array) -> T5DenseState:
        rng1, rng2 = random.split(rng, 2)
        return T5DenseState(
            wi=self.wi.init_state(rng1),
            wo=self.wo.init_state(rng2),
        )


class T5FeedForward(BaseModule):
    def __init__(self, n_in: int, d_ff: int, dropout: float=0.1):
        super().__init__()
        self.dense = T5Dense(n_in, d_ff, dropout)
        # T5LayerFF uses custom LayerNorm which is equivalent to RMSNorm
        self.norm = RMSNorm(n_in, eps=1e-6)
        self.dropout = dropout

    def forward(self, state: T5FeedForwardState, x: Array, rng: Array, training=True) -> Array:
        rng1, rng2 = random.split(rng, 2)

        z = self.norm(state.norm, x)
        z = self.dense(state.dense, z, rng1, training)
        x += dropout(z, self.dropout, rng2, training)
        return x

    def init_state(self, rng: Array) -> T5FeedForwardState:
        rng1, rng2 = random.split(rng, 2)
        return T5FeedForwardState(
            norm=self.norm.init_state(rng1),
            dense=self.dense.init_state(rng2),
        )
