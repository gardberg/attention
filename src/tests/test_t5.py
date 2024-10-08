import os
import json
import copy

import torch
import jax.numpy as jnp
from transformers.models.t5.modeling_t5 import (
    T5DenseActDense,
    T5LayerFF,
    T5Attention,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5Block,
    T5Stack,
)
from transformers.models.t5.modeling_t5 import T5Model as TorchT5Model
from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration as TorchT5ForConditionalGeneration,
)
from transformers.models.t5 import T5Tokenizer
from pytest import fixture, mark
from huggingface_hub import hf_hub_download
import jax

from utils import ROOT_DIR, count_params, torch_count_params
from testing_utils import TOL
from transformers.models.t5 import T5Config
from models.t5 import (
    T5Dense,
    T5FeedForward,
    T5MultiHeadAttention,
    T5SelfAttention,
    T5CrossAttention,
    T5EncoderBlock,
    T5DecoderBlock,
    T5Encoder,
    T5Decoder,
    T5BaseModel,
    T5Model,
)
from states import to_jax_state
from log_utils import logger

T5_REPO = "google-t5/t5-small"


@fixture(scope="function")
def t5_config() -> T5Config:
    CONFIG_NAME = "config.json"

    t5_config_path = hf_hub_download(repo_id=T5_REPO, filename=CONFIG_NAME)

    try:
        with open(t5_config_path, "r") as f:
            t5_config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading t5 config: {e}")
        raise e

    return T5Config.from_dict(t5_config)


torch.random.manual_seed(0)
BATCH_SIZE = 2
SEQ_LEN = 3
EMBED_SIZE = 512
D_FF = 2048

test_shape = (BATCH_SIZE, SEQ_LEN, EMBED_SIZE)

RNG = jax.random.PRNGKey(0)


def test_t5_dense(t5_config: T5Config):
    x = torch.randn(test_shape)
    x_jax = jnp.array(x)

    # torch
    torch_t5_dense = T5DenseActDense(t5_config).eval()  # disables dropout

    with torch.no_grad():
        y_torch = torch_t5_dense(x)

    # jax
    jax_state = to_jax_state(torch_t5_dense)
    jax_t5_dense = T5Dense(EMBED_SIZE, D_FF)

    y_jax = jax_t5_dense(jax_state, x_jax, RNG, training=False)

    assert y_torch.shape == y_jax.shape
    assert jnp.allclose(y_torch.numpy(), y_jax, atol=1e-4)


def test_t5_ff(t5_config: T5Config):
    x = torch.randn(test_shape)
    x_jax = jnp.array(x)

    # torch
    torch_t5_ff = T5LayerFF(t5_config).eval()

    with torch.no_grad():
        y_torch = torch_t5_ff(x)

    # jax
    jax_state = to_jax_state(torch_t5_ff)
    jax_t5_ff = T5FeedForward(EMBED_SIZE, D_FF)

    y_jax = jax_t5_ff(jax_state, x_jax, RNG, training=False)

    assert y_torch.shape == y_jax.shape
    logger.debug(f"y_torch: {y_torch}, y_jax: {y_jax}")
    assert jnp.allclose(y_torch.numpy(), y_jax, atol=TOL)


@mark.parametrize("q_len, k_len, bidirectional", [(2, 3, True), (4, 6, False)])
def test_t5_mha_relative_pos(
    t5_config: T5Config, q_len: int, k_len: int, bidirectional
):
    REL_ATN_BIAS = True

    t5_config = copy.deepcopy(t5_config)
    t5_config.is_decoder = not bidirectional
    # torch
    torch_t5_mha = T5Attention(t5_config, REL_ATN_BIAS).eval()
    with torch.no_grad():
        torch_pos_bias = torch_t5_mha.compute_bias(q_len, k_len).numpy()

    # jax
    jax_t5_mha = T5MultiHeadAttention(
        emb_size=t5_config.d_model,
        n_heads=t5_config.num_heads,
        use_rel_attn_bias=REL_ATN_BIAS,
        bidirectional=bidirectional,
    )
    pos_emb_state = to_jax_state(torch_t5_mha.relative_attention_bias)

    jax_pos_bias = jax_t5_mha.compute_pos_bias(pos_emb_state, q_len, k_len)

    assert jnp.allclose(torch_pos_bias, jax_pos_bias, atol=TOL)


@mark.parametrize("tgt_len, rel_pos", [(2, False), (3, True)])
def test_t5_self_mha(t5_config, tgt_len, rel_pos):
    xq = torch.randn((BATCH_SIZE, tgt_len, EMBED_SIZE))

    # Torch
    torch_t5_mha = T5Attention(t5_config, has_relative_attention_bias=rel_pos).eval()
    with torch.no_grad():
        attn_out, kv_state, pos_bias = torch_t5_mha(xq)

    # Jax
    rng = jax.random.PRNGKey(0)
    xq_jax = jnp.array(xq.detach().numpy())

    t5_jax = T5MultiHeadAttention(
        emb_size=t5_config.d_model,
        n_heads=t5_config.num_heads,
        use_rel_attn_bias=rel_pos,
    )

    state = to_jax_state(torch_t5_mha)

    attn_out_jax = t5_jax(state, xq_jax, xq_jax, xq_jax, rng, training=False)

    assert attn_out.shape == attn_out_jax.shape

    assert jnp.allclose(attn_out.numpy(), attn_out_jax, atol=TOL)


@mark.parametrize("tgt_len, src_len", [(2, 3), (3, 2)])
def test_t5_cross_mha(t5_config, tgt_len, src_len):
    xq = torch.randn((BATCH_SIZE, tgt_len, EMBED_SIZE))
    xkv = torch.randn((BATCH_SIZE, src_len, EMBED_SIZE))

    torch_t5_mha = T5Attention(t5_config, has_relative_attention_bias=False).eval()
    with torch.no_grad():
        attn_out, kv_state, pos_bias = torch_t5_mha(xq, key_value_states=xkv)

    # Jax
    rng = jax.random.PRNGKey(0)
    xq_jax = jnp.array(xq.detach().numpy())
    xkv_jax = jnp.array(xkv.detach().numpy())

    t5_jax = T5MultiHeadAttention(
        emb_size=t5_config.d_model, n_heads=t5_config.num_heads, use_rel_attn_bias=False
    )

    state = to_jax_state(torch_t5_mha)

    attn_out_jax = t5_jax(state, xq_jax, xkv_jax, xkv_jax, rng, training=False)

    assert attn_out.shape == attn_out_jax.shape

    assert jnp.allclose(attn_out.numpy(), attn_out_jax, atol=TOL)


@mark.parametrize("tgt_len, rel_attn", [(2, False), (2, True), (3, False), (3, True)])
def test_t5_self_attn(t5_config, tgt_len, rel_attn):
    xq = torch.randn((BATCH_SIZE, tgt_len, EMBED_SIZE))

    torch_t5_self_attn = T5LayerSelfAttention(
        t5_config, has_relative_attention_bias=rel_attn
    ).eval()
    with torch.no_grad():
        attn_out, kv_state, pos_bias = torch_t5_self_attn(xq)

    # Jax
    rng = jax.random.PRNGKey(0)
    xq_jax = jnp.array(xq.detach().numpy())

    jax_t5_self_attn = T5SelfAttention(
        emb_size=t5_config.d_model,
        n_heads=t5_config.num_heads,
        use_rel_attn_bias=rel_attn,
    )

    state = to_jax_state(torch_t5_self_attn)

    jax_attn_out = jax_t5_self_attn(state, xq_jax, rng, training=False)

    assert attn_out.shape == jax_attn_out.shape
    assert jnp.allclose(attn_out.numpy(), jax_attn_out, atol=TOL)


@mark.parametrize("tgt_len, src_len", [(2, 2), (3, 2)])
def test_t5_cross_attn(t5_config, tgt_len, src_len):
    xq = torch.randn((BATCH_SIZE, tgt_len, EMBED_SIZE))
    xkv = torch.randn((BATCH_SIZE, src_len, EMBED_SIZE))

    torch_t5_cross_attn = T5LayerCrossAttention(t5_config).eval()
    with torch.no_grad():
        attn_out, kv_state, pos_bias = torch_t5_cross_attn(xq, key_value_states=xkv)

    # Jax
    rng = jax.random.PRNGKey(0)
    xq_jax = jnp.array(xq.detach().numpy())
    xkv_jax = jnp.array(xkv.detach().numpy())

    jax_t5_cross_attn = T5CrossAttention(
        emb_size=t5_config.d_model,
        n_heads=t5_config.num_heads,
    )

    state = to_jax_state(torch_t5_cross_attn)

    jax_attn_out = jax_t5_cross_attn(state, xq_jax, xkv_jax, rng, training=False)

    assert attn_out.shape == jax_attn_out.shape
    assert jnp.allclose(attn_out.numpy(), jax_attn_out, atol=TOL)


@mark.parametrize("use_rel_attn_bias", [True, False])
def test_t5_encoder_block(t5_config, use_rel_attn_bias):
    t5_config = copy.deepcopy(t5_config)
    t5_config.is_decoder = False
    t5_config.use_cache = False
    t5_config.is_encoder_decoder = False

    xq = torch.randn((BATCH_SIZE, SEQ_LEN, EMBED_SIZE))

    torch_t5_block = T5Block(
        t5_config, has_relative_attention_bias=use_rel_attn_bias
    ).eval()
    with torch.no_grad():
        torch_out, kv_states = torch_t5_block(xq)

    # Jax
    rng = jax.random.PRNGKey(0)
    xq_jax = jnp.array(xq.detach().numpy())

    jax_t5_block = T5EncoderBlock(
        emb_size=t5_config.d_model,
        n_heads=t5_config.num_heads,
        use_rel_attn_bias=use_rel_attn_bias,
    )

    state = to_jax_state(torch_t5_block)

    jax_out = jax_t5_block(state, xq_jax, rng, training=False)

    assert torch_out.shape == jax_out.shape
    assert jnp.allclose(torch_out.numpy(), jax_out, atol=TOL)


@mark.parametrize("use_rel_attn_bias", [True, False])
def test_t5_decoder_block(t5_config, use_rel_attn_bias):
    t5_config = copy.deepcopy(t5_config)
    t5_config.is_decoder = True
    t5_config.is_encoder_decoder = False
    t5_config.num_layers = t5_config.num_decoder_layers

    xq = torch.randn((BATCH_SIZE, SEQ_LEN, EMBED_SIZE), dtype=torch.float32)
    xkv = torch.randn((BATCH_SIZE, SEQ_LEN + 1, EMBED_SIZE), dtype=torch.float32)

    torch_t5_block = T5Block(
        t5_config, has_relative_attention_bias=use_rel_attn_bias
    ).eval()
    with torch.no_grad():
        torch_out, kv_states, pos_bias = torch_t5_block(xq, encoder_hidden_states=xkv)

    # import code; code.interact(local=locals())
    # pos_bias: (1, n_heads, tgt_len, src_len)

    # Jax
    rng = jax.random.PRNGKey(0)
    xq_jax = jnp.array(xq.detach().numpy())
    xkv_jax = jnp.array(xkv.detach().numpy())

    jax_t5_block = T5DecoderBlock(
        emb_size=t5_config.d_model,
        n_heads=t5_config.num_heads,
        use_rel_attn_bias=use_rel_attn_bias,
    )

    state = to_jax_state(torch_t5_block)

    jax_out, self_pos_bias, cross_pos_bias = jax_t5_block(
        state, xq_jax, xkv_jax, rng, training=False, output_pos_bias=True
    )

    assert torch_out.shape == jax_out.shape

    assert jnp.allclose(pos_bias.numpy(), cross_pos_bias, atol=TOL)
    assert jnp.allclose(torch_out.numpy(), jax_out, atol=TOL)


def test_t5_encoder(t5_config):
    t5_config = copy.deepcopy(t5_config)
    t5_config.is_decoder = False
    t5_config.use_cache = False
    t5_config.is_encoder_decoder = False
    t5_config.output_attentions = False

    input_ids = torch.randint(0, t5_config.vocab_size, (BATCH_SIZE, SEQ_LEN))

    shared_emb = torch.nn.Embedding(t5_config.vocab_size, t5_config.d_model)

    torch_t5_encoder = T5Stack(t5_config, embed_tokens=shared_emb).eval()
    with torch.no_grad():
        torch_out = torch_t5_encoder(input_ids)

    torch_out = torch_out.last_hidden_state

    # Jax
    rng = jax.random.PRNGKey(0)
    input_ids_jax = jnp.array(input_ids.detach().numpy())

    jax_t5_encoder = T5Encoder(
        emb_size=t5_config.d_model,
        n_layers=t5_config.num_layers,
        vocab_size=t5_config.vocab_size,
    )

    state = to_jax_state(torch_t5_encoder)

    # (batch_size, seq_len, emb_size)
    jax_out = jax_t5_encoder(state, input_ids_jax, rng, training=False)
    # import code; code.interact(local=locals())

    # count params
    assert torch_count_params(torch_t5_encoder) == count_params(
        state
    ), f"{torch_count_params(torch_t5_encoder)} != {count_params(state)}"

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    # print(f"{torch_out.numpy()[0,0,:5]}\n{jax_out[0,0,:5]}")
    assert jnp.allclose(torch_out.numpy(), jax_out, atol=TOL)


def test_t5_decoder(t5_config):
    t5_config = copy.deepcopy(t5_config)
    t5_config.is_decoder = True
    t5_config.is_encoder_decoder = False
    t5_config.use_cache = False
    t5_config.output_attentions = False
    t5_config.num_layers = t5_config.num_decoder_layers

    input_ids = torch.randint(0, t5_config.vocab_size, (BATCH_SIZE, SEQ_LEN))
    encoder_hidden_states = torch.randn((BATCH_SIZE, SEQ_LEN + 1, EMBED_SIZE))

    shared_emb = torch.nn.Embedding(t5_config.vocab_size, t5_config.d_model)

    torch_t5_decoder = T5Stack(t5_config, embed_tokens=shared_emb).eval()
    with torch.no_grad():
        torch_out = torch_t5_decoder(
            input_ids, encoder_hidden_states=encoder_hidden_states
        )

    torch_out = torch_out.last_hidden_state

    # Jax
    rng = jax.random.PRNGKey(0)
    input_ids_jax = jnp.array(input_ids.detach().numpy())
    encoder_hidden_states_jax = jnp.array(encoder_hidden_states.detach().numpy())

    jax_t5_decoder = T5Decoder(
        emb_size=t5_config.d_model,
        n_layers=t5_config.num_layers,
        vocab_size=t5_config.vocab_size,
    )

    state = to_jax_state(torch_t5_decoder)

    assert torch_count_params(torch_t5_decoder) == count_params(
        state
    ), f"{torch_count_params(torch_t5_decoder)} != {count_params(state)}"

    # import code; code.interact(local=locals())
    jax_out = jax_t5_decoder(state, input_ids_jax, encoder_hidden_states_jax, rng)

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"

    print(f"{torch_out.numpy()[0,0,:5]}\n{jax_out[0,0,:5]}")
    assert jnp.allclose(torch_out.numpy(), jax_out, atol=TOL)


@mark.parametrize("use_pretrained", [True, False])
def test_t5_base_model(t5_config, use_pretrained):
    input_ids = torch.randint(0, t5_config.vocab_size, (BATCH_SIZE, SEQ_LEN))
    decoder_input_ids = torch.randint(0, t5_config.vocab_size, (BATCH_SIZE, SEQ_LEN))

    shared_emb = torch.nn.Embedding(t5_config.vocab_size, t5_config.d_model)

    if use_pretrained:
        torch_t5_model = TorchT5Model.from_pretrained(T5_REPO)
    else:
        torch_t5_model = TorchT5Model(t5_config).eval()
        torch_t5_model.set_input_embeddings(shared_emb)

    with torch.no_grad():
        torch_out = torch_t5_model(input_ids, decoder_input_ids=decoder_input_ids)

    torch_out = torch_out.last_hidden_state

    # Jax
    rng = jax.random.PRNGKey(0)
    input_ids_jax = jnp.array(input_ids.detach().numpy())
    decoder_input_ids_jax = jnp.array(decoder_input_ids.detach().numpy())

    jax_t5_model = T5BaseModel(
        emb_size=t5_config.d_model,
        n_layers=t5_config.num_layers,
        vocab_size=t5_config.vocab_size,
    )

    state = to_jax_state(torch_t5_model)

    nbr_torch_params = torch_count_params(torch_t5_model)
    nbr_jax_params = count_params(state)

    # We're counting embedding params twice, once for encoder, once for decoder
    nbr_emb_params = state.encoder.embedding.embeddings.size
    nbr_expected_params = nbr_jax_params - nbr_emb_params

    assert (
        nbr_torch_params == nbr_expected_params
    ), f"{nbr_torch_params} != {nbr_expected_params}"

    jax_out, _enc_out = jax_t5_model(state, input_ids_jax, decoder_input_ids_jax, rng)

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    print(f"{torch_out.numpy()[0,0,:5]}\n{jax_out[0,0,:5]}")
    assert jnp.allclose(torch_out.numpy(), jax_out, atol=TOL)


@mark.parametrize("use_pretrained", [True, False])
def test_t5_base_model_encoder(t5_config, use_pretrained):
    # Using pre-computed encoder output
    input_ids = torch.randint(0, t5_config.vocab_size, (BATCH_SIZE, SEQ_LEN))
    decoder_input_ids = torch.randint(0, t5_config.vocab_size, (BATCH_SIZE, SEQ_LEN))
    encoder_hidden_states = torch.randn((BATCH_SIZE, SEQ_LEN, EMBED_SIZE))

    if use_pretrained:
        torch_t5_model = TorchT5Model.from_pretrained(T5_REPO)
    else:
        shared_emb = torch.nn.Embedding(t5_config.vocab_size, t5_config.d_model)
        torch_t5_model = TorchT5Model(t5_config).eval()
        torch_t5_model.set_input_embeddings(shared_emb)

    with torch.no_grad():
        torch_out = torch_t5_model(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_hidden_states.unsqueeze(0),
        )  # add weird dim to make work with HF output dict wrapping

    torch_out = torch_out.last_hidden_state

    # Jax
    rng = jax.random.PRNGKey(42)
    input_ids_jax = jnp.array(input_ids.detach().numpy())
    decoder_input_ids_jax = jnp.array(decoder_input_ids.detach().numpy())
    encoder_hidden_states_jax = jnp.array(encoder_hidden_states.detach().numpy())

    jax_t5_model = T5BaseModel(
        emb_size=t5_config.d_model,
        n_layers=t5_config.num_layers,
        vocab_size=t5_config.vocab_size,
    )

    state = to_jax_state(torch_t5_model)

    jax_out, _enc_out = jax_t5_model(
        state,
        input_ids_jax,
        decoder_input_ids_jax,
        rng,
        encoder_output=encoder_hidden_states_jax,
    )

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    assert jnp.allclose(torch_out.numpy(), jax_out, atol=TOL)


@mark.parametrize("use_pretrained", [True, False])
def test_t5_model(t5_config, use_pretrained):
    input_ids = torch.randint(0, t5_config.vocab_size, (BATCH_SIZE, SEQ_LEN))
    decoder_input_ids = torch.randint(0, t5_config.vocab_size, (BATCH_SIZE, SEQ_LEN))
    # decoder_input_ids = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long)

    if use_pretrained:
        torch_t5_model = TorchT5ForConditionalGeneration.from_pretrained(T5_REPO)
    else:
        shared_emb = torch.nn.Embedding(t5_config.vocab_size, t5_config.d_model)
        torch_t5_model = TorchT5ForConditionalGeneration(t5_config).eval()
        torch_t5_model.set_input_embeddings(shared_emb)

    with torch.no_grad():
        torch_out = torch_t5_model(input_ids, decoder_input_ids=decoder_input_ids)

    torch_out = torch_out.logits

    # Jax
    rng = jax.random.PRNGKey(0)
    input_ids_jax = jnp.array(input_ids.detach().numpy())
    decoder_input_ids_jax = jnp.array(decoder_input_ids.detach().numpy())

    jax_t5_model = T5Model(vocab_size=t5_config.vocab_size, emb_size=t5_config.d_model)

    state = to_jax_state(torch_t5_model)

    jax_out, _enc_out = jax_t5_model(state, input_ids_jax, decoder_input_ids_jax, rng)

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"

    assert jnp.allclose(torch_out.numpy(), jax_out, atol=TOL)


@mark.parametrize("use_pretrained", [True, False])
def test_t5_generation(t5_config, use_pretrained):
    tokenizer = T5Tokenizer.from_pretrained(T5_REPO)

    TEST_SENTENCE = "translate English to French: Hello, how are you?"

    input_ids = tokenizer(TEST_SENTENCE, return_tensors="pt").input_ids

    # Torch
    if use_pretrained:
        torch_t5_model = TorchT5ForConditionalGeneration.from_pretrained(T5_REPO).eval()
    else:
        shared_emb = torch.nn.Embedding(t5_config.vocab_size, t5_config.d_model)
        torch_t5_model = TorchT5ForConditionalGeneration(t5_config).eval()
        torch_t5_model.set_input_embeddings(shared_emb)

    with torch.no_grad():
        torch_out = torch_t5_model.generate(input_ids, max_length=20)

    torch_out_decoded = tokenizer.decode(torch_out[0], skip_special_tokens=False)
    logger.debug(f"use_pretrained: {use_pretrained}, torch: {torch_out_decoded}")

    # Jax
    jax_t5_model = T5Model(vocab_size=t5_config.vocab_size, emb_size=t5_config.d_model)

    state = to_jax_state(torch_t5_model)

    input_ids_jnp = jnp.array(input_ids.detach().numpy())
    rng = jax.random.PRNGKey(0)

    jax_out = jax_t5_model.generate(state, input_ids_jnp, rng, max_length=20)

    jax_out_decoded = tokenizer.decode(jax_out[0], skip_special_tokens=False)
    logger.debug(f"use_pretrained: {use_pretrained}: jax: {jax_out_decoded}")

    assert (
        torch_out_decoded == jax_out_decoded
    ), f"{torch_out_decoded} != {jax_out_decoded}"
