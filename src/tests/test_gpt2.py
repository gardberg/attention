import jax
import json

from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP as TorchGPT2MLP
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as TorchGPT2Attention
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as TorchGPT2Block
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as TorchGPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from huggingface_hub import hf_hub_download

import torch.nn as nn
import torch
import jax.numpy as jnp
import pytest
from pytest import fixture
from typing import Tuple, Union

from models.gpt2 import GPT2BaseModel, GPT2Block, GPT2Dense
from models.gpt2 import GPT2Attention as JaxGPT2Attention
from attention import LayerNorm
from states import to_jax_state
from utils import count_params, torch_count_params
from base import Array

from log_utils import logger


def get_types_and_shapes(data: Union[Tuple, ...], array_type: Union[Array, torch.Tensor]):
    if isinstance(data, tuple):
        types = tuple(type(d) for d in data)
        shapes = tuple(d.shape if isinstance(d, array_type) else None for d in data)
    else:
        types = (type(data),)
        shapes = (data.shape if isinstance(data, array_type) else None,)
    return types, shapes


def torch_debug_hook(module, input, output):
    input_types, input_shapes = get_types_and_shapes(input, torch.Tensor)
    output_types, output_shapes = get_types_and_shapes(output, torch.Tensor)

    logger.debug(
        f"For module: {module.__class__.__name__} got input: {input_types}, output: {output_types}"
    )

    logger.debug(
        f"Called {module.__class__.__name__} with shapes: {input_shapes} -> {output_shapes}"
    )


def jax_debug_hook(module, input, output):
    input_types, input_shapes = get_types_and_shapes(input, Array)
    output_types, output_shapes = get_types_and_shapes(output, Array)

    logger.debug(
        f"For module: {module.__class__.__name__} got input: {input_types}, output: {output_types}"
    )
    logger.debug(
        f"Called {module.__class__.__name__}: {input_shapes} -> {output_shapes}"
    )

MODEL_NAME = "gpt2"
BATCH_SIZE = 2
SEQ_LEN = 8
EMBED_SIZE = 768

test_shape = (BATCH_SIZE, SEQ_LEN, EMBED_SIZE)

GPT2_REPO = "openai-community/gpt2"


@fixture(scope="function")
def gpt2_config() -> GPT2Config:
    CONFIG_NAME = "config.json"

    gpt2_config_path = hf_hub_download(repo_id=GPT2_REPO, filename=CONFIG_NAME)

    try:
        with open(gpt2_config_path, "r") as f:
            gpt2_config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading t5 config: {e}")
        raise e

    return GPT2Config.from_dict(gpt2_config)


@pytest.mark.skip
def test_load_gpt2(gpt2_config: GPT2Config):
    torch_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    torch_n_params = torch_count_params(torch_model)

    jax_state = to_jax_state(torch_model)
    jax_n_params = count_params(jax_state)

    assert torch_n_params == jax_n_params


# Redundant, already running in test_compare.py:204
def test_gpt2_norm(gpt2_config: GPT2Config):
    x = torch.randn(test_shape, requires_grad=False) * 10
    x_jax = jnp.array(x)

    torch_norm = nn.LayerNorm((EMBED_SIZE,), eps=1e-5)

    with torch.no_grad():
        torch_out = torch_norm(x)

    # Jax
    jax_state = to_jax_state(torch_norm)

    jax_norm = LayerNorm((EMBED_SIZE,))

    jax_out = jax_norm(jax_state, x_jax)

    assert (
        torch_out.shape == jax_out.shape
    ), f"torch_out.shape = {torch_out.shape}, jax_out.shape = {jax_out.shape}"
    assert jnp.allclose(
        jax_out, torch_out.numpy(), atol=1e-5
    ), f"jax_out = {jax_out}, torch_out = {torch_out}"


def test_gpt2_attention(gpt2_config: GPT2Config):
    rng = jax.random.PRNGKey(0)

    x = torch.randn((BATCH_SIZE, SEQ_LEN, gpt2_config.n_embd), requires_grad=False)
    x_jax = jnp.array(x)

    gpt2_attention = TorchGPT2Attention(gpt2_config).eval()

    with torch.no_grad():
        torch_out, _key_values = gpt2_attention(x)

    # Jax
    jax_gpt2_attention = JaxGPT2Attention(emb_size=gpt2_config.n_embd)
    states = to_jax_state(gpt2_attention)

    jax_out = jax_gpt2_attention(states, x_jax, rng)

    # import code; code.interact(local=locals())

    assert (
        torch_out.shape == jax_out.shape
    ), f"torch_out.shape = {torch_out.shape}, jax_out.shape = {jax_out.shape}"
    assert jnp.allclose(
        jax_out, torch_out.numpy(), atol=1e-5
    ), f"jax_out = {jax_out}, torch_out = {torch_out}"


def test_gpt2_block():
    pass


def test_gpt2_dense(gpt2_config: GPT2Config):
    # forward debug hook

    x = torch.randn((1, EMBED_SIZE), requires_grad=False) * 10
    x_jax = jnp.array(x)

    gpt2_dense = TorchGPT2MLP(4 * EMBED_SIZE, gpt2_config).eval()

    gpt2_dense.register_forward_hook(torch_debug_hook)

    with torch.no_grad():
        torch_out = gpt2_dense(x)

    # Jax
    rng = jax.random.PRNGKey(0)

    jax_state = to_jax_state(gpt2_dense)
    jax_gpt2_dense = GPT2Dense(4 * EMBED_SIZE)

    jax_gpt2_dense.register_forward_hook(jax_debug_hook)

    jax_out = jax_gpt2_dense(jax_state, x_jax, rng)

    assert (
        torch_out.shape == jax_out.shape
    ), f"torch_out.shape = {torch_out.shape}, jax_out.shape = {jax_out.shape}"
    assert jnp.allclose(
        jax_out, torch_out.numpy(), atol=1e-5
    ), f"jax_out = {jax_out}, torch_out = {torch_out}"


def test_gpt2_block(gpt2_config: GPT2Config):
    rng = jax.random.PRNGKey(0)

    x = torch.randn((BATCH_SIZE, SEQ_LEN, EMBED_SIZE), requires_grad=False)
    x_jax = jnp.array(x)

    gpt2_block = TorchGPT2Block(gpt2_config).eval()

    gpt2_block.register_forward_hook(torch_debug_hook)

    with torch.no_grad():
        torch_out = gpt2_block(x)[0]

    # Jax
    jax_gpt2_block = GPT2Block(emb_size=gpt2_config.n_embd)
    jax_state = to_jax_state(gpt2_block)

    jax_gpt2_block.register_forward_hook(jax_debug_hook)

    jax_out = jax_gpt2_block(jax_state, x_jax, rng)

    assert (
        torch_out.shape == jax_out.shape
    ), f"torch_out.shape = {torch_out.shape}, jax_out.shape = {jax_out.shape}"
    assert jnp.allclose(
        jax_out, torch_out.numpy(), atol=1e-5
    ), f"jax_out = {jax_out}, torch_out = {torch_out}"


def test_gpt2_base_model(gpt2_config: GPT2Config):
    rng = jax.random.PRNGKey(0)

    input_ids = torch.randint(0, gpt2_config.vocab_size, (BATCH_SIZE, SEQ_LEN))
    jax_input_ids = jnp.array(input_ids, dtype=jnp.int64)
   
    torch_gpt_base_model = TorchGPT2Model(gpt2_config).eval()

    torch_gpt_base_model.register_forward_hook(torch_debug_hook)

    with torch.no_grad():
        torch_out = torch_gpt_base_model(input_ids)[0]

    # Jax
    jax_gpt_base_model = GPT2BaseModel(vocab_size=gpt2_config.vocab_size, emb_size=gpt2_config.n_embd)
    jax_state = to_jax_state(torch_gpt_base_model)

    jax_gpt_base_model.register_forward_hook(jax_debug_hook)

    jax_out = jax_gpt_base_model(jax_state, jax_input_ids, rng)

    assert (
        torch_out.shape == jax_out.shape
    ), f"torch_out.shape = {torch_out.shape}, jax_out.shape = {jax_out.shape}"
    assert jnp.allclose(
        jax_out, torch_out.numpy(), atol=1e-5
    ), f"jax_out = {jax_out}, torch_out = {torch_out}"
