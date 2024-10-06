import jax
import json

from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2MLP, GPT2Attention
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from huggingface_hub import hf_hub_download

import torch.nn as nn
import torch
import jax.numpy as jnp
import pytest
from pytest import fixture

from models.gpt2 import GPT2, GPT2Block, GPT2Dense
from models.gpt2 import GPT2Attention as JaxGPT2Attention
from attention import LayerNorm
from states import to_jax_state
from tests.test_t5 import BATCH_SIZE, EMBED_SIZE
from utils import count_params, torch_count_params

from log_utils import logger


MODEL_NAME = "gpt2"
BATCH_SIZE = 2
SEQ_LEN = 8
EMBED_SIZE = 10

test_shape = (BATCH_SIZE, SEQ_LEN, EMBED_SIZE)

torch.random.manual_seed(0)


GPT2_REPO = "openai-community/gpt2"

@fixture
def gpt2_config() -> GPT2Config:

    CONFIG_NAME = "config.json"

    t5_config_path = hf_hub_download(repo_id=GPT2_REPO, filename=CONFIG_NAME)

    try:
        with open(t5_config_path, "r") as f:
            t5_config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading t5 config: {e}")
        raise e

    return GPT2Config.from_dict(t5_config)

def test_load_gpt2(gpt2_config: GPT2Config):
    torch_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    torch_n_params = torch_count_params(torch_model)

    jax_state = to_jax_state(torch_model)
    jax_n_params = count_params(jax_state)

    assert torch_n_params == jax_n_params
    

# Redundant, already running in test_compare.py:204
@pytest.mark.skip
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

    assert torch_out.shape == jax_out.shape, f"torch_out.shape = {torch_out.shape}, jax_out.shape = {jax_out.shape}"
    assert jnp.allclose(jax_out, torch_out.numpy(), atol=1e-5), f"jax_out = {jax_out}, torch_out = {torch_out}"


def test_gpt2_attention(gpt2_config: GPT2Config):
    x = torch.randn((BATCH_SIZE, SEQ_LEN, gpt2_config.n_embd), requires_grad=False)
    x_jax = jnp.array(x)

    gpt2_attention = GPT2Attention(gpt2_config).eval()

    with torch.no_grad():
        torch_out, _key_values = gpt2_attention(x)

    # Jax
    jax_gpt2_attention = JaxGPT2Attention(emb_size=gpt2_config.n_embd)
    states = to_jax_state(gpt2_attention)

    rng = jax.random.PRNGKey(0)
    jax_out = jax_gpt2_attention(states, x_jax, rng)

    # import code; code.interact(local=locals())

    assert torch_out.shape == jax_out.shape, f"torch_out.shape = {torch_out.shape}, jax_out.shape = {jax_out.shape}"
    assert jnp.allclose(jax_out, torch_out.numpy(), atol=1e-5), f"jax_out = {jax_out}, torch_out = {torch_out}"


def test_gpt2_block():
    pass 


def test_gpt2_dense(gpt2_config: GPT2Config):
    x = torch.randn((1, 768), requires_grad=False) * 10
    x_jax = jnp.array(x)

    gpt2_dense = GPT2MLP(4 * EMBED_SIZE, gpt2_config).eval()

    with torch.no_grad():
        torch_out = gpt2_dense(x)

    # Jax
    rng = jax.random.PRNGKey(0)

    jax_state = to_jax_state(gpt2_dense)    
    jax_gpt2_dense = GPT2Dense(4 * EMBED_SIZE)

    jax_out = jax_gpt2_dense(jax_state, x_jax, rng)

    assert torch_out.shape == jax_out.shape, f"torch_out.shape = {torch_out.shape}, jax_out.shape = {jax_out.shape}"
    assert jnp.allclose(jax_out, torch_out.numpy(), atol=1e-5), f"jax_out = {jax_out}, torch_out = {torch_out}"