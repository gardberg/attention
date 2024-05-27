import os
import json

import torch
import jax.numpy as jnp
from transformers.models.t5.modeling_t5 import T5DenseActDense, T5LayerFF, T5Attention
from pytest import fixture, mark
from huggingface_hub import hf_hub_download
import jax

from utils import ROOT_DIR
from testing_utils import TOL
from transformers.models.t5 import T5Config
from t5 import T5Dense, T5FeedForward, T5MultiHeadAttention
from states import to_jax_state
from log_utils import logger

@fixture
def t5_config() -> T5Config:

    T5_REPO = "google-t5/t5-small"
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
    torch_t5_dense = T5DenseActDense(t5_config).eval() # disables dropout

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

    assert(y_torch.shape == y_jax.shape)
    logger.debug(f"y_torch: {y_torch}, y_jax: {y_jax}")
    assert(jnp.allclose(y_torch.numpy(), y_jax, atol=TOL))

@mark.parametrize("q_len, k_len", [(2, 3), (3, 2)])
def test_t5_mha_relative_pos(t5_config: T5Config, q_len: int, k_len: int):
    REL_ATN_BIAS = True

    # torch
    torch_t5_mha = T5Attention(t5_config, REL_ATN_BIAS).eval()
    with torch.no_grad():
        torch_pos_bias = torch_t5_mha.compute_bias(q_len, k_len).numpy()

    # jax
    jax_t5_mha = T5MultiHeadAttention(
        emb_size=t5_config.d_model,
        n_heads=t5_config.num_heads, 
        use_rel_attn_bias=REL_ATN_BIAS
    ) 
    pos_emb_state = to_jax_state(torch_t5_mha.relative_attention_bias)

    jax_pos_bias = jax_t5_mha.compute_pos_bias(pos_emb_state, q_len, k_len)

    assert jnp.allclose(torch_pos_bias, jax_pos_bias, atol=TOL)
    

