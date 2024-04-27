import torch
import jax.numpy as jnp
from transformers.models.t5.modeling_t5 import T5DenseActDense, T5LayerFF
from states import to_jax_state

from utils import ROOT_DIR
from testing_utils import TOL
import os
from transformers.models.t5 import T5Config
import json

from t5 import T5Dense, T5FeedForward
import jax

from log_utils import logger

# get t5 config
REL_PATH = "..\hf_hub\models--google-t5--t5-small\snapshots\df1b051c49625cf57a3d0d8d3863ed4d13564fe4\config.json"
CONFIG_PATH = os.path.join(ROOT_DIR, REL_PATH)

try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"T5 config not found at {CONFIG_PATH}. Tip: symlink HF_HUB to ../hf_hub.")

t5_config = T5Config.from_dict(config)


torch.random.manual_seed(0)
BATCH_SIZE = 2
SEQ_LEN = 3
EMBED_SIZE = 512
D_FF = 2048

test_shape = (BATCH_SIZE, SEQ_LEN, EMBED_SIZE)

RNG = jax.random.PRNGKey(0)


def test_t5_dense():
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
    

def test_t5_ff():
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
