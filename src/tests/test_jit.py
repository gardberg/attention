from models.t5 import T5Model
from pytest import fixture, mark
from transformers.models.t5 import T5Config
from huggingface_hub import hf_hub_download
from log_utils import logger
import json
import jax
import pytest


T5_REPO = "google-t5/t5-small"


@fixture
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


BATCH_SIZE = 1
SEQ_LEN = 10


# benchmark is here a benchmark fixture
def test_t5_forward_jit(t5_config):
    jax_t5_model = T5Model(vocab_size=t5_config.vocab_size, emb_size=t5_config.d_model)

    state = jax_t5_model.init_state(rng=jax.random.PRNGKey(0))

    input_ids = jax.random.randint(
        jax.random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN), 0, t5_config.vocab_size
    )
    decoder_input_ids = jax.random.randint(
        jax.random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN), 0, t5_config.vocab_size
    )

    rng = jax.random.PRNGKey(0)

    jax_t5_model.set_use_jit(False)
    logits_nojit, enc_out_nojit = jax_t5_model.forward(
        state, input_ids, decoder_input_ids, rng
    )

    jax_t5_model.set_use_jit(True)
    logits_jit, enc_out_jit = jax_t5_model.forward(
        state, input_ids, decoder_input_ids, rng
    )

    assert jax.numpy.allclose(logits_nojit, logits_jit)


@pytest.mark.benchmark(group="jit", warmup=True)
@mark.parametrize("use_jit", [True, False])
def test_benchmark_t5_forward(t5_config, benchmark, use_jit: bool):
    jax_t5_model = T5Model(vocab_size=t5_config.vocab_size, emb_size=t5_config.d_model)

    state = jax_t5_model.init_state(rng=jax.random.PRNGKey(0))

    input_ids = jax.random.randint(
        jax.random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN), 0, t5_config.vocab_size
    )
    decoder_input_ids = jax.random.randint(
        jax.random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN), 0, t5_config.vocab_size
    )

    rng = jax.random.PRNGKey(0)

    jax_t5_model.set_use_jit(use_jit)
    output_no_jit = benchmark(
        jax_t5_model.forward, state, input_ids, decoder_input_ids, rng
    )
