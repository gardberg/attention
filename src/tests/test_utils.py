from utils import *
import pytest
import jax.numpy as jnp
from testing_utils import TOL
from log_utils import logger


@pytest.mark.parametrize(("text", "enc"), [("Hello, world!", [9906, 11, 1917, 0])])
def test_cl100k_base(text, enc):
    tokenizer = Tokenizer(name="cl100k_base")
    enc = jnp.array(enc)

    # jax arrays
    assert (jnp.array(tokenizer.encode(text)) == enc).all()
    # strings
    assert tokenizer.decode(enc) == text


def test_cl100k_batch():
    texts = ["Hello, world!", "Goodbye, world!"]
    enc = [[9906, 11, 1917, 0], [15571, 29474, 11, 1917, 0]]

    tokenizer = Tokenizer(name="cl100k_base")

    logger.debug(f"enc: {enc}")
    batch_enc = tokenizer.encode_batch(texts)
    logger.debug(f"batch_enc: {batch_enc}")

    texts_pred = tokenizer.decode_batch(enc)
    logger.debug(f"texts_pred: {texts_pred}")

    for e, pred in zip(enc, batch_enc):
        assert e == pred

    for t, pred in zip(texts, texts_pred):
        assert t == pred
