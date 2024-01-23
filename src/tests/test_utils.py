from utils import *
import pytest
import jax.numpy as jnp

@pytest.mark.parametrize(("text", "enc"), [("Hello, world!", [9906, 11, 1917, 0])])
def test_cl100k_base(text, enc):
    tokenizer = Tokenizer(name="cl100k_base")
    enc = jnp.array(enc)

    # jax arrays
    assert (tokenizer.encode(text) == enc).all()
    # strings
    assert tokenizer.decode(enc) == text
