from utils import get_tokenizer, count_params, state_to_str
import pytest
import jax.numpy as jnp
from testing_utils import TOL
from log_utils import logger
from states import EmbeddingState
from transformer import Transformer
import jax
from torch.nn import Transformer as TorchTransformer


@pytest.mark.parametrize(("text", "enc"), [("Hello, world!", [9906, 11, 1917, 0])])
def test_cl100k_base(text, enc):
    tokenizer = get_tokenizer(name="cl100k_base")
    enc = jnp.array(enc)

    # jax arrays
    assert (jnp.array(tokenizer.encode(text)) == enc).all()
    # strings
    assert tokenizer.decode(enc) == text


def test_cl100k_batch():
    texts = ["Hello, world!", "Goodbye, world!"]
    enc = [[9906, 11, 1917, 0], [15571, 29474, 11, 1917, 0]]

    tokenizer = get_tokenizer(name="cl100k_base")

    logger.debug(f"enc: {enc}")
    batch_enc = tokenizer.encode_batch(texts)
    logger.debug(f"batch_enc: {batch_enc}")

    texts_pred = tokenizer.decode_batch(enc)
    logger.debug(f"texts_pred: {texts_pred}")

    for e, pred in zip(enc, batch_enc):
        assert e == pred

    for t, pred in zip(texts, texts_pred):
        assert t == pred


def test_count_emb():
    state = EmbeddingState(embeddings=jnp.zeros((10, 10)))

    nbr_params = count_params(state)
    logger.debug(f"nbr_params: {nbr_params}, expected: 100")

    assert nbr_params == 100


@pytest.mark.skip(reason="Not working")
def test_count_transformer():
    # Torch
    emb_size = 1
    n_heads = 1
    n_layers = 1
    d_ff = 1
    dropout = 0

    torch_transformer = TorchTransformer(
        d_model=emb_size,
        nhead=n_heads,
        num_encoder_layers=n_layers,
        num_decoder_layers=n_layers,
        dim_feedforward=d_ff,
        norm_first=True,
        dropout=dropout,
    )

    # Count learnable parameters
    s = 0

    for name, p in torch_transformer.named_parameters():
        print(name, p.numel())
        s += p.numel()
    print(s)

    # Jax
    rng = jax.random.PRNGKey(0)
    model = Transformer(
        emb_size=emb_size,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    )

    model_state = model.init_state(rng)

    print(count_params(model_state))
    print(state_to_str(model_state))
