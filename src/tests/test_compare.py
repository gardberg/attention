import numpy as np
import jax
import jax.numpy as jnp
import torch
import pytest
from testing_utils import *

from log_utils import logger
from attention import *
from act import *
from states import to_jax_state
from utils import count_params, torch_count_params


np.random.seed(1337)
rng = jax.random.PRNGKey(0)
torch.manual_seed(1337)


# Compare pytorch linear network to custom implementation
@pytest.mark.parametrize("n_in, n_out", [(1, 1), (4, 1), (1, 4), (4, 4)])
def test_dense(n_in: int, n_out: int):
    x_in = torch.randn(n_in)

    torch_linear = torch.nn.Linear(n_in, n_out)

    torch_params = torch_count_params(torch_linear)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Linear(n_in, n_out)
    state = to_jax_state(torch_linear)

    jax_params = count_params(state)
    assert torch_params == jax_params, f"Got different number of parameters: {torch_params} vs {jax_params}"

    y_jax = dense(state, jnp.array(x_in))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("n_in, n_out, batch_size", [(1, 1, 1), (4, 4, 4)])
def test_dense_batch(n_in, n_out, batch_size):
    x_in = torch.randn(batch_size, n_in)
    logger.debug(f"x_in: {x_in.shape}")

    torch_linear = torch.nn.Linear(n_in, n_out)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Linear(n_in, n_out)
    state = to_jax_state(torch_linear)
    y_jax = dense(state, jnp.array(x_in))
    logger.debug(f"y_jax: {y_jax.shape}")

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("n_in, n_out, batch_size", [(1, 1, 1), (4, 4, 4)])
def test_dense_batch_no_bias(n_in, n_out, batch_size):
    x_in = torch.randn(batch_size, n_in)
    logger.debug(f"x_in: {x_in.shape}")

    torch_linear = torch.nn.Linear(n_in, n_out, bias=False)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Linear(n_in, n_out, bias=False)
    state = to_jax_state(torch_linear)
    y_jax = dense(state, jnp.array(x_in))
    logger.debug(f"y_jax: {y_jax.shape}")

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


def test_dense_square():
    n_in = (4, 4)
    n_out = 1

    x_in = torch.randn(*n_in)

    torch_linear = torch.nn.Linear(np.prod(n_in), np.prod(n_out))

    with torch.no_grad():
        y_torch = torch_linear(x_in.flatten()).numpy()

    # Jax
    dense = Linear(n_in, n_out)
    state = to_jax_state(torch_linear)
    y_jax = dense(state, jnp.array(x_in.flatten()))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("B, N", [(1, 2), (2, 1), (2, 3)])
def test_batchnorm_1d_inference_small(B: int, N: int):
    x = torch.randn(B, N)
    torch_bn = torch.nn.BatchNorm1d(N)
    torch_bn.eval()

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()

    # Jax
    state = BatchNormState()
    y, _state = batchnorm_1d(jnp.array(x), state, training=False)

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("B, N, L", [(1, 2, 2), (2, 2, 2), (1, 1, 1)])
def test_batchnorm_1d_inference(B: int, N: int, L: int):
    x = torch.randn(B, N, L)
    torch_bn = torch.nn.BatchNorm1d(N)
    torch_bn.eval()

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()

    # Jax
    state = BatchNormState()
    y, _state = batchnorm_1d(jnp.array(x), state, training=False)

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("B, N", [(2, 2), (2, 1), (2, 3)])
def test_batchnorm_1d_train(B: int, N: int):
    x = torch.randn(B, N)
    torch_bn = torch.nn.BatchNorm1d(N)

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()

    # Jax
    state = BatchNormState()
    y, _state = batchnorm_1d(jnp.array(x), state, training=True)

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("norm_dims", [(3,), (2, 3)])
def test_layernorm(norm_dims: tuple):
    # assume input: (context_len, batch_size, emb_dim)
    x = torch.randn(4, 2, 3, requires_grad=False) * 10

    torch_ln = torch.nn.LayerNorm(norm_dims)
    with torch.no_grad():
        y_torch = torch_ln(x).numpy()

    # Jax
    jax_ln = LayerNorm(norm_dims)
    state = to_jax_state(torch_ln)

    y_jax = jax_ln(state, jnp.array(x))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("size", [(4, 2, 16), (1, 1, 16), (16, 1, 512)])
def test_positional_encoding(size):
    # (context_len, batch_size, embed_dim)
    # Torch
    x = torch.zeros(*size, requires_grad=False)
    embed_dim = x.shape[-1]

    torch_posenc = TorchPositionalEncoding(embed_dim, dropout=0.0)
    y_torch = torch_posenc(x).numpy()

    # Jax
    jax_posenc = PositionalEncoding(embed_dim, dropout=0.0)
    y_jax = jax_posenc(jnp.array(x), rng)

    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("embed_dim", [1, 8])
def test_rmsnorm(embed_dim):
    x = torch.randn(4, 2, embed_dim, requires_grad=False)
    eps = 1e-5

    # Torch
    torch_rmsnorm = TRMSNorm(embed_dim, eps=eps)
    with torch.no_grad():
        y_torch = torch_rmsnorm(x).numpy()

    # Jax
    jax_rmsnorm = RMSNorm(embed_dim, eps=eps)
    state = jax_rmsnorm.init_state(rng)
    y_jax = jax_rmsnorm(state, jnp.array(x))

    logger.debug(f"y_torch.shape = {y_torch.shape}, y_jax.shape = {y_jax.shape}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("shape", [(2, 2, 1, 4)])
def test_rope(shape: tuple):
    # TODO: Put expected in parametrize

    x = jnp.ones(shape)
    expected = jnp.array(
        [
            [[[1.0, 1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0, 1.0]]],
            [
                [[-0.30116868, 0.9899502, 1.3817732, 1.0099498]],
                [[-0.30116868, 0.9899502, 1.3817732, 1.0099498]],
            ],
        ]
    )
    res = apply_rope(x)

    assert np.allclose(res, expected, atol=TOL), f"res = {res}, expected = {expected}"


@pytest.mark.parametrize("shape", [(2,), (2, 2)])
def test_learned_embeddings(shape: tuple):
    VOCAB_SIZE = 16
    EMB_DIM = 2

    indices = np.random.randint(0, VOCAB_SIZE, shape)

    # Torch
    torch_emb = torch.nn.Embedding(VOCAB_SIZE, EMB_DIM)

    with torch.no_grad():
        y_torch = torch_emb(torch.tensor(indices)).numpy()

    # Jax

    jax_emb = Embedding(VOCAB_SIZE, EMB_DIM)
    embedding_state = to_jax_state(torch_emb)

    y_jax = jax_emb(embedding_state, jnp.array(indices))

    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"
