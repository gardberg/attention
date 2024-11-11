import numpy as np
import jax
import jax.numpy as jnp
import torch
import pytest
import torch.nn.functional as F

from testing_utils import *

from log_utils import logger
from attention import *
from act import *
from states import to_jax_state


np.random.seed(1337)
rng = jax.random.PRNGKey(0)
torch.manual_seed(1337)


def test_attention_computation():
    BATCH_SIZE, LEN, N_HEADS, EMB_SIZE = 2, 10, 12, 768
    HEAD_DIM = EMB_SIZE // N_HEADS
    x = torch.randn(BATCH_SIZE, N_HEADS, LEN, HEAD_DIM)

    # torch
    torch_attn = F.scaled_dot_product_attention(x, x, x).numpy()

    # reshape to (*_len, batch_size, n_heads, d_k)
    x = jnp.array(x).transpose(2, 0, 1, 3)

    # jax einsum
    jax_einsum = jnp.einsum("cbhd,Cbhd->cCbh", x, x)
    scaled_jax_einsum = jax_einsum * (1 / jnp.sqrt(HEAD_DIM))
    softmax_jax_einsum = softmax_stable(scaled_jax_einsum, dim=1)
    jax_attn = jnp.einsum("cCbh,Cbhd->cbhd", softmax_jax_einsum, x)
    jax_attn = jax_attn.transpose(1, 2, 0, 3)

    # jax matrix
    x = x.transpose(1, 2, 0, 3)  # (batch_size, n_heads, *_len, d_k)
    jax_matrix = x @ x.transpose(0, 1, 3, 2) * (1 / jnp.sqrt(HEAD_DIM))
    jax_matrix = softmax_stable(jax_matrix, dim=-1)
    jax_attn_matrix = jax_matrix @ x

    assert jnp.array(torch_attn).shape == jax_attn.shape
    logger.debug(f"Torch vs Jax Einsum:")
    logger.debug(f"Max diff: {np.abs(torch_attn - jax_attn).max():.2e}")
    logger.debug(f"Norm diff: {np.linalg.norm(torch_attn - jax_attn):.2e}")
    logger.debug(f"\nTorch vs Jax Matrix: ")
    logger.debug(f"Max diff matrix: {np.abs(torch_attn - jax_attn_matrix).max():.2e}")
    logger.debug(f"Norm diff matrix: {np.linalg.norm(torch_attn - jax_attn_matrix):.2e}")
    logger.debug(f"\nJax Einsum vs Matrix:")
    logger.debug(f"Max diff: {np.abs(jax_attn - jax_attn_matrix).max():.2e}")
    logger.debug(f"Norm diff: {np.linalg.norm(jax_attn - jax_attn_matrix):.2e}")


@pytest.mark.parametrize(
    "affine, num_groups, num_channels",
    [(False, 2, 4), (True, 2, 4), (False, 4, 4), (True, 512, 512)],
)
def test_groupnorm(affine, num_groups, num_channels):
    B, N, H = 2, num_channels, 2
    x = torch.randn(B, N, H)
    torch_gn = torch.nn.GroupNorm(num_groups, num_channels, affine=affine)
    with torch.no_grad():
        y_torch = torch_gn(x)

    y_torch = jnp.array(y_torch)

    jax_gn = GroupNorm(num_groups, num_channels, affine=affine)
    state = to_jax_state(torch_gn)
    x_jax = jnp.array(x)
    y_jax = jax_gn(state, x_jax)

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert (
        y_torch.shape == y_jax.shape
    ), f"y_torch.shape = {y_torch.shape}, y_jax.shape = {y_jax.shape}"
    assert jnp.allclose(
        y_torch, y_jax, atol=TOL
    ), f"y_torch = {y_torch}, y_jax = {y_jax}"


# Compare pytorch linear network to custom implementation
@pytest.mark.parametrize("n_in, n_out", [(1, 1), (4, 1), (1, 4), (4, 4)])
# run with pytest -m paramtest
@pytest.mark.paramtest
def test_dense(n_in: int, n_out: int):
    x_in = torch.randn(n_in)

    torch_linear = torch.nn.Linear(n_in, n_out)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Linear(n_in, n_out)
    state = to_jax_state(torch_linear)

    y_jax = dense(state, jnp.array(x_in))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"

    # Count params
    jax_params, torch_params = get_nbr_params(state, torch_linear, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


@pytest.mark.parametrize("n_in, n_out, batch_size", [(1, 1, 1), (4, 4, 4)])
@pytest.mark.paramtest
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

    # Count params
    jax_params, torch_params = get_nbr_params(state, torch_linear, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


@pytest.mark.parametrize("n_in, n_out, batch_size", [(1, 1, 1), (4, 4, 4)])
@pytest.mark.paramtest
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

    # Count params
    jax_params, torch_params = get_nbr_params(state, torch_linear, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


@pytest.mark.paramtest
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

    # Count params
    jax_params, torch_params = get_nbr_params(state, torch_linear, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


@pytest.mark.parametrize("B, N", [(1, 2), (2, 1), (2, 3)])
@pytest.mark.paramtest
def test_batchnorm_1d_inference_small(B: int, N: int):
    x = torch.randn(B, N)
    torch_bn = torch.nn.BatchNorm1d(N)
    torch_bn.eval()

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()

    # Jax
    jax_bn = BatchNorm1d(N)
    state = jax_bn.init_state()

    y, _ = jax_bn(state, jnp.array(x), training=False)

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"

    # Batchnorm has 3 more params than the count we get from torch
    jax_params, torch_params = get_nbr_params(state, torch_bn, debug=True)
    assert (
        torch_params + 3 == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


@pytest.mark.parametrize("B, N, L", [(1, 2, 2), (2, 2, 2), (1, 1, 1)])
@pytest.mark.paramtest
def test_batchnorm_1d_inference(B: int, N: int, L: int):
    x = torch.randn(B, N, L)
    torch_bn = torch.nn.BatchNorm1d(N)
    torch_bn.eval()

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()

    # Jax
    jax_bn = BatchNorm1d(N)
    state = jax_bn.init_state()

    y, _ = jax_bn(state, jnp.array(x), training=False)

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"

    jax_params, torch_params = get_nbr_params(state, torch_bn, debug=True)
    assert (
        torch_params + 3 == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


@pytest.mark.parametrize("B, N", [(2, 2), (2, 1), (2, 3)])
@pytest.mark.paramtest
def test_batchnorm_1d_train(B: int, N: int):
    x = torch.randn(B, N)
    torch_bn = torch.nn.BatchNorm1d(N)

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()

    # Jax
    jax_bn = BatchNorm1d(N)
    state = jax_bn.init_state()
    y, _ = jax_bn(state, jnp.array(x), training=True)

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"

    jax_params, torch_params = get_nbr_params(state, torch_bn, debug=True)
    assert (
        torch_params + 3 == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


@pytest.mark.parametrize("norm_dims", [(3,), (2, 3)])
@pytest.mark.paramtest
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

    jax_params, torch_params = get_nbr_params(state, torch_ln, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


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
@pytest.mark.paramtest
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

    jax_params, torch_params = get_nbr_params(state, torch_rmsnorm, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


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
@pytest.mark.paramtest
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

    jax_params, torch_params = get_nbr_params(embedding_state, torch_emb, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"
