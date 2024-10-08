import numpy as np
import jax
import jax.numpy as jnp
import torch
from attention import *
from log_utils import logger
import pytest
from testing_utils import TOL, get_nbr_params
from states import to_jax_state

np.random.seed(1337)
rng = jax.random.PRNGKey(0)
torch.manual_seed(0)

CONTEXT_LEN = 3


@pytest.mark.parametrize(
    "emb_size, n_heads, use_bias",
    [(1, 1, False), (1, 1, True), (4, 4, False), (4, 4, True)],
)
def test_pre_attention(emb_size, n_heads, use_bias):
    # No batch
    seq_len = 10
    d_k = emb_size // n_heads
    x = torch.randn(seq_len, emb_size, requires_grad=False)
    linear = torch.nn.Linear(emb_size, n_heads * d_k, bias=use_bias)

    with torch.no_grad():
        head_shape = x.shape[:-1]
        y_torch = linear(x)
        y_torch = y_torch.view(*head_shape, n_heads, d_k)

    # Jax
    # weight = jnp.array(linear.weight.detach())
    # bias = jnp.array(linear.bias.detach()) if use_bias else None
    # state = LinearState(weight, bias)

    state = to_jax_state(linear)

    preattn = PreAttention(n_heads=n_heads, emb_size=emb_size, d_k=d_k, bias=use_bias)
    y_jax = preattn(state, jnp.array(x))

    logger.debug(f"y_torch.shape = {y_torch.shape}, y_jax.shape = {y_jax.shape}")
    assert np.allclose(
        y_torch, y_jax, atol=TOL
    ), f"y_torch = {y_torch}, y_jax = {y_jax}"


@pytest.mark.parametrize(
    "n_heads, emb_size, batch_size", [(1, 2, 3), (2, 4, 3), (8, 16, 8)]
)
@pytest.mark.paramtest
def test_multihead_attn(n_heads, emb_size, batch_size):
    bias = False

    x = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)

    # Torch
    torch_mha = torch.nn.MultiheadAttention(emb_size, n_heads, bias=bias)

    with torch.no_grad():
        y_torch = torch_mha(x, x, x, need_weights=False)[0].detach().numpy()

    # Jax
    jax_mha = MultiHeadAttention(emb_size, n_heads, out_bias=bias, v_bias=False)

    jax_mha_state = to_jax_state(torch_mha)

    x_jnp = jnp.array(x.detach().numpy())

    logger.debug(
        f"Calling jax_mha.forward with x_jnp.shape = {x_jnp.shape} and type = {type(x_jnp)}"
    )
    y_jax = jax_mha.forward(jax_mha_state, x_jnp, x_jnp, x_jnp)

    logger.debug(f"y_torch.shape = {y_torch.shape}, y_jax.shape = {y_jax.shape}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"

    jax_params, torch_params = get_nbr_params(jax_mha_state, torch_mha, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


@pytest.mark.parametrize(
    "n_heads, emb_size, batch_size", [(1, 2, 3), (2, 4, 3), (8, 16, 8)]
)
@pytest.mark.paramtest
def test_multihead_attn_kv_cache(n_heads, emb_size, batch_size):
    xs = jax.random.normal(jax.random.PRNGKey(0), (CONTEXT_LEN, batch_size, emb_size))
    x_single = xs[0][None, ...]

    mha = MultiHeadAttention(emb_size, n_heads, out_bias=False, v_bias=False)
    state = mha.init_state(rng)

    kv_cache = None
    for _ in range(4):
        y, kv_cache = mha(state, xs, xs, xs, kv_cache=kv_cache, use_cache=True)
        y2 = mha(state, xs, xs, xs, use_cache=False)
        xs = jnp.concatenate([xs, x_single], axis=0)

        assert np.allclose(y, y2, atol=TOL), f"y = {y}, y2 = {y2}"


@pytest.mark.parametrize(
    "n_heads, emb_size, batch_size", [(1, 1, 1), (1, 2, 3), (2, 4, 3), (8, 16, 8)]
)
@pytest.mark.paramtest
def test_attention_with_mask(n_heads, emb_size, batch_size):
    x = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)
    mha_torch = torch.nn.MultiheadAttention(emb_size, n_heads, bias=False, dropout=0.0)

    # Jax
    x_jnp = jnp.array(x.detach().numpy())
    jax_mha = MultiHeadAttention(emb_size, n_heads, out_bias=False, v_bias=False)
    state = to_jax_state(mha_torch)

    causal_mask_jax = create_causal_mask(CONTEXT_LEN, CONTEXT_LEN)

    jax_out = jax_mha(state, x_jnp, x_jnp, x_jnp, mask=causal_mask_jax)

    # Torch
    # Invert, and only use 2d, since by pytorch standard true => mask
    causal_mask_torch = torch.from_numpy(np.array(causal_mask_jax))

    with torch.no_grad():
        out_torch = (
            mha_torch(x, x, x, attn_mask=causal_mask_torch, need_weights=False)[0]
            .detach()
            .numpy()
        )

    assert np.allclose(
        jax_out, out_torch, atol=TOL
    ), f"jax_out = {jax_out}, out_torch = {out_torch}"

    jax_params, torch_params = get_nbr_params(state, mha_torch, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"


@pytest.mark.paramtest
def test_cross_attention():
    tgt_len = 3
    src_len = 4

    n_heads = 2
    emb_size = 4
    batch_size = 2
    target = torch.randn(tgt_len, batch_size, emb_size, requires_grad=False)
    source = torch.randn(src_len, batch_size, emb_size, requires_grad=False)

    torch_mha = torch.nn.MultiheadAttention(emb_size, n_heads, bias=False)

    with torch.no_grad():
        y_torch = (
            torch_mha(target, source, source, need_weights=False)[0].detach().numpy()
        )

    # Jax
    jax_mha = MultiHeadAttention(emb_size, n_heads, out_bias=False, v_bias=False)
    jax_mha_state = to_jax_state(torch_mha)

    target_jnp = jnp.array(target.detach().numpy())
    source_jnp = jnp.array(source.detach().numpy())

    logger.debug(
        f"""Calling jax_mha.forward with target_jnp.shape = {target_jnp.shape} and type = {type(target_jnp)},
        source_jnp.shape = {source_jnp.shape} and type = {type(source_jnp)}"""
    )

    y_jax = jax_mha.forward(jax_mha_state, target_jnp, source_jnp, source_jnp)

    logger.debug(f"y_torch.shape = {y_torch.shape}, y_jax.shape = {y_jax.shape}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"

    jax_params, torch_params = get_nbr_params(jax_mha_state, torch_mha, debug=True)
    assert (
        torch_params == jax_params
    ), f"Got different number of parameters: {torch_params} vs {jax_params}"
