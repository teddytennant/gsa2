"""Block shape, causality (including the conv), and finite grads."""

import numpy as np
import jax
import jax.numpy as jnp

from gsa2 import GSA2Block, apply_block, init_block
from gsa2.block import depthwise_causal_conv


def test_param_shapes_and_block_output_shape():
    d_model, n_slots, d_k, d_v = 32, 8, 16, 12
    rank, width = 16, 4
    key = jax.random.PRNGKey(0)
    params = init_block(key, d_model, n_slots, d_k, d_v, rank=rank, conv_kernel=width)
    assert params["W_q"].shape == (d_model, d_k)
    assert params["W_k"].shape == (d_model, d_k)
    assert params["W_v"].shape == (d_model, d_v)
    assert params["W_w1"].shape == (d_model, rank)
    assert params["W_w2"].shape == (rank, n_slots)
    assert params["conv_q"].shape == (width, d_k)
    assert params["conv_v"].shape == (width, d_v)
    assert params["conv_w"].shape == (width, n_slots)
    assert params["W_alpha1"].shape == (d_model, n_slots)
    assert params["W_b2"].shape == (d_model, n_slots)
    assert params["W_c1"].shape == (d_model, d_k)
    assert params["W_c2"].shape == (d_model, d_v)
    assert params["W_out_gate"].shape == (d_model, d_v)
    assert params["W_out"].shape == (d_v, d_model)

    B, T = 2, 7
    x = jax.random.normal(jax.random.PRNGKey(1), (B, T, d_model))
    y = apply_block(params, x)
    assert y.shape == (B, T, d_model)

    block = GSA2Block(d_model, n_slots, d_k, d_v, rank=rank, conv_kernel=width, key=key)
    y2 = block(x)
    assert y2.shape == (B, T, d_model)
    np.testing.assert_allclose(y, y2, atol=1e-6)


def test_unbatched_shape():
    params = init_block(jax.random.PRNGKey(2), 16, 4, 8, 6, rank=8, conv_kernel=4)
    x = jax.random.normal(jax.random.PRNGKey(3), (5, 16))
    y = apply_block(params, x)
    assert y.shape == (5, 16)


def test_conv_is_causal_and_prefix_stable():
    T, C, K = 6, 3, 4
    x = jax.random.normal(jax.random.PRNGKey(4), (2, T, C))
    kernel = jax.random.normal(jax.random.PRNGKey(5), (K, C))
    y = depthwise_causal_conv(x, kernel)
    y_short = depthwise_causal_conv(x[:, :3], kernel)
    np.testing.assert_allclose(y[:, :3], y_short, atol=1e-6)
    x2 = x.at[:, 4].set(x[:, 4] + 5.0)
    y2 = depthwise_causal_conv(x2, kernel)
    np.testing.assert_allclose(y[:, :4], y2[:, :4], atol=1e-6)
    assert not np.allclose(np.asarray(y[:, 4]), np.asarray(y2[:, 4]), atol=1e-5)


def test_block_prefix_includes_conv():
    params = init_block(jax.random.PRNGKey(6), 16, 5, 8, 7, rank=8, conv_kernel=4)
    x = jax.random.normal(jax.random.PRNGKey(7), (2, 9, 16))
    y_long = apply_block(params, x)
    y_short = apply_block(params, x[:, :4])
    np.testing.assert_allclose(y_long[:, :4], y_short, atol=1e-5)


def test_future_token_does_not_change_past_outputs():
    params = init_block(jax.random.PRNGKey(8), 16, 5, 8, 7, rank=8, conv_kernel=4)
    x = jax.random.normal(jax.random.PRNGKey(9), (2, 8, 16))
    t_change = 5
    x2 = x.at[:, t_change].set(x[:, t_change] + 3.0)
    y1 = apply_block(params, x)
    y2 = apply_block(params, x2)
    np.testing.assert_allclose(y1[:, :t_change], y2[:, :t_change], atol=1e-5)
    # The changed token itself moves the residual, so that position must move.
    assert not np.allclose(np.asarray(y1[:, t_change]), np.asarray(y2[:, t_change]), atol=1e-4)


def test_past_token_reaches_beyond_conv_window():
    width = 4
    params = init_block(jax.random.PRNGKey(10), 16, 5, 8, 7, rank=8, conv_kernel=width)
    x = jax.random.normal(jax.random.PRNGKey(11), (2, 8, 16))
    x2 = x.at[:, 0].set(x[:, 0] + 4.0)
    y1 = apply_block(params, x)
    y2 = apply_block(params, x2)
    # Position `width` is outside the conv window of token 0, so a change
    # there has to come from the recurrent state, not the convolution.
    diff = jnp.max(jnp.abs(y1[:, width] - y2[:, width]))
    assert float(diff) > 1e-6


def test_grads_are_finite():
    params = init_block(jax.random.PRNGKey(12), 16, 4, 8, 6, rank=8, conv_kernel=4)
    x = jax.random.normal(jax.random.PRNGKey(13), (2, 6, 16))

    def loss(p, inp):
        y = apply_block(p, inp)
        return jnp.sum(y * y)

    grads = jax.grad(loss)(params, x)
    leaves = jax.tree.leaves(grads)
    assert leaves
    for g in leaves:
        assert g.shape  # nonempty
        assert bool(jnp.all(jnp.isfinite(g)))
    assert any(bool(jnp.any(g != 0)) for g in leaves)
