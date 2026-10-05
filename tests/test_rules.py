"""Independent checks of the GSA2 recurrences."""

import numpy as np
import jax
import jax.numpy as jnp

from gsa2 import gdn2_step, gsa2_scan, gsa2_step, gsa_step, oja2_step, oja_step


def _np_outer(a, b):
    return a[:, None] * b[None, :]


def _np_silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def numpy_gsa2_step(S1, S2, w, b1, b2, alpha1, alpha2, c1, c2, k, v, q):
    """Eqs. 11-13 in numpy. Does not call the library."""
    M = w.shape[0]
    eye = np.eye(M, dtype=np.float32)
    S1_t = (eye - _np_outer(w, b1 * w)) @ (np.diag(alpha1) @ S1) + _np_outer(w, c1 * k)
    S2_t = (S2 @ np.diag(alpha2)) @ (eye - _np_outer(b2 * w, w)) + _np_outer(c2 * v, w)
    slot = S1_t @ q
    o = S2_t @ _np_silu(slot)
    return S1_t, S2_t, o


def numpy_gsa_step(S1, S2, alpha, k, v, q):
    """Eqs. 4-6 in numpy. Write term is outer(1 - alpha, k), as printed."""
    S1_t = np.diag(alpha) @ S1 + _np_outer(1.0 - alpha, k)
    S2_t = S2 @ np.diag(alpha) + _np_outer(v, 1.0 - alpha)
    slot = S1_t @ q
    slot = slot - np.max(slot)
    weights = np.exp(slot)
    weights = weights / np.sum(weights)
    o = S2_t @ weights
    return S1_t, S2_t, o


def _rand(rng, shape):
    return rng.standard_normal(shape).astype(np.float32)


def test_gsa2_step_matches_numpy_reference():
    rng = np.random.default_rng(0)
    M, Dk, Dv = 5, 4, 3
    S1 = _rand(rng, (M, Dk))
    S2 = _rand(rng, (Dv, M))
    w = _rand(rng, (M,))
    b1 = _rand(rng, (M,))
    b2 = _rand(rng, (M,))
    alpha1 = _rand(rng, (M,))
    alpha2 = _rand(rng, (M,))
    c1 = _rand(rng, (Dk,))
    c2 = _rand(rng, (Dv,))
    k = _rand(rng, (Dk,))
    v = _rand(rng, (Dv,))
    q = _rand(rng, (Dk,))

    S1_n, S2_n, o_n = numpy_gsa2_step(
        S1, S2, w, b1, b2, alpha1, alpha2, c1, c2, k, v, q
    )
    S1_j, S2_j, o_j = gsa2_step(
        jnp.asarray(S1),
        jnp.asarray(S2),
        jnp.asarray(w),
        jnp.asarray(b1),
        jnp.asarray(b2),
        jnp.asarray(alpha1),
        jnp.asarray(alpha2),
        jnp.asarray(c1),
        jnp.asarray(c2),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(q),
    )
    np.testing.assert_allclose(np.asarray(S1_j), S1_n, atol=1e-5)
    np.testing.assert_allclose(np.asarray(S2_j), S2_n, atol=1e-5)
    np.testing.assert_allclose(np.asarray(o_j), o_n, atol=1e-5)
    assert o_j.shape == (Dv,)


def test_gsa_matches_numpy_reference():
    rng = np.random.default_rng(1)
    M, Dk, Dv = 4, 3, 5
    S1 = _rand(rng, (M, Dk))
    S2 = _rand(rng, (Dv, M))
    alpha = _rand(rng, (M,))
    k = _rand(rng, (Dk,))
    v = _rand(rng, (Dv,))
    q = _rand(rng, (Dk,))
    S1_n, S2_n, o_n = numpy_gsa_step(S1, S2, alpha, k, v, q)
    S1_j, S2_j, o_j = gsa_step(
        jnp.asarray(S1),
        jnp.asarray(S2),
        jnp.asarray(alpha),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(q),
    )
    np.testing.assert_allclose(np.asarray(S1_j), S1_n, atol=1e-5)
    np.testing.assert_allclose(np.asarray(S2_j), S2_n, atol=1e-5)
    np.testing.assert_allclose(np.asarray(o_j), o_n, atol=1e-5)


def test_oja2_reduces_to_oja():
    rng = np.random.default_rng(2)
    Dv, Dk = 6, 4
    S = jnp.asarray(_rand(rng, (Dv, Dk)))
    alpha = jnp.asarray(_rand(rng, (Dv,)))
    v = jnp.asarray(_rand(rng, (Dv,)))
    k = jnp.asarray(_rand(rng, (Dk,)))
    beta = jnp.asarray(0.37, dtype=jnp.float32)
    b = beta * jnp.ones((Dv,), dtype=jnp.float32)
    c = beta * jnp.ones((Dk,), dtype=jnp.float32)
    S_oja = oja_step(S, alpha, v, k, beta)
    S_oja2 = oja2_step(S, alpha, v, k, b, c)
    np.testing.assert_allclose(S_oja2, S_oja, atol=1e-5)


def test_gsa2_stage1_reduces_to_oja2():
    rng = np.random.default_rng(3)
    M, Dk, Dv = 5, 4, 3
    S1 = jnp.asarray(_rand(rng, (M, Dk)))
    S2 = jnp.asarray(_rand(rng, (Dv, M)))
    w = jnp.asarray(_rand(rng, (M,)))
    b1 = jnp.asarray(_rand(rng, (M,)))
    b2 = jnp.asarray(_rand(rng, (M,)))
    alpha1 = jnp.asarray(_rand(rng, (M,)))
    alpha2 = jnp.asarray(_rand(rng, (M,)))
    c1 = jnp.asarray(_rand(rng, (Dk,)))
    c2 = jnp.asarray(_rand(rng, (Dv,)))
    k = jnp.asarray(_rand(rng, (Dk,)))
    v = jnp.asarray(_rand(rng, (Dv,)))
    q = jnp.asarray(_rand(rng, (Dk,)))
    S1_t, _, _ = gsa2_step(S1, S2, w, b1, b2, alpha1, alpha2, c1, c2, k, v, q)
    S1_oja2 = oja2_step(S1, alpha1, w, k, b1, c1)
    np.testing.assert_allclose(S1_t, S1_oja2, atol=1e-5)


def test_gsa2_stage2_reduces_to_gdn2():
    rng = np.random.default_rng(4)
    M, Dk, Dv = 5, 4, 3
    S1 = jnp.asarray(_rand(rng, (M, Dk)))
    S2 = jnp.asarray(_rand(rng, (Dv, M)))
    w = jnp.asarray(_rand(rng, (M,)))
    b1 = jnp.asarray(_rand(rng, (M,)))
    b2 = jnp.asarray(_rand(rng, (M,)))
    alpha1 = jnp.asarray(_rand(rng, (M,)))
    alpha2 = jnp.asarray(_rand(rng, (M,)))
    c1 = jnp.asarray(_rand(rng, (Dk,)))
    c2 = jnp.asarray(_rand(rng, (Dv,)))
    k = jnp.asarray(_rand(rng, (Dk,)))
    v = jnp.asarray(_rand(rng, (Dv,)))
    q = jnp.asarray(_rand(rng, (Dk,)))
    _, S2_t, _ = gsa2_step(S1, S2, w, b1, b2, alpha1, alpha2, c1, c2, k, v, q)
    # GDN2 with the key replaced by w.
    S2_gdn2 = gdn2_step(S2, alpha2, b2, w, c2, v)
    np.testing.assert_allclose(S2_t, S2_gdn2, atol=1e-5)


def test_silu_read_differs_from_softmax():
    rng = np.random.default_rng(5)
    M, Dk, Dv = 4, 3, 5
    S1 = jnp.asarray(_rand(rng, (M, Dk)))
    S2 = jnp.asarray(_rand(rng, (Dv, M)))
    w = jnp.asarray(_rand(rng, (M,)))
    b1 = jnp.asarray(_rand(rng, (M,)))
    b2 = jnp.asarray(_rand(rng, (M,)))
    alpha1 = jnp.asarray(_rand(rng, (M,)))
    alpha2 = jnp.asarray(_rand(rng, (M,)))
    c1 = jnp.asarray(_rand(rng, (Dk,)))
    c2 = jnp.asarray(_rand(rng, (Dv,)))
    k = jnp.asarray(_rand(rng, (Dk,)))
    v = jnp.asarray(_rand(rng, (Dv,)))
    q = jnp.asarray(_rand(rng, (Dk,)))
    S1_t, S2_t, o = gsa2_step(S1, S2, w, b1, b2, alpha1, alpha2, c1, c2, k, v, q)
    slot = S1_t @ q
    o_silu = S2_t @ (slot * jax.nn.sigmoid(slot))
    o_soft = S2_t @ jax.nn.softmax(slot)
    np.testing.assert_allclose(o, o_silu, atol=1e-5)
    assert not np.allclose(np.asarray(o_silu), np.asarray(o_soft), atol=1e-5)
    assert float(jnp.max(jnp.abs(o_silu - o_soft))) > 1e-3


def _seq(T, M, Dk, Dv, key):
    keys = jax.random.split(key, 10)
    def n(k, shape):
        return jax.random.normal(k, (T,) + shape)
    return (
        n(keys[0], (M,)),
        n(keys[1], (M,)),
        n(keys[2], (M,)),
        n(keys[3], (M,)),
        n(keys[4], (M,)),
        n(keys[5], (Dk,)),
        n(keys[6], (Dv,)),
        n(keys[7], (Dk,)),
        n(keys[8], (Dv,)),
        n(keys[9], (Dk,)),
    )


def test_state_shape_independent_of_T():
    M, Dk, Dv = 4, 3, 5
    key = jax.random.PRNGKey(6)
    outs_a, S1_a, S2_a = gsa2_scan(*_seq(3, M, Dk, Dv, key))
    outs_b, S1_b, S2_b = gsa2_scan(*_seq(9, M, Dk, Dv, key))
    assert outs_a.shape == (3, Dv)
    assert outs_b.shape == (9, Dv)
    assert S1_a.shape == S1_b.shape == (M, Dk)
    assert S2_a.shape == S2_b.shape == (Dv, M)


def test_scan_prefix_matches_shorter_run():
    M, Dk, Dv, T, t = 4, 3, 5, 8, 5
    args = _seq(T, M, Dk, Dv, jax.random.PRNGKey(7))
    outs_long, _, _ = gsa2_scan(*args)
    outs_short, S1_s, S2_s = gsa2_scan(*(a[:t] for a in args))
    np.testing.assert_allclose(outs_long[:t], outs_short, atol=1e-5)
    # A hand loop from a zero state must agree, including the final state.
    S1 = jnp.zeros((M, Dk))
    S2 = jnp.zeros((Dv, M))
    manual = []
    for i in range(t):
        S1, S2, o = gsa2_step(
            S1, S2, *[a[i] for a in args]
        )
        manual.append(o)
    np.testing.assert_allclose(jnp.stack(manual), outs_short, atol=1e-5)
    np.testing.assert_allclose(S1, S1_s, atol=1e-5)
    np.testing.assert_allclose(S2, S2_s, atol=1e-5)
