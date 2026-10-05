"""Recurrences for Gated Slot Attention-2 (arXiv 2610.02816).

Vectors are columns. State orientation matches the paper: S @ q, not q^T @ S.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def outer(a, b):
    """Outer product a b^T for column vectors."""
    return a[:, None] * b[None, :]


def row_scale(alpha, M):
    """diag(alpha) @ M, scale rows of M."""
    return alpha[:, None] * M


def col_scale(M, alpha):
    """M @ diag(alpha), scale columns of M."""
    return M * alpha[None, :]


def silu(x):
    """SiLU activation, x * sigmoid(x)."""
    return x * jax.nn.sigmoid(x)


def gdn2_step(S_prev, alpha, b, k, c, v):
    """Gated Delta Rule-2, Eq. 3.

    S_prev: (Dv, Dk)
    alpha, b, k: (Dk,)
    c, v: (Dv,)
    Returns S_t: (Dv, Dk)
    """
    Dk = k.shape[-1]
    eye = jnp.eye(Dk, dtype=S_prev.dtype)
    erase = eye - outer(b * k, k)
    return col_scale(S_prev, alpha) @ erase + outer(c * v, k)


def oja_step(S_prev, alpha, v, k, beta):
    """Gated Oja Rule, Eq. 9.

    S_prev: (Dv, Dk)
    alpha, v: (Dv,)
    k: (Dk,)
    beta: scalar
    Returns S_t: (Dv, Dk)
    """
    Dv = v.shape[-1]
    eye = jnp.eye(Dv, dtype=S_prev.dtype)
    left = eye - beta * outer(v, v)
    return left @ row_scale(alpha, S_prev) + beta * outer(v, k)


def oja2_step(S_prev, alpha, v, k, b, c):
    """Gated Oja Rule-2, Eq. 10.

    S_prev: (Dv, Dk)
    alpha, v, b: (Dv,)
    k, c: (Dk,)
    Returns S_t: (Dv, Dk)

    When b = beta * ones(Dv) and c = beta * ones(Dk), this is Eq. 9.
    """
    Dv = v.shape[-1]
    eye = jnp.eye(Dv, dtype=S_prev.dtype)
    left = eye - outer(v, b * v)
    return left @ row_scale(alpha, S_prev) + outer(v, c * k)


def gsa_step(S1_prev, S2_prev, alpha, k, v, q):
    """Gated Slot Attention, Eqs. 4-6.

    S1_prev: (M, Dk), S2_prev: (Dv, M)
    alpha: (M,), k, q: (Dk,), v: (Dv,)

    The write term is the outer product of (1 - alpha) with k, as printed.
    Returns S1_t, S2_t, and o = S2_t @ softmax(S1_t @ q).
    """
    S1 = row_scale(alpha, S1_prev) + outer(1.0 - alpha, k)
    S2 = col_scale(S2_prev, alpha) + outer(v, 1.0 - alpha)
    o = S2 @ jax.nn.softmax(S1 @ q)
    return S1, S2, o


def gsa2_step(S1_prev, S2_prev, w, b1, b2, alpha1, alpha2, c1, c2, k, v, q):
    """Gated Slot Attention-2, Eqs. 11-13.

    S1_prev: (M, Dk), S2_prev: (Dv, M)
    w, b1, b2, alpha1, alpha2: (M,)
    c1, k, q: (Dk,)
    c2, v: (Dv,)

    Stage 1 is OJA2 with the value vector replaced by w.
    Stage 2 is GDN2 with the key replaced by w.
    The read is S2 @ silu(S1 @ q), not softmax.
    """
    M = w.shape[-1]
    eye = jnp.eye(M, dtype=S1_prev.dtype)
    S1 = (eye - outer(w, b1 * w)) @ row_scale(alpha1, S1_prev) + outer(w, c1 * k)
    S2 = col_scale(S2_prev, alpha2) @ (eye - outer(b2 * w, w)) + outer(c2 * v, w)
    o = S2 @ silu(S1 @ q)
    return S1, S2, o


def gsa2_scan(w, b1, b2, alpha1, alpha2, c1, c2, k, v, q):
    """Scan Eqs. 11-13 over time with a zero initial state.

    Each argument is time-major, shape (T, dim). Returns outputs (T, Dv)
    and the final states S1 (M, Dk), S2 (Dv, M). State shapes do not
    depend on T.
    """
    M = w.shape[-1]
    Dk = k.shape[-1]
    Dv = v.shape[-1]
    S1_0 = jnp.zeros((M, Dk), dtype=w.dtype)
    S2_0 = jnp.zeros((Dv, M), dtype=w.dtype)

    def step(carry, xs):
        S1, S2 = carry
        w_t, b1_t, b2_t, a1_t, a2_t, c1_t, c2_t, k_t, v_t, q_t = xs
        S1, S2, o = gsa2_step(
            S1, S2, w_t, b1_t, b2_t, a1_t, a2_t, c1_t, c2_t, k_t, v_t, q_t
        )
        return (S1, S2), o

    (S1_f, S2_f), outputs = jax.lax.scan(
        step,
        (S1_0, S2_0),
        (w, b1, b2, alpha1, alpha2, c1, c2, k, v, q),
    )
    return outputs, S1_f, S2_f
