"""One GSA2 token-mixer block.

Pre-norm, linear projections, a short depthwise causal convolution, the
Eqs. 11-13 scan, then an output RMSNorm, sigmoid gate, and residual add.

rank and conv_kernel are stand-ins. The paper uses a low-rank slot projection
and a short causal convolution but does not state those widths. Ablations
report that rank 8 is worse than the default, so 16 is a reasonable default,
not a number taken from the paper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .rules import gsa2_scan, silu

_INIT_STD = 0.02
_L2_EPS = 1e-6
_RMS_EPS = 1e-6


def rms_norm(x, eps=_RMS_EPS):
    """Parameter-free RMSNorm over the last axis."""
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps)


def l2_normalize(x, eps=_L2_EPS):
    """L2-normalize over the last axis."""
    sq = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(sq + eps)


def depthwise_causal_conv(x, kernel):
    """Depthwise causal convolution.

    x: (..., T, C), kernel: (K, C).
    Position t sees only tokens t-K+1 through t, with zeros before the start.
    kernel[0] multiplies the oldest visible token, kernel[-1] the current one.
    """
    K = kernel.shape[0]
    T = x.shape[-2]
    pad_width = [(0, 0)] * (x.ndim - 2) + [(K - 1, 0), (0, 0)]
    x_pad = jnp.pad(x, pad_width)
    acc = None
    for i in range(K):
        term = x_pad[..., i : i + T, :] * kernel[i]
        acc = term if acc is None else acc + term
    return acc


def _linear(key, in_dim, out_dim):
    return jax.random.normal(key, (in_dim, out_dim)) * _INIT_STD


def _conv(key, channels, width):
    return jax.random.normal(key, (width, channels)) * _INIT_STD


def init_block(key, d_model, n_slots, d_k, d_v, rank=16, conv_kernel=4):
    """Initialize a GSA2 block. Returns a pytree of arrays, std 0.02."""
    keys = jax.random.split(key, 17)
    return {
        "W_q": _linear(keys[0], d_model, d_k),
        "W_k": _linear(keys[1], d_model, d_k),
        "W_v": _linear(keys[2], d_model, d_v),
        "W_w1": _linear(keys[3], d_model, rank),
        "W_w2": _linear(keys[4], rank, n_slots),
        "conv_q": _conv(keys[5], d_k, conv_kernel),
        "conv_k": _conv(keys[6], d_k, conv_kernel),
        "conv_v": _conv(keys[7], d_v, conv_kernel),
        "conv_w": _conv(keys[8], n_slots, conv_kernel),
        "W_alpha1": _linear(keys[9], d_model, n_slots),
        "W_alpha2": _linear(keys[10], d_model, n_slots),
        "W_b1": _linear(keys[11], d_model, n_slots),
        "W_b2": _linear(keys[12], d_model, n_slots),
        "W_c1": _linear(keys[13], d_model, d_k),
        "W_c2": _linear(keys[14], d_model, d_v),
        "W_out_gate": _linear(keys[15], d_model, d_v),
        "W_out": _linear(keys[16], d_v, d_model),
    }


def _scan_leading(w, b1, b2, alpha1, alpha2, c1, c2, k, v, q):
    """Run gsa2_scan over the time axis, vmapped across leading batch axes."""
    leading = w.shape[:-2]
    T = w.shape[-2]

    def flat(a):
        return a.reshape((-1, T) + a.shape[-1:])

    outs, _, _ = jax.vmap(gsa2_scan)(
        flat(w),
        flat(b1),
        flat(b2),
        flat(alpha1),
        flat(alpha2),
        flat(c1),
        flat(c2),
        flat(k),
        flat(v),
        flat(q),
    )
    return outs.reshape(leading + (T, outs.shape[-1]))


def apply_block(params, x):
    """Apply one GSA2 residual block.

    x: (..., T, d_model) -> y of the same shape.
    The residual is the original input, not the pre-norm activation.
    """
    residual = x
    h = rms_norm(x)

    q = depthwise_causal_conv(h @ params["W_q"], params["conv_q"])
    k = depthwise_causal_conv(h @ params["W_k"], params["conv_k"])
    v = depthwise_causal_conv(h @ params["W_v"], params["conv_v"])
    w = depthwise_causal_conv((h @ params["W_w1"]) @ params["W_w2"], params["conv_w"])

    q = l2_normalize(silu(q))
    k = l2_normalize(silu(k))
    v = silu(v)
    w = l2_normalize(silu(w))

    alpha1 = jax.nn.sigmoid(h @ params["W_alpha1"])
    alpha2 = jax.nn.sigmoid(h @ params["W_alpha2"])
    b1 = jax.nn.sigmoid(h @ params["W_b1"])
    b2 = jax.nn.sigmoid(h @ params["W_b2"])
    c1 = jax.nn.sigmoid(h @ params["W_c1"])
    c2 = jax.nn.sigmoid(h @ params["W_c2"])

    o = _scan_leading(w, b1, b2, alpha1, alpha2, c1, c2, k, v, q)
    gate = jax.nn.sigmoid(h @ params["W_out_gate"])
    y = rms_norm(o) * gate
    y = y @ params["W_out"]
    return residual + y


class GSA2Block:
    """GSA2 token mixer. Parameters are a pytree, not an nn.Module."""

    def __init__(self, d_model, n_slots, d_k, d_v, rank=16, conv_kernel=4, key=None):
        if key is None:
            raise ValueError("key is required to initialize GSA2Block")
        self.d_model = d_model
        self.n_slots = n_slots
        self.d_k = d_k
        self.d_v = d_v
        self.rank = rank
        self.conv_kernel = conv_kernel
        self.params = init_block(
            key, d_model, n_slots, d_k, d_v, rank=rank, conv_kernel=conv_kernel
        )

    def __call__(self, x):
        return apply_block(self.params, x)
