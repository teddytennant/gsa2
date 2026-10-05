"""Toy associative-recall smoke for one GSA2 block.

Not the 1.3B language-model experiment. A delayed copy (token t-3) is enough
to require memory. Forty steps of SGD, CPU only.
"""

import os
import sys
from pathlib import Path

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax
import jax.numpy as jnp

from gsa2 import apply_block, init_block

VOCAB = 16
SEQ_LEN = 32
D_MODEL = 32
N_SLOTS = 8
D_K = 16
D_V = 16
RANK = 8
LR = 0.05
STEPS = 40
BATCH = 32
DELAY = 3


def init_params(key):
    k_embed, k_block, k_head = jax.random.split(key, 3)
    return {
        "embed": jax.random.normal(k_embed, (VOCAB, D_MODEL)) * 0.02,
        "block": init_block(
            k_block, D_MODEL, N_SLOTS, D_K, D_V, rank=RANK, conv_kernel=4
        ),
        # Block weights stay at std 0.02. The head scale is larger so forty
        # steps at lr 0.05 can move the readout off a flat softmax.
        "head": jax.random.normal(k_head, (D_MODEL, VOCAB)) * 2.0,
    }


def make_batch(key):
    tokens = jax.random.randint(key, (BATCH, SEQ_LEN), 0, VOCAB)
    pad = jnp.zeros((BATCH, DELAY), dtype=jnp.int32)
    targets = jnp.concatenate([pad, tokens[:, :-DELAY]], axis=1)
    return tokens, targets


def loss_fn(params, tokens, targets):
    x = params["embed"][tokens]
    y = apply_block(params["block"], x)
    logits = y @ params["head"]
    log_probs = jax.nn.log_softmax(logits)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
    return jnp.mean(nll)


@jax.jit
def sgd_step(params, tokens, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets)
    params = jax.tree.map(lambda p, g: p - LR * g, params, grads)
    return params, loss


def main():
    key = jax.random.PRNGKey(1)
    key, k_init, k_data = jax.random.split(key, 3)
    params = init_params(k_init)
    tokens, targets = make_batch(k_data)

    first = float(loss_fn(params, tokens, targets))
    print(f"step 0 loss {first:.6f}", flush=True)
    last = first
    for step in range(1, STEPS + 1):
        params, loss = sgd_step(params, tokens, targets)
        last = float(loss)
        if step % 10 == 0:
            print(f"step {step} loss {last:.6f}", flush=True)

    if last < first:
        raise SystemExit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
