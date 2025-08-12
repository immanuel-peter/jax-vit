# mini_jax_vit.py

"""
Tiny ViT on CIFAR-10 using JAX.
Runs on Apple Silicon GPUs via JAX-Metal.
"""

import os
for _var in ("JAX_BACKEND", "JAX_DEFAULT_DEVICE", "JAX_DEFAULT_BACKEND"):
    if _var in os.environ:
        os.environ.pop(_var, None)
os.environ["JAX_PLATFORM_NAME"] = "METAL"
os.environ["JAX_PLATFORMS"] = "METAL"
os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
jax.config.update("jax_platform_name", "METAL")
jax.config.update("jax_platforms", "METAL")

import jax.numpy as jnp
import numpy as np
from jax import random, jit, value_and_grad
import flax.linen as nn
from flax.training import train_state
import optax
from keras.datasets import cifar10
from einops import rearrange

# ========== DATA ==========

def make_ds_from_keras(split="train", batch_size=256, shuffle=True, limit=None):
    """
    Make a dataset from Keras CIFAR-10.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if split == "train":
        X, y = x_train, y_train
    else:
        X, y = x_test, y_test

    X = X.astype(jnp.float32) / 255.0
    y = y.flatten().astype(jnp.int32)

    if limit is not None:
        X, y = X[:limit], y[:limit]
    
    def iterator():
        idx = np.arange(len(X))
        if shuffle:
            rng = np.random.default_rng(42)
            rng.shuffle(idx)
        for i in range(0, len(X), batch_size):
            j = idx[i:i+batch_size]
            yield X[j], y[j]

    return iterator, len(X)

# ========== MODEL ==========

class MLP(nn.Module):
    """
    Two-layer feedforward network used inside Transformer blocks.
    Expands channel dimension by `mlp_ratio`, applies GELU and dropout, then projects back to `dim` with optional dropout.
    """
    dim: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):
        hidden = int(self.dim * self.mlp_ratio)
        x = nn.Dense(hidden)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, deterministic=not train)(x)
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(self.dropout, deterministic=not train)(x)
        return x

class EncoderBlock(nn.Module):
    """
    A Transformer encoder block with pre-layernorm self-attention followed by an MLP.
    Each sublayer output is added back via residual connections, with configurable heads, dropout, and MLP expansion.
    """
    dim: int
    heads: int
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.dim,
            dropout_rate=self.attn_dropout,
            deterministic=not train,
        )(y)
        y = nn.Dropout(self.drop, deterministic=not train)(y)
        x = x + y

        y = nn.LayerNorm()(y)
        y = MLP(self.dim, self.mlp_ratio, self.drop)(y, train=train)
        return x + y

class PatchEmbed(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.
    Non-overlapping image patches are flattened and projected to `dim` with a linear layer to form token embeddings.
    """
    patch: int
    dim: int
    
    @nn.compact
    def __call__(self, x): # x: [B, H, W, C]
        # Extract non-overlapping patches and flatten: [B, (H/p)*(W/p), p*p*C]
        patches = rearrange(
            x,
            "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
            p1=self.patch,
            p2=self.patch,
        )
        return nn.Dense(self.dim)(patches)


class ViT(nn.Module):
    """
    Minimal Vision Transformer for CIFAR-10 classification.
    Images are patch-embedded, a learnable class token and positional embeddings are added, then several encoder blocks produce a class token used for logits.
    """
    num_classes: int = 10
    img_size: int = 32
    patch: int = 4
    dim: int = 192
    depth: int = 4
    heads: int = 3
    mlp_ratio: float = 4.0
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):
        assert self.img_size % self.patch == 0, "image has to be divisible by patch"
        n = (self.img_size // self.patch) ** 2

        x = PatchEmbed(self.patch, self.dim)(x) # [B, n, dim]

        cls = self.param("cls", nn.initializers.normal(stddev=1e-2), (1, 1, self.dim))
        cls_tok = jnp.tile(cls, (x.shape[0], 1, 1))
        x = jnp.concatenate([cls_tok, x], axis=1)

        pos = self.param("pos", nn.initializers.normal(stddev=1e-2), (1, n + 1, self.dim))
        x = x + pos

        for _ in range(self.depth):
            x = EncoderBlock(self.dim, self.heads, self.mlp_ratio, drop=self.drop)(x, train=train)

        x = nn.LayerNorm()(x)
        cls_token = x[:, 0]
        logits = nn.Dense(self.num_classes)(cls_token)
        return logits

# ========== TRAINING ==========

def cross_entropy_loss(logits, labels):
    """
    Computes mean softmax cross-entropy between predicted logits and ground-truth labels.
    Labels are converted to one-hot vectors and the batch-average loss is returned.
    """
    onehot = jax.nn.one_hot(labels, logits.shape[-1])
    return optax.softmax_cross_entropy(logits, onehot).mean()

def accuracy(logits, labels):
    """
    Computes top-1 accuracy by comparing the argmax class of logits to integer labels.
    Returns the mean fraction of correct predictions over the batch.
    """
    return (logits.argmax(axis=-1) == labels).mean()

def create_state(rng, lr):
    model = ViT()
    dummy = jnp.zeros((1, 32, 32, 3), jnp.float32)
    variables = model.init({"params": rng}, dummy, train=False)
    params = variables["params"]
    tx = optax.adamw(lr, weight_decay=1e-4)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

@jit
def train_step(state, batch, rng):
    """
    Single JIT-compiled training step that runs a forward pass with dropout, computes loss and gradients, and applies an optimizer update.
    Returns the updated state along with the scalar loss and accuracy for logging.
    """
    imgs, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, imgs, train=True, rngs={"dropout": rng})
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = accuracy(logits, labels)
    return state, loss, acc

@jit
def eval_step(state, batch):
    """
    JIT-compiled evaluation step that runs a deterministic forward pass without dropout.
    Returns the mean cross-entropy loss and top-1 accuracy for the batch.
    """
    imgs, labels = batch
    logits = state.apply_fn({"params": state.params}, imgs, train=False)
    return cross_entropy_loss(logits, labels), accuracy(logits, labels)

# ========== MAIN ==========

def main():
    """
    Entry point that prepares CIFAR-10 data, initializes the ViT model and optimizer, and runs a short training loop with per-epoch evaluation.
    Prints backend info and epoch summaries so the run is easy to monitor.
    """
    import time
    from tqdm import tqdm

    bs = 256

    # Keep caps modest so total run is under 2 hours on a MacBook
    train_iter, n_train = make_ds_from_keras("train", batch_size=bs, shuffle=True, limit=20_000)
    test_iter, n_test = make_ds_from_keras("test", batch_size=bs, shuffle=False, limit=None)

    rng = random.PRNGKey(42)
    state = create_state(rng, lr=3e-4)

    print(f"Backend: {jax.default_backend()} | train size: {n_train} | test size: {n_test}")
    
    steps = 0

    for epoch in range(1, 15):
        t0 = time.time()
        train_losses, train_accs = [], []
        n_train_batches = (n_train + bs - 1) // bs
        for imgs, labels in tqdm(train_iter(), total=n_train_batches, desc=f"Epoch {epoch} [Train]"):
            imgs = jnp.asarray(imgs)
            labels = jnp.asarray(labels)
            rng, drop_rng = random.split(rng)
            state, loss, acc = train_step(state, (imgs, labels), drop_rng)
            train_losses.append(float(loss))
            train_accs.append(float(acc))
            steps += 1

        test_losses, test_accs = [], []
        n_test_batches = (n_test + bs - 1) // bs
        for imgs, labels in tqdm(test_iter(), total=n_test_batches, desc=f"Epoch {epoch} [Test]"):
            imgs = jnp.asarray(imgs)
            labels = jnp.asarray(labels)
            loss, acc = eval_step(state, (imgs, labels))
            test_losses.append(float(loss))
            test_accs.append(float(acc))

        print(
            f"epoch {epoch:02d} | "
            f"train loss {np.mean(train_losses):.4f} acc {np.mean(train_accs)*100:.1f}% | "
            f"test loss {np.mean(test_losses):.4f} acc {np.mean(test_accs)*100:.1f}% | "
            f"{time.time()-t0:.1f}s"
        )

if __name__ == "__main__":
    main()