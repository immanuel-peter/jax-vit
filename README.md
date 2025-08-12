## Mini JAX Vision Transformer

A tiny Vision Transformer implemented in JAX + Flax, trained on CIFAR‑10. The whole model and training loop live in a single file: `mini_jax_vit.py`. It is pre‑configured to run on Apple Silicon GPUs via JAX‑Metal.

### Highlights
- **Single file**: minimal, readable reference (`mini_jax_vit.py`)
- **Apple Silicon GPU**: uses Metal backend (M‑series Macs)
- **Standard dataset**: CIFAR‑10 via Keras
- **Modern stack**: Flax Linen, Optax, JIT‑compiled train/eval

### Requirements
- **OS**: macOS on Apple Silicon (M1–M4)
- **Python**: 3.10–3.12 recommended
- **Packages**: pinned in `requirements.txt`

### Setup
```bash
git clone https://github.com/immanuel-peter/jax-vit.git
cd jax-vit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
python3 mini_jax_vit.py
```
You should see a line like:
```
Backend: metal | train size: 20000 | test size: 10000
```
followed by per‑epoch train/test metrics.

### What this script does
- **Data**: loads CIFAR‑10 from Keras; normalizes to [0,1]; uses a simple Python iterator with optional shuffling and a `limit` for quick runs.
- **Model**: minimal ViT (`patch=4`, `dim=192`, `depth=4`, `heads=3`, `mlp_ratio=4.0`).
- **Training**: AdamW (`lr=3e-4`, `weight_decay=1e-4`), batch size 256, JIT‑compiled steps with dropout RNG. Default training samples are capped (`limit=20_000`) to keep runtime modest.

### Change the knobs
Open `mini_jax_vit.py` and tweak these:
- **Batch size**: edit `bs` in `main()`
- **Epochs**: edit the `for epoch in range(1, 15):` loop
- **Learning rate / optimizer**: see `create_state(rng, lr)`
- **Train set size**: change `limit` in `make_ds_from_keras("train", ..., limit=20_000)` to `None` for all 50k
- **Model size**: either change `ViT` defaults or instantiate with args in `create_state`:
```python
# inside create_state(...)
model = ViT(dim=256, depth=6, heads=4, drop=0.1)
```

### Apple Silicon notes
The script explicitly targets Metal:
```python
os.environ["JAX_PLATFORM_NAME"] = "METAL"
os.environ["JAX_PLATFORMS"] = "METAL"
jax.config.update("jax_platform_name", "METAL")
jax.config.update("jax_platforms", "METAL")
```
This ensures GPU use on M‑series Macs with `jax-metal` installed (already in `requirements.txt` for macOS/arm64).

### CPU or non‑macOS
If you want to run on CPU (or on a system without Metal), edit the lines above to use CPU or comment them out. Example CPU configuration:
```python
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_platforms", "cpu")
```

### Expected output
Per‑epoch logs look like:
```
epoch 01 | train loss 1.7xxx acc 4x.x% | test loss 1.6xxx acc 4x.x% | 12.3s
...
epoch 14 | train loss 0.xxxx acc 7x.x% | test loss 0.xxxx acc 7x.x% | 11.8s
```
Exact numbers vary by machine and hyperparameters.

### Project layout
- `mini_jax_vit.py`: model, training loop, and dataset iterator
- `requirements.txt`: pinned dependencies

### Troubleshooting
- **Backend shows CPU**: ensure you’re on Apple Silicon and `jax-metal` is installed (`pip show jax-metal`). Re‑install requirements if needed.
- **Metal backend errors**: upgrade to recent macOS and Xcode CLT; try newer `jax`/`jax-metal` versions; or run on CPU by switching to the CPU config above.
- **CIFAR‑10 download issues**: Keras downloads to `~/.keras/`. If behind a firewall, set `KERAS_HOME` to a writable cache directory.

### Extending
- Swap CIFAR‑10 with your own iterator by adapting `make_ds_from_keras`
- Increase model depth/width and train longer by removing the dataset `limit`
- Add augmentation or Mixup/CutMix upstream of the iterator for stronger baselines
