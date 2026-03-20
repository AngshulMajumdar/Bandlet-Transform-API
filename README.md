# bandlet-tf

GPU-first redundant directional tight-frame bandlet-like transform for PyTorch.

This repository exposes two API layers:

1. `BandletTransform` for direct analysis, synthesis, packing, and denoising.
2. `nn.Module` wrappers for easy integration into VAEs, diffusion pipelines, and tokenizers.

## Install

```bash
pip install -e .
```

## Core usage

```python
import torch
from bandlet_tf import BandletConfig, BandletTransform

T = BandletTransform(BandletConfig(levels=2, block_size=8, device='cuda'))
x = torch.rand(1, 1, 256, 256, device=T.device, dtype=T.dtype)

enc = T.encode(x)
xrec = T.reconstruct(enc)

vec, meta = T.encode_packed(x)
xrec2 = T.decode_packed(vec, meta)

xden = T.denoise(x, tau=0.05)
stats = T.stats(enc)
```

## Module wrappers

```python
import torch
from bandlet_tf import BandletConfig, BandletPackedLayer

layer = BandletPackedLayer(BandletConfig(levels=2, block_size=8, device='cuda'))
x = torch.rand(4, 1, 128, 128, device='cuda')
vec, meta = layer(x)
```

Available wrappers:
- `BandletAnalysisLayer`
- `BandletSynthesisLayer`
- `BandletPackedLayer`
- `BandletDenoiseLayer`

## Accepted input shapes

- `[H, W]`
- `[B, H, W]`
- `[B, 1, H, W]`

## Smoke test

```bash
python examples/smoke_test.py
pytest -q
```

## Notes

- Perfect reconstruction and Parseval-style energy preservation are tested.
- CPU/CUDA parity is tested when CUDA is available.
- The API is designed so a model can work either with structured coefficients (`EncodedBandlet`) or with flat packed vectors.
