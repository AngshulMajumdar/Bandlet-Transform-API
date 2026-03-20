from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

from .config import BandletConfig
from .transform import BandletTransform
from .types import EncodedBandlet


class BandletAnalysisLayer(nn.Module):
    def __init__(self, config: BandletConfig | None = None):
        super().__init__()
        self.transform = BandletTransform(config)

    def forward(self, x) -> EncodedBandlet:
        return self.transform.encode(x)


class BandletSynthesisLayer(nn.Module):
    def __init__(self, config: BandletConfig | None = None):
        super().__init__()
        self.transform = BandletTransform(config)

    def forward(self, enc: EncodedBandlet) -> torch.Tensor:
        return self.transform.reconstruct(enc)


class BandletPackedLayer(nn.Module):
    """Model-friendly layer returning a flat coefficient tensor and metadata."""

    def __init__(self, config: BandletConfig | None = None):
        super().__init__()
        self.transform = BandletTransform(config)

    def forward(self, x) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.transform.encode_packed(x)


class BandletDenoiseLayer(nn.Module):
    def __init__(self, tau: float, config: BandletConfig | None = None):
        super().__init__()
        self.transform = BandletTransform(config)
        self.tau = float(tau)

    def forward(self, x) -> torch.Tensor:
        return self.transform.denoise(x, tau=self.tau)
