from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple
import torch


@dataclass
class PackedDirectionalCoeffs:
    coeffs: torch.Tensor          # [B, N, K, G, L]
    valid_mask: torch.Tensor      # [K, G, L]
    coeff_mask: torch.Tensor      # [K, G, L]
    line_count: int
    line_len: int
    tight_scale: float

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.coeffs.shape)

    @property
    def device(self) -> torch.device:
        return self.coeffs.device

    @property
    def dtype(self) -> torch.dtype:
        return self.coeffs.dtype

    def clone(self) -> 'PackedDirectionalCoeffs':
        return PackedDirectionalCoeffs(
            coeffs=self.coeffs.clone(),
            valid_mask=self.valid_mask.clone(),
            coeff_mask=self.coeff_mask.clone(),
            line_count=self.line_count,
            line_len=self.line_len,
            tight_scale=self.tight_scale,
        )

    def to(self, device=None, dtype=None) -> 'PackedDirectionalCoeffs':
        return PackedDirectionalCoeffs(
            coeffs=self.coeffs.to(device=device, dtype=dtype),
            valid_mask=self.valid_mask.to(device=device),
            coeff_mask=self.coeff_mask.to(device=device),
            line_count=self.line_count,
            line_len=self.line_len,
            tight_scale=self.tight_scale,
        )


@dataclass
class EncodedSubband:
    level: int
    subband: str
    orig_shape: Tuple[int, int]
    padded_shape: Tuple[int, int]
    num_blocks_h: int
    num_blocks_w: int
    block_size: int
    num_angles: int
    packed: PackedDirectionalCoeffs

    @property
    def coeff_shape(self) -> Tuple[int, ...]:
        return tuple(self.packed.coeffs.shape)

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def dtype(self) -> torch.dtype:
        return self.packed.dtype

    def clone(self) -> 'EncodedSubband':
        return replace(self, packed=self.packed.clone())

    def to(self, device=None, dtype=None) -> 'EncodedSubband':
        return replace(self, packed=self.packed.to(device=device, dtype=dtype))


@dataclass
class EncodedBandlet:
    approx: torch.Tensor
    detail_bands: List[Tuple[EncodedSubband, EncodedSubband, EncodedSubband]]
    meta: Dict[str, Any]

    @property
    def device(self) -> torch.device:
        return self.approx.device

    @property
    def dtype(self) -> torch.dtype:
        return self.approx.dtype

    @property
    def image_shape(self) -> Tuple[int, int]:
        return tuple(self.meta.get('orig_image_shape', ()))

    def clone(self) -> 'EncodedBandlet':
        return EncodedBandlet(
            approx=self.approx.clone(),
            detail_bands=[tuple(sub.clone() for sub in triplet) for triplet in self.detail_bands],
            meta=dict(self.meta),
        )

    def to(self, device=None, dtype=None) -> 'EncodedBandlet':
        return EncodedBandlet(
            approx=self.approx.to(device=device, dtype=dtype),
            detail_bands=[tuple(sub.to(device=device, dtype=dtype) for sub in triplet) for triplet in self.detail_bands],
            meta=dict(self.meta),
        )
