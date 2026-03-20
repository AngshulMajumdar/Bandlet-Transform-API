from __future__ import annotations
from typing import Any, Dict, List, Literal, Tuple
import copy
import torch
import torch.nn as nn

from .blocks import assemble_blocks_2d, crop_to_shape, extract_blocks_2d, pad_to_multiple
from .config import BandletConfig
from .directional_ops import analyze_blocks, soft_threshold_packed, synthesize_blocks_with_spec
from .haar import HaarLevel, dwt2_haar, idwt2_haar
from .packing import export_template_meta, pack_encoded, unpack_encoded
from .stats import encoded_stats
from .types import EncodedBandlet, EncodedSubband


class BandletTransform(nn.Module):
    """GPU-first directional tight-frame transform with a model-friendly API.

    Main entry points:
        - encode(x) / analysis(x)
        - reconstruct(enc) / synthesis(enc)
        - pack(enc), unpack(vec, meta)
        - encode_packed(x), decode_packed(vec, meta)
        - denoise(x, tau)

    Input shapes accepted:
        [H, W], [B, H, W], [B, 1, H, W]
    """

    def __init__(self, config: BandletConfig | None = None):
        super().__init__()
        self.config = config or BandletConfig()
        self.device = self._resolve_device(self.config.device)
        self.dtype = getattr(torch, self.config.dtype)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def extra_repr(self) -> str:
        return (
            f'levels={self.config.levels}, block_size={self.config.block_size}, '
            f'angles={len(self.config.angles)}, device={self.device}, dtype={self.dtype}'
        )

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            out = x.to(device=self.device, dtype=self.dtype)
        else:
            out = torch.tensor(x, device=self.device, dtype=self.dtype)
        if out.ndim == 2:
            out = out.unsqueeze(0).unsqueeze(0)
        elif out.ndim == 3:
            out = out.unsqueeze(1)
        if out.ndim != 4 or out.shape[1] != 1:
            raise ValueError(f'Expected image with shape [H,W], [B,H,W], or [B,1,H,W], got {tuple(out.shape)}')
        if torch.is_floating_point(out) and out.numel() > 0 and out.max() > 1.5:
            out = out / 255.0
        return out.contiguous()

    def _pad_image_for_haar(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        mult = 2 ** self.config.levels
        x_pad, _ = pad_to_multiple(x, mult, mult, mode=self.config.pad_mode_image)
        return x_pad, x.shape[-2:]

    def _encode_subband(self, band: torch.Tensor, level: int, subband_name: str) -> EncodedSubband:
        blocks, orig_shape, padded_shape, nh, nw = extract_blocks_2d(band, self.config.block_size)
        packed = analyze_blocks(blocks, self.config.angles)
        return EncodedSubband(
            level=level,
            subband=subband_name,
            orig_shape=orig_shape,
            padded_shape=padded_shape,
            num_blocks_h=nh,
            num_blocks_w=nw,
            block_size=self.config.block_size,
            num_angles=len(self.config.angles),
            packed=packed,
        )

    def _decode_subband(self, encoded: EncodedSubband) -> torch.Tensor:
        rec_blocks = synthesize_blocks_with_spec(encoded.packed, encoded.block_size, self.config.angles)
        band_pad = assemble_blocks_2d(
            rec_blocks,
            padded_shape=encoded.padded_shape,
            num_blocks_hw=(encoded.num_blocks_h, encoded.num_blocks_w),
            block_size=encoded.block_size,
        )
        return crop_to_shape(band_pad, encoded.orig_shape)

    def encode(self, x) -> EncodedBandlet:
        x = self._to_tensor(x)
        x_pad, orig_image_shape = self._pad_image_for_haar(x)
        approx, coeffs = dwt2_haar(x_pad, levels=self.config.levels)
        detail_bands: List[Tuple[EncodedSubband, EncodedSubband, EncodedSubband]] = []
        for i, level in enumerate(coeffs, start=1):
            detail_bands.append((
                self._encode_subband(level.lh, i, 'LH'),
                self._encode_subband(level.hl, i, 'HL'),
                self._encode_subband(level.hh, i, 'HH'),
            ))
        meta: Dict[str, Any] = {
            'orig_image_shape': tuple(orig_image_shape),
            'padded_image_shape': tuple(x_pad.shape[-2:]),
            'angles': tuple(self.config.angles),
            'block_size': self.config.block_size,
            'levels': self.config.levels,
        }
        return EncodedBandlet(approx=approx, detail_bands=detail_bands, meta=meta)

    def analysis(self, x) -> EncodedBandlet:
        return self.encode(x)

    def reconstruct(self, enc: EncodedBandlet) -> torch.Tensor:
        coeffs: List[HaarLevel] = []
        for lh_enc, hl_enc, hh_enc in enc.detail_bands:
            coeffs.append(HaarLevel(
                lh=self._decode_subband(lh_enc),
                hl=self._decode_subband(hl_enc),
                hh=self._decode_subband(hh_enc),
            ))
        x_pad = idwt2_haar(enc.approx, coeffs)
        return crop_to_shape(x_pad, tuple(enc.meta['orig_image_shape']))

    def synthesis(self, enc: EncodedBandlet) -> torch.Tensor:
        return self.reconstruct(enc)

    def threshold(self, enc: EncodedBandlet, tau) -> EncodedBandlet:
        out = copy.deepcopy(enc)
        for li, triplet in enumerate(out.detail_bands):
            new_triplet = []
            for sub in triplet:
                packed = soft_threshold_packed(sub.packed, tau=tau, keep_dc=self.config.keep_dc_on_threshold)
                new_triplet.append(EncodedSubband(
                    level=sub.level,
                    subband=sub.subband,
                    orig_shape=sub.orig_shape,
                    padded_shape=sub.padded_shape,
                    num_blocks_h=sub.num_blocks_h,
                    num_blocks_w=sub.num_blocks_w,
                    block_size=sub.block_size,
                    num_angles=sub.num_angles,
                    packed=packed,
                ))
            out.detail_bands[li] = tuple(new_triplet)
        return out

    def denoise(self, x, tau) -> torch.Tensor:
        enc = self.encode(x)
        enc_thr = self.threshold(enc, tau=tau)
        return self.reconstruct(enc_thr)

    def stats(self, enc: EncodedBandlet) -> dict:
        return encoded_stats(enc)

    def pack(self, enc: EncodedBandlet) -> torch.Tensor:
        return pack_encoded(enc)

    def flatten(self, enc: EncodedBandlet) -> torch.Tensor:
        return self.pack(enc)

    def export_template_meta(self, enc: EncodedBandlet) -> Dict[str, Any]:
        return export_template_meta(enc)

    def unpack(self, vec: torch.Tensor, template_meta: Dict[str, Any]) -> EncodedBandlet:
        enc = unpack_encoded(vec, template_meta, device=vec.device, dtype=vec.dtype)
        refreshed = []
        for triplet in enc.detail_bands:
            new_triplet = []
            for sub in triplet:
                from .directional_spec import get_packed_spec
                spec = get_packed_spec(sub.block_size, sub.block_size, self.config.angles, sub.packed.coeffs.device, sub.packed.coeffs.dtype)
                new_triplet.append(EncodedSubband(
                    level=sub.level,
                    subband=sub.subband,
                    orig_shape=sub.orig_shape,
                    padded_shape=sub.padded_shape,
                    num_blocks_h=sub.num_blocks_h,
                    num_blocks_w=sub.num_blocks_w,
                    block_size=sub.block_size,
                    num_angles=sub.num_angles,
                    packed=type(sub.packed)(
                        coeffs=sub.packed.coeffs,
                        valid_mask=spec.valid_mask,
                        coeff_mask=spec.coeff_mask,
                        line_count=spec.line_count,
                        line_len=spec.line_len,
                        tight_scale=sub.packed.tight_scale,
                    ),
                ))
            refreshed.append(tuple(new_triplet))
        enc.detail_bands = refreshed
        return enc

    def unflatten(self, vec: torch.Tensor, template_meta: Dict[str, Any]) -> EncodedBandlet:
        return self.unpack(vec, template_meta)

    def encode_packed(self, x) -> Tuple[torch.Tensor, Dict[str, Any]]:
        enc = self.encode(x)
        return self.pack(enc), self.export_template_meta(enc)

    def decode_packed(self, vec: torch.Tensor, template_meta: Dict[str, Any]) -> torch.Tensor:
        return self.reconstruct(self.unpack(vec, template_meta))

    def coeff_shapes(self, enc: EncodedBandlet) -> Dict[str, Any]:
        return {
            'approx': tuple(enc.approx.shape),
            'detail': [
                {
                    sub.subband: tuple(sub.packed.coeffs.shape)
                    for sub in triplet
                }
                for triplet in enc.detail_bands
            ],
        }

    def forward(
        self,
        x,
        mode: Literal['encode', 'pack', 'reconstruct', 'denoise'] = 'encode',
        tau=None,
        template_meta: Dict[str, Any] | None = None,
    ):
        if mode == 'encode':
            return self.encode(x)
        if mode == 'pack':
            enc = self.encode(x)
            return self.pack(enc), self.export_template_meta(enc)
        if mode == 'reconstruct':
            if isinstance(x, EncodedBandlet):
                return self.reconstruct(x)
            if template_meta is not None:
                return self.decode_packed(x, template_meta)
            return self.reconstruct(self.encode(x))
        if mode == 'denoise':
            if tau is None:
                raise ValueError('tau must be provided when mode="denoise".')
            return self.denoise(x, tau=tau)
        raise ValueError(f'Unknown mode: {mode}')
