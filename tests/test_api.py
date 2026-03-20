import torch
from bandlet_tf import (
    BandletAnalysisLayer,
    BandletConfig,
    BandletDenoiseLayer,
    BandletPackedLayer,
    BandletSynthesisLayer,
    BandletTransform,
)


def test_encode_pack_decode_api():
    t = BandletTransform(BandletConfig(levels=2, block_size=8, device='cpu'))
    x = torch.rand(2, 1, 64, 64)
    vec, meta = t.encode_packed(x)
    xrec = t.decode_packed(vec, meta)
    assert torch.allclose(x, xrec, atol=1e-5, rtol=1e-5)


def test_nn_wrappers():
    cfg = BandletConfig(levels=2, block_size=8, device='cpu')
    x = torch.rand(1, 1, 64, 64)

    analysis = BandletAnalysisLayer(cfg)
    synthesis = BandletSynthesisLayer(cfg)
    packed = BandletPackedLayer(cfg)
    denoise = BandletDenoiseLayer(0.01, cfg)

    enc = analysis(x)
    xrec = synthesis(enc)
    vec, meta = packed(x)
    xden = denoise(x)

    assert xrec.shape == x.shape
    assert vec.ndim == 1
    assert meta['meta']['levels'] == cfg.levels
    assert xden.shape == x.shape
