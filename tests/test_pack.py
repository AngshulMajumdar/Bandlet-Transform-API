import torch
from bandlet_tf import BandletConfig, BandletTransform


def test_pack_unpack_roundtrip():
    t = BandletTransform(BandletConfig(levels=2, block_size=8, device='cpu'))
    x = torch.rand(1, 1, 64, 64)
    enc = t.encode(x)
    vec = t.pack(enc)
    meta = t.export_template_meta(enc)
    enc2 = t.unpack(vec, meta)
    vec2 = t.pack(enc2)
    assert torch.allclose(vec, vec2)
    xrec = t.reconstruct(enc2)
    assert torch.allclose(x, xrec, atol=1e-5, rtol=1e-5)
