import torch
from bandlet_tf import BandletConfig, BandletTransform


def _detail_energy(enc):
    e = enc.approx.square().sum()
    for triplet in enc.detail_bands:
        for sub in triplet:
            e = e + sub.packed.coeffs.square().sum()
    return e


def _run(device: str):
    cfg = BandletConfig(levels=2, block_size=8, device=device)
    t = BandletTransform(cfg)
    x = torch.rand(2, 1, 64, 64, device=t.device, dtype=t.dtype)
    enc = t.encode(x)
    ex = x.square().sum()
    ec = _detail_energy(enc)
    rel = (ex - ec).abs() / ex.abs().clamp_min(1e-12)
    assert rel.item() < 1e-5


def test_parseval_cpu():
    _run('cpu')


def test_parseval_cuda():
    if torch.cuda.is_available():
        _run('cuda')
