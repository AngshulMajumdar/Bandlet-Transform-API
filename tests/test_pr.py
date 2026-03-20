import torch
from bandlet_tf import BandletConfig, BandletTransform


def _run(device: str):
    cfg = BandletConfig(levels=2, block_size=8, device=device)
    t = BandletTransform(cfg)
    x = torch.rand(2, 1, 64, 64, device=t.device, dtype=t.dtype)
    enc = t.encode(x)
    xrec = t.reconstruct(enc)
    rel = (x - xrec).norm() / x.norm()
    assert rel.item() < 1e-5


def test_pr_cpu():
    _run('cpu')


def test_pr_cuda():
    if torch.cuda.is_available():
        _run('cuda')
